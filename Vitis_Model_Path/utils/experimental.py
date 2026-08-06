# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            # Thay vì dùng sigmoid, ta khởi tạo trọng số mặc định là 1.0
            self.w = nn.Parameter(torch.ones(n - 1), requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            # Sử dụng ReLU thay cho Sigmoid để đảm bảo DPU không bị ngắt subgraph
            w = torch.relu(self.w) 
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        n = len(k)
        if equal_ch:  # equal channels per group
            self.m = nn.ModuleList([nn.Conv2d(c1 // n, c2 // n, k[j], s, k[j] // 2, bias=False) for j in range(n)])
        else:  # custom channels
            self.m = nn.ModuleList([nn.Conv2d(c1, c2, k[j], s, k[j] // 2, bias=False) for j in range(n)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Ép kiểu contiguous() giúp DPU đọc địa chỉ RAM liên tục, chống Segfault 100%
        xs = torch.chunk(x, len(self.m), 1)
        return self.act(self.bn(torch.cat([m(xi.contiguous()) for m, xi in zip(self.m, xs)], 1)))

class CrossConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, (1, k), (1, s), (0, k // 2), groups=g, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, (k, 1), (s, 1), (k // 2, 0), groups=g, bias=False)
        self.act = nn.ReLU(inplace=True) 
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.act(self.cv2(self.cv1(x))) if self.add else self.act(self.cv2(self.cv1(x)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # CHỈ dùng torch.cat, tuyệt đối không dùng .mean(0) hoặc .max(0) ở đây
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=False):
    from models.yolo import Detect, Model
    model = nn.ModuleList()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())

    for m in model.modules():
        t = type(m)
        # Ép tất cả về ReLU, loại bỏ các loại activation không được DPU hỗ trợ
        if t in (nn.ReLU, nn.ReLU6, Detect, Model):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    return model[-1] if len(model) == 1 else model
