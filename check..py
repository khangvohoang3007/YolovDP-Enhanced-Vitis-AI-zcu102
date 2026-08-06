import json, os
import numpy as np

# Thử nghiệm với ảnh 0039.txt bạn vừa đưa
iid = 39 
lbl_path = r'dataset\dataset\LL\labels\0039.txt'
IMG_SIZE = 640

# Load GT
gt_boxes = []
data = np.atleast_2d(np.genfromtxt(lbl_path))
for l in data:
    w, h = l[3] * IMG_SIZE, l[4] * IMG_SIZE
    xmin = (l[1] * IMG_SIZE) - (w / 2)
    ymin = (l[2] * IMG_SIZE) - (h / 2)
    gt_boxes.append([xmin, ymin, w, h])

# Load Pred từ out_results.txt (Tìm các dòng có image_id: 39)
with open(r'out_results.txt', 'r') as f:
    preds = json.load(f)
    img_preds = [p['bbox'] for p in preds if p['image_id'] == iid]

print(f"--- KIỂM TRA ẢNH {iid} ---")
print(f"Số lượng nhãn thật: {len(gt_boxes)}")
print(f"Số lượng dự đoán: {len(img_preds)}")

if len(img_preds) > 0:
    print(f"GT Box 1: {gt_boxes[0]}")
    print(f"Pred Box 1: {img_preds[0]}")
    
    # Tính thử IoU tay
    b1 = gt_boxes[0]
    b2 = img_preds[0]
    i_x1, i_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    i_x2, i_y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
    print(f"Diện tích phần giao: {inter}")