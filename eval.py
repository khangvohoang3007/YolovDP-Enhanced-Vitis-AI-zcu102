import json
import os
import numpy as np
from scipy.integrate import trapezoid

# ================= CONFIGURATION =================
LABEL_DIR = r'dataset\dataset\LL\labels'
PRED_JSON = r'out_results.txt' 
IMG_SIZE = 640
# =================================================

def box_iou(box1, box2):
    i_x1, i_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    i_x2, i_y2 = min(box1[0]+box1[2], box2[0]+box2[2]), min(box1[1]+box1[3], box2[1]+box2[3])
    inter = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return inter / union if union > 0 else 0

def calculate_ap(tp_fp_list, total_gt):
    if not tp_fp_list or total_gt == 0: return 0
    tp_fp_list.sort(key=lambda x: x[0], reverse=True)
    tps = np.array([x[1] for x in tp_fp_list])
    fps = 1 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / total_gt
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    m_pre = np.concatenate(([1.0], precisions, [0.0]))
    m_pre = np.maximum.accumulate(m_pre[::-1])[::-1]
    return trapezoid(m_pre, m_rec)

def run_eval():
    if not os.path.exists(PRED_JSON):
        print(f"❌ Không tìm thấy file dự đoán: {PRED_JSON}")
        return

    with open(PRED_JSON, 'r') as f:
        predictions = json.load(f)
    
    preds_by_id = {}
    for p in predictions:
        iid = p['image_id']
        if iid not in preds_by_id: preds_by_id[iid] = []
        preds_by_id[iid].append(p)

    all_gt = {}
    total_gt_all = 0
    # Lấy max ID từ dữ liệu thực tế
    max_id = max(preds_by_id.keys()) if preds_by_id else 0
    
    print(f"📂 Đang load nhãn từ {LABEL_DIR}...")
    for iid in range(max_id + 1):
        lbl_path = os.path.join(LABEL_DIR, f"{iid:04d}.txt")
        gt_boxes = []
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            data = np.atleast_2d(np.genfromtxt(lbl_path))
            for l in data:
                if len(l) == 5:
                    w, h = l[3] * IMG_SIZE, l[4] * IMG_SIZE
                    xmin = (l[1] * IMG_SIZE) - (w / 2)
                    ymin = (l[2] * IMG_SIZE) - (h / 2)
                    gt_boxes.append([int(l[0]), xmin, ymin, w, h])
                    total_gt_all += 1
        all_gt[iid] = gt_boxes

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    p_05, r_05 = 0, 0
    
    print(f"🚀 Đang tính toán mAP cho {max_id + 1} ảnh (Tổng GT: {total_gt_all})...")

    for iou_thr in iou_thresholds:
        all_tp_fp = []
        for iid in range(max_id + 1):
            gt_boxes = all_gt.get(iid, [])
            ########################## SỬA Ở ĐÂY
            #img_preds = sorted(preds_by_id.get(iid, []), key=lambda x: x['score'], reverse=True)
            img_preds = [p for p in preds_by_id.get(iid, []) if p['score'] >= 0.4]
            img_preds = sorted(img_preds, key=lambda x: x['score'], reverse=True)

            matched_gt = set()
            for p in img_preds:
                best_iou = -1
                best_idx = -1
                
                # Duyệt qua các GT của ảnh này để tìm cái khớp nhất
                for idx, g in enumerate(gt_boxes):
                    if idx in matched_gt: continue
                    iou = box_iou(p['bbox'], g[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_iou >= iou_thr:
                    all_tp_fp.append((p['score'], 1))
                    matched_gt.add(best_idx)
                else:
                    all_tp_fp.append((p['score'], 0))
        
        ap = calculate_ap(all_tp_fp, total_gt_all)
        aps.append(ap)
        
        # Lưu P, R tại ngưỡng 0.5
        if abs(iou_thr - 0.5) < 1e-7:
            tps_list = [x[1] for x in all_tp_fp]
            tp_sum = sum(tps_list)
            p_05 = tp_sum / len(all_tp_fp) if all_tp_fp else 0
            r_05 = tp_sum / total_gt_all if total_gt_all else 0

    print("\n" + "="*45)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("-" * 45)
    print(f"Precision (@0.5):  {p_05:.4f}")
    print(f"Recall (@0.5):     {r_05:.4f}")
    print(f"mAP @0.5:          {aps[0]:.4f}")
    print(f"mAP @0.5:0.95:     {np.mean(aps):.4f}")
    print("="*45)

if __name__ == "__main__":
    run_eval()