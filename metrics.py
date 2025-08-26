import numpy as np
import torch

def iou(box1, box2):
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    intersection = max(0, x_b - x_a) * max (0, y_b - y_a)
    area_1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area_2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    return intersection / (area_1 + area_2 - intersection + 1e-9)

# 1. Sort all boxes by score descending.
# 2. Select the box with the highest score → add to final list.
# 3. Remove all boxes that overlap (IoU > threshold) with this box.
# 4. Repeat until no boxes left.
def nms_np(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        idxs = idxs[1:]
        idxs = [j for j in idxs if iou(boxes[i], boxes[j])< iou_threshold]
    return keep


def soft_nms(boxes, scores, iou_thresh=0.5, sigma=0.5, method="gaussian"):
    N = len(boxes)
    for i in range(N):
        maxpos = i + np.argmax(scores[i:])
        boxes[i], boxes[maxpos] = boxes[maxpos], boxes[i]
        scores[i], scores[maxpos] = scores[maxpos], scores[i]

        for j in range(i+1, N):
            ov = iou(boxes[i], boxes[j])
            if method == "linear":
                if ov > iou_thresh:
                    scores[j] *= (1 - ov)
            elif method == "gaussian":
                scores[j] *= np.exp(-(ov**2) / sigma)
            else: # hard nms
                if ov > iou_thresh:
                    scores[j] = 0
    keep = [i for i, s in enumerate(scores) if s > 0.001]
    return keep

def eval_detection(preds, gts, iou_thresh=0.5):
    """
    preds: list of [x1, y1, x2, y2, scores]
    gts: list of [x1, y1, x2, y2]
    """
    preds = sorted(preds, key=lambda x: x[-1], reverse=True)
    matched_gt = set()

    tp, fp = 0, 0

    for p in preds:
        best_iou, best_gt = 0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou_eval = iou(p[:4], gt)
            print("iou", iou_eval)
            if iou_eval > best_iou:
                best_iou, best_gt = iou_eval, gi
        
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp +=1
    
    fn = len(gts) - len(matched_gt)

    return tp, fp, fn

def compute_precision_recall_f1(fp, tp, fn):
    precision = tp / (fp + tp + 1e-9)
    recall = tp / (fn + tp + 1e-9)
    f1 = 2*precision*recall / (precision + recall + 1e-9)
    return precision, recall, f1


if __name__ == "__main__":
    # Ground truth boxes
    gts = [
        [50, 50, 150, 150],  # one object
        [200, 200, 300, 300] # another object
    ]

    # Predictions [x1,y1,x2,y2,score]
    preds = [
        [48, 48, 152, 152, 0.95],  # good match (TP)
        [205, 205, 295, 295, 0.90], # good match (TP)
        [60, 60, 140, 140, 0.60],   # duplicate (FP)
        [400, 400, 500, 500, 0.50]  # false alarm (FP)
    ]

    tp, fp, fn = eval_detection(preds, gts, iou_thresh=0.5)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

