# src/utils/metrics.py
from typing import Tuple, List

def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Box format: (x_min, y_min, x_max, y_max)
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_detections(predictions: List[Tuple[int, int, int, int]], 
                        ground_truths: List[Tuple[int, int, int, int]], 
                        iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Compares predictions to ground truth to find True Positives (TP), 
    False Positives (FP), and False Negatives (FN).
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Keep track of which ground truths have been matched
    matched_gt = set()
    
    for pred_box in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt_box in enumerate(ground_truths):
            if idx in matched_gt:
                continue
                
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
                
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    # Any ground truth not matched is a False Negative (missed defect)
    fn = len(ground_truths) - len(matched_gt)
    
    return tp, fp, fn