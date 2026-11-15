from __future__ import annotations

from typing import List, Tuple

import numpy as np


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


def evaluate_detections(
    predictions: List[Tuple[int, int, int, int]],
    ground_truths: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    tp = 0
    fp = 0
    matched_gt = set()

    for pred_box in predictions:
        best_iou = 0.0
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

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def binary_confusion_counts(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, int]:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: predicted {pred.shape}, target {target.shape}")
    tp = int(np.logical_and(pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def segmentation_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def evaluate_segmentation_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float | int]:
    counts = binary_confusion_counts(pred_mask, target_mask)
    return segmentation_metrics_from_counts(
        tp=counts["tp"],
        tn=counts["tn"],
        fp=counts["fp"],
        fn=counts["fn"],
    )


def aggregate_segmentation_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    total_tp = int(sum(int(row["tp"]) for row in rows))
    total_tn = int(sum(int(row["tn"]) for row in rows))
    total_fp = int(sum(int(row["fp"]) for row in rows))
    total_fn = int(sum(int(row["fn"]) for row in rows))
    summary = segmentation_metrics_from_counts(
        tp=total_tp,
        tn=total_tn,
        fp=total_fp,
        fn=total_fn,
    )
    summary["image_count"] = len(rows)
    return summary
