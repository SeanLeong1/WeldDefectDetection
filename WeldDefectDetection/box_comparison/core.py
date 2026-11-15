from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.temporal_window_dataset import TemporalWindowDataset
from src.utils.image_preprocessing import filter_small_components


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def median_reference(context):
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3 or len(context) != 4 or not np.isfinite(context).all():
        raise ValueError('Expected four finite, target-excluded neighbour crops.')
    return np.median(context, axis=0).astype(np.float32)


def model_input(target, reference, hybrid):
    target, reference = np.asarray(target), np.asarray(reference)
    if target.shape != reference.shape or target.ndim != 2:
        raise ValueError('Target and reference must have matching two-dimensional shapes.')
    return np.stack([target, reference, np.abs(target-reference)] if hybrid else [target]).astype(np.float32)


def boxes_from_mask(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), 8)
    return [[int(x), int(y), int(x+w), int(y+h)] for x, y, w, h, area in stats[1:n] if area]


def truth_box(mask):
    """Boxes around all 8-connected labelled regions, with no truth-area filter.

    These are region annotations, not physical instance identities: adjacent
    injections can merge and a disconnected injection can yield several boxes.
    """
    return boxes_from_mask(np.asarray(mask) > 0)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_boxes(predicted, truth, threshold=.5):
    if not predicted or not truth:
        return dict(tp=0, fp=len(predicted), fn=len(truth), matched_iou_sum=0.)
    ious = np.array([[box_iou(p, t) for t in truth] for p in predicted])
    eligible = ious >= threshold
    # Maximise match count first, then total IoU. Each prediction/instance matches once.
    weight = eligible * (min(len(predicted), len(truth)) + 1 + ious)
    pi, ti = linear_sum_assignment(-weight)
    selected = [(p, t) for p, t in zip(pi, ti) if eligible[p, t]]
    tp = len(selected)
    return dict(tp=tp, fp=len(predicted)-tp, fn=len(truth)-tp,
                matched_iou_sum=float(sum(ious[p, t] for p, t in selected)))


def threshold_mask(score, threshold, min_area, opening=1):
    mask = (score >= threshold).astype(np.uint8)
    if opening > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8))
    return filter_small_components(mask, min_area).astype(np.uint8)


def morphology_score(image, kernel_size, polarity='both'):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = np.asarray(image, dtype=np.float32)
    dark = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    if polarity == 'dark':
        return dark
    bright = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    if polarity == 'bright':
        return bright
    if polarity != 'both':
        raise ValueError('Unknown morphology polarity.')
    return np.maximum(dark, bright)


def load_windows(root):
    ds = TemporalWindowDataset(root, 'all')
    targets, references, masks, meta = [], [], [], []
    for i in range(len(ds)):
        item = ds[i]
        window = ds.windows[i]
        record = ds.record_map[window['target_sample_id']]
        physical = record['realized_non_noise_parameters']
        if abs(physical['beam']['e0_kev']-60) > 1e-6 or abs(record['pixel_size_mm']-.05) > 1e-8:
            raise ValueError('Benchmark requires 60 keV and 0.05 mm pixels.')
        layout = record['renderer_parameters']['weld_layout']
        if (layout['mode'] != 'straight_seam' or abs(layout['seam_rotation_deg']) > 1e-6):
            raise ValueError('This protocol requires straight, row-aligned seam crops. Circular registration is a separate experiment.')
        context_ids = window['previous_sample_ids'] + window['future_sample_ids']
        for sid in context_ids:
            neighbour = ds.record_map[sid]
            if neighbour['crop_y0_px'] != record['crop_y0_px'] or neighbour['pixel_size_mm'] != record['pixel_size_mm']:
                raise ValueError('Context must have matching cross-seam coordinates and spacing.')
        target = item['target'][0].numpy()
        mask = item['mask'][0].numpy().astype(np.uint8)
        reference = median_reference(item['context'].numpy())
        if not np.isfinite(target).all() or target.shape != mask.shape:
            raise ValueError('Nonfinite image or misaligned mask.')
        targets.append(target); references.append(reference); masks.append(mask)
        meta.append(dict(window_id=item['window_id'], sequence_id=item['sequence_id'], split=item['split'],
            source_class=item['planned_class'], sample_id=window['target_sample_id'],
            context_sample_ids=context_ids, crop_x0_px=record['crop_x0_px'], crop_y0_px=record['crop_y0_px'],
            pixel_size_mm=record['pixel_size_mm'], truth_boxes=truth_box(mask),
            context_defective_count=sum(bool(ds.record_map[s]['is_defective']) for s in context_ids)))
    return dict(target=np.stack(targets), reference=np.stack(references), mask=np.stack(masks), meta=meta)


def subset(data, split):
    ids = [i for i, m in enumerate(data['meta']) if split == 'all' or m['split'] == split]
    return {k: [v[i] for i in ids] if k == 'meta' else v[ids] for k, v in data.items()}


def evaluate_scores(data, scores, parameters):
    result = []
    if len(scores) != len(data['meta']):
        raise ValueError('Score count must match the target population.')
    for meta, truth, score in zip(data['meta'], data['mask'], scores):
        predicted_mask = threshold_mask(score, parameters['threshold'], parameters['min_area'], parameters.get('opening', 1))
        boxes = boxes_from_mask(predicted_mask)
        box_counts = match_boxes(boxes, meta['truth_boxes'])
        truth, pred = truth > 0, predicted_mask > 0
        result.append(dict(**meta, boxes_xyxy=boxes, **box_counts,
            pixel_tp=int(np.sum(truth & pred)), pixel_fp=int(np.sum(~truth & pred)), pixel_fn=int(np.sum(truth & ~pred)),
            target_positive=bool(truth.any()), predicted_positive=bool(boxes)))
    return result


def summarize(rows):
    totals = {k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'pixel_tp', 'pixel_fp', 'pixel_fn')}
    tp, fp, fn = [totals[k] for k in ('tp', 'fp', 'fn')]
    pt, pf, pn = [totals[k] for k in ('pixel_tp', 'pixel_fp', 'pixel_fn')]
    positive = [r for r in rows if r['target_positive']]
    negative = [r for r in rows if not r['target_positive']]
    sensitivity = sum(r['predicted_positive'] for r in positive)/len(positive) if positive else None
    specificity = sum(not r['predicted_positive'] for r in negative)/len(negative) if negative else None
    return dict(**totals, targets=len(rows), sequences=len({r['sequence_id'] for r in rows}),
        box_precision=tp/max(tp+fp, 1), box_recall=tp/max(tp+fn, 1), box_f1=2*tp/max(2*tp+fp+fn, 1),
        matched_box_mean_iou=sum(r['matched_iou_sum'] for r in rows)/max(tp, 1),
        pixel_iou=pt/max(pt+pf+pn, 1), pixel_f1=2*pt/max(2*pt+pf+pn, 1),
        image_sensitivity=sensitivity, image_specificity=specificity,
        balanced_accuracy=(sensitivity+specificity)/2 if positive and negative else None,
        false_positive_boxes_per_target=fp/max(len(rows), 1),
        clean_targets=len(negative), clean_false_alarms=sum(r['predicted_positive'] for r in negative))


def selection_key(summary):
    return (summary['box_f1'], summary['pixel_iou'], -summary['fp'])


def select_threshold(data, scores, thresholds, min_areas=(3, 8, 15), openings=(1,)):
    if any(m['split'] != 'val' for m in data['meta']):
        raise ValueError('Parameter selection is restricted to development validation windows.')
    best, candidates = None, []
    for threshold in thresholds:
        for min_area in min_areas:
            for opening in openings:
                parameters = dict(threshold=float(threshold), min_area=int(min_area), opening=int(opening))
                summary = summarize(evaluate_scores(data, scores, parameters))
                record = dict(parameters=parameters, metrics=summary)
                candidates.append(record)
                if best is None or selection_key(summary) > selection_key(best['metrics']):
                    best = record
    return best, candidates
