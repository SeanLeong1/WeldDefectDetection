import cv2
import numpy as np
from box_comparison.core import summarize, selection_key, match_boxes
from box_refinement.methods import select, evaluate, predicted_regions
from box_refinement.training import scores_for as old_scores
from box_refinement.campaign import evaluate_spec as old_evaluate


def aligned_reference(target, context):
    """Correct smooth cross-seam profiles, retaining local neighbour deviations."""
    target_profile = np.median(target, axis=1, keepdims=True)
    aligned = []
    for neighbour in context:
        difference = target_profile - np.median(neighbour, axis=1, keepdims=True)
        correction = cv2.GaussianBlur(difference, (1, 9), sigmaX=0, sigmaY=2).reshape(-1, 1)
        aligned.append(neighbour + correction)
    return np.median(aligned, axis=0).astype(np.float32)


def residual(target, context, arm, background_sigma=8.):
    if arm == 'thresholding':
        # Remove smooth local background after the continuous seam profile.
        centred = target - np.median(target, axis=1, keepdims=True)
        value = centred - cv2.GaussianBlur(centred, (0, 0), background_sigma)
    else:
        value = target - aligned_reference(target, context)
    return cv2.GaussianBlur(value, (0, 0), 1.).astype(np.float32)


def fit_scale(train, arm, background_sigma=8.):
    if any(m['split'] != 'train' for m in train['meta']):
        raise ValueError('Noise scale must be fitted on training data only.')
    ids = [i for i, m in enumerate(train['meta'])
           if not train['mask'][i].any() and m['context_defective_count'] == 0]
    if len(ids) < 10:
        raise ValueError('Too few clean training windows for a variation estimate.')
    values = np.stack([residual(train['target'][i], train['context'][i], arm, background_sigma) for i in ids])
    centre = np.median(values, axis=0)
    scale = 1.4826*np.median(np.abs(values-centre), axis=0)
    scale = np.maximum(.003, cv2.GaussianBlur(scale, (0, 0), 2.)).astype(np.float32)
    return scale, dict(clean_training_windows=len(ids), minimum=float(scale.min()), maximum=float(scale.max()),
                       rule='Smoothed across-clean-training residual MAD, floor 0.003; no label use at inference.')


def simple_scores(data, spec):
    scale = np.asarray(spec['scale'], dtype=np.float32)
    return np.stack([np.abs(residual(t, c, spec['arm'], spec['background_sigma']))/scale
                     for t, c in zip(data['target'], data['context'])])


def neural_inputs(data, variant, scale=None):
    target = data['target']
    if variant == 'cnn':
        return target[:, None].astype(np.float32)
    if variant == 'cnn_median':
        reference = data['reference']
        return np.stack([target, reference, np.abs(target-reference)], axis=1).astype(np.float32)
    if variant != 'context_scaled':
        raise ValueError('Unknown input variant.')
    reference = np.stack([aligned_reference(t, c) for t, c in zip(target, data['context'])])
    scaled = np.stack([np.abs(residual(t, c, 'median')) for t, c in zip(target, data['context'])])
    scaled = np.clip(scaled/np.asarray(scale)/6., 0, 1)
    return np.stack([target, reference, np.abs(target-reference), scaled], axis=1).astype(np.float32)


def make_network(channels, rectangle=False):
    import torch
    from src.algorithms.classical.cnn_detector import TinyDefectSegNet
    model = TinyDefectSegNet(in_channels=channels)
    if rectangle:
        model.head = torch.nn.Conv2d(16, 2, kernel_size=1)
    return model


def predict(model, x):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i+32]))).numpy()
                               for i in range(0, len(x), 32)])


def scores_for(data, spec):
    if spec['implementation'] == 'historical':
        return old_scores(data, spec['spec'])[:, None]
    if spec['implementation'] == 'local_variation':
        return simple_scores(data, spec)[:, None]
    import torch
    from box_comparison.core import sha
    if sha(spec['checkpoint']) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch.')
    x = neural_inputs(data, spec['variant'], spec.get('scale'))
    model = make_network(x.shape[1], spec.get('rectangle', False))
    model.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)['state_dict'])
    return predict(model, x)


def evaluate_new(data, scores, parameters, box_channel=0):
    rows = evaluate(data, scores[:, 0], parameters)
    if box_channel:
        for row, score in zip(rows, scores[:, box_channel]):
            _, boxes = predicted_regions(score, **parameters)
            row.update(boxes_xyxy=boxes, predicted_positive=bool(boxes), **match_boxes(boxes, row['truth_boxes']))
    return rows


def evaluate_spec(data, spec):
    if spec['implementation'] == 'historical':
        return old_evaluate(data, spec['spec'])
    return evaluate_new(data, scores_for(data, spec), spec['parameters'], spec.get('box_channel', 0))


def select_scores(val, scores):
    """Small fixed box policy search; rectangle channel cannot inflate mask IoU."""
    candidates = []
    for channel in range(scores.shape[1]):
        _, trials = select(val, scores[:, channel], (.35, .5, .65, .8),
                           min_areas=(3, 15), closings=(1, 3), low_ratios=(1.,), paddings=(0,))
        best_f1 = max(t['metrics']['box_f1'] for t in trials)
        for chosen in trials:
            if chosen['metrics']['box_f1'] != best_f1:
                continue
            # Re-score actual masks for tied box results, including rectangle output.
            rows = evaluate_new(val, scores, chosen['parameters'], channel)
            candidates.append(dict(parameters=chosen['parameters'], metrics=summarize(rows), box_channel=channel))
    return max(candidates, key=lambda c: selection_key(c['metrics']))


def rectangle_targets(masks):
    from box_comparison.core import truth_box
    output = np.zeros_like(masks, dtype=np.float32)
    for out, mask in zip(output, masks):
        for x0, y0, x1, y1 in truth_box(mask):
            out[y0:y1, x0:x1] = 1.
    return output
