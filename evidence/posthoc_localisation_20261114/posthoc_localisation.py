"""Post-hoc diagnostics on the frozen final-comparison outputs (sets A and B).

Reads saved per-scene predicted and reference boxes; no detector is re-run.
1. Reproduce Table 5.2 counts at box IoU >= 0.5.
2. Re-score with looser matching criteria (IoU >= 0.3, >= 0.1, any overlap).
3. Recall by nominal pore-diameter bin and by radial position.
4. Classify median misses: matched / overlap-only / no overlap, by size.
"""
import ast
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = '/mnt/user-data/uploads/Codebase2/DataSimulation/data/output/'
SETS = {
    'A': dict(median=BASE + 'median_localisation_20261110_v1/confirmation/combined.json',
              cnn=BASE + 'median_localisation_20261110_v1/confirmation/cnn.json',
              index=BASE + 'median_localisation_20261110_v1/datasets/test/index.json'),
    'B': dict(median=BASE + 'median_region_growth_20261110_v1/confirmation/previous.json',
              cnn=BASE + 'median_region_growth_20261110_v1/confirmation/cnn.json',
              index=BASE + 'median_region_growth_20261110_v1/datasets/test/index.json'),
}
PERIOD = 5024
BINS = [(0.15, 0.30), (0.30, 0.60), (0.60, 1.00)]


def circular_iou(a, b, period=PERIOD):
    """Same definition as circular_study/metrics.py (evidence E26)."""
    vertical = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    horizontal = sum(max(0, min(a[2], b[2] + s) - max(a[0], b[0] + s)) for s in (-period, 0, period))
    inter = vertical * horizontal
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match(pred, truth, threshold):
    """One-to-one matching that maximises qualifying pairs (as in E26).

    threshold=None means any positive overlap qualifies."""
    if not pred or not truth:
        return 0, len(pred), len(truth), [False] * len(truth)
    ious = np.array([[circular_iou(p, t) for t in truth] for p in pred])
    ok = ious > 0 if threshold is None else ious >= threshold
    ii, jj = linear_sum_assignment(-(ok * (min(len(pred), len(truth)) + 1 + ious)))
    pairs = [(i, j) for i, j in zip(ii, jj) if ok[i, j]]
    hit = [False] * len(truth)
    for _, j in pairs:
        hit[j] = True
    return len(pairs), len(pred) - len(pairs), len(truth) - len(pairs), hit


def load():
    scenes = []
    for set_name, paths in SETS.items():
        index = {r['scene_id']: r for r in json.load(open(paths['index']))}
        rows = {m: {r['scene_id']: r for r in json.load(open(paths[m]))['rows']} for m in ('median', 'cnn')}
        for sid, r_med in rows['median'].items():
            r_cnn = rows['cnn'][sid]
            assert r_cnn['truth_boxes'] == r_med['truth_boxes']
            params = index[sid]['parameters']
            params = params if isinstance(params, dict) else ast.literal_eval(params)
            radial = None
            if params.get('defects'):
                radial = params['defects'][0]['params'].get('radial_offset_mm')
            scenes.append(dict(set=set_name, scene=sid, cls=r_med['source_class'],
                               count=int(r_med['count_multiplier']), diameter=float(r_med['diameter_mm']),
                               radial=radial, truth=r_med['truth_boxes'],
                               pred=dict(median=r_med['predicted_boxes'], cnn=r_cnn['predicted_boxes']),
                               saved=dict(median=(r_med['tp'], r_med['fp'], r_med['fn']),
                                          cnn=(r_cnn['tp'], r_cnn['fp'], r_cnn['fn']))))
    return scenes


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan')


def main():
    scenes = load()
    out = {}
    # 1. Reproduce saved counts exactly at IoU >= 0.5.
    for s in scenes:
        for m in ('median', 'cnn'):
            tp, fp, fn, _ = match(s['pred'][m], s['truth'], 0.5)
            assert (tp, fp, fn) == tuple(s['saved'][m]), (s['scene'], m)
    print('Reproduced saved TP/FP/FN for all', len(scenes), 'scenes')

    # 2. Matching-criterion sensitivity.
    crit = {'0.5': 0.5, '0.3': 0.3, '0.1': 0.1, 'overlap': None}
    sens = defaultdict(dict)
    for name, t in crit.items():
        for m in ('median', 'cnn'):
            for c in (1, 4, 64):
                tot = np.zeros(3, int)
                for s in scenes:
                    if s['count'] == c:
                        tot += match(s['pred'][m], s['truth'], t)[:3]
                sens[name][f'{m}_{c}'] = dict(tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), f1=round(f1(*tot), 3))
    out['criterion'] = sens

    # 3. Recall by size bin and radial position (pore scenes only).
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in scenes:
        if s['cls'] == 'clean':
            continue
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= s['diameter'] <= hi + 1e-9)
        for m in ('median', 'cnn'):
            _, _, _, hit = match(s['pred'][m], s['truth'], 0.5)
            for key in (f'size{b}', f'radial{s["radial"]}'):
                by[(m, s['count'])][key][0] += int(all(hit))
                by[(m, s['count'])][key][1] += 1
    out['recall'] = {f'{m}_{c}': {k: dict(matched=v[0], n=v[1], recall=round(v[0] / v[1], 3)) for k, v in sorted(d.items())}
                     for (m, c), d in sorted(by.items())}

    # 4. Median miss types at 64x by size bin.
    types = defaultdict(lambda: defaultdict(int))
    for s in scenes:
        if s['cls'] == 'clean' or s['count'] != 64:
            continue
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= s['diameter'] <= hi + 1e-9)
        for m in ('median', 'cnn'):
            matched = all(match(s['pred'][m], s['truth'], 0.5)[3])
            overlap = all(match(s['pred'][m], s['truth'], None)[3])
            kind = 'matched' if matched else ('overlap_only' if overlap else 'no_overlap')
            types[m][f'size{b}_{kind}'] += 1
            types[m][f'all_{kind}'] += 1
    out['miss_types_64'] = {m: dict(sorted(v.items())) for m, v in types.items()}
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()


def bootstrap(scenes, threshold, reps=2000, seed=20261114):
    """Paired background bootstrap of CNN minus median F1 (48 backgrounds)."""
    per = defaultdict(lambda: defaultdict(lambda: np.zeros(3, int)))
    for s in scenes:
        bg = s['set'] + s['scene'].split('-')[1]
        for m in ('median', 'cnn'):
            per[(bg, s['count'])][m] += match(s['pred'][m], s['truth'], threshold)[:3]
    bgs = sorted({k[0] for k in per})
    rng = np.random.default_rng(seed)
    res = {}
    for c in (1, 4, 64):
        def delta(sample):
            t = {m: sum(per[(b, c)][m] for b in sample) for m in ('median', 'cnn')}
            return f1(*t['cnn']) - f1(*t['median'])
        point = delta(bgs)
        draws = [delta(rng.choice(bgs, len(bgs), replace=True)) for _ in range(reps)]
        res[c] = (round(point, 3), [round(float(x), 3) for x in np.quantile(draws, [.025, .975])])
    return res


if __name__ == '__main__' and len(sys.argv) > 2:
    sc = load()
    print(len({s['set'] + s['scene'].split('-')[1] for s in sc}), 'backgrounds')
    for name, t in (('0.5', 0.5), ('0.1', 0.1), ('overlap', None)):
        print(name, bootstrap(sc, t))
