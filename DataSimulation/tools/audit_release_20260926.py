"""Read-only campaign verification and current development diagnostics.

Writes only to the requested audit directory, never reseals a historical campaign.
"""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'DataSimulation'), str(ROOT/'WeldDefectDetection')]
import numpy as np
from inspection.engine import digest, write_json, LEGACY_CAMPAIGN as DEFAULT_CAMPAIGN


def verify_manifest(root):
    inventory = json.loads((root/'FINAL_MANIFEST.json').read_text())['files']
    changed = []
    for filename, expected in inventory.items():
        p = Path(filename)
        if not p.exists() or digest(p) != expected:
            changed.append(filename)
    return dict(root=str(root), checked=len(inventory), changed=changed, passed=not changed)


def coordinate_check():
    from bse_renderer.validation.casino_emission_fit import _fit_entry_frame, _direction_to_angles
    errors = []
    for tilt in (0., 2., 25., 60.):
        for az in (0., 45., 137.):
            a, p = np.radians([tilt, az])
            n = np.array([np.sin(a)*np.cos(p), np.sin(a)*np.sin(p), np.cos(a)])
            t = n*n[2]-np.array([0., 0., 1.])
            t = t/np.linalg.norm(t) if np.linalg.norm(t) > 1e-10 else np.array([1., 0., 0.])
            u = np.cross(n, t)
            points = [t*x+u*y for x, y in [(-1,-1), (-1,1), (1,-1), (1,1), (.3,.5)]]
            frame = _fit_entry_frame(points, [[0.,0.,-1.]]*len(points))
            q = np.array([.4, .3, np.sqrt(.75)])
            phi, azimuth, mu = _direction_to_angles(q, *frame[:3])
            renderer_az = np.arctan2(q@u, q@t)
            # Normal incidence has no identifiable beam-projection azimuth; B sin(alpha)=0.
            err = abs(np.arctan2(np.sin(azimuth-renderer_az), np.cos(azimuth-renderer_az))) if tilt else 0.
            errors.append(dict(tilt_deg=tilt, orientation_deg=az, normal_error=float(np.linalg.norm(n-frame[0])),
                               polar_cosine_error=abs(float(mu-n@q)), azimuth_error_rad=float(err)))
    assert max(r['azimuth_error_rad'] for r in errors) < 1e-10
    assert max(r['normal_error'] for r in errors) < 1e-10
    return errors


def median_diagnostics():
    from variability_campaign.campaign import load
    from box_development.methods import residual
    from box_refinement.methods import predicted_regions
    spec = json.loads((DEFAULT_CAMPAIGN/'frozen_models.json').read_text())['models']['median']
    data = load(DEFAULT_CAMPAIGN, 'development')
    scale = np.array(spec['scale'])
    rows = []
    for split in ('train', 'val'):
        ids = [i for i, m in enumerate(data['meta']) if m['split'] == split and m['source_class'] == 'clean']
        residuals = np.stack([residual(data['target'][i], data['context'][i], 'median', spec['background_sigma']) for i in ids])
        scores = abs(residuals)/scale
        flags = [bool(predicted_regions(s, **spec['parameters'])[1]) for s in scores]
        rows.append(dict(split=split, clean_windows=len(ids), false_alarm_fraction=float(np.mean(flags)),
                         absolute_score_quantiles=np.quantile(scores, [.5,.9,.95,.99]).tolist(),
                         residual_center_absolute_median=float(np.median(abs(np.median(residuals, axis=0)))),
                         scale_quantiles=np.quantile(scale, [0,.5,1]).tolist(), frozen_parameters=spec['parameters']))
    from box_comparison.core import match_boxes, summarize
    ids = [i for i, m in enumerate(data['meta']) if m['split'] == 'val']
    scores = [abs(residual(data['target'][i], data['context'][i], 'median', spec['background_sigma']))/scale for i in ids]
    curve = []
    for threshold in (1.5, 2., 2.5, 3., 4., 5., 6., 8.):
        observations = []
        clean_flags = []
        for i, s in zip(ids, scores):
            mask, boxes = predicted_regions(s, **{**spec['parameters'], 'threshold': threshold})
            truth = data['mask'][i] > 0
            pred = mask > 0
            observations.append(dict(sequence_id=data['meta'][i]['sequence_id'], **match_boxes(boxes, data['meta'][i]['truth_boxes']),
                                     target_positive=bool(truth.any()), predicted_positive=bool(boxes),
                                     pixel_tp=int((pred & truth).sum()), pixel_fp=int((pred & ~truth).sum()),
                                     pixel_fn=int((~pred & truth).sum())))
            if data['meta'][i]['source_class'] == 'clean':
                clean_flags.append(bool(boxes))
        curve.append(dict(threshold=threshold, clean_false_alarm_fraction=float(np.mean(clean_flags)),
                          metrics=summarize(observations)))
    return dict(scope='Development-only diagnosis. No frozen parameters changed. Validation curve is exploratory, not confirmation.',
                rows=rows, validation_operating_curve=curve)


def main():
    out = ROOT/'data/audit_20260926'
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifests = [verify_manifest(DEFAULT_CAMPAIGN), verify_manifest(ROOT/'DataSimulation/data/output/pore_campaign_20260921_v1')]
    protocol = json.loads((DEFAULT_CAMPAIGN/'protocol.json').read_text())
    drift = [p for p, h in protocol['source_hashes'].items() if not Path(p).exists() or digest(p) != h]
    # All frozen source versions remain recoverable from the original source archive.
    sources = [p for base in ('DataSimulation/bse_renderer','WeldDefectDetection/inspection') for p in (ROOT/base).rglob('*.py')]
    sources += [ROOT/'launch_weld_studio.py', ROOT/'WeldDefectDetection/inspect_images.py']
    result = dict(manifests=manifests, source_drift_from_frozen_campaign=drift,
                  coordinate_checks=coordinate_check(), median_development_diagnostics=median_diagnostics(),
                  release_sources={str(p.relative_to(ROOT)):digest(p) for p in sources},
                  physical_validation=False, seconds=time.perf_counter()-start)
    write_json(out/'audit_measurements.json', result)
    print(json.dumps(dict(manifests=manifests, source_drift_count=len(drift),
                         coordinate_cases=len(result['coordinate_checks']),
                         median_diagnostics=result['median_development_diagnostics'], seconds=result['seconds'])))
    return int(any(not m['passed'] for m in manifests))


if __name__ == '__main__':
    raise SystemExit(main())
