from dataclasses import replace
import json
from pathlib import Path
import cv2
import numpy as np
from box_comparison.core import box_iou, write, subset
from box_comparison.audit import visibility_audit
from .methods import evaluate_spec


def diagnose(data, models, path):
    output = {}
    for arm, spec in models.items():
        rows = evaluate_spec(data, spec)
        details = []
        for row in rows:
            for truth in row['truth_boxes']:
                overlaps = [box_iou(truth, box) for box in row['boxes_xyxy']]
                best = max(overlaps, default=0.)
                details.append(dict(window_id=row['window_id'], source_class=row['source_class'], truth_box=truth,
                                    best_iou=best, category='matched_extent' if best >= .5 else
                                    'inaccurate_extent' if best >= .1 else 'missed_or_negligible_overlap'))
        output[arm] = dict(regions=details, counts={c:sum(r['category']==c for r in details)
            for c in ('matched_extent','inaccurate_extent','missed_or_negligible_overlap')},
            false_positive_boxes=sum(r['fp'] for r in rows),
            note='Per-truth maximum IoU diagnoses extent; official success uses one-to-one matching.')
    write(path, output)
    return output


def replay_visibility(dataset_root, output, config_dir, emission_path):
    """Three deterministic train sequences/class; every positive eligible crop retained."""
    from bse_renderer.config_loader import load_config_bundle
    from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
    from bse_renderer.models import WeldLayoutRuntime
    from bse_renderer.renderer.scan_engine import ScanEngine
    from bse_renderer.validation.visibility_pilot import _quiet_sample
    root, output = Path(dataset_root), Path(output); output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    sequences = {}
    for r in records:
        if r['split']=='train': sequences.setdefault(r['sequence_id'], []).append(r)
    chosen = {}; selected = []
    for sid, frames in sequences.items():
        positives = [r for r in frames if r['is_defective'] and 2 <= r['frame_index'] <= 6]
        if not positives: continue
        label = positives[0]['planned_defect_parameters'][0]['type']
        if chosen.get(label, 0) < 3:
            chosen[label] = chosen.get(label, 0)+1; selected.append((label, positives))
    cfg = load_config_bundle(config_dir); rows = []
    for label, positives in selected:
        r = positives[0]; values = r['realized_non_noise_parameters']
        sample = sample_parameters(cfg, seed=r['seed'], max_defects=0)
        sample = replace(sample, **{name:replace(getattr(sample,name), **values[name]) for name in ('beam','material','detector','slice')},
                         defects=values['defects'], randomize_defect_location=values['randomize_defect_location'])
        sample = _quiet_sample(sample)
        engine = ScanEngine(image_size=r['source_scene_size_px'], pixel_size_mm=r['pixel_size_mm'],
            render_model_mode='formula_only', emission_b_mode='assumed_constant', emission_b_path=str(emission_path),
            render_view_mode='top_surface', weld_layout=WeldLayoutRuntime(**r['renderer_parameters']['weld_layout']))
        on = engine.run(sample); off = engine.run(replace(sample, defects=[]))
        for r in positives:
            y,x = r['crop_y0_px'],r['crop_x0_px']; cut = np.s_[y:y+64,x:x+64]
            mask = on.defect_mask[cut]>0
            saved = cv2.imread(str(root/r['mask_path']), cv2.IMREAD_UNCHANGED)>0
            if not np.array_equal(mask,saved): raise ValueError('Replay mask mismatch.')
            delta = np.abs(on.channels.topo[cut]-off.channels.topo[cut])
            noisy = np.load(root/r['raw_channels']['topo'])
            noise = noisy-on.channels.topo[cut]
            sigma = max(1e-8,float(np.std(noise)))
            visible = delta > 2*sigma
            rows.append(dict(source_class=label, sample_id=r['sample_id'], sequence_id=r['sequence_id'], seed=r['seed'],
                mask_reproduced=True, truth_pixels=int(mask.sum()), noise_rms=float(np.sqrt(np.mean(noise**2))),
                noise_std=sigma, median_change_in_mask=float(np.median(delta[mask])),
                fraction_mask_change_above_2_noise_std=float(np.mean(visible[mask])),
                change_energy_fraction_in_mask=float(delta[mask].sum()/max(delta.sum(),1e-12))))
            np.savez_compressed(output/(r['sample_id']+'.npz'),defect_on=on.channels.topo[cut],defect_off=off.channels.topo[cut],
                                noisy=noisy,mask=mask,delta=delta)
        print(f'Visibility replay {label} seed={r["seed"]}',flush=True)
    if len(chosen)!=6 or any(v!=3 for v in chosen.values()): raise ValueError('Incomplete six-class visibility coverage.')
    report=dict(status='ok',sequences=len(selected),crops=len(rows),rows=rows,
        scope='First three defective training sequences per class, every positive eligible target crop. All defects removed together.',
        limitations='Two-noise-std rule is a descriptive proxy, not a detectability threshold. No labels changed or weak cases removed. No physical SEM validation.')
    write(output/'report.json',report)
    return report
