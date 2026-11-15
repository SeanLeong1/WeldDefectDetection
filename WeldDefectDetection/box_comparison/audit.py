"""Visibility proxies and diagnosis of historical morphology assumptions."""
import cv2
import numpy as np
from src.algorithms.classical.morphology_detector import MorphologyDetector
from .core import morphology_score, write


def visibility_audit(data, output):
    detector = MorphologyDetector({'classical': dict(blackhat_kernel_size=41, opening_kernel_size=3,
        thresh_c=48, min_defect_area=15, max_defect_area_fraction=.05,
        max_component_aspect_ratio=30, max_component_span_fraction=1)})
    rows = []
    for image, mask, meta in zip(data['target'], data['mask'], data['meta']):
        if meta['split'] not in ('train', 'val') or not mask.any():
            continue
        truth = mask > 0
        ring = (cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0) & ~truth
        bg = image[ring] if ring.any() else image[~truth]
        contrast = float(np.median(image[truth])-np.median(bg)) if len(bg) else 0.
        mad = float(1.4826*np.median(np.abs(bg-np.median(bg)))) if len(bg) else 0.
        _, stages = detector.detect_with_stages((image*255).astype(np.uint8))
        old_mask = detector.predict_binary_mask((image*255).astype(np.uint8)) > 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rejected = sum(area < 15 or area > mask.size*.05 or max(w,h)/max(1,min(w,h)) > 30
                       for x,y,w,h,area in stats[1:n])
        rows.append(dict(window_id=meta['window_id'], source_class=meta['source_class'],
            signed_local_contrast=contrast, robust_contrast_ratio=abs(contrast)/max(mad, 1/65535),
            truth_pixels=int(truth.sum()), historical_roi_truth_fraction=float(np.mean(stages['weld_roi'][truth]>0)),
            historical_threshold_truth_fraction=float(np.mean(stages['threshold'][truth]>0)),
            historical_opened_truth_fraction=float(np.mean(stages['opened'][truth]>0)),
            historical_final_truth_fraction=float(np.mean(old_mask[truth])),
            true_regions=n-1, true_regions_rejected_by_old_size_shape_rules=int(rejected),
            dark_response=float(np.mean(morphology_score(image,17,'dark')[truth])),
            bright_response=float(np.mean(morphology_score(image,17,'bright')[truth]))))
    grouped = {}
    for label in sorted({r['source_class'] for r in rows}):
        part = [r for r in rows if r['source_class']==label]
        grouped[label] = dict(targets=len(part), weak_contrast_proxy_count=sum(r['robust_contrast_ratio']<1 for r in part),
            median_robust_contrast_ratio=float(np.median([r['robust_contrast_ratio'] for r in part])),
            bright_signed_contrast_count=sum(r['signed_local_contrast']>0 for r in part),
            roi_truth_fraction=float(np.mean([r['historical_roi_truth_fraction'] for r in part])),
            old_final_truth_fraction=float(np.mean([r['historical_final_truth_fraction'] for r in part])),
            true_regions=sum(r['true_regions'] for r in part),
            old_size_shape_rejected_regions=sum(r['true_regions_rejected_by_old_size_shape_rules'] for r in part))
    report = dict(status='ok', scope='Development train/validation 64px targets; historical policy diagnosed on this new population.',
        interpretation='Local contrast is a visibility proxy, not causal defect evidence or measured SEM validation. No examples removed.',
        weak_proxy_rule='absolute median contrast / robust annulus MAD < 1', classes=grouped, rows=rows)
    write(output, report)
    return report


def counterfactual_audit(dataset_root, output_root, config_dir, emission_path):
    """Six deterministic noiseless defect-on/off replays, one per source class.

    Sampled beam/material/detector/geometry are held fixed. This demonstrates
    causal renderer response for these examples, not visibility of every label.
    """
    import json
    from pathlib import Path
    from dataclasses import replace
    from bse_renderer.config_loader import load_config_bundle
    from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
    from bse_renderer.models import WeldLayoutRuntime
    from bse_renderer.renderer.scan_engine import ScanEngine
    from bse_renderer.validation.visibility_pilot import _quiet_sample
    root=Path(dataset_root);output=Path(output_root);output.mkdir(parents=True,exist_ok=True)
    records=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    chosen={}
    for r in records:
        plan=r['planned_defect_parameters']
        if r['split']=='train' and r['is_defective'] and len(plan)==1:
            chosen.setdefault(plan[0]['type'],r)
    if len(chosen)!=6: raise ValueError('Expected one replay example per defect class.')
    cfg=load_config_bundle(config_dir);rows=[]
    for label,r in sorted(chosen.items()):
        values=r['realized_non_noise_parameters']
        sample=sample_parameters(cfg,seed=r['seed'],max_defects=0)
        sample=replace(sample,**{name:replace(getattr(sample,name),**values[name]) for name in ('beam','material','detector','slice')},
                       defects=values['defects'],randomize_defect_location=values['randomize_defect_location'])
        sample=_quiet_sample(sample)
        engine=ScanEngine(image_size=r['source_scene_size_px'],pixel_size_mm=r['pixel_size_mm'],
            render_model_mode='formula_only',emission_b_mode='assumed_constant',emission_b_path=str(emission_path),
            render_view_mode='top_surface',weld_layout=WeldLayoutRuntime(**r['renderer_parameters']['weld_layout']))
        on=engine.run(sample);off=engine.run(replace(sample,defects=[]))
        y,x=r['crop_y0_px'],r['crop_x0_px'];cut=np.s_[y:y+64,x:x+64]
        saved=cv2.imread(str(root/r['mask_path']),cv2.IMREAD_UNCHANGED)>0
        mask=on.defect_mask[cut]>0
        if not np.array_equal(saved,mask): raise ValueError(f'Replayed geometry differs: {r["sample_id"]}')
        delta=np.abs(on.channels.topo[cut]-off.channels.topo[cut])
        background=delta[~mask]
        row=dict(source_class=label,sample_id=r['sample_id'],seed=r['seed'],mask_reproduced=True,
            mask_pixels=int(mask.sum()),median_absolute_change_inside=float(np.median(delta[mask])),
            max_absolute_change_inside=float(np.max(delta[mask])),
            fraction_mask_above_1e_6=float(np.mean(delta[mask]>1e-6)),
            mean_absolute_change_outside=float(np.mean(background)) if len(background) else None)
        rows.append(row)
        np.savez_compressed(output/f'{label}.npz',defect_on=on.channels.topo[cut],defect_off=off.channels.topo[cut],mask=mask)
        print(f'Counterfactual replay {label}: identical mask; median signal change {row["median_absolute_change_inside"]:.6g}',flush=True)
    write(output/'report.json',dict(status='ok',scope='Six train examples, identical settings and seeds, noise disabled, all defects removed in counterfactual.',
        limitations='Small diagnostic sample; not a universal visibility qualification. All original weak-contrast cases are retained.',rows=rows))
