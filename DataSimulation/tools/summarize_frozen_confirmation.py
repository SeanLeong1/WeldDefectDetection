"""Report completed confirmation artifacts without fitting or selecting models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frozen_confirmation import sha, verify_frozen, verify_receipt, write


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def aggregate(items):
    counts = {k: sum(int(r[k]) for r in items) for k in ('tp', 'tn', 'fp', 'fn')}
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    truth = np.array([r['target_is_defective'].lower() == 'true' for r in items])
    pred = np.array([r['predicted_is_defective'].lower() == 'true' for r in items])
    sensitivity = float(pred[truth].mean()) if truth.any() else None
    specificity = float((~pred[~truth]).mean()) if (~truth).any() else None
    return {'count': len(items), 'pixel': {
        **counts, 'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
        'precision': tp / (tp + fp) if tp + fp else 0.,
        'recall': tp / (tp + fn) if tp + fn else 0.,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
    }, 'image_classification': {
        'sensitivity': sensitivity, 'specificity': specificity,
        'balanced_accuracy': (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None else None,
        'tp': int((truth & pred).sum()), 'tn': int((~truth & ~pred).sum()),
        'fp': int((~truth & pred).sum()), 'fn': int((truth & ~pred).sum()),
    }}


def paired_noise(noisy, noiseless, seed=915910002, iterations=2000):
    paired = {r['sample_id']: r for r in noiseless}
    if len(paired) != len(noiseless) or {r['sample_id'] for r in noisy} != set(paired):
        raise ValueError('Paired noise comparison requires unique matching sample IDs.')
    a = np.array([[int(r[k]) for k in ('tp', 'fp', 'fn')] for r in noisy])
    b = np.array([[int(paired[r['sample_id']][k]) for k in ('tp', 'fp', 'fn')] for r in noisy])
    truth = np.array([r['target_is_defective'].lower() == 'true' for r in noisy])
    def iou(counts):
        t = counts.sum(axis=0)
        return float(t[0] / max(t.sum(), 1))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(~truth), np.flatnonzero(truth)]
    samples = []
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(g, len(g)) for g in groups])
        samples.append(iou(b[indices]) - iou(a[indices]))
    return {'direction': 'noiseless minus noisy pixel IoU',
            'point_estimate': iou(b) - iou(a),
            'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'bootstrap_unit': 'paired independent image, stratified clean/defective',
            'seed': seed, 'iterations': iterations, 'purpose': 'descriptive noise sensitivity only'}


def baseline_points(metric_rows):
    counts = np.zeros(3, dtype=np.int64)
    for row in metric_rows:
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError('Missing mask: ' + row['mask_path'])
        mask = mask > 0
        weld = np.zeros_like(mask)
        weld[3 * mask.shape[0] // 8:5 * mask.shape[0] // 8, :] = True
        counts += [np.sum(weld & mask), np.sum(weld & ~mask), np.sum(~weld & mask)]
    return {'all_background_pixel_iou': 0., 'all_background_balanced_accuracy': .5,
            'fixed_weld_region_pixel_iou': float(counts[0] / counts.sum()),
            'fixed_weld_region_balanced_accuracy': .5}


def summarize(root):
    protocol = verify_frozen(root)
    for stage in ('confirmation', 'evaluate', 'verify'):
        verify_receipt(root, read(root / 'receipts' / f'{stage}.json'))
    integrity = read(root / 'confirmation_integrity.json')
    if integrity['protocol_sha256'] != sha(root / 'confirmation_protocol.json'):
        raise ValueError('Confirmation protocol changed.')
    frozen = read(root / 'frozen_models.json')
    primary = frozen['primary_static_method']
    summaries, metrics, per_class = {}, {}, {}
    for variant in ('static_confirmation', 'static_noiseless'):
        source_classes = {r['sample_id']: r.get('planned_class', r.get('defect_type', 'clean'))
                          for r in map(json.loads, (root / 'datasets' / variant / 'samples.jsonl').read_text().splitlines())}
        for method in ('morphology', 'cnn'):
            name = variant + '_' + method
            directory = root / 'experiments' / name
            result = read(directory / 'summary.json')
            metric_rows = rows(directory / 'metrics.csv')
            summaries[name] = {**aggregate(metric_rows), 'bootstrap': result['bootstrap'],
                               'model_identity': result['model_identity'],
                               'inference_runtime_seconds': result['inference_runtime_seconds']}
            metrics[name] = metric_rows
            groups = defaultdict(list)
            for row in metric_rows:
                groups[source_classes[row['sample_id']]].append(row)
            per_class[name] = {name: aggregate(group) for name, group in sorted(groups.items())}
    for name in ('mean_confirmation', 'context_confirmation'):
        directory = root / 'experiments' / name
        result = read(directory / 'summary.json')
        metric_rows = rows(directory / 'evaluation_metrics.csv')
        summaries[name] = {**aggregate(metric_rows),
                           'bootstrap': result['evaluation_summary']['bootstrap'],
                           'frozen_parameters': result.get('frozen_parameters', result.get('selected_parameters'))}
        groups = defaultdict(list)
        for row in metric_rows:
            groups[row['planned_class']].append(row)
        per_class[name] = {name: aggregate(group) for name, group in sorted(groups.items())}
    noise = {method: paired_noise(metrics['static_confirmation_' + method],
                                 metrics['static_noiseless_' + method])
             for method in ('morphology', 'cnn')}
    retries = {}
    for dataset in ('static_confirmation', 'static_noiseless', 'sequence_confirmation'):
        records = [json.loads(line) for line in (root / 'datasets' / dataset / 'samples.jsonl').read_text().splitlines()]
        source = {r['sequence_id']: r for r in records}
        manifest = root / 'datasets' / dataset / 'sequence_manifest.json'
        classes = {r['sequence_id']: r['planned_class'] for r in read(manifest)} if manifest.exists() else {}
        groups = defaultdict(lambda: {'sources': 0, 'retried_sources': 0, 'rejected_attempts': 0})
        for sid, record in source.items():
            group = groups[classes.get(sid, record.get('planned_class', record.get('defect_type', 'unknown')))]
            attempts = int(record.get('generation_attempt', 1))
            group['sources'] += 1
            group['retried_sources'] += int(attempts > 1)
            group['rejected_attempts'] += attempts - 1
        retries[dataset] = dict(groups)
    primary_rows = metrics['static_confirmation_' + primary]
    skill = read(root / 'jobs/skill/result.json')
    report_sources = root / 'confirmation_report_sources'
    report_sources.mkdir(exist_ok=True)
    source_hashes = {}
    for name in ('run_frozen_confirmation.py', 'summarize_frozen_confirmation.py', 'make_confirmation_examples.py'):
        source = Path(__file__).parent / name
        target = report_sources / name
        target.write_bytes(source.read_bytes())
        source_hashes[str(target.relative_to(root))] = sha(target)
    report = {
        'schema': 'synthetic-confirmation-report/v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'operational_pipeline': 'completed', 'emission_qualification': 'not_established',
        'emission_policy': 'engineering_assumption', 'primary_method': primary,
        'independent_synthetic_detection_skill': skill['demonstrated_skill'], 'skill_gate': skill,
        'baselines': baseline_points(primary_rows), 'methods': summaries,
        'per_source_class': per_class, 'paired_noise': noise, 'retries_by_class': retries,
        'frozen_artifacts_unchanged': integrity['all_frozen_artifacts_unchanged'],
        'protocol_sha256': sha(root / 'confirmation_protocol.json'),
        'report_source_hashes': source_hashes,
        'stage_receipts': {stage: sha(root / 'receipts' / f'{stage}.json')
                           for stage in ('pilot', 'generate', 'audit', 'train', 'freeze', 'confirmation', 'evaluate', 'verify')},
        'audits': read(root / 'jobs/audit_confirmation/result.json'),
        'scope': 'Independent random seeds within this synthetic generator. No measured SEM transfer or physically qualified emission claim.',
        'limitations': ['Static images and spatial target windows are different populations.',
                        'Spatial sequences can contain clean target windows even in a defective source class.',
                        'No paired noiseless spatial dataset was generated by this campaign.',
                        'Any subsequent tuning requires a new independently seeded confirmation campaign.'],
    }
    write(root / 'confirmation_results.json', report)
    fmt = lambda x: 'undefined' if x is None else f'{x:.4f}'
    ci = lambda pair: f'[{pair[0]:.4f}, {pair[1]:.4f}]'
    primary_classification = summaries['static_confirmation_' + primary]['image_classification']
    primary_pixels = summaries['static_confirmation_' + primary]['pixel']
    lines = ['# Independent synthetic confirmation', '',
        'The separately seeded confirmation campaign completed. ' +
        ('The validation-selected primary method passed all prespecified synthetic skill criteria.'
         if skill['demonstrated_skill'] else 'The validation-selected primary method did not pass all prespecified skill criteria; detection skill remains unproven.'), '',
        'The fixed constant B remains an engineering assumption. These results measure performance on new synthetic samples from the same generator at 60 keV for 304 stainless steel. They do not establish physical emission validity or performance on measured SEM images.', '',
        f"The primary detector has limited practical accuracy: it falsely flags {primary_classification['fp']} of 100 clean images and misses {primary_classification['fn']} of 600 defective images. Its pixel IoU is {primary_pixels['iou']:.4f}. Passing the statistical gate does not establish readiness for inspection use.", '',
        '## Preserved settings and independent data', '',
        '- Primary static method: ' + primary + ', selected using development validation before confirmation.',
        '- Static confirmation: 700 noisy images and 700 geometry-matched noiseless images; seed 915210001; 100 images per source class.',
        '- Spatial confirmation: 70 sequences, 630 frames and 350 target windows; seed 915310001; ten sequences per source class.',
        '- Training normalization, thresholds, component filters and selected checkpoint hashes were preserved. All pinned files and archived package sources passed post-run hash verification.',
        '- Static evaluation config copies add only the development-dataset path required for the independence audit. See `jobs/evaluate_static/adapter.json` for the exact differences.',
        '- Confirmation audits check source independence, frozen normalization, mask alignment, physical spacing, split grouping and static noisy/noiseless pairing. Class-specific generation retry counts are in `confirmation_results.json`.', '',
        '## Primary skill gate', '',
        '| Requirement | Observed 95% interval | Pass |', '|---|---|---|',
        f"| Image balanced accuracy lower bound > 0.5 | {ci(skill['balanced_accuracy_ci95'])} | {skill['balanced_accuracy_ci95'][0] > .5} |",
        f"| Pixel IoU improvement over background > 0 | {ci(skill['iou_improvement_over_background_ci95'])} | {skill['iou_improvement_over_background_ci95'][0] > 0} |",
        f"| Paired pixel IoU improvement over fixed weld region > 0 | {ci(skill['iou_improvement_over_weld_region_ci95'])} | {skill['iou_improvement_over_weld_region_ci95'][0] > 0} |", '',
        'The gate uses 2,000 paired bootstrap samples of independent images, stratified by clean/defective status, with seed 915910001. The weld baseline covers the central quarter of image height across its full width. The background baseline predicts zero defect pixels.', '',
        '## Static results', '',
        '| Input / method | Pixel IoU | Precision | Recall | F1 | Image sensitivity | Specificity | Balanced accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, result in summaries.items():
        if not key.startswith('static'):
            continue
        p, c = result['pixel'], result['image_classification']
        values = [p[k] for k in ('iou', 'precision', 'recall', 'f1')] + [c[k] for k in ('sensitivity', 'specificity', 'balanced_accuracy')]
        lines.append('| ' + key + ' | ' + ' | '.join(map(fmt, values)) + ' |')
    lines += ['', f"Fixed weld-region pixel IoU: {report['baselines']['fixed_weld_region_pixel_iou']:.4f}. All-background pixel IoU: 0. Both baselines have image balanced accuracy 0.5.", '',
              '## Spatial results', '',
              'Each sequence contributes five target windows. Confidence intervals resample entire sequences to account for their shared source geometry. These windows have a different target population from the static images.', '',
              '| Method | Pixel IoU | Precision | Recall | F1 | Window sensitivity | Specificity | Balanced accuracy |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('mean_confirmation', 'context_confirmation'):
        result = summaries[key]
        p, c = result['pixel'], result['image_classification']
        values = [p[k] for k in ('iou', 'precision', 'recall', 'f1')] + [c[k] for k in ('sensitivity', 'specificity', 'balanced_accuracy')]
        lines.append('| ' + key + ' | ' + ' | '.join(map(fmt, values)) + ' |')
    lines += ['', 'Both spatial balanced-accuracy intervals include 0.5. Their pixel IoUs remain low, so these results do not establish useful spatial detection skill. No method or setting was replaced after inspecting confirmation scores.']
    lines += ['', '## Confidence intervals', '', '| Method | Pixel IoU 95% interval | Balanced accuracy 95% interval | Resampling unit |', '|---|---|---|---|']
    for key, result in summaries.items():
        b = result['bootstrap']
        interval = lambda name: ci([b[name]['ci_lower'], b[name]['ci_upper']])
        lines.append(f"| {key} | {interval('pixel_iou')} | {interval('image_balanced_accuracy')} | {b['resampling_unit']} |")
    lines += ['', 'These descriptive intervals use the saved evaluator bootstrap settings. The primary acceptance decision uses the prespecified 2,000-draw stratified intervals in the skill table.', '',
              '## Paired noise sensitivity', '', '| Method | Noiseless minus noisy IoU | Paired 95% interval |', '|---|---:|---|']
    for method, result in noise.items():
        lines.append(f"| {method} | {result['point_estimate']:.4f} | {ci(result['ci95'])} |")
    lines += ['', 'The geometry, labels and normalization match within each pair. This comparison describes sensitivity to the configured noise; it does not select a physical model or a new detection threshold. This campaign has no paired noiseless spatial dataset.', '',
              '## Results by source class', '', 'An undefined sensitivity or specificity means that subgroup has no positive or negative targets respectively. A source class describes the generated defect, while the spatial target window can be clean.', '']
    for key in ('static_confirmation_morphology', 'static_confirmation_cnn', 'mean_confirmation', 'context_confirmation'):
        lines += ['### ' + key, '', '| Source class | N | Pixel IoU | Sensitivity | Specificity | False positives | Misses |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, result in per_class[key].items():
            c = result['image_classification']
            lines.append(f"| {name} | {result['count']} | {fmt(result['pixel']['iou'])} | {fmt(c['sensitivity'])} | {fmt(c['specificity'])} | {c['fp']} | {c['fn']} |")
        lines.append('')
    lines += ['## Selected error examples', '',
              '[Inspect the CNN error examples](confirmation_error_examples.png). The figure selects the largest clean-image false-positive area and the lowest-IoU example within each defective source class. It is an error inspection, not a random sample or an estimate of overall performance. Each displayed prediction was checked against its saved evaluation confusion counts.', '',
              '## Reproduce and inspect', '',
              'Run from the Codebase2 directory:', '', '```powershell',
              f"& '.\\.venv\\Scripts\\python.exe' '.\\DataSimulation\\tools\\run_frozen_confirmation.py' --root '.\\DataSimulation\\data\\output\\{root.name}'",
              f"& '.\\.venv\\Scripts\\python.exe' '.\\DataSimulation\\tools\\summarize_frozen_confirmation.py' --root '.\\DataSimulation\\data\\output\\{root.name}'", '```', '',
              'Completed stages resume only after their artifact hashes verify. The report command reads saved scores and performs descriptive calculations; it does not retrain or retune.', '',
              'The complete numeric report is `confirmation_results.json`. `confirmation_protocol.json` records the pre-run settings; `confirmation_integrity.json` records post-run preservation. Experiment directories contain per-image/window CSVs, summaries and overlays. Historical development reports remain unchanged.', '',
              'Any subsequent tuning requires a new confirmation campaign.']
    (root / 'CONFIRMATION_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'demonstrated_skill': skill['demonstrated_skill'],
                      'report': str(root / 'CONFIRMATION_RESULTS.md')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root.resolve())
