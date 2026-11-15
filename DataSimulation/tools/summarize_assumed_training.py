"""Build an artifact-derived development report after generation, audits and training."""
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bse_renderer.validation.campaign_execution import verify_receipt, require_model
from bse_renderer.domain_randomisation.dataset_forms import _sha256_path


def summarize(root):
    root = root.resolve()
    require_model(root)
    for stage in ('pilot', 'generate', 'audit', 'train'):
        verify_receipt(root, json.loads((root/'receipts'/f'{stage}.json').read_text()))
    job = lambda name: json.loads((root/'jobs'/name/'result.json').read_text())
    audit = job('audit_dev')['checks']
    rows = []
    retry_tables = {}
    for name in ('static_dev', 'sequence_dev'):
        path = Path(job(name)['records'])
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_source = {}
        for record in records:
            by_source.setdefault(record['sequence_id'], record)
        counts = defaultdict(lambda: {'sources': 0, 'retried_sources': 0, 'rejected_attempts': 0})
        sequence_manifest = root/'datasets'/name/'sequence_manifest.json'
        planned = ({r['sequence_id']: r['planned_class'] for r in json.loads(sequence_manifest.read_text())}
                   if sequence_manifest.exists() else {})
        for record in by_source.values():
            key = planned.get(record['sequence_id'], record.get('planned_class', record.get('defect_type', 'unknown')))
            row = counts[key]
            attempts = int(record.get('generation_attempt', 1))
            row['sources'] += 1
            row['retried_sources'] += int(attempts > 1)
            row['rejected_attempts'] += attempts-1
        retry_tables[name] = dict(counts)
    import torch
    for i in range(3):
        result = job(f'cnn_{i}')
        checkpoint = torch.load(result['best_checkpoint'], map_location='cpu', weights_only=False)
        history = next(r for r in checkpoint['history'] if r['epoch'] == checkpoint['epoch'])
        rows.append({'method': 'cnn', 'seed': result['seed'], 'epochs': result['epochs'],
            'selected_epoch': checkpoint['epoch'], 'validation_pixel_iou': result['best_val_iou'],
            'validation_pixel_f1': history['val_f1'], 'threshold': checkpoint['threshold'],
            'min_area': checkpoint['min_defect_area'], 'checkpoint': result['best_checkpoint'],
            'checkpoint_sha256': _sha256_path(Path(result['best_checkpoint'])),
            'runtime_seconds': result['training_runtime_seconds']})
    for i in range(3):
        result = job(f'context_{i}')
        rows.append({'method': 'spatial_predictor', 'seed': result['seed'], 'epochs': result['epochs'],
            'selected_epoch': result['best_epoch'],
            'validation_pixel_iou': result['selection_summary']['pixel']['iou'],
            'validation_balanced_accuracy': result['selection_summary']['image_classification']['balanced_accuracy'],
            **result['selected_parameters'], 'checkpoint': result['checkpoint'],
            'checkpoint_sha256': _sha256_path(Path(result['checkpoint'])),
            'runtime_seconds': result['training_runtime_seconds']})
    report = {'status': 'generation_audit_training_completed', 'emission_policy': 'engineering_assumption',
        'physical_qualification': False, 'independent_detection_confirmation_performed': False,
        'audit': audit, 'retries_by_class': retry_tables, 'trained_models': rows,
        'classical': job('classical'), 'mean_context': job('mean_context'),
        'scope': 'Development results only. Static and spatial workflows use different target populations.',
        'stage_receipt_hashes': {stage: _sha256_path(root/'receipts'/f'{stage}.json') for stage in ('pilot', 'generate', 'audit', 'train')}}
    (root/'development_results.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Development datasets, audits and training', '',
        'The generation, audit and training stages completed with fixed B = 0.9206948532270666, 304SS and 60 keV. '
        'Constant B is an engineering assumption. No physical emission qualification or independent detection confirmation is claimed.', '',
        '## Dataset checks', '', '| Dataset | Records | Audit |', '|---|---:|---|']
    for name in ('static', 'sequence'):
        a = audit[name]
        lines.append(f'| {name} | {a["record_count"]} | {a["status"]} |')
    lines += ['', 'The saved audits include split counts, duplicate seeds and raw hashes, image variation, normalization checks and context/target overlap. '
        'Detailed source retries are in [development_results.json](development_results.json).', '',
        '## Static workflows', '',
        f'Classical morphology was selected on validation data: pixel IoU {report["classical"]["best_validation_metrics"]["pixel_iou"]:.4f}, '
        f'image balanced accuracy {report["classical"]["best_validation_metrics"]["image_balanced_accuracy"]:.4f}.', '',
        '| CNN seed | Epochs | Selected epoch | Validation pixel IoU | Threshold |', '|---:|---:|---:|---:|---:|']
    for r in rows[:3]:
        lines.append(f'| {r["seed"]} | {r["epochs"]} | {r["selected_epoch"]} | {r["validation_pixel_iou"]:.4f} | {r["threshold"]:.3f} |')
    lines += ['', '## Spatial workflows', '',
        'The learned predictor trains on clean sequences. It predicts a target from four adjacent, non-overlapping context frames. '
        'The mean-context baseline is evaluated separately. Validation labels select residual thresholds and component filters.', '',
        f'Mean-context validation: pixel IoU {report["mean_context"]["selection_summary"]["pixel"]["iou"]:.4f}, '
        f'image balanced accuracy {report["mean_context"]["selection_summary"]["image_classification"]["balanced_accuracy"]:.4f}.', '',
        f'On its separate development-test windows, the frozen mean-context baseline achieved pixel IoU '
        f'{report["mean_context"]["evaluation_summary"]["pixel"]["iou"]:.4f} and balanced accuracy '
        f'{report["mean_context"]["evaluation_summary"]["image_classification"]["balanced_accuracy"]:.4f}. '
        'This test result was not used to retune it.', '',
        '| Predictor seed | Epochs | Selected epoch | Validation pixel IoU | Validation balanced accuracy |', '|---:|---:|---:|---:|---:|']
    for r in rows[3:]:
        lines.append(f'| {r["seed"]} | {r["epochs"]} | {r["selected_epoch"]} | {r["validation_pixel_iou"]:.4f} | {r["validation_balanced_accuracy"]:.4f} |')
    lines += ['', 'Validation scores were used for selection and are not unbiased final performance estimates. '
        'Static and spatial scores must not be ranked against each other as if they used the same images and targets.', '',
        '## Next stage', '',
        ('Model choices are frozen in [frozen_models.json](frozen_models.json). Run the separately seeded synthetic confirmation campaign before making a detection-skill claim. '
         if (root/'frozen_models.json').exists() else
         'Freeze model choices and run the separately seeded synthetic confirmation campaign before making a detection-skill claim. ')
        +
        'Preserve the saved preprocessing, thresholds, component filters and model hashes.', '']
    (root/'DEVELOPMENT_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = summarize(parser.parse_args().root)
    print(json.dumps({'status': result['status'], 'trained_models': len(result['trained_models'])}, indent=2))
