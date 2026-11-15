"""Finish a frozen campaign with audit-only static evaluation config copies.

The original campaign worker omits data.development_dataset_root when invoking
static confirmation. This adapter supplies that audit reference without changing
the archived implementation, saved configs, or any inference setting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'DataSimulation'))
sys.path.insert(0, str(REPO / 'WeldDefectDetection'))

from bse_renderer.validation.campaign_execution import (
    execute_stage, inventory, require_model, verify_receipt,
)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def verify_frozen(root):
    protocol = json.loads((root / 'confirmation_protocol.json').read_text())
    for relative, digest in protocol['frozen_artifacts'].items():
        path = root / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Pre-confirmation frozen artifact changed: {path}')
    identity = json.loads((root / 'receipts/execution_identity.json').read_text())
    actual = {str(p.relative_to(REPO)): sha(p)
              for package in ('DataSimulation/bse_renderer', 'WeldDefectDetection/src')
              for p in (REPO / package).rglob('*.py')}
    if actual != identity['sources']:
        raise ValueError('Archived execution source identity changed.')
    configs = {p.name: sha(p) for p in (REPO / 'DataSimulation/config').glob('*.yaml')}
    if configs != identity['configuration']:
        raise ValueError('Renderer configuration identity changed.')
    require_model(root)
    for stage in ('pilot', 'generate', 'audit', 'train', 'freeze'):
        verify_receipt(root, json.loads((root / 'receipts' / f'{stage}.json').read_text()))
    return protocol


def evaluate_static_with_audit_reference(root):
    import yaml
    from src.evaluation.evaluate_dataset import evaluate_dataset

    verify_frozen(root)
    verify_receipt(root, json.loads((root / 'receipts/confirmation.json').read_text()))
    receipt_path = root / 'receipts/job_evaluate_static.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        verify_receipt(root, receipt)
        if receipt.get('adapter_sha256') != sha(__file__):
            raise ValueError('Completed static evaluation used a different adapter.')
        return
    job_dir = root / 'jobs/evaluate_static'
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError('Partial evaluation exists. Preserve it; do not overwrite scores.')
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'run_frozen_confirmation.py').write_bytes(Path(__file__).read_bytes())
    frozen = json.loads((root / 'frozen_models.json').read_text())
    selected = frozen['selected']
    original = {
        'morphology': Path(selected['classical']['tuned_config']),
        'cnn': root / 'jobs' / f"cnn_{selected['cnn']['seed'] - 915510001}" / 'config.yaml',
    }
    configs, changes = {}, {}
    for algorithm, source in original.items():
        config = yaml.safe_load(source.read_text())
        evaluation = deepcopy(config)
        evaluation['data']['development_dataset_root'] = str(root / 'datasets/static_dev')
        roundtrip = deepcopy(evaluation)
        roundtrip['data'].pop('development_dataset_root')
        original_without_reference = deepcopy(config)
        original_without_reference['data'].pop('development_dataset_root', None)
        if roundtrip != original_without_reference:
            raise ValueError('Evaluation configuration altered an inference setting.')
        target = job_dir / f'{algorithm}_evaluation.yaml'
        target.write_text(yaml.safe_dump(evaluation, sort_keys=True), encoding='utf-8')
        configs[algorithm] = target
        changes[algorithm] = {
            'original': str(source), 'original_sha256': sha(source),
            'evaluation': str(target), 'evaluation_sha256': sha(target),
            'only_override': {'data.development_dataset_root': str(root / 'datasets/static_dev')},
        }
    write(job_dir / 'adapter.json', {
        'schema': 'static-confirmation-audit-reference-adapter/v1',
        'adapter': str(Path(__file__).resolve()), 'adapter_sha256': sha(__file__),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'configs': changes, 'inference_settings_changed': False,
    })
    results = {}
    for variant in ('static_confirmation', 'static_noiseless'):
        for algorithm in ('morphology', 'cnn'):
            key = f'{variant}_{algorithm}'
            output = root / 'experiments' / key
            if output.exists() and any(output.iterdir()):
                raise ValueError(f'Existing evaluation must be preserved: {output}')
            print('EVALUATE ' + key, flush=True)
            results[key] = evaluate_dataset(
                config_path=configs[algorithm], algorithm=algorithm, split='all',
                checkpoint=selected['cnn']['best_checkpoint'] if algorithm == 'cnn' else None,
                dataset_root_override=root / 'datasets' / variant, output_dir=output,
            )
    verify_frozen(root)
    result = {'status': 'ok', 'results': results,
              'artifact_directories': [str(root / 'experiments' / key) for key in results],
              'emission_policy': 'engineering_assumption',
              'emission_model_sha256': sha(root / 'model/assumed_constant.json')}
    write(job_dir / 'result.json', result)
    directories = [job_dir] + [Path(p) for p in result['artifact_directories']]
    artifacts = {str((directory / relative).relative_to(root)): digest
                 for directory in directories for relative, digest in inventory(directory).items()}
    write(receipt_path, {'status': 'completed', 'job': 'evaluate_static',
                         'adapter_sha256': sha(__file__), 'artifacts': artifacts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = verify_frozen(root)
    print(f"Verified {len(protocol['frozen_artifacts'])} frozen artifacts and prior stages", flush=True)
    if args.check_only:
        return
    started = time.monotonic()
    result = execute_stage(root, 'confirmation')
    if result['status'] != 'completed':
        raise RuntimeError('Confirmation generation did not complete.')
    evaluate_static_with_audit_reference(root)
    for stage in ('evaluate', 'verify'):
        print('START ' + stage, flush=True)
        result = execute_stage(root, stage)
        print(json.dumps({k: result[k] for k in ('status', 'stage', 'seconds') if k in result}), flush=True)
        if result['status'] != 'completed':
            raise RuntimeError(f'{stage} did not complete.')
    verify_frozen(root)
    write(root / 'confirmation_integrity.json', {
        'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(root / 'confirmation_protocol.json'),
        'frozen_artifacts': protocol['frozen_artifacts'],
        'all_frozen_artifacts_unchanged': True, 'archived_sources_unchanged': True,
        'adapter_sha256': sha(__file__), 'seconds': time.monotonic() - started,
    })


if __name__ == '__main__':
    main()
