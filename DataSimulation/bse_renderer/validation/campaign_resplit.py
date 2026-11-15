"""Rebuild grouped splits from verified raw renders without rerunning the renderer."""
import json
from pathlib import Path
import shutil
from collections import defaultdict, Counter

import yaml

from ..domain_randomisation.dataset_forms import (
    _sha256_path, _split_for_repetition, _apply_training_global_topo_normalisation,
    _write_contract_bundle,
)
from .casino_compact import _write_json


def planned_split_counts(total):
    return dict(Counter(_split_for_repetition(i, total) for i in range(total)))


def check_campaign_splits(dataset, expected_sources_per_class):
    """Require the campaign's 60/20/20 policy separately within every source class."""
    dataset = Path(dataset)
    records = [json.loads(x) for x in (dataset/'samples.jsonl').read_text().splitlines() if x.strip()]
    manifest = dataset/'sequence_manifest.json'
    planned = ({x['sequence_id']: x['planned_class'] for x in json.loads(manifest.read_text())}
               if manifest.exists() else {})
    groups = {}
    for row in records:
        key = row['sequence_id']
        value = (planned.get(key, row['defect_type']), row['split'])
        if key in groups and groups[key] != value:
            raise ValueError('A source sequence crosses classes or splits.')
        groups[key] = value
    counts = defaultdict(Counter)
    for label, split in groups.values():
        counts[label][split] += 1
    expected = planned_split_counts(expected_sources_per_class)
    if len(counts) != 7 or any(dict(c) != expected for c in counts.values()):
        raise ValueError(f'Campaign requires 60/20/20 source splits per class: expected {expected}, found {dict(counts)}')
    return {'status': 'ok', 'expected_per_class': expected, 'actual_by_class': dict(counts)}


def resplit_dataset(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Split rebuilding requires a new dataset directory.')
    descriptor = yaml.safe_load((source/'dataset_contract.yaml').read_text())
    records = [json.loads(x) for x in (source/descriptor['records']).read_text().splitlines() if x.strip()]
    sequence_file = source/'sequence_manifest.json'
    sequences = json.loads(sequence_file.read_text()) if sequence_file.exists() else []
    planned = {x['sequence_id']: x['planned_class'] for x in sequences}
    by_class = defaultdict(set)
    for row in records:
        by_class[planned.get(row['sequence_id'], row['defect_type'])].add(row['sequence_id'])
    mapping = {sid: _split_for_repetition(i, len(ids)) for ids in by_class.values()
               for i, sid in enumerate(sorted(ids))}
    destination.mkdir(parents=True)
    copied = []
    for row in records:
        old_split = row['split']
        row['split'] = mapping[row['sequence_id']]
        for container, key in ((row['raw_channels'], 'topo'), (row, 'mask_path'), (row['channels'], 'topo')):
            old = Path(container[key])
            if old.is_absolute() or '..' in old.parts:
                raise ValueError('Input dataset path is outside its root.')
            new = Path(*(row['split'] if part == old_split else part for part in old.parts))
            container[key] = new.as_posix()
            if container is row['channels']:
                continue  # Refit normalization using the new training split below.
            target = destination/new
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/old, target)
            digest = _sha256_path(source/old)
            if _sha256_path(target) != digest:
                raise ValueError('Copied raw render or mask hash differs.')
            copied.append({'source': old.as_posix(), 'destination': new.as_posix(), 'sha256': digest})
    for name in ('generation_provenance.json', 'generation_source.zip'):
        shutil.copy2(source/name, destination/name)
    normalization = _apply_training_global_topo_normalisation(destination, records)
    descriptor['split_fractions'] = [.6, .2, .2]
    descriptor['postprocessing_provenance'] = 'split_rebuild.json'
    artifacts = _write_contract_bundle(destination, records, dataset_kind=descriptor['dataset_kind'], extra_descriptor=descriptor)
    if sequences:
        for row in sequences:
            row['split'] = mapping[row['sequence_id']]
        _write_json(destination/'sequence_manifest.json', sequences)
        windows = [json.loads(x) for x in (source/descriptor['windows']).read_text().splitlines() if x.strip()]
        for row in windows:
            row['split'] = mapping[row['sequence_id']]
        (destination/'windows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in windows))
    shutil.copy2(Path(__file__), destination/'split_rebuild_source.py')
    _write_json(destination/'split_rebuild.json', {
        'status': 'completed', 'operation': 'source-grouped 60/20/20 resplit and training-only renormalization',
        'source_dataset': str(source), 'source_records_sha256': _sha256_path(source/descriptor['records']),
        'original_generation_provenance_sha256': _sha256_path(source/'generation_provenance.json'),
        'transformation_code_sha256': _sha256_path(destination/'split_rebuild_source.py'),
        'raw_renders_changed': False, 'masks_changed': False, 'prior_model_scores_used': False,
        'source_split_mapping': mapping, 'copied_artifacts': copied})
    summary = json.loads((source/'generation_summary.json').read_text())
    summary.update({'status': 'completed', 'output_dir': str(destination), 'normalization': normalization,
                    'postprocessing_provenance': str(destination/'split_rebuild.json'), **artifacts})
    if sequences:
        summary['sequence_manifest'] = str(destination/'sequence_manifest.json')
        summary['windows'] = str(destination/'windows.jsonl')
    _write_json(destination/'generation_summary.json', summary)
    return summary


def reuse_completed_job(source_root, root, job):
    from .campaign_execution import verify_receipt, require_model
    source_root, root = Path(source_root).resolve(), Path(root).resolve()
    if _sha256_path(require_model(source_root)[1]) != _sha256_path(require_model(root)[1]):
        raise ValueError('Reusing renders requires the identical fixed emission artifact.')
    receipt = source_root/'receipts'/f'job_{job}.json'
    saved_receipt = json.loads(receipt.read_text())
    if saved_receipt.get('status') != 'completed':
        raise ValueError('Only a completed source job can be reused.')
    verify_receipt(source_root, saved_receipt)
    if job == 'pilot':
        destination = root/'integrated_pilot'
        shutil.copytree(source_root/'integrated_pilot', destination)
        result = json.loads((destination/'summary.json').read_text())
        if result.get('accepted_count') != 84:
            raise ValueError('Reused pilot must contain all 84 accepted cases.')
        result['artifact_directories'] = [str(destination)]
    else:
        destination = root/'datasets'/job
        result = resplit_dataset(source_root/'datasets'/job, destination)
        result['artifact_directories'] = [str(destination)]
    result['reuse_lineage'] = {'source_campaign': str(source_root), 'source_job': job,
                               'source_job_receipt_sha256': _sha256_path(receipt)}
    return result
