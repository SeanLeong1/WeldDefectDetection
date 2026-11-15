import json
from pathlib import Path
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from bse_renderer.domain_randomisation.dataset_forms import (
    DATASET_CLASSES, _split_for_repetition, _write_contract_bundle, _sha256_path,
)
from bse_renderer.validation.campaign_resplit import resplit_dataset, check_campaign_splits


@pytest.mark.parametrize('total,expected', [(200, {'train':120,'val':40,'test':40}),
                                         (30, {'train':18,'val':6,'test':6}),
                                         (3, {'train':1,'val':1,'test':1})])
def test_campaign_split_policy(total, expected):
    assert dict(Counter(_split_for_repetition(i,total) for i in range(total))) == expected


def make_source(root, sequential):
    root.mkdir()
    records, sequences, windows = [], [], []
    for label in DATASET_CLASSES:
        for i in range(10):
            split = 'train' if i<7 else 'val' if i<9 else 'test'
            sid = f'{label}-{i:03d}'
            sequences.append({'sequence_id':sid,'planned_class':label,'split':split})
            windows.append({'sequence_id':sid,'split':split})
            for frame in range(2 if sequential else 1):
                sample = f'{sid}-f{frame}'
                raw = Path('raw')/split/f'{sample}.npy'
                mask = Path('masks')/split/f'{sample}.png'
                (root/raw).parent.mkdir(parents=True,exist_ok=True)
                (root/mask).parent.mkdir(parents=True,exist_ok=True)
                # Validation/test intensity outliers must not enter the new normalization fit.
                np.save(root/raw,np.array([[0,1],[2,3]],dtype=np.float32)+(100 if i>=6 else i))
                Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(root/mask)
                records.append({'sample_id':sample,'sequence_id':sid,'frame_index':frame,
                    'defect_type':'clean' if sequential else label,'split':split,
                    'raw_channels':{'topo':raw.as_posix()},'mask_path':mask.as_posix(),
                    'channels':{'topo':f'images/topo/{split}/{sample}.png'}})
    _write_contract_bundle(root,records,dataset_kind='sequential' if sequential else 'static_balanced',
                           extra_descriptor={'windows':'windows.jsonl'} if sequential else {})
    if sequential:
        (root/'sequence_manifest.json').write_text(json.dumps(sequences))
        (root/'windows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in windows))
    (root/'generation_provenance.json').write_text('{}')
    (root/'generation_source.zip').write_bytes(b'original fixture archive')
    (root/'generation_summary.json').write_text(json.dumps({'status':'ok'}))
    return records


@pytest.mark.parametrize('sequential',[False,True])
def test_rebuild_preserves_raws_and_refits_only_new_training(tmp_path,sequential):
    source = tmp_path/'source'
    original = make_source(source,sequential)
    with pytest.raises(ValueError,match='60/20/20'):
        check_campaign_splits(source,10)
    old_hash = _sha256_path(source/'samples.jsonl')
    target = tmp_path/'rebuilt'
    result = resplit_dataset(source,target)
    assert result['normalization']['raw_high'] < 20
    assert _sha256_path(source/'samples.jsonl') == old_hash
    assert check_campaign_splits(target,10)['expected_per_class'] == {'train':6,'val':2,'test':2}
    current = [json.loads(x) for x in (target/'samples.jsonl').read_text().splitlines()]
    for before,after in zip(original,current):
        assert _sha256_path(source/before['raw_channels']['topo']) == _sha256_path(target/after['raw_channels']['topo'])
        assert _sha256_path(source/before['mask_path']) == _sha256_path(target/after['mask_path'])
    if sequential:
        splits={r['sequence_id']:r['split'] for r in current}
        for row in json.loads((target/'sequence_manifest.json').read_text()):
            assert row['split'] == splits[row['sequence_id']]
        for line in (target/'windows.jsonl').read_text().splitlines():
            row=json.loads(line)
            assert row['split'] == splits[row['sequence_id']]
