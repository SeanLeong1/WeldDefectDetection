"""Integrity checks of the executed comparison, independent of outcome."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from circular_study import metrics
from . import campaign as c, learning as learn


def run():
    frozen=learn.verify_freeze(); c.verify(historical=True)
    rows=c.read(c.OUT/'datasets/train/index.json'); controls=[]
    for seed in c.SEEDS:
        a=next(x for x in frozen['arms']['coverage']['cnn_candidates'] if x['seed']==seed)
        b=next(x for x in frozen['arms']['hard_negative']['cnn_candidates'] if x['seed']==seed)
        assert a['history'][:5]==b['history'][:5], 'Matched warm-up training diverged'
        assert len(a['history'])==len(b['history'])==20
        assert not a['mining'] and [x['after_epoch'] for x in b['mining']]==[5,10,15]
        for bank in b['mining']:
            assert bank['candidate_count']==24*168 and bank['retained']==24*16
            coordinates=np.asarray(bank['coordinates']); unique,counts=np.unique(coordinates[:,0],return_counts=True)
            assert len(unique)==24 and np.all(counts==16)
            for ring,y,x in coordinates:
                meta=rows[int(ring)]
                assert meta['split']=='train' and meta['source_class']=='clean'
                assert y in (32,64,96,128) and (x-16)%32==0 and 0<=x<5024
        controls.append(dict(seed=seed,identical_first_five_epochs=True,epochs=20,
            hard_banks_training_only=True,hard_bank_backgrounds=24,tiles_per_background=16))
    timestamp=datetime.strptime(frozen['frozen_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    earliest={}
    for split in ('test','stress'):
        records=c.read(c.OUT/f'datasets/{split}/index.json')
        earliest[split]=min(Path(row['npz_path']).stat().st_mtime for row in records)
        assert earliest[split]>=timestamp, 'Confirmation image predates model freeze'
    result=c.read(c.OUT/'results.json')
    for arm in c.ARMS:
        for spec in frozen['arms'][arm]['cnn_candidates']:
            assert spec==c.read(c.OUT/f'training/{arm}/cnn_{spec["seed"]}.json')
        expected=max(frozen['arms'][arm]['cnn_candidates'],key=lambda x:metrics.key(x['selection']['best']['metrics']))
        assert expected['seed']==frozen['arms'][arm]['cnn']['seed']
        assert result['selected'][arm]==f'{arm}_{expected["seed"]}'
    for value in result['results'].values():
        assert value['inputs']==dict(frozen_sha256=c.sha(c.OUT/'frozen_models.json'),test_index_sha256=c.sha(c.OUT/'datasets/test/index.json'))
        assert value['metrics']==metrics.summary(value['rows'])
    robustness=c.read(c.OUT/'robustness_results.json')
    for method,groups in robustness['groups'].items():
        assert groups['nominal']['metrics']['scenes']==groups['image_estimated']['metrics']['scenes']==84
        assert groups['nominal']['individual_count']==groups['image_estimated']['individual_count']==120
        assert groups['combined']['metrics']['scenes']==groups['nominal_combined_subset']['metrics']['scenes']==24
        assert {x['scene_id'] for x in groups['combined']['rows']}=={x['scene_id'] for x in groups['nominal_combined_subset']['rows']}
    record=dict(status='passed',matched_training_controls=controls,confirmation_after_freeze=True,
        frozen_utc=frozen['frozen_utc'],earliest_confirmation_mtime_utc=earliest,
        selections_match_validation=True,reported_metrics_recomputed=True,matched_robustness_populations=True,physical_validation=False)
    manifest=[]
    for split in c.COUNTS:
        for row in c.read(c.OUT/f'datasets/{split}/index.json'):
            manifest.append({key:row[key] for key in ('scene_id','background_id','split','seed','source_class','dwell_bin','dwell_time_us','npz_path','npz_sha256')})
    c.write(c.OUT/'dataset_manifest.json',dict(schema='raised-acquisition-dataset/v1',records=manifest,
        raw_channels=['TOPO','Tx','Ty'],raw_shape=[3,160,5024],cnn_channel='TOPO',input_shape=[160,5024],scored_rows=[16,144],pixel_size_mm=.05,
        geometry_version='raised-crown-v1',split_unit='independent background',physical_validation=False))
    c.write(c.OUT/'comparison_integrity.json',record); return record
