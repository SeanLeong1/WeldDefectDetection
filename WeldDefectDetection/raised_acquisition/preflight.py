"""Inspect acquisition coverage and paired-background separation before fitting."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import campaign as c


def run():
    assert not (c.OUT/'datasets/test/index.json').exists()
    coverage={}; seeds={}
    for split in ('train','val'):
        rows=c.read(c.OUT/f'datasets/{split}/index.json')
        clean=[row for row in rows if row['source_class']=='clean']; seeds[split]={row['seed'] for row in rows}
        assert len(clean)==c.COUNTS[split] and len(rows)==4*len(clean)
        assert all(sum(row['dwell_bin']==b for row in clean)==6 for b in range(4))
        for row in rows:
            assert row['dwell_time_us']==row['parameters']['beam']['dwell_time_us']
            assert (row['dwell_time_us'],row['dwell_bin'])==c.dwell(split,row['index'])
            assert c.sha(row['npz_path'])==row['npz_sha256']
        for seed in seeds[split]:
            siblings=[row for row in rows if row['seed']==seed]
            assert {row['source_class'] for row in siblings}==set(c.prior.KINDS)
            assert len({row['dwell_time_us'] for row in siblings})==1
        coverage[split]={str(b):dict(backgrounds=6,dwell_us=[row['dwell_time_us'] for row in clean if row['dwell_bin']==b],
            amplitude_counts=dict(Counter(str(row['amplitude']) for row in clean if row['dwell_bin']==b)),
            beam_current_counts=dict(Counter(str(row['parameters']['beam']['beam_current_ma']) for row in clean if row['dwell_bin']==b))) for b in range(4)}
    assert not seeds['train']&seeds['val']
    result=dict(status='passed',coverage=coverage,paired_conditions_share_acquisition=True,split_overlap=False,
        before_training=not any((c.OUT/'training').rglob('*.pt')),before_confirmation=True)
    c.write(c.OUT/'development_preflight.json',result); print('DEVELOPMENT PREFLIGHT PASSED',flush=True); return result
