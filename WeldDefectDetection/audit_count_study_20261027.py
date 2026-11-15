"""Independent read-only checks of the count study's data and saved selections.

This auditor never selects a new model or changes a frozen experiment artifact.
Only its audit receipt is written. Use --final after the complete campaign.
"""
from pathlib import Path
import argparse
import collections
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
from count_study import campaign as c, learning as learn
from circular_study import metrics
from raised_crown import learning as base


def same(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,(int,float)):
        assert np.isclose(a,b,rtol=0,atol=1e-12),(path,a,b)
    else:assert a==b,(path,a,b)


def rank(x):return x['macro_f1'],x['metrics']['pixel_iou'],-x['metrics']['fp']


def data_audit(final):
    files={};splits={};seeds={};acquisition={};geometries={}
    for split,n in c.COUNTS.items():
        index=c.OUT/f'datasets/{split}/index.json'
        if not index.exists():
            assert not final,split
            continue
        rows=c.read(index);assert len(rows)==18*n
        assert len({r['scene_id'] for r in rows})==len(rows)
        groups=collections.defaultdict(list)
        for r in rows:
            groups[r['background_id']].append(r)
            assert r['split']==split and r['seed']==c.BASES[split]+r['index']
            assert r['acquisition_seed']==r['seed']+100000000*r['repeat']
            files[r['raw_path']]=r['raw_sha256'];files[r['geometry_path']]=r['geometry_sha256']
            geometries[r['geometry_path']]=r
        assert len(groups)==n
        for group in groups.values():
            assert {(r['source_class'],r['count_multiplier'],r['repeat']) for r in group}=={
                (k,m,j) for k in c.KINDS for m in c.MULTIPLIERS for j in range(2)}
            assert len({r['dwell_time_us'] for r in group})==1
            rates=[r['profile']['effective_count_rate']/r['count_multiplier'] for r in group]
            assert np.allclose(rates,rates[0])
            for kind in c.KINDS:
                assert len({r['geometry_sha256'] for r in group if r['source_class']==kind})==1
        seeds[split]={r['seed'] for r in rows};acquisition[split]={r['acquisition_seed'] for r in rows}
        splits[split]=dict(backgrounds=n,scenes=len(rows))
    for a in splits:
        for b in splits:
            if a!=b:
                assert not seeds[a]&seeds[b]
                assert not acquisition[a]&acquisition[b]
    for path,digest in files.items():assert c.sha(path)==digest,path
    minimum=None
    for path,row in geometries.items():
        with np.load(path) as a:
            mask=a['mask'];assert mask.shape==(160,5024)
            assert int(mask[16:144].sum())==row['area_px']
            assert bool(mask.any())==(row['source_class']!='clean')
            if row['source_class']!='clean':minimum=row['area_px'] if minimum is None else min(minimum,row['area_px'])
    return dict(splits=splits,checked_file_hashes=len(files),minimum_pore_area_px=minimum,
                background_and_acquisition_seeds_disjoint=True,matched_variants=True)


def replay_selection(arm,item,method):
    folder=c.OUT/'development'/arm;spec=item[method]
    op=spec['selection']['best']
    same(op,max(spec['selection']['trials'],key=rank))
    data=learn.Data(learn.selected_rows('val',arm),c.read(c.OUT/'normalization.json'))
    if method=='cnn':
        path=folder/f'cnn_{spec["seed"]}_e{spec["epoch"]}_val.npy'
        assert c.sha(spec['checkpoint'])==spec['checkpoint_sha256']
        model=base.model_for(spec);fresh=base.predict_image(data.image(0),model)
    else:
        path=folder/f'median_sigma{spec["spec"]["sigma"]:g}_val.npy'
        fresh=learn.median_score(data.image(0,spec['spec']['sigma']),spec['spec'])
    scores=np.load(path,mmap_mode='r')
    assert np.allclose(fresh,scores[0],rtol=0,atol=1e-6),str(path)
    # Use the original metric implementation, not the study's accelerated scorer.
    rows=[]
    for i,row in enumerate(data.rows):
        reference=metrics.evaluate(dict(mask=data.masks[i][None,16:144],meta=[row]),scores[i:i+1],op)[0]
        reference['count_multiplier']=row['count_multiplier'];rows.append(reference)
    same(learn.objective(rows),{k:op[k] for k in ('macro_f1','metrics','by_count')})
    return dict(method=method,validation_scenes=len(rows),original_metric_replay=True,fresh_prediction_parity=True)


def selection_audit(final):
    output={}
    for arm in c.ARMS:
        path=c.OUT/'development'/arm/'selection.json'
        if not path.exists():
            assert not final,arm
            continue
        item=c.read(path);candidates=item['cnn_candidates']
        assert {x['seed'] for x in candidates}==set(c.SEEDS)
        assert item['cnn']['seed']==max(candidates,key=lambda x:rank(x['selection']['best']))['seed']
        for candidate in candidates:
            assert len(candidate['history'])==20 and candidate['parameters']==262777
            assert all(np.isfinite(x['loss']) for x in candidate['history'])
            epochs=[c.read(path.parent/f'cnn_{candidate["seed"]}_e{e}.json') for e in (10,20)]
            winner=max(epochs,key=lambda x:rank(x['selection']['best']))
            assert candidate['epoch']==winner['epoch']
            assert c.sha(candidate['checkpoint'])==candidate['checkpoint_sha256']
        medians=[c.read(path.parent/f'median_sigma{s:g}.json') for s in c.SIGMAS]
        winner=max(medians,key=lambda x:rank(x['selection']['best']))
        assert item['median']['spec']['sigma']==winner['spec']['sigma']
        output[arm]=[replay_selection(arm,item,kind) for kind in ('cnn','median')]
    return output


def confirmation_audit():
    results=c.read(c.OUT/'results.json')
    assert len(results)==26
    expected={r['scene_id'] for r in c.read(c.OUT/'datasets/test/index.json')}
    for name,result in results.items():
        rows=result['rows'];assert len(rows)==864
        assert {r['scene_id'] for r in rows}==expected,name
        same(learn.objective(rows),{k:result[k] for k in ('macro_f1','metrics','by_count')},name)
    assert c.read(c.OUT/'audit.json')['status']=='passed'
    assert c.read(c.OUT/'completion.json')['status']=='complete'
    return dict(results=len(results),scenes_per_result=864,saved_row_aggregates_match=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    c.verify();learn.verify_development()
    if not (c.OUT/'frozen_models.json').exists():
        assert not (c.OUT/'datasets/test/index.json').exists()
    else:learn.verify_freeze()
    receipt=dict(status='passed',scope='final' if args.final else 'completed_development_only',
        data=data_audit(args.final),selections=selection_audit(args.final),
        no_model_or_selection_changes=True,source_sha256=c.sha(__file__),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if args.final:receipt['confirmation']=confirmation_audit()
    out=c.OUT/('independent_final_audit.json' if args.final else 'independent_development_audit.json')
    c.write(out,receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
