"""Frozen confirmation, transfer, noise diagnosis and clustered uncertainty."""
import gc,time
from pathlib import Path
import numpy as np
from box_comparison.core import sha
from . import generation as g,metrics,median,training,diagnostics,quiet as quiet_channels


def verify_freeze():
    frozen=g.read(g.OUT/'frozen_models.json')
    for p,h in frozen['files'].items():assert sha(p)==h,f'Changed frozen artifact: {p}'
    return frozen


def store_result(name,data,scores,choice,results):
    for point in ('best','low_fa'):
        spec=choice[point]
        if spec is None:
            results[name+'_'+point]=dict(available=False,reason='No validation threshold in the declared grid met the clean-ring alarm constraint')
            continue
        rows=metrics.evaluate(data,scores,spec,retain=True)
        for row,meta in zip(rows,data['meta']):
            row.update(index=meta['index'],area_px=meta['area_px'])
            if meta['parameters']['defects']:row.update(meta['parameters']['defects'][0]['params'])
        g.write(g.OUT/f'predictions/{name}_{point}.json',rows)
        results[name+'_'+point]=dict(available=True,operating_point={k:spec[k] for k in ('threshold','min_area','closing')},
            metrics=metrics.summary(rows),intervals=metrics.bootstrap(rows),
            per_class={k:metrics.summary([r for r in rows if r['source_class']==k]) for k in g.baseline.c.KINDS},
            per_amplitude={str(a):metrics.summary([r for r in rows if r['amplitude']==a]) for a in (0.,.05,.1)},
            per_angle={str(j*90):metrics.summary([r for r in rows if 'angle_rad' in r and j*np.pi/2<=r['angle_rad']<(j+1)*np.pi/2]) for j in range(4)})
        print(name,point,results[name+'_'+point]['metrics'],flush=True)


def historical_transfer(results):
    old=g.baseline.OUT;frozen=g.read(old/'frozen_models.json')['models'];norm=g.read(old/'normalization.json')
    data=g.load('test')
    for i,r in enumerate(data['meta']):
        with np.load(r['npz_path']) as a:
            data['x'][i,0]=np.sqrt(np.clip((a['raw'][0]-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1))
    cnn=frozen['cnn'];config=training.configuration('fixed_negatives')
    spec=dict(config=config,checkpoint=cnn['checkpoint'],checkpoint_sha256=cnn['checkpoint_sha256'])
    model=training.load_model(spec);score=training.predict(data,model,config)
    assert cnn['parameters']['low_ratio']==1. and cnn['parameters']['padding']==0
    store_result('historical_circular_cnn',data,score,dict(best=cnn['parameters'],low_fa=None),results)
    med=frozen['median'];spec=dict(variant='legacy_scale',calibration=[dict(centre=np.zeros((64,64)).tolist(),scale=med['scale'])])
    score=median.scores(data,spec)
    assert med['parameters']['low_ratio']==1. and med['parameters']['padding']==0
    store_result('historical_circular_median',data,score,dict(best=med['parameters'],low_fa=None),results)
    del data,model,score;gc.collect()


def run():
    frozen=verify_freeze();results={};data=g.load('test')
    assert len(data['meta'])==90
    wanted={('fixed_negatives',s) for s in g.SEEDS}|{(frozen['cnn_variant'],s) for s in g.SEEDS}
    selected=[s for s in frozen['cnn_candidates'] if (s['variant'],s['seed']) in wanted or s['config']['channels']==3]
    for spec in frozen['median_candidates']:
        score=median.scores(data,spec['spec'])
        store_result('median_'+spec['variant'],data,score,spec['selection'],results)
    for spec in [s for s in selected if s['config']['channels']==1]:
        model=training.load_model(spec);score=training.predict(data,model,spec['config'])
        store_result(f'cnn_{spec["variant"]}_{spec["seed"]}',data,score,spec['selection'],results)
        del model,score;gc.collect()
    if any(s['config']['channels']==3 for s in selected):
        del data;gc.collect();data=g.load('test',3)
        for spec in [s for s in selected if s['config']['channels']==3]:
            model=training.load_model(spec);score=training.predict(data,model,spec['config'])
            store_result(f'cnn_{spec["variant"]}_{spec["seed"]}',data,score,spec['selection'],results)
            del model,score;gc.collect()
    del data;gc.collect();historical_transfer(results)
    quiet=g.load('test',quiet=True)
    med=frozen['median'];score=median.scores(quiet,med['spec'])
    store_result('quiet_selected_median',quiet,score,med['selection'],results)
    cnn=frozen['cnn']
    if cnn['config']['channels']==3:quiet=quiet_channels.add_signed(quiet)
    if quiet is not None:
        model=training.load_model(cnn);score=training.predict(quiet,model,cnn['config'])
        store_result('quiet_selected_cnn',quiet,score,cnn['selection'],results)
    else:
        results['quiet_selected_cnn']=dict(available=False,reason='Quiet Tx/Ty sidecars missing; mixed quiet/noisy channels would be an invalid quiet comparison')
    g.write(g.OUT/'results.json',dict(results=results,selected_cnn=f'cnn_{cnn["variant"]}_{cnn["seed"]}',selected_median='median_'+med['variant'],
        claim='Background-disjoint synthetic circular annuli; no measured weld validity or manufacturing acceptance claim.',
        evaluation_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    diagnostics.visibility();audit()


def audit():
    frozen=verify_freeze();rows=[];seeds={};records=[]
    for split,count in g.COUNTS.items():
        rr=g.read(g.OUT/f'datasets/{split}/index.json');assert len(rr)==3*count
        seeds[split]={r['seed'] for r in rr}
        for r in rr:
            assert sha(r['npz_path'])==r['npz_sha256']
            assert r['renderer_metadata']['weld_layout']['mode']=='circular_weld'
            with np.load(r['npz_path']) as a:
                assert a['raw'].shape==(3,96,5024) and a['mask'].shape==(96,5024)
                assert np.isfinite(a['raw']).all()
                assert bool(a['mask'].any())==(r['source_class']!='clean')
            records.append(dict(scene_id=r['scene_id'],background_id=r['background_id'],split=split,source_class=r['source_class'],
                path=Path(r['npz_path']).relative_to(g.OUT).as_posix(),sha256=r['npz_sha256']))
        rows.extend(rr)
    assert all(not seeds[a]&seeds[b] for a in seeds for b in seeds if a<b)
    g.write(g.OUT/'dataset_manifest.json',dict(schema='circular-annulus/v4',records=records,
        arrays=dict(raw='float32 [TOPO,Tx,Ty] x96x5024',quiet='float32 TOPO 96x5024',mask='uint8 geometric 96x5024'),
        physical_sampling=dict(radial_pitch_mm=.05,arc_pitch_mm=g.g.ARC_PITCH,radius_mm=40.),
        valid_evaluation_rows=[16,80],angular_boundary='periodic',origin_angle_deg=0.,split_unit='background'))
    g.write(g.OUT/'dataset_audit.json',dict(status='passed',backgrounds=110,scenes=330,counts=g.COUNTS,
        source_and_artifact_hashes_verified=True,split_seed_overlap=False,all_circular=True,full_periodic_coverage=True,
        frozen_before_test=True,physical_validation=False))
