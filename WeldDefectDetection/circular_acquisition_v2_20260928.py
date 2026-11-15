"""Matched circular geometry with v2 acquisition, then frozen v2 CV inference."""
from pathlib import Path
from dataclasses import replace, asdict
from unittest.mock import patch
import argparse
import gc
import time
import numpy as np
import torch
import circular_examples_20260928 as old
from run_acquisition_campaign_20260928 import OUT as CAMPAIGN
from bse_renderer.noise.raster_acquisition import AcquisitionProfile, acquire
from bse_renderer.renderer.image_formation import form_channels
from box_comparison.core import write, sha, truth_box, match_boxes, summarize
from box_refinement.methods import predicted_regions
from inspection.engine import FrozenInspector

OUT=old.ROOT/'DataSimulation/data/output/circular_acquisition_20260928_v2'
CASES=[(0.,'clean'),(.05,'clean'),(.1,'clean'),(.1,'keyhole_pore'),(.1,'intrinsic_pore')]


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=old.load_config_bundle(old.ROOT/'DataSimulation/config')
    sample,_=old.sampled(cfg,1360100000,0)
    sample=replace(sample,randomize_defect_location=False,slice=replace(sample.slice,slice_offset_mm=0.))
    profile=AcquisitionProfile()
    protocol=dict(radius_mm=40.,diameter_mm=80.,image_size=old.N,pixel_size_mm=.05, sample=asdict(sample),
        acquisition=asdict(profile),acquisition_model='raster_v2',physical_validation=False,
        inference='Frozen v2 straight-weld models, annular rectification, no circular tuning.',
        comparison='Same previously inspected geometry as circular_examples_20260928_v1; exploratory paired illustration, not independent confirmation.',
        source_sha256=sha(__file__),helper_sha256=sha(old.__file__),campaign=str(CAMPAIGN),
        cases=CASES,coverage='78 non-overlapping sectors, +/-1.6 mm ring band with small gaps; not whole-specimen inspection.')
    path=OUT/'render_protocol.json'
    if path.exists():
        assert old.json.loads(path.read_text())==old.json.loads(old.json.dumps(protocol)),'Circular protocol changed'
    else: write(path,protocol)
    engine=old.CircularEngine(); engine.acquisition_model='raster_v2'; engine.acquisition_profile=profile
    for amplitude,kind in CASES:
        name=f'a{round(amplitude*100):02d}_{kind}'; target=OUT/f'{name}.npz'; mp=OUT/f'{name}.render.json'
        engine.amplitude=amplitude
        if target.exists() and mp.exists():
            assert sha(target)==old.json.loads(mp.read_text())['npz_sha256']; print('Verified render '+name,flush=True); continue
        if engine.base is None and kind!='clean':
            with patch.object(old.se,'build_voxel_volume',old.top_volume),patch.object(old.se,'compute_quadrant_signals',old.chunked_quadrants): engine.run(sample)
        plan=old.pore_parameters(kind,np.random.default_rng(sample.seed+310003)); started=time.monotonic()
        print('Rendering '+name,flush=True)
        with patch.object(old.se,'build_voxel_volume',old.top_volume),patch.object(old.se,'compute_quadrant_signals',old.chunked_quadrants):
            result=engine.run(replace(sample,defects=plan))
        crops,masks=old.crop_ring(result.channels.topo,result.defect_mask)
        assert bool(masks.any())==(kind!='clean')
        q=engine.acquisition_quiet_quadrants
        gain=float(np.interp(sample.beam.focusing_current_norm,[0.,1.],[1.10,.92]))
        quiet=form_channels(q).topo*gain
        np.savez_compressed(target,raw=result.channels.topo,quiet=quiet,mask=result.defect_mask,
                            relief=engine.last_relief,crops_raw=crops,crops_mask=masks)
        if name=='a10_clean':
            np.savez_compressed(OUT/'clean_channel_reference.npz',quiet_quadrants=q,
                tx=result.channels.tx,ty=result.channels.ty,compo=result.channels.compo)
        write(mp,dict(name=name,kind=kind,amplitude=amplitude,renderer=result.metadata,npz_sha256=sha(target),
                      seconds=time.monotonic()-started))
        print(f'Completed {name} in {time.monotonic()-started:.1f}s',flush=True)
        del result,crops,masks,quiet,q; gc.collect()


def infer():
    torch.set_num_threads(4)
    inspector=FrozenInspector(campaign=CAMPAIGN); inspector._network('cnn')
    protocol=old.json.loads((OUT/'render_protocol.json').read_text())
    protocol.update(normalization=inspector.norm,model_sha256=inspector.model_hashes,
                    frozen_models_sha256=sha(CAMPAIGN/'frozen_models.json'))
    write(OUT/'protocol.json',protocol)
    summary=[]; replayed=0
    for amplitude,kind in CASES:
        name=f'a{round(amplitude*100):02d}_{kind}'
        meta=old.json.loads((OUT/f'{name}.render.json').read_text()); assert sha(OUT/f'{name}.npz')==meta['npz_sha256']
        with np.load(OUT/f'{name}.npz') as a: crops=a['crops_raw']; masks=a['crops_mask']
        n=inspector.norm; normalized=np.sqrt(np.clip((crops-n['raw_low'])/(n['raw_high']-n['raw_low']),0,1)).astype(np.float32)
        predictions={}; arrays={}
        for method in ('cnn','median'):
            rows=[]; scores=[]
            for i,target in enumerate(normalized):
                ix=[(i+k)%old.FRAMES for k in (-2,-1,1,2)]
                assert i not in ix and len(set(ix))==4
                score,mask,boxes=inspector.predict(dict(target=target,context=normalized[ix]),method)
                gt=truth_box(masks[i])
                rows.append(dict(window_id=f'{name}-sector-{i:02d}',scene_id=name,sequence_id='circular-background-000',
                    sector=i,angle_deg=float((90+360*i/old.FRAMES)%360),source_class=kind,truth_boxes=gt,boxes_xyxy=boxes,
                    target_positive=bool(gt),predicted_positive=bool(boxes),pixel_tp=int(np.sum((mask>0)&(masks[i]>0))),
                    pixel_fp=int(np.sum((mask>0)&(masks[i]==0))),pixel_fn=int(np.sum((mask==0)&(masks[i]>0))),**match_boxes(boxes,gt)))
                scores.append(score)
                _,bb=predicted_regions(score,**inspector.frozen['models'][method]['parameters'])
                assert bb==boxes; replayed+=1
            predictions[method]=rows; arrays[method+'_scores']=np.stack(scores)
        np.savez_compressed(OUT/f'{name}.scores.npz',**arrays)
        meta.update(predictions=predictions,metrics={m:summarize(r) for m,r in predictions.items()},
                    scores_sha256=sha(OUT/f'{name}.scores.npz'))
        write(OUT/f'{name}.json',meta); summary.append(dict(name=name,metrics=meta['metrics']))
        print(old.json.dumps(summary[-1]),flush=True)
    write(OUT/'summary.json',summary)
    write(OUT/'verification.json',dict(saved_decisions_replayed=replayed,physical_validation=False,source_sha256=sha(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['render','infer','all'],default='all'); args=parser.parse_args()
    if args.stage in ('render','all'): render()
    if args.stage in ('infer','all'): infer()
