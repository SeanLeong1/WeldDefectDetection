"""Matched renderer sensitivity; frozen detectors, no tuning on these outcomes."""
from pathlib import Path
import sys,os,json,time,argparse,subprocess,concurrent.futures
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import median_exploration_20261115 as ex
from raised_crown import rendering as render
from bse_renderer.renderer import quadrant_detector as q
from bse_renderer.noise.raster_acquisition import acquire
from bse_renderer.renderer.image_formation import form_channels
OUT=ex.OUT/'renderer_sensitivity'
INDICES=tuple(range(8))
MODES=('original','B0971','bounded_yield')

def prepare():
    ex.verify(True);OUT.mkdir(exist_ok=True)
    p=dict(created_utc=ex.utc(),indices=INDICES,modes=MODES,count_multiplier=64,repeat=0,
      comparison='Same new background seeds, geometry, acquisition parameters, masks and frozen detectors. No retraining or threshold selection.',
      B0971='Replace angular emission coefficient with .971 everywhere. This extrapolates the scalar law beyond the CASINO tilt coverage and is a sensitivity assumption, not qualified physics.',
      bounded_yield='Replace eta0/cos(alpha)^p with eta0*(2/(1+cos(alpha)))^p, preserving eta0 at normal incidence; clip [0,1]. This tests a bounded tilt dependence, not a calibrated physical replacement.',
      dependencies={str(p):ex.sha(p) for p in (Path(__file__),Path(ex.__file__),ex.OUT/'frozen.json')})
    target=OUT/'protocol.json'
    if target.exists():assert ex.read(target)['dependencies']==p['dependencies']
    else:ex.write(target,p)

def verify():
    ex.verify(True)
    for p,h in ex.read(OUT/'protocol.json')['dependencies'].items():assert ex.sha(p)==h,p

def background(index):
    verify();folder=OUT/f'{index:03d}';folder.mkdir(exist_ok=True)
    if (folder/'index.json').exists():return
    rows=ex.read(ex.OUT/f'datasets/test/{index:03d}/index.json')
    originals={r['source_class']:r for r in rows if r['repeat']==0 and r['count_multiplier']==64}
    seed=ex.SEED_BASE+index
    sample,_=render.oldg.baseline.c.sampled(render.old.load_config_bundle(ex.ROOT/'DataSimulation/config'),seed,index)
    r0=originals['clean'];sample=replace(sample,beam=replace(sample.beam,dwell_time_us=r0['dwell_time_us']))
    engine=render.Engine(seed,r0['amplitude']);records=[]
    original_emission=q.emission_distribution
    def emitted(phi,varphi,alpha,b):return original_emission(phi,varphi,alpha,.971)
    def bounded(eta0,alpha,z,exponent_scale=1.):
        p=exponent_scale*9/np.sqrt(np.maximum(z,1e-6))
        return np.clip(eta0*(2/(1+np.clip(np.cos(alpha),0,1)))**p,0,1)
    # Retain computed geometric horizons, but never reuse an old mode's signals.
    for mode in MODES:
        if hasattr(engine,'base_visibility'):engine.base_visibility.output=None
        context=patch.object(q,'emission_distribution',emitted) if mode=='B0971' else patch.object(q,'eta_oblique',bounded) if mode=='bounded_yield' else __import__('contextlib').nullcontext()
        with context:
            for kind in ex.c.KINDS:
                ref=originals[kind];sample_k=replace(sample,defects=ex.c.pore_plan(seed,index,kind))
                arrays,metadata=render.render(engine,sample_k)
                with np.load(ref['geometry_path']) as z:assert np.array_equal(arrays['mask'],z['mask'])
                scaled=replace(engine.acquisition_profile,effective_count_rate=engine.acquisition_profile.effective_count_rate*64)
                measured,_=acquire(engine.acquisition_quiet_quadrants,sample_k.noise,seed=seed,dwell_us=r0['dwell_time_us'],profile=scaled)
                gain=float(np.interp(sample_k.beam.focusing_current_norm,[0,1],[1.10,.92]))
                topo=render.extract(form_channels(measured).topo*gain).astype('f4')
                if mode=='original':assert np.array_equal(topo,np.load(ref['raw_path'])),'Original renderer replay mismatch'
                target=folder/f'{mode}_{kind}.npy';np.save(target,topo)
                quietpath=folder/f'{mode}_{kind}_quiet.npy';np.save(quietpath,arrays['quiet'])
                records.append(dict(**{k:ref[k] for k in ('background_id','source_class','count_multiplier','diameter_mm','repeat','geometry_path')},scene_id=f'{mode}-{ref["scene_id"]}',mode=mode,raw_path=str(target),quiet_path=str(quietpath),raw_sha256=ex.sha(target)))
                print('PHYSICS',index,mode,kind,flush=True)
    ex.write(folder/'index.json',records)

def generate():
    verify();pending=list(INDICES);active=[]
    while pending or active:
        while pending and len(active)<2:
            i=pending.pop(0)
            if (OUT/f'{i:03d}/index.json').exists():continue
            fh=(OUT/f'worker_{i:03d}.log').open('a',encoding='utf8')
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],stdout=fh,stderr=subprocess.STDOUT,cwd=ex.ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((p,fh,i))
        for p,fh,i in active[:]:
            if p.poll() is not None:
                fh.close();active.remove((p,fh,i))
                if p.returncode:raise RuntimeError(f'Physics worker {i} failed')
                print('PHYSICS BACKGROUND COMPLETE',i,flush=True)
        if active:time.sleep(2)

def evaluate():
    verify();policies=ex.read(ex.OUT/'frozen.json')['policies'];cal=ex.read(ex.OUT/'calibration.json')
    ops={k:v['op'] for k,v in policies.items() if k!='per_count'};ops['per_count']=policies['per_count']['64']['op']
    spec=ex.read(ex.OUT/'protocol.json')['cnn'];model=ex.base.model_for(spec);results={}
    for i in INDICES:
        for r in ex.read(OUT/f'{i:03d}/index.json'):
            raw=np.load(r['raw_path']);maps=ex.score_maps(raw,cal,{o['method'] for o in ops.values()});truth=ex.truth_of(r)
            meta={k:r[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')}
            for name,op in ops.items():
                pred=ex.reconstruct(ex.components(maps[op['method']],op['threshold'],op['closing']),op)
                results.setdefault(r['mode'],{}).setdefault(name,[]).append(ex.result_row(meta,truth,pred))
            s=ex.base.predict_image(ex.base.normalize(raw,ex.NORM),model);o=spec['selection']['best']
            _,pred=ex.metrics.binary(s,o['threshold'],o['min_area'],o['closing'])
            results[r['mode']].setdefault('cnn',[]).append(ex.result_row(meta,truth,pred))
        print('PHYSICS EVALUATED',i,flush=True)
    out={mode:{name:dict(**ex.summary(rows),rows=rows) for name,rows in methods.items()} for mode,methods in results.items()}
    ex.write(OUT/'results.json',out)
    print('PHYSICS RESULTS',json.dumps({m:{n:x['by_count']['64']['f1'] for n,x in v.items()} for m,v in out.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','background','generate','evaluate']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='background':background(a.index)
    else:globals()[a.action]()
