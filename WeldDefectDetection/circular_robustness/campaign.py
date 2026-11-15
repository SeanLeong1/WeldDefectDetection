"""Fresh paired multi-pore scenes; frozen v4 detectors, no fitting or tuning."""
from pathlib import Path
from dataclasses import replace,asdict
from unittest.mock import patch
import json,time,zipfile,gc
import numpy as np
from scipy.ndimage import map_coordinates
from circular_study import generation as g,geometry as geo,evaluation
from box_comparison.core import sha

ROOT=g.ROOT
OUT=ROOT/'DataSimulation/data/output/circular_robustness_20261004_v1'
COUNT=24
BASE=2240400000
ROWS=128
PAD=5.
VARIANTS=('clean','single_keyhole','single_intrinsic','pair_far','pair_near','pair_overlap','triple')
PERTURBATIONS=[dict(name='nominal',dx=0.,dy=0.,dr=0.)]
for delta in (.1,.25,.5):
    for axis in ('dx','dy','dr'):
        for sign in (-1,1):
            value=dict(dx=0.,dy=0.,dr=0.);value[axis]=sign*delta
            PERTURBATIONS.append(dict(name=f'{axis}_{sign*delta:+.2f}',**value))
PERTURBATIONS += [dict(name='combined',dx=.25,dy=-.25,dr=.25)]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):g.write(p,d)

def prepare():
    frozen=evaluation.verify_freeze();g.baseline.c.verify_sources(g.OUT)
    sources=[Path(__file__),Path(__file__).with_name('assessment.py')]
    evidence=dict(schema='circular-robustness/v1',backgrounds=COUNT,seed_base=BASE,variants=VARIANTS,
        profile='Same v4 recessed circular profile; unfamiliar profiles deliberately excluded',
        radius_mm=40.,source_rows=ROWS,model_rows=96,evaluation_rows=64,columns=geo.COLS,pitch_mm=.05,
        separations_mm=dict(pair_far=12.,pair_near=1.4,pair_overlap=.35,triple=[-12.,0.,12.]),
        perturbations=PERTURBATIONS,
        labels='Both individual occupancy masks and their union. Primary union-region IoU>=0.5; secondary one-to-one instance boxes.',
        acquisition='Same v4 raster assumptions; wider computed annulus for valid mislocalisation sampling',
        mapping='Resample noisy physical image into estimated ring coordinates, infer, map scores to fixed true inspection band. Uncovered pixels score zero; labels never cropped out of denominator.',
        model_policy='Frozen v4 selected CNN/median and both preselected operating points. No fitting or threshold search.',
        uncertainty='Paired whole-background bootstrap; stress conditions are repeated measures, not new specimens',
        inference_source_hashes={p:sha(p) for p in frozen['files'] if Path(p).suffix=='.py'},
        frozen_models_sha256=sha(g.OUT/'frozen_models.json'),source_hashes={str(p):sha(p) for p in sources},
        physical_validation=False,multi_pore_physics='Existing minimum yield, maximum void fraction and additive relief composition; not a physical pore-interaction solver')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==json.loads(json.dumps(evidence)),'Changed robustness protocol or sources'
    else:
        write(path,evidence)
        with zipfile.ZipFile(OUT/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sources:z.write(p,str(p.relative_to(ROOT)))
    return frozen

def unwrap(array,rows=ROWS,order=1):
    radius=40+(np.arange(rows)-(rows-1)/2)*.05
    angle=2*np.pi*np.arange(geo.COLS)/geo.COLS
    y,x=np.broadcast_arrays(radius[:,None]*np.sin(angle),radius[:,None]*np.cos(angle))
    return map_coordinates(array,[(y+(g.old.N-1)*.025)/.05,(x+(g.old.N-1)*.025)/.05],order=order,mode='nearest')

def plans(seed,index):
    rng=np.random.default_rng(seed+610001)
    angle=0. if index%6==0 else (index%4+rng.random())*np.pi/2
    diameter=.5+.5*((index%3+rng.random())/3)
    def pore(kind,offset,identity):
        return dict(type=kind,params=dict(diameter_mm=diameter,depth_mm=.25+.35*rng.random(),
            angle_rad=float((angle+offset/40)%(2*np.pi)),radial_offset_mm=(-.15,0.,.15)[index%3],instance_seed=seed+710001+identity))
    a=pore('keyhole_pore',0,0);b=pore('intrinsic_pore',0,1)
    def shifted(p,arc):
        return dict(type=p['type'],params=dict(p['params'],angle_rad=float((angle+arc/40)%(2*np.pi))))
    return dict(clean=[],single_keyhole=[a],single_intrinsic=[b],pair_far=[a,shifted(b,12.)],
        pair_near=[a,shifted(b,1.4)],pair_overlap=[a,shifted(b,.35)],triple=[a,shifted(b,12.),shifted(pore('keyhole_pore',0,2),-12.)])

def inject(volume,plan,rng,**kwargs):
    for p in plan:geo.inject(volume,[p],np.random.default_rng(p['params']['instance_seed']))

class Engine(g.Engine):
    def _top_surface_section(self,volume,**kwargs):
        section,defects=super()._top_surface_section(volume,**kwargs)
        self.instance_masks=np.stack([unwrap(d.mask,order=0) for d in defects]).astype('u1') if defects else np.zeros((0,ROWS,geo.COLS),'u1')
        return section,defects

    def _apply_adaptive_quadrant_blur(self,quadrants,*,sigma_px,blur_scale_map):
        reach=np.sqrt(2)*(int(4*sigma_px*min(4.,max(1.,float(blur_scale_map.max())))+.5)+2)*.05
        if ROWS*.025+reach>=PAD:raise ValueError('Insufficient robustness halo')
        return g.old.se.ScanEngine._apply_adaptive_quadrant_blur(self,quadrants,sigma_px=sigma_px,blur_scale_map=blur_scale_map)

def render(engine,sample,full=False):
    cached={}
    def profile(**kw):
        if not cached:cached['height']=geo.profile(**kw,amplitude=engine.amplitude)
        return cached['height'].copy()
    with patch.object(g.old,'generate_weld_profile',profile),patch.object(g.old.se,'build_voxel_volume',g.old.top_volume),patch.object(g.old.se,'apply_defect_bodies',inject),patch.object(g.baseline,'PAD_MM',PAD),patch.object(g.old.se,'compute_quadrant_signals',g.old.chunked_quadrants if full else lambda **kw:g.baseline.annular_quadrants(engine.quadrature_cache,**kw)):
        result=engine.run(sample)
    gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
    quiet=g.baseline.form_channels(engine.acquisition_quiet_quadrants).topo*gain
    raw=np.stack([unwrap(getattr(result.channels,k)) for k in ('topo','tx','ty')]).astype('f4')
    mask=unwrap(result.defect_mask,order=0).astype('u1')
    assert np.array_equal(mask,engine.instance_masks.any(axis=0))
    # Retain original metadata but explicitly correct the constant-coefficient reporting label.
    meta=result.metadata
    meta['emission_b']['coefficient_evaluation_tilt_stats_deg']=meta['emission_b'].pop('local_incidence_tilt_stats_deg',None)
    meta['emission_b']['local_incidence_range_deg']=[meta['operating_domain']['incidence_deg_quantiles'][0],meta['operating_domain']['incidence_deg_quantiles'][-1]]
    return raw,unwrap(quiet).astype('f4'),mask,engine.instance_masks.copy(),meta

def generate():
    prepare();folder=OUT/'datasets';folder.mkdir(exist_ok=True);rows=[]
    cfg=g.old.load_config_bundle(ROOT/'DataSimulation/config')
    for i in range(COUNT):
        seed=BASE+i;sample,_=g.baseline.c.sampled(cfg,seed,i);amplitude=(0.,.05,.1)[i%3]
        engine=Engine(seed,amplitude);plan=plans(seed,i)
        for variant in VARIANTS:
            sid=f'robust-{i:03d}-{variant}';path=folder/f'{sid}.npz';metadata=folder/f'{sid}.json'
            if metadata.exists():
                row=read(metadata);assert sha(path)==row['npz_sha256'];rows.append(row);continue
            start=time.monotonic();s=replace(sample,defects=plan[variant],randomize_defect_location=True)
            raw,quiet,mask,instances,meta=render(engine,s)
            assert np.isfinite(raw).all() and len(instances)==len(plan[variant])
            np.savez_compressed(path,raw=raw,quiet=quiet,mask=mask,instances=instances,pixel_size_mm=.05)
            row=dict(scene_id=sid,background_id=f'robust-{i:03d}',split='stress',index=i,seed=seed,variant=variant,
                source_class='clean' if variant=='clean' else 'multiple' if len(instances)>1 else plan[variant][0]['type'],
                amplitude=amplitude,parameters=asdict(s),renderer_metadata=meta,npz_path=str(path),npz_sha256=sha(path),
                pore_count=len(instances),instance_areas_px=instances[:,32:96].sum(axis=(1,2)).tolist())
            write(metadata,row);rows.append(row);print(sid,round(time.monotonic()-start,1),'s',flush=True)
            del raw,quiet,mask,instances;gc.collect()
        del engine;gc.collect()
    write(OUT/'index.json',rows)

if __name__=='__main__':generate()
