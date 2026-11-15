"""Prospective execution gates and separately frozen raised-crown datasets."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,gc,json,time,zipfile
import numpy as np
from box_comparison.core import sha,write
from . import geometry as geo,rendering as r

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'DataSimulation/data/output/raised_crown_20261005_v5'
COUNTS=dict(train=24,val=12,test=24,stress=12)
BASES=dict(pilot=2300500000,train=2310500000,val=2320500000,test=2330500000,stress=2340500000)
SEEDS=[100505001,100505002,100505003]
KINDS=('clean','open_keyhole','open_intrinsic','bulge')
STRESS=('clean','pair_far','pair_near','pair_overlap','triple','small_open','hidden')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def plans(seed,index,stress=False):
    rng=np.random.default_rng(seed+901)
    angle=0. if index%6==0 else float(rng.uniform(0,2*np.pi))
    # Diameter and radial offset use different factors, not the same index cycle.
    diam=.5+.5*((index%3+rng.random())/3)
    off=(-1.05,0.,1.05)[(index//3)%3]
    def pore(kind='intrinsic_pore',effect='open',arc=0.,diameter=diam):
        return dict(type=kind,params=dict(kind=kind,effect=effect,diameter_mm=diameter,
            depth_mm=float(.15+.2*rng.random()),angle_rad=float((angle+arc/40)%(2*np.pi)),
            radial_offset_mm=off,coupling_fraction=1. if effect=='bulge' else 0.,roof_mm=.15))
    a=pore('keyhole_pore');b=pore()
    def shifted(p,arc):return dict(type=p['type'],params=dict(p['params'],angle_rad=float((angle+arc/40)%(2*np.pi))))
    return dict(clean=[],open_keyhole=[a],open_intrinsic=[b],bulge=[pore(effect='bulge')],
        hidden=[pore(effect='hidden')],pair_far=[a,shifted(b,12)],pair_near=[a,shifted(b,1.4)],
        pair_overlap=[a,shifted(b,.35)],triple=[shifted(a,-12),b,shifted(a,12)],
        small_open=[pore(diameter=.15+.15*(index%3)/2)])

def sources():
    return sorted(set(r.oldg.baseline.c.source_paths()+[Path(__file__).with_name(k+'.py') for k in ('geometry','rendering','campaign','validation','shadow')]+
        [Path(r.oldg.__file__),Path(r.oldg.baseline.__file__),Path(r.old.__file__),ROOT/'WeldDefectDetection/run_acquisition_campaign_20260928.py']))

def verify_sources():
    p=read(OUT/'protocol.json')
    for path,digest in p['source_hashes'].items():
        assert sha(path)==digest,f'Frozen renderer source changed: {path}'
    return p

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return verify_sources()
    p=dict(schema='raised-crown-study/v1',geometry=geo.specification(),counts=COUNTS,seed_bases=BASES,
        classes=KINDS,stress_classes=STRESS,split_unit='independent background including all paired conditions',
        diameter_mm=80.,sampling_mm=.05,columns=r.COLS,rows=r.ROWS,scored_rows=[16,144],
        material='304SS',energy_kev=60.,emission='Explicit uncalibrated Lambertian B=0 fallback',
        rationale='Original assumed B domain stops at 60 degrees; 3 mm crown exceeds it. No fit extrapolation.',
        researched_electron_depth_implemented=False,electron_transport_simulated=False,
        acquisition='Preserved native Cartesian raster_v2, inherited unmeasured settings',
        primary_task='Surface anomaly footprints, not inference of all buried pores',
        bulge='Counterfactual V_surface=V_pore, not validated weld mechanics',
        hidden='Enclosed pore without surface change is a negative control; separate from clean in reports',
        material_contrast='Uniform alloy; surface geometry is the only defect cue before acquisition',
        overlap='Minimum open-pore floor; additive displacements. No fluid interaction solver.',
        visibility='Nonlocal ray horizons on physical polar grid, refined-grid convergence gate; parallel detector directions and constant local working distance remain approximations',
        training=dict(architecture='TinyDefectSegNet',base_channels=24,seeds=SEEDS,epochs=20,examples_per_epoch=1200,
            input_size=64,radial_centres=[32,64,96,128],angular_stride=32,
            selection='Final epoch, validation-only threshold and seed selection. No architecture search.'),
        sample_size='24 training, 12 validation, 24 confirmation backgrounds; exploratory precision reported with background-cluster intervals, not a power claim',
        test_gate='No test/stress generation before frozen_models.json; no retuning after opening confirmation',
        physical_validation=False,source_hashes={str(p):sha(p) for p in sources()})
    write(OUT/'protocol.json',p)
    with zipfile.ZipFile(OUT/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(path,str(path.relative_to(ROOT)))
    # Record old artifacts before any promotion work. These remain read-only.
    oldroot=r.oldg.OUT
    frozen=read(oldroot/'frozen_models.json')
    paths=[oldroot/'frozen_models.json',oldroot/'normalization.json',ROOT/'data/releases/CURRENT_RELEASE.json']
    paths += [Path(p) for p in frozen['files']]
    write(OUT/'historical_inventory.json',{str(p):sha(p) for p in paths if p.exists()})
    return p

def pilot():
    from .validation import run_pilot
    run_pilot()

def generate(split):
    verify_sources()
    assert read(OUT/'pilot/gates.json')['numerical_gate']=='passed'
    if split in ('test','stress'):
        from .learning import verify_freeze
        verify_freeze()
    cfg=r.old.load_config_bundle(ROOT/'DataSimulation/config')
    folder=OUT/'datasets'/split;folder.mkdir(parents=True,exist_ok=True);rows=[]
    kinds=STRESS if split=='stress' else KINDS+(('hidden',) if split=='test' else ())
    for i in range(COUNTS[split]):
        seed=BASES[split]+i;amplitude=(0.,.05,.1)[(i//3+i)%3]
        sample,_=r.oldg.baseline.c.sampled(cfg,seed,i);engine=None
        for kind in kinds:
            sid=f'{split}-{i:03d}-{kind}';meta_path=folder/(sid+'.json');path=folder/(sid+'.npz')
            if meta_path.exists():
                row=read(meta_path);assert sha(path)==row['npz_sha256'];rows.append(row);continue
            if engine is None:engine=r.Engine(seed,amplitude)
            started=time.monotonic();s=replace(sample,defects=plans(seed,i)[kind])
            arrays,metadata=r.render(engine,s,retain_native=split=='stress')
            assert np.isfinite(arrays['raw']).all()
            assert bool(arrays['mask'].any())==(kind not in ('clean','hidden'))
            np.savez_compressed(path,**arrays)
            row=dict(scene_id=sid,background_id=f'{split}-{i:03d}',split=split,seed=seed,index=i,
                source_class=kind,amplitude=amplitude,parameters=asdict(s),metadata=metadata,
                npz_path=str(path),npz_sha256=sha(path),area_px=int(arrays['mask'][r.SCORE].sum()),
                seconds=time.monotonic()-started)
            write(meta_path,row);rows.append(row);print(sid,f'{row["seconds"]:.1f}s',flush=True)
            del arrays;gc.collect()
        del engine;gc.collect()
    write(folder/'index.json',rows)

def audit():
    from .learning import verify_freeze
    verify_sources();verify_freeze();sets={};counts={};records=[]
    for split in COUNTS:
        rows=read(OUT/f'datasets/{split}/index.json');sets[split]={x['seed'] for x in rows};counts[split]=len(rows)
        assert len(sets[split])==COUNTS[split]
        for row in rows:
            assert sha(row['npz_path'])==row['npz_sha256']
            with np.load(row['npz_path']) as a:
                assert a['raw'].shape==(3,160,5024)
                assert np.array_equal(a['mask'],a['instances'].any(0))
            records.append(dict(scene_id=row['scene_id'],split=split,background_id=row['background_id'],path=row['npz_path'],sha256=row['npz_sha256']))
    assert all(not sets[a]&sets[b] for a in sets for b in sets if a<b)
    changed=[p for p,h in read(OUT/'historical_inventory.json').items() if sha(p)!=h]
    assert not changed,changed
    write(OUT/'dataset_manifest.json',dict(records=records,geometry=geo.specification()))
    write(OUT/'audit.json',dict(status='passed',counts=counts,backgrounds={k:len(v) for k,v in sets.items()},
        split_overlap=False,historical_artifacts_unchanged=True,physical_validation=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pilot','train_data','train','test_data','evaluate','audit','report','all'])
    stage=parser.parse_args().stage
    if stage in ('prepare','all'):prepare()
    if stage in ('pilot','all'):pilot()
    if stage in ('train_data','all'):generate('train');generate('val')
    if stage in ('train','all'):
        from .learning import train
        train()
    if stage in ('test_data','all'):generate('test');generate('stress')
    if stage in ('evaluate','all'):
        from .assessment import evaluate
        evaluate()
    if stage in ('audit','all'):audit()
    if stage in ('report','all'):
        from .reporting import report
        report()
