"""Matched native-raster acquisition, with old campaigns kept read-only."""
from dataclasses import asdict, replace
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import time
import zipfile
import numpy as np
from raised_acquisition import campaign as old, learning as oldlearn
from raised_crown import campaign as crown, rendering as render
from bse_renderer.noise.raster_acquisition import AcquisitionProfile, acquire
from bse_renderer.renderer.image_formation import form_channels

ROOT = old.ROOT
OUT = ROOT / 'DataSimulation/data/output/count_diversity_20261027_v2'
COUNTS = dict(train=96, val=32, test=48)
BASES = dict(train=2710100000, val=2720100000, test=2730100000, diagnostic=2740100000)
MULTIPLIERS = (1, 4, 64)
KINDS = ('clean', 'open_intrinsic', 'open_keyhole')
SEEDS = (1027001, 1027002, 1027003)
ARMS = dict(single1_96=(96, (1,)), single4_96=(96, (4,)),
            single64_96=(96, (64,)), mixed_96=(96, MULTIPLIERS),
            single1_24=(24, (1,)), mixed_24=(24, MULTIPLIERS))
SIGMAS = (0., .5, 1., 2., 4., 8.)
MED_THRESHOLDS = (2., 3., 4., 6., 8., 12.)
CNN_THRESHOLDS = (.2, .35, .5, .65, .8, .9, .97, .995)
AREAS = (3, 8, 15)
CLOSINGS = (1, 3)
read, write, sha = old.read, old.write, old.sha


def verify():
    protocol = read(OUT / 'protocol.json')
    for path, digest in protocol['source_hashes'].items():
        assert sha(path) == digest, f'Frozen generation dependency changed: {path}'
    return protocol


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return verify()
    frozen = oldlearn.verify_freeze()
    files = set(old.generation_sources()) | {Path(__file__), Path(acquire.__code__.co_filename)}
    files |= set((ROOT / 'DataSimulation/config').glob('*.yaml'))
    historical = set(Path(p) for p in frozen['files'])
    historical |= {old.OUT / n for n in ('frozen_models.json', 'normalization.json', 'results.json')}
    for split in old.COUNTS:
        index = old.OUT / f'datasets/{split}/index.json'
        historical.add(index)
        historical.update(Path(x['npz_path']) for x in read(index))
    write(OUT / 'historical_inventory.json', {str(p): sha(p) for p in historical if p.exists()})
    protocol = dict(campaign=OUT.name, counts=COUNTS, seed_bases=BASES, multipliers=MULTIPLIERS,
        classes=KINDS, repeats=2, total_scenes=3168, independent_backgrounds=176,
        arms=ARMS, seeds=SEEDS, epochs=20, patches_per_epoch=1200,
        architecture='TinyDefectSegNet in_channels=1 base_channels=24; unchanged',
        optimizer='AdamW lr .001 weight_decay .0001; cosine to .00005',
        hard_negatives='Mine after epochs 5/10/15; same 1200 draws/epoch in every arm',
        checkpoint_selection='Validation at epochs 10 and 20; seed and operating point by mean F1 over applicable count levels, then pooled pixel IoU, fewer FP',
        median=dict(sigmas=SIGMAS, thresholds=MED_THRESHOLDS, areas=AREAS, closing=CLOSINGS,
                    offsets=[-256,-192,-128,-64,64,128,192,256], calibration='clean training only; per-row median/MAD floor .003'),
        cnn_thresholds=CNN_THRESHOLDS, postprocessing_areas=AREAS, postprocessing_closing=CLOSINGS,
        input='Raw TOPO; shared training-only sqrt normalization fitted on nested 24 training backgrounds across all counts',
        exposure='Multiply effective_count_rate only. Preserve per-background dwell/profile/geometry. Not calibrated primary electrons.',
        noise='Full raster_v2 acquisition; named independent repeat seeds shared across matched scene conditions and count multipliers',
        geometry='Nominal coordinates, 80 mm circular weld, .05 mm pitch, 160x5024 TOPO',
        pore_design='Random independent diameter bin [.15,.30], [.30,.60], [.60,1.0] mm; depth .15-.35 mm; angular seam cases; radial positions -1.05/0/+1.05 mm',
        source_hashes={str(p):sha(p) for p in sorted(files)},
        split_rule='Split by background before variants; no seed overlap with historical manifest backgrounds',
        confirmation_gate='All six arms, three CNN seeds and medians frozen before test generation',
        evaluation='Full-ring one-to-one circular box IoU >= .5; pixel IoU, clean alarms, per-size/per-count; paired background bootstrap',
        sample_size='Exploratory; not a power calculation or manufacturing acceptance claim',
        controlled_subset='Separate noiseless/shot-only/full-acquisition fixed-small-pore diagnostic; excluded from training, selection and confirmation',
        physical_validation=False, two_layer_cnn=False)
    seen=set()
    for path in (ROOT/'DataSimulation/data/output').glob('*/datasets/*/index.json'):
        try:
            records=read(path)
            if isinstance(records,list): seen.update(x['seed'] for x in records if 'seed' in x)
        except (KeyError,TypeError): pass
    new={BASES[s]+i for s,n in COUNTS.items() for i in range(n)}
    assert not new & seen
    write(OUT / 'protocol.json', protocol)
    with zipfile.ZipFile(OUT/'generation_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p,p.relative_to(ROOT))
    return protocol


def dwell(split,index):
    b=index%4
    rng=np.random.default_rng(BASES[split]+index+701)
    # Independent within-bin draws keep the nested 24/96 samples comparable.
    return float(np.exp(rng.uniform(np.log(old.EDGES[b]),np.log(old.EDGES[b+1])))), b


def pore_plan(seed,index,kind):
    if kind=='clean': return []
    rng=np.random.default_rng(seed+901)
    # Independent factors avoid tying small pores to one dwell or radial position.
    size_bin=int(rng.integers(3)); limits=((.15,.30),(.30,.60),(.60,1.0))[size_bin]
    diameter=float(rng.uniform(*limits))
    angle=0. if index%8==0 else float(rng.uniform(0,2*np.pi))
    offset=float(rng.choice([-1.05,0.,1.05]))
    subtype='intrinsic_pore' if kind=='open_intrinsic' else 'keyhole_pore'
    return [dict(type=subtype,params=dict(kind=subtype,effect='open',diameter_mm=diameter,
        depth_mm=float(rng.uniform(.15,.35)),angle_rad=angle,radial_offset_mm=offset,
        coupling_fraction=0.,roof_mm=.15))]


def background(split,index):
    verify()
    if split=='test':
        from .learning import verify_freeze
        verify_freeze()
    folder=OUT/'datasets'/split/f'{index:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'index.json'
    if done.exists():
        for row in read(done): assert sha(row['raw_path'])==row['raw_sha256']
        return
    seed=BASES[split]+index
    sample,_=render.oldg.baseline.c.sampled(render.old.load_config_bundle(ROOT/'DataSimulation/config'),seed,index)
    d,bin_id=dwell(split,index)
    sample=replace(sample,beam=replace(sample.beam,dwell_time_us=d))
    amplitude=float(np.random.default_rng(seed+912).choice([0.,.05,.1]))
    engine=render.Engine(seed,amplitude)
    profile=engine.acquisition_profile
    rows=[]
    for kind in KINDS:
        start=time.monotonic()
        changed=replace(sample,defects=pore_plan(seed,index,kind))
        arrays,metadata=render.render(engine,changed)
        geometry_path=folder/f'{kind}_geometry.npz'
        np.savez_compressed(geometry_path,mask=arrays['mask'],height=arrays['height'],quiet=arrays['quiet'])
        geometry_sha=sha(geometry_path)
        assert bool(arrays['mask'].any())==(kind!='clean')
        q=engine.acquisition_quiet_quadrants
        gain=float(np.interp(changed.beam.focusing_current_norm,[0,1],[1.10,.92]))
        quiet=arrays['quiet']; baseline=arrays['raw'][0]
        for repeat in range(2):
            for multiplier in MULTIPLIERS:
                acquisition_seed=seed+repeat*100000000
                scaled=replace(profile,effective_count_rate=profile.effective_count_rate*multiplier)
                measured,receipt=acquire(q,changed.noise,seed=acquisition_seed,dwell_us=d,profile=scaled)
                topo=render.extract(form_channels(measured).topo*gain).astype('f4')
                if multiplier==1 and repeat==0:
                    assert np.array_equal(topo,baseline),'Reacquisition must reproduce original renderer exactly'
                sid=f'{split}-{index:03d}-{kind}-c{multiplier}-r{repeat}'
                target=folder/f'{kind}_c{multiplier}_r{repeat}.npy'
                np.save(target,topo)
                row=dict(scene_id=sid,background_id=f'{split}-{index:03d}',index=index,seed=seed,split=split,
                    source_class=kind,amplitude=amplitude,dwell_bin=bin_id,dwell_time_us=d,count_multiplier=multiplier,
                    repeat=repeat,acquisition_seed=acquisition_seed,profile=asdict(scaled),parameters=asdict(changed),
                    raw_path=str(target),raw_sha256=sha(target),geometry_path=str(geometry_path),geometry_sha256=geometry_sha,
                    area_px=int(arrays['mask'][render.SCORE].sum()),
                    diameter_mm=0. if kind=='clean' else changed.defects[0]['params']['diameter_mm'],
                    residual_rms=float(np.sqrt(np.mean((topo[render.SCORE]-quiet[render.SCORE])**2))),
                    physical_validation=False)
                rows.append(row)
                del measured,topo
        write(folder/f'{kind}_metadata.json',metadata)
        print(split,index,kind,round(time.monotonic()-start,1),'s',flush=True)
        del arrays,q
        gc.collect()
    write(done,rows)


def generate(split,workers=3):
    verify()
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    pending=list(range(COUNTS[split]));active=[]
    while pending or active:
        while pending and len(active)<workers:
            i=pending.pop(0)
            handle=(logs/f'generate_{split}_{i:03d}.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(ROOT/'WeldDefectDetection/run_count_study_20261027.py'),
                                   'background','--split',split,'--index',str(i),'--campaign',OUT.name],stdout=handle,stderr=subprocess.STDOUT,
                                   cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((proc,handle,i))
        for proc,handle,i in active[:]:
            if proc.poll() is not None:
                handle.close();active.remove((proc,handle,i))
                if proc.returncode:
                    for other,fh,_ in active: other.terminate();fh.close()
                    raise RuntimeError(f'Generation failed: {split}/{i}; see log')
                print('Completed background',split,i,flush=True)
        if active: time.sleep(2)
    rows=[x for i in range(COUNTS[split]) for x in read(OUT/f'datasets/{split}/{i:03d}/index.json')]
    write(OUT/f'datasets/{split}/index.json',rows)


def noise_check():
    """Moment checks test the acquisition law, not learned performance."""
    verify()
    sample,_=render.oldg.baseline.c.sampled(render.old.load_config_bundle(ROOT/'DataSimulation/config'),BASES['diagnostic'],0)
    noise=replace(sample.noise,thermal_enabled=False,contamination_enabled=False,poisson_enabled=True)
    profile=AcquisitionProfile()
    q=np.full((256,256,4),.05,'f4'); records=[]
    for m in MULTIPLIERS:
        p=replace(profile,effective_count_rate=profile.effective_count_rate*m)
        x,_=acquire(q,noise,seed=1027,dwell_us=8.,profile=p)
        expected=.05/(p.effective_count_rate*8e-6)
        actual=float(np.var(x-q)); bias=float(np.mean(x-q))
        assert abs(actual/expected-1)<.025
        assert abs(bias)<5*np.sqrt(expected/q.size)
        records.append(dict(multiplier=m,measured_variance=actual,expected_variance=expected,bias=bias))
    zero,_=acquire(q,replace(noise,poisson_enabled=False),seed=1027,dwell_us=8.,profile=profile)
    assert np.array_equal(q,zero)
    write(OUT/'noise_checks.json',dict(status='passed',records=records,no_noise_identity=True,
        expected_std_ratios=[1.,.5,.125],calibrated_primary_electrons=False))


def audit():
    verify()
    from .learning import verify_freeze
    verify_freeze()
    sets={};counts={};physical_files={}; pairs={}
    for split,n in COUNTS.items():
        rows=read(OUT/f'datasets/{split}/index.json')
        assert len(rows)==n*18
        sets[split]={x['seed'] for x in rows};counts[split]=len(rows)
        assert len(sets[split])==n
        for row in rows:
            physical_files[row['raw_path']]=row['raw_sha256']
            physical_files[row['geometry_path']]=row['geometry_sha256']
            a=np.load(row['raw_path'],mmap_mode='r')
            assert a.shape==(160,5024) and np.isfinite(a).all()
            key=(row['background_id'],row['source_class'])
            pairs.setdefault(key,set()).add(row['geometry_sha256'])
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    assert all(len(x)==1 for x in pairs.values())
    for path,digest in physical_files.items(): assert sha(path)==digest,path
    for path,digest in read(OUT/'historical_inventory.json').items(): assert sha(path)==digest,path
    write(OUT/'audit.json',dict(status='passed',counts=counts,independent_backgrounds={k:len(v) for k,v in sets.items()},
        matched_geometry=True,historical_preserved=True,checked_files=len(physical_files)))
