"""Versioned study; all physical rendering is delegated to frozen crown code."""
from dataclasses import replace, asdict
from pathlib import Path
import gc, json, subprocess, sys, time, zipfile
import numpy as np
from raised_crown import campaign as prior, rendering as r
from box_comparison.core import sha, write

ROOT = prior.ROOT
OUT = ROOT/'DataSimulation/data/output/raised_acquisition_20261006_v1'
COUNTS = dict(train=24, val=24, test=24, stress=12)
BASES = dict(train=2410600000, val=2420600000, test=2430600000, stress=2440600000)
EDGES = (.1, .5, 2., 8., 20.)
SEEDS = (100606001, 100606002, 100606003)
ARMS = ('coverage', 'hard_negative')
read = prior.read


def dwell(split, index):
    """Equal bin counts, stratified log dwell, with training endpoint coverage."""
    b = index % 4; rank = index // 4; n = COUNTS[split] // 4
    rng = np.random.default_rng(BASES[split]+index+701)
    q = (rank + rng.uniform(.1, .9))/n
    value = float(np.exp(np.log(EDGES[b]) + q*np.log(EDGES[b+1]/EDGES[b])))
    if split == 'train' and index == 0: value = EDGES[0]
    if split == 'train' and index == COUNTS[split]-1: value = EDGES[-1]
    return value, b


def generation_sources():
    return sorted(set(prior.sources()+[Path(__file__), ROOT/'DataSimulation/config/beam_params.yaml']))


def verify(historical=False):
    p = read(OUT/'protocol.json')
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, f'Frozen generation source changed: {path}'
    if historical:
        for path, digest in read(OUT/'historical_inventory.json').items():
            assert sha(path) == digest, f'Historical artifact changed: {path}'
    return p


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists(): return verify()
    from raised_crown.learning import verify_freeze
    old = verify_freeze(); prior.verify_sources()
    assert read(prior.OUT/'audit.json')['status'] == 'passed'
    cfg = r.old.load_config_bundle(ROOT/'DataSimulation/config')
    # This range is already in the project configuration, not a new noise law.
    p = dict(schema='raised-acquisition-study/v1', counts=COUNTS, seed_bases=BASES,
        geometry=prior.geo.specification(), renderer='Unchanged raised_crown v5; B=0, raster_v2, same defect law',
        dwell_bins_us=EDGES, dwell_sampling='Equal backgrounds in four bins; stratified log spacing within bins',
        training_endpoints_us=[.1,20.], classes=prior.KINDS, stress_classes=prior.STRESS,
        split_unit='Independent background; all paired scene conditions stay together',
        target_population='Balanced synthetic acquisition challenge, not natural deployment prevalence',
        primary_comparison='hard_negative versus coverage, same images, seeds, optimizer updates and architecture',
        historical_comparison='Frozen 5 Oct selected CNN and median on the same fresh confirmation; different training draws, not a pure causal dwell ablation',
        primary_endpoint='Full-ring box F1 at IoU >= 0.5, with clean-ring false alarms and recall',
        selection='Within each arm: final epoch only, validation F1 then pixel IoU then fewer FP; select seed and operating point before test',
        reporting='All seeds, per-dwell-bin results, paired stratified background-bootstrap differences; no test-based model selection',
        hard_negative='After epochs 5,10,15: rank 168 clean training tiles per background by mean highest 64 probabilities, retain 16 per background; replace 300 of 600 negative draws per epoch',
        training=dict(seeds=SEEDS, arms=ARMS, epochs=20, patches_per_epoch=1200, positive_fraction=.5,
            architecture='TinyDefectSegNet base24', optimizer='AdamW lr .001 wd .0001', patch_size=64),
        test_gate='Generate fresh test/stress only after both arms and median are frozen',
        robustness='Same multiple-pore layouts, nominal and image-estimated coordinates; injected combined error for matched clean/close-pair scenes',
        sample_size='24 backgrounds per development/confirmation split, 12 stress; exploratory, not a power or manufacturing claim',
        physical_validation=False, electron_depth_cutoff_implemented=False,
        source_hashes={str(x):sha(x) for x in generation_sources()})
    paths = [prior.OUT/'frozen_models.json', prior.OUT/'normalization.json', prior.OUT/'protocol.json',
        prior.OUT/'results.json', prior.OUT/'robustness_results.json', prior.OUT/'delivery_manifest.json',
        ROOT/'data/releases/CURRENT_RAISED_RELEASE.json', ROOT/'data/releases/CURRENT_RELEASE.json']
    paths += [ROOT/x if old.get('portable') else Path(x) for x in old['files']]
    for split in prior.COUNTS:
        paths.append(prior.OUT/f'datasets/{split}/index.json')
        paths += [Path(x['npz_path']) for x in read(paths[-1])]
    write(OUT/'historical_inventory.json', {str(x):sha(x) for x in set(paths) if x.exists()})
    write(OUT/'protocol.json', p)
    with zipfile.ZipFile(OUT/'generation_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in generation_sources(): z.write(path, path.relative_to(ROOT))
    return p


def generate_background(split, index):
    verify()
    if split in ('test','stress'):
        from .learning import verify_freeze
        verify_freeze()
    seed = BASES[split]+index
    cfg = r.old.load_config_bundle(ROOT/'DataSimulation/config')
    sample, _ = r.oldg.baseline.c.sampled(cfg, seed, index)
    value, bin_id = dwell(split,index)
    sample = replace(sample, beam=replace(sample.beam,dwell_time_us=value))
    amplitude = (0.,.05,.1)[(index//3+index)%3]
    kinds = prior.STRESS if split == 'stress' else prior.KINDS+(('hidden',) if split == 'test' else ())
    folder = OUT/'datasets'/split; folder.mkdir(parents=True,exist_ok=True)
    engine = None; rows=[]
    for kind in kinds:
        sid=f'{split}-{index:03d}-{kind}'; path=folder/(sid+'.npz'); mp=path.with_suffix('.json')
        if mp.exists():
            row=read(mp); assert sha(path)==row['npz_sha256']; rows.append(row); continue
        if engine is None: engine=r.Engine(seed,amplitude)
        started=time.monotonic(); changed=replace(sample,defects=prior.plans(seed,index)[kind])
        arrays, metadata=r.render(engine,changed,retain_native=split=='stress')
        assert np.isfinite(arrays['raw']).all()
        assert bool(arrays['mask'].any()) == (kind not in ('clean','hidden'))
        np.savez_compressed(path,**arrays)
        row=dict(scene_id=sid,background_id=f'{split}-{index:03d}',split=split,seed=seed,index=index,
            source_class=kind,amplitude=amplitude,dwell_bin=bin_id,dwell_time_us=value,
            parameters=asdict(changed),metadata=metadata,npz_path=str(path),npz_sha256=sha(path),
            area_px=int(arrays['mask'][r.SCORE].sum()),seconds=time.monotonic()-started)
        write(mp,row); rows.append(row); print(sid,round(row['seconds'],1),'s',flush=True)
        del arrays; gc.collect()
    write(folder/f'background_{index:03d}.json',rows)


def generate(split, workers=4):
    verify(); logs=OUT/'logs'; logs.mkdir(exist_ok=True)
    pending=list(range(COUNTS[split])); active=[]
    while pending or active:
        while pending and len(active)<workers:
            index=pending.pop(0); handle=(logs/f'{split}_{index:03d}.log').open('a',encoding='utf-8')
            args=[sys.executable,str(ROOT/'WeldDefectDetection/run_raised_acquisition_20261006.py'),
                  'background','--split',split,'--index',str(index)]
            proc=subprocess.Popen(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((proc,handle,index))
        for proc,handle,index in active[:]:
            if proc.poll() is not None:
                handle.close(); active.remove((proc,handle,index))
                if proc.returncode: raise RuntimeError(f'{split} background {index} failed; see log')
                print('Generated',split,index,flush=True)
        if active: time.sleep(3)
    rows=[row for i in range(COUNTS[split]) for row in read(OUT/f'datasets/{split}/background_{i:03d}.json')]
    write(OUT/f'datasets/{split}/index.json',rows)


def audit():
    verify(historical=True)
    from .learning import verify_freeze
    verify_freeze(); sets={}; counts={}; coverage={}
    previous={x['seed'] for s in prior.COUNTS for x in read(prior.OUT/f'datasets/{s}/index.json')}
    for split,n in COUNTS.items():
        rows=read(OUT/f'datasets/{split}/index.json'); sets[split]={x['seed'] for x in rows}
        assert len(sets[split])==n and not sets[split]&previous
        assert len({x['scene_id'] for x in rows})==len(rows)
        counts[split]=len(rows); clean=[x for x in rows if x['source_class']=='clean']
        coverage[split]={str(b):dict(backgrounds=sum(x['dwell_bin']==b for x in clean),
            dwell_us=[x['dwell_time_us'] for x in clean if x['dwell_bin']==b]) for b in range(4)}
        for row in rows:
            assert sha(row['npz_path'])==row['npz_sha256']
            with np.load(row['npz_path']) as a:
                assert a['raw'].shape==(3,160,5024)
                assert np.array_equal(a['mask'],a['instances'].any(0))
                if row['source_class']=='hidden':
                    with np.load(Path(row['npz_path']).with_name(row['background_id']+'-clean.npz')) as b:
                        assert np.array_equal(a['raw'],b['raw'])
    assert all(not sets[a]&sets[b] for a in sets for b in sets if a<b)
    record=dict(status='passed',scenes=counts,backgrounds=COUNTS,dwell_coverage=coverage,
        split_overlap=False,old_background_overlap=False,historical_artifacts_unchanged=True,
        hidden_matches_clean=True,physical_validation=False)
    write(OUT/'audit.json',record); return record
