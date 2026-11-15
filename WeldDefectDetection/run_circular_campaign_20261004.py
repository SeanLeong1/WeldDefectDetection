"""Circular-only, background-disjoint synthetic CNN/median benchmark.

Run with .venv/Scripts/python.exe. Historical experiments are read-only.
Only the declared inspection annulus is rendered; a padded annulus preserves
the native Cartesian gradients and interaction blur inside the inspected band.
"""
from pathlib import Path
from dataclasses import replace, asdict
from unittest.mock import patch
import argparse
import gc
import json
import time
import zipfile
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import circular_examples_20260928 as old
from variability_campaign import campaign as c
from run_acquisition_campaign_20260928 import profile_for, OUT as STRAIGHT
from bse_renderer.renderer.image_formation import form_channels
from box_development.methods import fit_scale, simple_scores, evaluate_spec, scores_for
from box_development.training import fit
from box_refinement.methods import select
from box_comparison.core import write, sha, subset, truth_box, selection_key

ROOT = old.ROOT
OUT = ROOT/'DataSimulation/data/output/circular_campaign_20261004_v1'
COUNTS = {'train': 24, 'val': 9, 'test': 12}
SEEDS = (100481001, 100481002, 100481003)
PAD_MM = 3.2
FIELDS = ('material_z_map', 'yield_modifier_map', 'void_fraction_map', 'relief_mm_map', 'defect_mask')


class CircularDatasetEngine(old.CircularEngine):
    def __init__(self, seed, amplitude):
        super().__init__()
        self.sample_seed = seed
        self.amplitude = amplitude
        self.acquisition_model = 'raster_v2'
        self.acquisition_profile = profile_for(seed)
        self.quadrature_cache = {}

    def _top_surface_section(self, volume, **kwargs):
        # Initialize the parent's baseline without applying its fixed example fields.
        if self.base is None:
            amplitude = self.amplitude
            self.amplitude = 0.
            super()._top_surface_section(volume, **kwargs)
            knots = np.linspace(-np.pi, np.pi, 41)
            rng = np.random.default_rng(self.sample_seed + 170003)
            self.fields = []
            for _ in range(2):
                values = rng.uniform(-1, 1, len(knots)); values[-1] = values[0]
                spline = CubicSpline(knots, values, bc_type='periodic')
                scale = np.max(np.abs(spline(np.linspace(-np.pi, np.pi, 20001))))
                self.fields.append((spline(self.theta)/scale).astype(np.float32))
            self.amplitude = amplitude
        return super()._top_surface_section(volume, **kwargs)

    def _apply_adaptive_quadrant_blur(self, quadrants, *, sigma_px, blur_scale_map):
        # scipy Gaussian kernels have support round(4*sigma) in EACH Cartesian axis.
        max_scale = min(4., max(1., float(np.max(blur_scale_map))))
        reach_mm = np.sqrt(2)*(int(4*sigma_px*max_scale + .5)+2)*old.SPACING
        if 1.6 + reach_mm >= PAD_MM:
            raise ValueError(f'Annulus padding inadequate: {reach_mm}')
        return super()._apply_adaptive_quadrant_blur(quadrants, sigma_px=sigma_px, blur_scale_map=blur_scale_map)


def annular_quadrants(cache=None, **kw):
    """Exact native quadrature on a padded ROI, zero placeholder outside it.

    The placeholder is never exported as a valid measurement. Full raster timing
    and independent electronic fields still run on the original Cartesian grid.
    """
    h, w = kw['normals'].shape[:2]
    axis = (np.arange(h)-(h-1)/2)*old.SPACING
    roi = np.abs(np.hypot(axis[:, None], axis[None, :])-old.RADIUS) <= PAD_MM
    arrays={k:v for k,v in kw.items() if isinstance(v,np.ndarray) and v.shape[:2]==(h,w)}
    scalars={k:v for k,v in kw.items() if k not in arrays}
    if cache and cache['scalars']==scalars and cache['arrays'].keys()==arrays.keys():
        changed=np.zeros((h,w),bool)
        for key,value in arrays.items():
            difference=value!=cache['arrays'][key]
            if difference.ndim>2:difference=np.any(difference,axis=tuple(range(2,difference.ndim)))
            changed|=difference
        ids=np.flatnonzero(roi & changed)
        out=cache['output'].copy()
    else:
        ids = np.flatnonzero(roi)
        out = np.zeros((h*w, 4), np.float32)
    for start in range(0, len(ids), 8192):
        ix = ids[start:start+8192]
        part = {k: (v.reshape((-1,)+v.shape[2:])[ix][None]
                    if isinstance(v, np.ndarray) and v.shape[:2] == (h,w) else v)
                for k,v in kw.items()}
        out[ix] = old.compute_quadrant_signals(**part)[0]
    if cache is not None and not cache:
        cache.update(arrays={k:v.copy() for k,v in arrays.items()},scalars=scalars,output=out.copy())
    return out.reshape(h,w,4)


def run_render(engine, sample, full=False):
    with patch.object(old.se, 'build_voxel_volume', old.top_volume), patch.object(
            old.se, 'compute_quadrant_signals', old.chunked_quadrants if full else lambda **kw: annular_quadrants(engine.quadrature_cache,**kw)):
        result = engine.run(sample)
    gain = float(np.interp(sample.beam.focusing_current_norm, [0.,1.], [1.10,.92]))
    quiet = form_channels(engine.acquisition_quiet_quadrants).topo*gain
    raw, masks = old.crop_ring(result.channels.topo, result.defect_mask)
    quiet_crops, _ = old.crop_ring(quiet, result.defect_mask)
    return result, raw, quiet_crops, masks


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = c.source_paths()+[Path(__file__), Path(old.__file__), ROOT/'WeldDefectDetection/run_acquisition_campaign_20260928.py']
    protocol = dict(schema='circular-only-cv/v1', counts=COUNTS, methods=['cnn','median'],
        geometry=dict(layout='circular_weld', diameter_mm=80., cartesian_shape=[old.N,old.N],
            pixel_size_mm=.05, sectors=78, crop_shape=[64,64], inspected_half_width_mm=1.6,
            detector_quadrature_half_width_mm=PAD_MM, sector_centres_deg='90 + 360*i/78',
            coverage='Annular band only; 0.022 mm nominal centreline gaps. Outside band is not valid image data.'),
        classes=list(c.KINDS), amplitudes=[0.,.05,.1],
        pairing='Three classes share background, amplitude, acquisition and split. Fresh periodic fields per background.',
        defects='One surface-expressed pore; random native weld location. No label/contrast filtering.',
        splits='Disjoint latent background seeds. All sectors and class variants stay together.',
        seed_bases=dict(train=2100400000,val=2110400000,test=2120400000),
        acquisition='raster_v2, same assumed profile ranges as straight v2, full Cartesian raster clock',
        roi_noise='Uncomputed exterior quadrant values are zero before acquisition. Shot-noise realizations differ from full-frame rendering; same local Poisson law. Quiet retained sectors must match full rendering.',
        quadrature_cache='Reuse baseline only at pixels with exactly identical complete quadrature inputs; changed pixels recomputed.',
        training=dict(seeds=list(SEEDS),epochs=20,normalization='train pixels only, 1/99 quantiles with 8% padding, sqrt',
            cnn_sampling='All positive sectors plus eight deterministic random negative sectors per training scene; existing balanced sampler.',
            median_sampling='All clean training sectors with four clean neighbours.',
            validation='All 78 sectors; checkpoint/threshold selection only on validation.',
            architecture='TinyDefectSegNet, target only'),
        evaluation='All 78 sectors for every held-out scene; IoU>=0.5 one-to-one box matching; pixel IoU; clean-source and negative-sector false alarms; ring-cluster bootstrap.',
        test_generation='After model freeze. No tuning on test results.',
        physical_validation=False,
        source_hashes={str(p.resolve()):sha(p) for p in sorted(set(paths))})
    if (OUT/'protocol.json').exists():
        assert c.read(OUT/'protocol.json') == protocol, 'Frozen protocol or sources changed'
    else:
        write(OUT/'protocol.json',protocol)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in paths: z.write(p,str(p.relative_to(ROOT)))


def pilot():
    """Full native versus annular quiet sectors, including both pore classes."""
    cfg = old.load_config_bundle(ROOT/'DataSimulation/config')
    sample,_ = c.sampled(cfg,2130400000,0)
    sample = replace(sample,randomize_defect_location=True)
    errors = {}
    engines = [CircularDatasetEngine(sample.seed,.1),CircularDatasetEngine(sample.seed,.1)]
    for kind in c.KINDS:
        s = replace(sample,defects=c.pore_parameters(kind,np.random.default_rng(sample.seed+310003)))
        outputs = []
        for e,full in zip(engines,(True,False)):
            start=time.monotonic(); result,raw,quiet,masks=run_render(e,s,full)
            outputs.append((quiet,masks))
            print(f'pilot {kind} full={full} seconds={time.monotonic()-start:.1f}',flush=True)
            del result,raw;gc.collect()
        error=float(np.max(np.abs(outputs[0][0]-outputs[1][0])))
        assert error < 1e-7, error
        assert np.array_equal(outputs[0][1],outputs[1][1])
        assert bool(outputs[0][1].any()) == (kind!='clean')
        errors[kind]=dict(max_quiet_crop_error=error,labels_identical=True)
    write(OUT/'pilot.json',dict(status='passed',checks=errors))


def generate(split):
    if split=='test':
        c.require(OUT,'train')
    cfg=old.load_config_bundle(ROOT/'DataSimulation/config')
    protocol=c.read(OUT/'protocol.json')
    folder=OUT/'datasets'/split;folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(COUNTS[split]):
        seed=protocol['seed_bases'][split]+i
        amplitude=(0.,.05,.1)[(i//3+i%3)%3]
        sample,_=c.sampled(cfg,seed,i)
        sample=replace(sample,randomize_defect_location=True)
        e=CircularDatasetEngine(seed,amplitude)
        for kind in c.KINDS:
            sid=f'{split}-{i:03d}-{kind}'
            path=folder/f'{sid}.npz';meta=folder/f'{sid}.json'
            if path.exists() and meta.exists():
                row=c.read(meta);assert sha(path)==row['npz_sha256'];rows.append(row)
                if kind!='clean' or all((folder/f'{split}-{i:03d}-{k}.json').exists() for k in c.KINDS):
                    continue
            s=replace(sample,defects=c.pore_parameters(kind,np.random.default_rng(seed+310003)))
            started=time.monotonic()
            result,raw,quiet,masks=run_render(e,s)
            assert np.isfinite(raw).all() and np.ptp(raw)>0
            assert bool(masks.any()) == (kind!='clean'), 'Pore absent from inspection band; investigate, do not resample'
            assert result.metadata['acquisition']['version']=='raster_v2'
            row=dict(scene_id=sid,background_id=f'{split}-{i:03d}',split=split,seed=seed,
                source_class=kind,amplitude=amplitude,parameters=asdict(s),
                renderer_metadata=result.metadata,npz_path=str(path.resolve()),
                positive_sectors=int(masks.reshape(78,-1).any(1).sum()),
                seconds=time.monotonic()-started)
            # A full-grid annulus preview is saved for the first ring of each split.
            # Invalid exterior is explicitly NaN, never passed into a CV method.
            if i==0:
                axis=(np.arange(old.N)-(old.N-1)/2)*old.SPACING
                valid=np.abs(np.hypot(axis[:,None],axis[None,:])-old.RADIUS)<=1.6
                np.savez_compressed(folder/f'{sid}.ring.npz',raw=np.where(valid,result.channels.topo,np.nan),
                                    mask=result.defect_mask,valid=valid)
            np.savez_compressed(path,raw=raw,quiet=quiet,mask=masks)
            row['npz_sha256']=sha(path);write(meta,row)
            # Replace a reused clean metadata row when rebuilding baseline on resume.
            rows=[r for r in rows if r['scene_id']!=sid]+[row]
            print(f'{sid}: {row["seconds"]:.1f}s, positive sectors={row["positive_sectors"]}',flush=True)
            del result,raw,quiet,masks;gc.collect()
        del e;gc.collect()
    write(folder/'index.json',rows)


def normalize():
    rows=c.read(OUT/'datasets/train/index.json')
    values=[]
    for r in rows:
        with np.load(r['npz_path']) as a: values.append(a['raw'].reshape(-1))
    lo,hi=np.quantile(np.concatenate(values),[.01,.99]);pad=.08*(hi-lo)
    write(OUT/'normalization.json',dict(raw_low=max(0.,float(lo-pad)),raw_high=float(hi+pad),
        tone_curve='sqrt',source_split='train',scene_ids=[r['scene_id'] for r in rows]))


def load(split, quiet=False, norm=None):
    norm=norm or c.read(OUT/'normalization.json')
    targets=[];contexts=[];masks=[];meta=[]
    for r in c.read(OUT/f'datasets/{split}/index.json'):
        assert sha(r['npz_path'])==r['npz_sha256']
        with np.load(r['npz_path']) as a:
            frames=np.sqrt(np.clip((a['quiet' if quiet else 'raw']-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
            mm=a['mask']
        for t in range(78):
            ix=[(t+k)%78 for k in (-2,-1,1,2)]
            assert len(set(ix))==4 and t not in ix
            targets.append(frames[t]);contexts.append(frames[ix]);masks.append(mm[t])
            meta.append(dict(window_id=f'{r["scene_id"]}-s{t:02d}',sequence_id=r['background_id'],
                scene_id=r['scene_id'],split=split,source_class=r['source_class'],amplitude=r['amplitude'],
                target_frame=t,angle_deg=float((90+360*t/78)%360),truth_boxes=truth_box(mm[t]),
                context_defective_count=int(mm[ix].reshape(4,-1).any(1).sum())))
    context=np.stack(contexts)
    return dict(target=np.stack(targets),context=context,reference=np.median(context,axis=1),mask=np.stack(masks),meta=meta)


def cnn_training_subset(data):
    rng=np.random.default_rng(100499001);ids=[]
    for sid in sorted({m['scene_id'] for m in data['meta']}):
        scene=[i for i,m in enumerate(data['meta']) if m['scene_id']==sid]
        positive=[i for i in scene if data['mask'][i].any()]
        negative=[i for i in scene if not data['mask'][i].any()]
        ids.extend(positive+sorted(rng.choice(negative,min(8,len(negative)),replace=False).tolist()))
    return {k: [v[i] for i in ids] if k=='meta' else v[ids] for k,v in data.items()}


def train():
    normalize();data=load('train');val=load('val');folder=OUT/'training';folder.mkdir(exist_ok=True)
    scale,diagnostic=fit_scale(data,'median',8.)
    spec=dict(arm='median',implementation='local_variation',background_sigma=8.,scale=scale.tolist(),scale_fit=diagnostic)
    chosen,trials=select(val,simple_scores(val,spec),(1.,1.5,2.,3.,4.,6.,8.),min_areas=(3,15),closings=(1,3),low_ratios=(1.,),paddings=(0,))
    spec.update(chosen);write(folder/'median.json',dict(spec=spec,trials=trials))
    small=cnn_training_subset(data)
    write(OUT/'training_population.json',dict(all_training_sectors=len(data['meta']),cnn_selected_sectors=len(small['meta']),
        cnn_positive_sectors=int(small['mask'].reshape(len(small['meta']),-1).any(1).sum()),
        selected_window_ids=[m['window_id'] for m in small['meta']],validation_sectors=len(val['meta'])))
    del data;gc.collect()
    candidates=[fit(small,val,folder,'cnn',seed,1,'cnn',epochs=20) for seed in SEEDS]
    models=dict(median=spec,cnn=max(candidates,key=lambda s:selection_key(s['metrics'])))
    write(OUT/'model_candidates.json',dict(cnn=candidates,median=[spec]))
    write(OUT/'frozen_models.json',dict(models=models,normalization_sha256=sha(OUT/'normalization.json'),
        protocol_sha256=sha(OUT/'protocol.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    c.seal(OUT,'train',[OUT/'frozen_models.json',OUT/'normalization.json',OUT/'model_candidates.json',*folder.glob('*')])


def evaluate():
    c.require(OUT,'train');models=c.read(OUT/'frozen_models.json')['models'];results={}
    for quiet in (False,True):
        label='quiet' if quiet else 'noisy';data=load('test',quiet)
        rows={m:evaluate_spec(data,s) for m,s in models.items()}
        for m,rr in rows.items():write(OUT/f'predictions/{label}_{m}.json',rr)
        results[label]=dict(metrics={m:c.extended_summary(rr) for m,rr in rows.items()},uncertainty=c.clustered_intervals(rows),
            per_pore={m:{k:c.extended_summary([r for r in rr if r['source_class']==k]) for k in c.KINDS[1:]} for m,rr in rows.items()},
            per_amplitude={str(a):{m:c.extended_summary([r for r in rr if r['amplitude']==a]) for m,rr in rows.items()} for a in (0.,.05,.1)})
        del data;gc.collect()
    # A fair transfer comparison uses the same test rings and each model's own normalization.
    previous=c.read(STRAIGHT/'frozen_models.json')['models']
    data=load('test',norm=c.read(STRAIGHT/'normalization.json'))
    transfer={m:evaluate_spec(data,s) for m,s in previous.items()}
    for m,rr in transfer.items():write(OUT/f'predictions/straight_transfer_{m}.json',rr)
    results['straight_trained_same_test']={m:c.extended_summary(rr) for m,rr in transfer.items()}
    results['claim']='Held-out synthetic circular ring sectors, 12 independent test backgrounds; no measured SEM or acceptance validation.'
    write(OUT/'results.json',results)


def audit():
    allrows=[r for s in COUNTS for r in c.read(OUT/f'datasets/{s}/index.json')]
    seeds={s:{r['seed'] for r in allrows if r['split']==s} for s in COUNTS}
    assert all(not seeds[a]&seeds[b] for a in COUNTS for b in COUNTS if a<b)
    assert len(allrows)==3*sum(COUNTS.values())
    for r in allrows:
        assert sha(r['npz_path'])==r['npz_sha256']
        assert r['renderer_metadata']['acquisition']['version']=='raster_v2'
        assert r['renderer_metadata']['weld_layout']['mode']=='circular_weld'
    write(OUT/'dataset_audit.json',dict(status='passed',backgrounds=sum(COUNTS.values()),scenes=len(allrows),
        sectors=len(allrows)*78,split_backgrounds=COUNTS,split_scenes={s:3*n for s,n in COUNTS.items()},
        split_sectors={s:234*n for s,n in COUNTS.items()},seed_overlap=False,all_layouts='circular_weld',
        source_hashes_verified=True,all_scene_hashes_verified=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','prepare','pilot','development','train','test','evaluate','audit'])
    args=parser.parse_args();torch.set_num_threads(4)
    stages=['prepare','pilot','development','train','test','evaluate','audit'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage!='prepare':c.verify_sources(OUT)
        if (OUT/f'receipts/{stage}.json').exists():
            c.require(OUT,stage);print('Verified '+stage,flush=True);continue
        print('Starting '+stage,flush=True)
        if stage=='development':generate('train');generate('val')
        elif stage=='test':generate('test')
        else:globals()[stage]()
        if stage!='train':
            files = list((OUT/'datasets').rglob('*')) if stage in ('development','test') else [OUT/f'{"protocol" if stage=="prepare" else "results" if stage=="evaluate" else "dataset_audit" if stage=="audit" else stage}.json']
            c.seal(OUT,stage,[p for p in files if p.is_file()])
        print('Completed '+stage,flush=True)


if __name__=='__main__':main()
