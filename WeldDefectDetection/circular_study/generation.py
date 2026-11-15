"""Resumable full-ring dataset with prospective split and renderer provenance."""
from pathlib import Path
from dataclasses import replace,asdict
from unittest.mock import patch
import gc,json,time,zipfile
import numpy as np
from scipy.ndimage import map_coordinates
import run_circular_campaign_20261004 as baseline
from box_comparison.core import write,sha
from . import geometry as g

old=baseline.old
ROOT=old.ROOT
OUT=ROOT/'DataSimulation/data/output/circular_study_20261004_v4'
COUNTS={'train':60,'val':20,'test':30}
BASES={'train':2200400000,'val':2210400000,'test':2220400000}
SEEDS=[100482001,100482002,100482003]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    paths=baseline.c.source_paths()+[Path(__file__),Path(g.__file__),Path(baseline.__file__),Path(old.__file__),ROOT/'WeldDefectDetection/run_acquisition_campaign_20260928.py']
    protocol=dict(schema='circular-thesis-study/v4',counts=COUNTS,seed_bases=BASES,
        methods=['neighbouring_median','TinyDefectSegNet'],channels=['TOPO','Tx','Ty'],
        primary_channel='TOPO',geometry='periodic physical-coordinate cylindrical surrogate; recessed main population',
        pixel_size_mm=.05,arc_pitch_mm=g.ARC_PITCH,radius_mm=40.,annulus_shape=[96,g.COLS],evaluation_rows=[16,80],
        classes=list(baseline.c.KINDS),amplitudes=[0.,.05,.1],
        paired='Same background and acquisition across clean/keyhole/intrinsic; no visibility exclusion',
        population='Latin-style deterministic size/angle/offset coverage with independent within-bin draws; split by background',
        orientation='keyhole long axis and tail tangent to local circle, intrinsic random local angle',
        physics='Existing 304SS 60 keV formula renderer and raster_v2; B retained as an assumed constant, not a calibrated fit',
        geometry_changes=['periodic fixed-physical-grid texture','finite fusion/HAZ transitions','locally oriented pore','continuous annular output'],
        geometry_limit='Joint revised geometry distribution; no isolated causal claim versus historical generator. Closed-fusion and B=0 are sensitivity diagnostics only.',
        labels='Existing occupancy > .08, all nonempty surface intersections retained; do not resample weak signals',
        training=dict(seeds=SEEDS,epochs=20,batch_size=32,examples_per_epoch=1200,
            design='Screen ordered one-factor CNN alternatives on seed 1, then repeat baseline and validation-selected alternative with seeds 2/3. Freeze before new test generation.',
            variants=['fixed_negatives','rotating_negatives','hard_negatives','augmentation','larger_context','focal_loss','capacity24','topo_tx_ty'],
            selection='Validation periodic box F1 at IoU .5, then pixel IoU; second operating point maximises recall with <=10% clean-ring alarms',
            inputs='64 square; larger-context 96 square with central64 supervised; inference stride32, stitched before components'),
        median=dict(variants=['legacy_scale','centred','aligned','conditional','neighbours8','context_adaptive'],
            context='Disjoint 64-wide windows at offsets +/-64,+/-128; eight-neighbour variant extends to +/-256',
            adaptation='Context only, bounded scale multiplier .75 to 2.0; no labels at inference'),
        test='Generate only after frozen model hashes. Thirty fresh independent backgrounds. Old 12 backgrounds diagnostic only.',
        physical_validation=False,source_hashes={str(p.resolve()):sha(p) for p in sorted(set(paths))})
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==protocol,'Renderer or protocol changed after freeze'
    else:
        write(OUT/'protocol.json',protocol)
        with zipfile.ZipFile(OUT/'renderer_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(paths)):z.write(p,str(p.relative_to(ROOT)))


class Engine(baseline.CircularDatasetEngine):
    def _top_surface_section(self,volume,**kwargs):
        section,defects=old.se.ScanEngine._top_surface_section(volume,min_visible_area_px=1)
        self.last_relief=section.relief_mm_map
        return section,defects

    def _apply_adaptive_quadrant_blur(self,quadrants,*,sigma_px,blur_scale_map):
        reach=np.sqrt(2)*(int(4*sigma_px*min(4.,max(1.,float(blur_scale_map.max())))+.5)+2)*.05
        if 2.4+reach>=4.:raise ValueError('Insufficient annular quadrature halo')
        return old.se.ScanEngine._apply_adaptive_quadrant_blur(self,quadrants,sigma_px=sigma_px,blur_scale_map=blur_scale_map)


def unwrap(array,order=1):
    radial=g.RADIUS+(np.arange(96)-47.5)*.05
    angle=2*np.pi*np.arange(g.COLS)/g.COLS
    coords=np.stack(np.broadcast_arrays((radial[:,None]*np.sin(angle)+(old.N-1)*.025)/.05,
                                       (radial[:,None]*np.cos(angle)+(old.N-1)*.025)/.05))
    return map_coordinates(array,coords,order=order,mode='nearest')


def render(engine,sample,full=False,legacy=False):
    cached={}
    def profile(**kw):
        if not cached:cached['height']=g.profile(**kw,amplitude=engine.amplitude)
        return cached['height'].copy()
    # Patches are scoped; no historical source or global configuration is edited.
    with patch.object(old,'generate_weld_profile',old.generate_weld_profile if legacy else profile),patch.object(old.se,'build_voxel_volume',old.top_volume),patch.object(old.se,'apply_defect_bodies',old.vv.apply_defect_bodies if legacy else g.inject),patch.object(baseline,'PAD_MM',4.),patch.object(old.se,'compute_quadrant_signals',old.chunked_quadrants if full else lambda **kw:baseline.annular_quadrants(engine.quadrature_cache,**kw)):
        result=engine.run(sample)
    gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
    quiet=baseline.form_channels(engine.acquisition_quiet_quadrants).topo*gain
    return result,np.stack([unwrap(getattr(result.channels,k)) for k in ('topo','tx','ty')]).astype('f4'),unwrap(quiet).astype('f4'),unwrap(result.defect_mask,0).astype('u1')


def plan(kind,seed,index):
    if kind=='clean':return []
    rng=np.random.default_rng(seed+310003)
    diamlo,diamhi=(.5,1.2) if kind=='keyhole_pore' else (.5,1.)
    diam=diamlo+(diamhi-diamlo)*((index//3)%3+rng.random())/3
    depth=(.25+(.35 if kind=='keyhole_pore' else .40)*rng.random())
    angle=2*np.pi*(index%4+rng.random())/4
    offset=(-.15,0.,.15)[(index//4)%3]
    return [dict(type=kind,params=dict(diameter_mm=float(diam),depth_mm=float(depth),angle_rad=float(angle),radial_offset_mm=offset))]


def generate(split):
    baseline.c.verify_sources(OUT)
    if split=='test':
        assert (OUT/'frozen_models.json').exists(),'Freeze before test generation'
        for p,h in read(OUT/'frozen_models.json')['files'].items():assert sha(p)==h
    folder=OUT/'datasets'/split;folder.mkdir(parents=True,exist_ok=True)
    cfg=old.load_config_bundle(ROOT/'DataSimulation/config');rows=[]
    for i in range(COUNTS[split]):
        seed=BASES[split]+i;amp=(0.,.05,.1)[(i//3+i%3)%3]
        complete=all((folder/f'{split}-{i:03d}-{k}.json').exists() for k in baseline.c.KINDS)
        if complete:
            for kind in baseline.c.KINDS:
                r=read(folder/f'{split}-{i:03d}-{kind}.json');assert sha(r['npz_path'])==r['npz_sha256'];rows.append(r)
            continue
        sample,_=baseline.c.sampled(cfg,seed,i);engine=Engine(seed,amp);clean=None
        for kind in baseline.c.KINDS:
            start=time.monotonic();sid=f'{split}-{i:03d}-{kind}'
            s=replace(sample,defects=plan(kind,seed,i),randomize_defect_location=True)
            result,raw,quiet,mask=render(engine,s)
            assert np.isfinite(raw).all() and np.isfinite(quiet).all()
            assert bool(mask.any())==(kind!='clean'),'Absent label retained as failure, never resampled'
            if kind=='clean':clean=quiet.copy()
            valid=mask[16:80]>0;delta=quiet-clean
            path=folder/f'{sid}.npz'
            np.savez_compressed(path,raw=raw,quiet=quiet,mask=mask)
            row=dict(scene_id=sid,background_id=f'{split}-{i:03d}',split=split,index=i,seed=seed,
                source_class=kind,amplitude=amp,parameters=asdict(s),renderer_metadata=result.metadata,
                npz_path=str(path.resolve()),npz_sha256=sha(path),
                area_px=int(valid.sum()),signal_change_rms=float(np.sqrt(np.mean(delta**2))),
                signal_change_in_mask=float(np.mean(abs(delta[16:80][valid]))) if valid.any() else 0.,
                noise_rms=float(np.sqrt(np.mean((raw[0]-quiet)**2))),seconds=time.monotonic()-start)
            write(folder/f'{sid}.json',row);rows.append(row)
            print(f'{sid} {row["seconds"]:.1f}s area={row["area_px"]}',flush=True)
            del result,raw,quiet,mask;gc.collect()
        del engine;gc.collect()
    write(folder/'index.json',rows)


def normalize():
    rows=read(OUT/'datasets/train/index.json');values=[];signed=[]
    for r in rows:
        with np.load(r['npz_path']) as a:
            raw=a['raw'];values.append(raw[0,16:80,::8].ravel());signed.append(abs(raw[1:,16:80,::8]).ravel())
    lo,hi=np.quantile(np.concatenate(values),[.01,.99]);pad=.08*(hi-lo)
    write(OUT/'normalization.json',dict(low=max(0.,float(lo-pad)),high=float(hi+pad),signed_scale=float(np.quantile(np.concatenate(signed),.995)),source='train only, deterministic every eighth angular sample',tone='sqrt TOPO; signed Tx/Ty linear clipped +/-1'))


def load(split,channels=1,quiet=False):
    norm=read(OUT/'normalization.json');rows=read(OUT/f'datasets/{split}/index.json');images=[];masks=[];clipping=[]
    for r in rows:
        with np.load(r['npz_path']) as a:
            raw=a['raw'][:channels];t=a['quiet'] if quiet else raw[0]
            clipping.append(dict(scene_id=r['scene_id'],low=float(np.mean(t<norm['low'])),high=float(np.mean(t>norm['high']))))
            x=np.sqrt(np.clip((t-norm['low'])/(norm['high']-norm['low']),0,1))[None]
            if channels==3:x=np.concatenate([x,np.clip(raw[1:]/norm['signed_scale'],-1,1)])
            images.append(x.astype('f4'));masks.append(a['mask'][16:80].astype('u1'))
    write(OUT/f'clipping_{split}_{"quiet" if quiet else "raw"}.json',clipping)
    return dict(x=np.stack(images),mask=np.stack(masks),meta=rows)
