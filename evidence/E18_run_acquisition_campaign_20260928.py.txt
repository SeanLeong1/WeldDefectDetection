"""Prospective raster-v2 campaign, independent of the frozen legacy campaign.

Run with the repository .venv Python. Reuses audited geometry/CV primitives,
freezes the new assumptions before development, then freezes models before
confirmation. Historical datasets and checkpoints are never overwritten.
"""
from pathlib import Path
from dataclasses import asdict, replace
from unittest.mock import patch
import argparse
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'DataSimulation'), str(ROOT/'WeldDefectDetection')]
import numpy as np
import torch
from bse_renderer.noise.raster_acquisition import AcquisitionProfile, stream, acquire
from bse_renderer.renderer.image_formation import form_channels
from bse_renderer.renderer import scan_engine as se
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.models import WeldLayoutRuntime
from variability_campaign import campaign as c
from variability_campaign.renderer import VariableSurfaceEngine
from circular_examples_20260928 import top_volume, chunked_quadrants

OUT = ROOT/'DataSimulation/data/output/acquisition_campaign_20260928_v2'
SEEDS = (928810001, 928810002, 928810003)
ARMS = ('median', 'cnn')


def profile_for(seed):
    r = stream(seed, 'acquisition_profile')
    return AcquisitionProfile(effective_count_rate=float(np.exp(r.uniform(np.log(5e8), np.log(2e9)))),
        readout_rms=float(r.uniform(.0002, .0008)), baseline_rms=float(r.uniform(.0001, .0004)),
        common_gain_rms=float(r.uniform(.001, .004)), correlation_time_s=float(r.uniform(.1, 1.)),
        contamination_per_s=float(r.uniform(.0001, .0004)), evidence='assumed_sensitivity_range_not_measured')


class Engine(VariableSurfaceEngine):
    def run(self, sample, session_state=None):
        self.acquisition_profile = profile_for(sample.seed)
        return super().run(sample, session_state)


def engine(method='finite_difference'):
    return Engine(image_size=c.FRAME*c.FRAMES, pixel_size_mm=c.SPACING,
        render_model_mode='formula_only', emission_b_mode='assumed_constant', emission_b_path=str(c.MODEL),
        render_view_mode='top_surface', weld_layout=WeldLayoutRuntime(mode='straight_seam'),
        frame_size=c.FRAME, halo=32, gradient_method=method, acquisition_model='raster_v2')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        c.verify_sources(root); return
    paths = c.source_paths()+[Path(__file__), ROOT/'WeldDefectDetection/circular_examples_20260928.py',
                             ROOT/'docs/acquisition_model_v2.md']
    p = dict(schema='raster-acquisition-cv/v2', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        objective='Evaluate CNN and neighbour median under a declared raster-timed acquisition model.',
        development=dict(backgrounds=100, seed=1262800000, train=60, val=20, test=20, amplitudes=[0.,.05,.10]),
        confirmation=dict(backgrounds=30, seed=1362800000, amplitudes=[0.,.05,.10], generated_after_model_freeze=True),
        pilot=dict(backgrounds=3, seed=1462800000), methods=list(ARMS),
        training=dict(seeds=list(SEEDS), epochs=20, threads=4, architecture='TinyDefectSegNet target-only'),
        domain=dict(material='304SS', energy_kev=60., B=c.read(c.MODEL)['fitted_b'], B_status='engineering_assumption',
                    pixel_size_mm=.05, frame_size=64, frames=9, gradient_method='finite_difference'),
        acquisition=dict(version='raster_v2', measured_calibration=False, reference=asdict(AcquisitionProfile()),
            sensitivity_ranges=dict(effective_count_rate=[5e8,2e9], readout_rms=[.0002,.0008], baseline_rms=[.0001,.0004],
                common_gain_rms=[.001,.004], correlation_time_s=[.1,1.], contamination_per_s=[.0001,.0004]),
            sampling='Count rate log-uniform, other ranges uniform, named stream per background, paired across classes.',
            geometric_velocity_mm_s=[0.,0.], geometry_motion='Implemented and tested separately; excluded from this experiment.',
            raster='Each 128x576 detector strip is a raster acquisition; extracted target is its central 64 rows.'),
        pairing='Every background has clean, keyhole and intrinsic scenes in one split, identical electronic random fields; Poisson samples depend on signal.',
        variation='Bounded smooth procedural width/relief changes at 0, 5 and 10 percent, not a manufacturing acceptance standard.',
        labels='Unfiltered geometric surface-slab masks; no invisible-case exclusion.',
        selection='Train-only raw percentile normalization; validation-only thresholds/checkpoints, no confirmation retuning.',
        metrics='Box IoU>=0.5 one-to-one F1 and recall, clean-source false alarms, 2000 paired background bootstrap draws.',
        real_reference='User-supplied poor-quality screenshot with multiple test rings; qualitative only, unidentified channels, no raw noise calibration.',
        source_hashes={str(p.resolve()):c.sha(p) for p in sorted(set(paths))})
    c.write(root/'protocol.json', p)
    with zipfile.ZipFile(root/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in p['source_hashes']: z.write(name, str(Path(name).relative_to(ROOT)))
    c.seal(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def pilot(root):
    # Verify slim top-surface allocation against the full native volume,
    # including both pore types, for the straight layout used in this campaign.
    from bse_renderer.geometry import voxel_volume as vv
    cfg=load_config_bundle(ROOT/'DataSimulation/config'); sample,_=c.sampled(cfg,1462800000,0)
    kw=dict(beam=sample.beam,material=sample.material,image_size=96,pixel_size_mm=.05,
        voxel_depth_mm=.08,crown_height_mm=.012,seam_depth_mm=.14,bead_width_mm=1.1,haz_width_mm=1.,
        beam_focus_quality=.35,asymmetry_offset_mm=.02,weld_geometry_mode='recessed_seam',
        weld_layout=WeldLayoutRuntime(mode='straight_seam'),seed=51)
    errors={}
    for kind in c.KINDS:
        native,slim=vv.build_voxel_volume(**kw),top_volume(**kw)
        plan=c.pore_parameters(kind,np.random.default_rng(7))
        for v in (native,slim): vv.apply_defect_bodies(v,plan,np.random.default_rng(51),randomize_position=False,anchor_to_surface=True)
        a,_=se.ScanEngine._top_surface_section(native); b,_=se.ScanEngine._top_surface_section(slim)
        errors[kind]={f:float(np.max(np.abs(getattr(a,f).astype(float)-getattr(b,f)))) for f in
            ('relief_mm_map','material_z_map','yield_modifier_map','void_fraction_map','defect_mask')}
        assert max(errors[kind].values())==0
    rows=[]
    for i in range(3):
        sample,_=c.sampled(cfg,1462800000+i,i); e=engine(); e.set_variation(.1)
        for kind in c.KINDS:
            s=replace(sample,defects=c.pore_parameters(kind,np.random.default_rng(sample.seed+310003)))
            with patch.object(se,'build_voxel_volume',top_volume): result=e.run(s)
            assert np.isfinite(result.channels.topo).all()
            assert bool(result.defect_mask.any()) == (kind!='clean')
            assert result.metadata['acquisition']['version']=='raster_v2'
            rows.append(dict(seed=sample.seed,kind=kind,noise_rms=float(np.sqrt(np.mean((result.channels.topo-e.quiet_topo)**2)))))
    c.write(root/'pilot.json',dict(status='passed',surface_equivalence=errors,scenes=rows))
    c.seal(root,'pilot',[root/'pilot.json'])


def train(root):
    c.verify_sources(root); c.require(root,'development'); c.require(root,'pilot')
    c.fit_normalization(root)
    data=c.load(root,'development'); training=c.subset(data,'train'); val=c.subset(data,'val')
    folder=root/'training'; folder.mkdir(exist_ok=True)
    scale,diagnostic=c.fit_scale(training,'median',8.)
    spec=dict(arm='median',implementation='local_variation',background_sigma=8.,scale=scale.tolist(),scale_fit=diagnostic)
    chosen,trials=c.select(val,c.simple_scores(val,spec),(1.,1.5,2.,3.,4.,6.,8.),min_areas=(3,15),closings=(1,3),low_ratios=(1.,),paddings=(0,))
    spec.update(chosen); c.write(folder/'median.json',dict(spec=spec,trials=trials))
    candidates=dict(median=[spec],cnn=[])
    for seed in SEEDS: candidates['cnn'].append(c.fit(training,val,folder,'cnn',seed,1,'cnn',epochs=20))
    selected={a:max(candidates[a],key=lambda s:c.selection_key(s['metrics'])) for a in ARMS}
    c.write(root/'model_candidates.json',candidates)
    c.write(root/'frozen_models.json',dict(models=selected,primary='cnn',normalization_sha256=c.sha(root/'normalization.json'),
        protocol_sha256=c.sha(root/'protocol.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    c.seal(root,'train',[root/'model_candidates.json',root/'frozen_models.json',root/'normalization.json',*folder.glob('*.pt'),*folder.glob('*.json')])


def evaluate(root):
    c.verify_sources(root); c.require(root,'confirmation'); c.require(root,'train')
    models=c.read(root/'frozen_models.json')['models']; results={}
    for quiet in (False,True):
        label='quiet' if quiet else 'noisy'; data=c.load(root,'confirmation',quiet)
        rows={a:c.evaluate_spec(data,s) for a,s in models.items()}; results[label]={}
        for a,rr in rows.items(): c.write(root/'predictions'/f'{label}_{a}.json',rr)
        for level in (0.,.05,.10):
            grouped={a:[r for r in rr if r['amplitude']==level] for a,rr in rows.items()}
            results[label][str(level)]=dict(metrics={a:c.extended_summary(rr) for a,rr in grouped.items()},
                uncertainty=c.clustered_intervals(grouped),per_pore={a:{k:c.extended_summary([r for r in rr if r['source_class']==k])
                    for k in c.KINDS[1:]} for a,rr in grouped.items()})
    test=c.subset(c.load(root,'development'),'test')
    results['reserved_development_test']={a:c.extended_summary(c.evaluate_spec(test,s)) for a,s in models.items()}
    results['claim']='Held-out synthetic evidence under unmeasured acquisition assumptions; no real-image or acceptance validation.'
    c.write(root/'results.json',results)
    c.seal(root,'evaluate',[root/'results.json',*(root/'predictions').glob('*.json')])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',default='all',choices=['all','prepare','pilot','development','train','confirmation','evaluate'])
    args=parser.parse_args(); torch.set_num_threads(4)
    for stage in (['prepare','pilot','development','train','confirmation','evaluate'] if args.stage=='all' else [args.stage]):
        if (OUT/'receipts'/f'{stage}.json').exists():
            c.verify_sources(OUT); c.require(OUT,stage); print('Verified '+stage,flush=True); continue
        print('Starting '+stage,flush=True)
        if stage in ('development','confirmation'):
            with patch.object(c,'engine',engine),patch.object(se,'build_voxel_volume',top_volume): c.generate(OUT,stage)
        else: globals()[stage](OUT)
        print('Completed '+stage,flush=True)


if __name__=='__main__': main()
