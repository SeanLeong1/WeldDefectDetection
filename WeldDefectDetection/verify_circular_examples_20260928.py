"""Check saved circular examples and preserve their provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import circular_examples_20260928 as run
from box_refinement.methods import predicted_regions

ROOT, OUT = run.ROOT, run.OUT
frozen = ROOT/'DataSimulation/data/output/variability_campaign_20260926_v1'
old_manifest = json.loads((frozen/'FINAL_MANIFEST.json').read_text())['files']
spec = json.loads((frozen/'frozen_models.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('frozen_models.json','normalization.json'):
    path=frozen/name
    assert sha(path)==old_manifest[str(path.resolve())]
for method in ('cnn',):
    model=spec['models'][method]
    path=frozen/'training'/Path(model['checkpoint']).name
    assert sha(path)==model['checkpoint_sha256']

# Check both pore families and clean geometry against the native volume path.
cfg=run.load_config_bundle(ROOT/'DataSimulation/config')
sample,_=run.sampled(cfg,1360100000,0)
kw=dict(beam=sample.beam,material=sample.material,image_size=96,pixel_size_mm=.05,
    voxel_depth_mm=.08,crown_height_mm=.012,seam_depth_mm=.14,bead_width_mm=1.1,
    haz_width_mm=1.,beam_focus_quality=.35,asymmetry_offset_mm=.02,
    weld_geometry_mode='recessed_seam',weld_layout=run.WeldLayoutRuntime(mode='circular_weld',circle_radius_mm=4.),seed=51)
equivalence={}
for kind in ('clean','keyhole_pore','intrinsic_pore'):
    native,slim=run.vv.build_voxel_volume(**kw),run.top_volume(**kw)
    plan=run.pore_parameters(kind,np.random.default_rng(7))
    for volume in (native,slim):
        run.vv.apply_defect_bodies(volume,plan,np.random.default_rng(51),randomize_position=False,anchor_to_surface=True)
    a,_=run.se.ScanEngine._top_surface_section(native)
    b,_=run.se.ScanEngine._top_surface_section(slim)
    errors={field:float(np.max(np.abs(getattr(a,field).astype(float)-getattr(b,field))))
            for field in ('relief_mm_map','material_z_map','yield_modifier_map','void_fraction_map','defect_mask')}
    assert max(errors.values())==0
    equivalence[kind]=errors

replayed=0
for name in ('a00_clean','a05_clean','a10_clean','a10_keyhole_pore','a10_intrinsic_pore'):
    meta=json.loads((OUT/f'{name}.json').read_text())
    path=OUT/f'{name}.npz'
    assert sha(path)==meta['npz_sha256']
    with np.load(path) as data:
        assert data['raw'].shape==(1921,1921)
        assert data['crops_raw'].shape==(78,64,64)
        assert np.isfinite(data['raw']).all()
        assert np.isfinite(data['relief']).all()
        assert bool(data['mask'].any())==('pore' in name)
        for method in ('cnn','median'):
            for score,row in zip(data[f'{method}_scores'],meta['predictions'][method]):
                mask,boxes=predicted_regions(score,**spec['models'][method]['parameters'])
                assert boxes==row['boxes_xyxy']
                replayed+=1
    assert meta['renderer']['operating_domain']['incidence_deg_quantiles'][-1]<=60
    assert abs(meta['renderer']['effective_pixel_size_mm']-.05)<1e-12

figures=ROOT/'docs/figures/circular_weld_20260928'
assert len(list(figures.glob('*.png')))==3
record=dict(status='passed', surface_plane_equivalence=equivalence, saved_decisions_replayed=replayed,
    frozen_models_unchanged=True, native_pixel_size_mm=.05, all_local_incidence_within_60_degrees=True,
    physical_validation=False,
    source_hashes={str(p.resolve()):sha(p) for p in [
        *ROOT.glob('WeldDefectDetection/*circular_examples_20260928.py'),
        *(ROOT/'DataSimulation/bse_renderer').rglob('*.py'),
        *(ROOT/'DataSimulation/config').glob('*.yaml'),
        *(ROOT/'WeldDefectDetection/box_development').glob('*.py'),
        *(ROOT/'WeldDefectDetection/box_refinement').glob('*.py'),
        ROOT/'WeldDefectDetection/inspection/engine.py', run.MODEL]},
    outputs={str(p.resolve()):sha(p) for folder in (OUT,figures) for p in folder.iterdir()
             if p.is_file() and p.name!='verification.json'})
(OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','outputs')},indent=2))
