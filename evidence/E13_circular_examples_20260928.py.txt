"""Exploratory 80 mm circular-weld transfer examples, without retraining.

Keeps native 0.05 mm Cartesian sampling. Surface-only allocation reproduces
the two original z planes consumed by recessed top-surface rendering. Quadrant
integration is evaluated in bounded pixel batches with unchanged equations.
"""
from pathlib import Path
from dataclasses import replace, asdict
from unittest.mock import patch
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DataSimulation'))
sys.path.insert(0, str(ROOT / 'WeldDefectDetection'))
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from bse_renderer.geometry import voxel_volume as vv
from bse_renderer.geometry.weld_profile import generate_weld_profile, build_weld_layout_maps, resolve_layout_sampling_grid
from bse_renderer.models import VoxelVolumeState, WeldLayoutRuntime
from bse_renderer.renderer import scan_engine as se
from bse_renderer.renderer.quadrant_detector import compute_quadrant_signals
from bse_renderer.config_loader import load_config_bundle
from variability_campaign.campaign import sampled, MODEL
from pore_campaign.generation import pore_parameters
from inspection.engine import FrozenInspector
from box_comparison.core import truth_box, match_boxes, summarize

OUT = ROOT / 'DataSimulation/data/output/circular_examples_20260928_v1'
OUT.mkdir(parents=True, exist_ok=True)
N, SPACING, RADIUS, FRAMES = 1921, .05, 40., 78
START = time.monotonic()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def top_volume(**kw):
    """Exact recessed-mode surface planes, avoiding unused deep voxel arrays."""
    assert kw['weld_geometry_mode'] == 'recessed_seam'
    n, nominal, dz = kw['image_size'], kw['pixel_size_mm'], kw['voxel_depth_mm']
    layout = kw['weld_layout']
    _, _, gm = resolve_layout_sampling_grid(width_px=n, height_px=n, pixel_size_mm=nominal, layout=layout)
    spacing = gm['layout_pixel_size_mm']
    beam, material = kw['beam'], kw['material']
    profile_args = {k: kw[k] for k in ('crown_height_mm', 'seam_depth_mm', 'bead_width_mm',
                    'haz_width_mm', 'beam_focus_quality', 'asymmetry_offset_mm', 'weld_geometry_mode', 'seed')}
    relief = generate_weld_profile(width_px=n, height_px=n, pixel_size_mm=spacing,
        travel_speed_mm_min=beam.travel_speed_mm_min, beam_power_kw=beam.power_w/1000.,
        layout=layout, **profile_args).astype(np.float32)
    x = vv._mm_coords(n, spacing)
    depth = float(max(.55*(n-1)*spacing, 3.2*max(kw['bead_width_mm'], spacing), 2.4))
    original_nz = max(16, int(np.ceil(depth/max(dz, 1e-6)))+1)
    z = vv._mm_coords(original_nz, dz)[-2:]
    distance, _, lm = build_weld_layout_maps(width_px=n, height_px=n, pixel_size_mm=spacing,
        layout=layout, asymmetry_offset_mm=kw['asymmetry_offset_mm'])
    fusion = np.abs(distance) <= max(1.5*spacing, .5*kw['bead_width_mm'])
    phase = 2*np.pi*(z/max(depth, 1e-6))
    modulation = (1+.025*np.sin(phase+.7)+.015*np.cos(1.7*phase-.5)).astype(np.float32)
    height = (relief[None]*modulation[:, None, None]).astype(np.float32)
    height += (.0035*kw['seam_depth_mm']*np.sin(.65*phase)[:, None, None]).astype(np.float32)
    shape = (2, n, n)
    material2d = np.where(fusion, material.z+.2, material.z).astype(np.float32)
    return VoxelVolumeState(material_z=np.repeat(material2d[None], 2, axis=0),
        yield_modifier=np.ones(shape, np.float32), void_fraction=np.zeros(shape, np.float32),
        relief_mm=height, defect_mask=np.zeros(shape, bool), fusion_mask=np.repeat(fusion[None], 2, axis=0),
        x_mm=x, y_mm=x.copy(), z_mm=z, spacing_mm=(spacing, spacing, dz),
        origin_mm=(float(x[0]), float(x[0]), float(z[0])),
        metadata=dict(volume_shape=list(shape), original_volume_shape=[original_nz,n,n],
            depth_extent_mm=depth, bead_width_mm=kw['bead_width_mm'], haz_width_mm=kw['haz_width_mm'],
            pixel_size_mm=spacing, nominal_pixel_size_mm=nominal, weld_geometry_mode='recessed_seam',
            weld_layout=lm, fusion_body_half_depth_mm=depth*.5, surface_plane_index=1))


def chunked_quadrants(**kw):
    h, w = kw['normals'].shape[:2]
    output = np.empty((h*w, 4), np.float32)
    for start in range(0, h*w, 8192):
        stop = min(start+8192, h*w)
        part = {}
        for key, value in kw.items():
            if isinstance(value, np.ndarray) and value.shape[:2] == (h, w):
                part[key] = value.reshape((-1,)+value.shape[2:])[start:stop][None]
            else:
                part[key] = value
        output[start:stop] = compute_quadrant_signals(**part)[0]
    return output.reshape(h, w, 4)


class CircularEngine(se.ScanEngine):
    def __init__(self):
        super().__init__(image_size=N, pixel_size_mm=SPACING, render_model_mode='formula_only',
            emission_b_mode='assumed_constant', emission_b_path=str(MODEL), render_view_mode='top_surface',
            weld_layout=WeldLayoutRuntime(mode='circular_weld', circle_radius_mm=RADIUS, circle_position_deg=90.))
        self.amplitude = 0.
        self.base = self.base_material = None

    def _top_surface_section(self, volume, **kw):
        section, defects = super()._top_surface_section(volume, **kw)
        if self.base is None:
            assert not defects
            self.base = section.relief_mm_map.copy()
            self.base_material = section.material_z_map.copy()
            yy, xx = np.meshgrid(volume.y_mm, volume.x_mm, indexing='ij')
            self.radial = np.hypot(xx, yy)
            self.theta = np.arctan2(yy, xx)
            knots = np.linspace(-np.pi, np.pi, 41)
            rng = np.random.default_rng(1360100000+170003)
            self.fields = []
            for _ in range(2):
                values = rng.uniform(-1, 1, len(knots)); values[-1] = values[0]
                spline = CubicSpline(knots, values, bc_type='periodic')
                scale = np.max(np.abs(spline(np.linspace(-np.pi, np.pi, 20001))))
                self.fields.append((spline(self.theta)/scale).astype(np.float32))
                assert np.isclose(spline(-np.pi), spline(np.pi))
            self.axis = volume.x_mm
        delta = section.relief_mm_map-self.base
        material_delta = section.material_z_map-self.base_material
        if self.amplitude:
            width = 1+self.amplitude*self.fields[0]
            radial = RADIUS+(self.radial-RADIUS)/width
            coords = np.stack([(radial*np.sin(self.theta)-self.axis[0])/SPACING,
                               (radial*np.cos(self.theta)-self.axis[0])/SPACING])
            base = map_coordinates(self.base, coords, order=1, mode='nearest')
            base *= 1+self.amplitude*self.fields[1]
            material = map_coordinates(self.base_material, coords, order=0, mode='nearest')
            section.relief_mm_map = (base+delta).astype(np.float32)
            section.material_z_map = (material+material_delta).astype(np.float32)
        self.last_relief = section.relief_mm_map.copy()
        return section, defects


def validate(sample):
    kw = dict(beam=sample.beam, material=sample.material, image_size=96, pixel_size_mm=.05,
        voxel_depth_mm=.08, crown_height_mm=.012, seam_depth_mm=.14, bead_width_mm=1.1,
        haz_width_mm=1., beam_focus_quality=.35, asymmetry_offset_mm=.02,
        weld_geometry_mode='recessed_seam', weld_layout=WeldLayoutRuntime(mode='circular_weld', circle_radius_mm=4.), seed=51)
    native, slim = vv.build_voxel_volume(**kw), top_volume(**kw)
    errors = {}
    plan = pore_parameters('keyhole_pore', np.random.default_rng(7))
    for volume in (native, slim):
        vv.apply_defect_bodies(volume, plan, np.random.default_rng(51), randomize_position=False, anchor_to_surface=True)
    a, _ = se.ScanEngine._top_surface_section(native)
    b, _ = se.ScanEngine._top_surface_section(slim)
    for field in ('relief_mm_map', 'material_z_map', 'yield_modifier_map', 'void_fraction_map', 'defect_mask'):
        errors[field] = float(np.max(np.abs(getattr(a,field).astype(float)-getattr(b,field))))
        assert errors[field] == 0, (field, errors[field])
    normals = se.ScanEngine._normals_from_heightmap(a.relief_mm_map, native.spacing_mm[0])
    args = dict(normals=normals, detector=sample.detector, material=sample.material,
                material_z_map=a.material_z_map, b_coeff=.920694853227)
    errors['quadrant_batch_max_error'] = float(np.max(np.abs(compute_quadrant_signals(**args)-chunked_quadrants(**args))))
    assert errors['quadrant_batch_max_error'] < 1e-7
    write('numerical_checks.json', errors)
    print('Surface-plane and quadrant-batch equivalence checked.', flush=True)


def crop_ring(raw, mask):
    axis = (np.arange(64)-31.5)*SPACING
    radial, along = np.meshgrid(RADIUS+axis, axis, indexing='ij')
    images, masks = [], []
    for i in range(FRAMES):
        angle = np.pi/2+2*np.pi*i/FRAMES+along/RADIUS
        coords = np.stack([(radial*np.sin(angle)+(N-1)*SPACING/2)/SPACING,
                           (radial*np.cos(angle)+(N-1)*SPACING/2)/SPACING])
        images.append(map_coordinates(raw, coords, order=1, mode='nearest'))
        masks.append(map_coordinates(mask, coords, order=0, mode='nearest'))
    assert 2*np.pi*RADIUS/FRAMES > 64*SPACING  # Target/context sectors do not overlap.
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.uint8)


def main():
    torch.set_num_threads(4)
    cfg = load_config_bundle(ROOT/'DataSimulation/config')
    sample, _ = sampled(cfg, 1360100000, 0)
    sample = replace(sample, randomize_defect_location=False, slice=replace(sample.slice, slice_offset_mm=0.))
    validate(sample)
    inspector = FrozenInspector()
    inspector._network('cnn')
    write('protocol.json', dict(radius_mm=RADIUS, diameter_mm=2*RADIUS, image_size=N, pixel_size_mm=SPACING,
        sample=asdict(sample), circular_variability='Periodic cubic angular fields, radial width warp and relief scaling; pores unchanged.',
        levels=[0,.05,.1], cases='clean at 0/5/10 percent; keyhole and intrinsic at 10 percent',
        inference='Frozen straight-weld CNN and median. No tuning. Annular rectification uses known nominal radius, not defect labels.',
        patches=FRAMES, patch_shape=[64,64], center_spacing_mm=2*np.pi*RADIUS/FRAMES,
        sampling='0.05 mm radial and centerline arc sampling. Tangential scale varies by radius/R, about +/-4 percent across a patch.',
        coverage='Small inter-sector gaps (~0.022 mm centerline); inspection band +/-1.6 mm. Not whole-specimen inspection.',
        median='Four distinct neighbours at offsets -2,-1,+1,+2, wrapping around circumference.',
        physical_validation=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=inspector.model_hashes, normalization=inspector.norm))
    engine = CircularEngine()
    summaries = []
    for amplitude, kind in [(0.,'clean'),(.05,'clean'),(.1,'clean'),(.1,'keyhole_pore'),(.1,'intrinsic_pore')]:
        name = f'a{round(amplitude*100):02d}_{kind}'
        npz, meta_path = OUT/f'{name}.npz', OUT/f'{name}.json'
        if npz.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert hashlib.sha256(npz.read_bytes()).hexdigest() == meta['npz_sha256']
            summaries.append(meta)
            print('Reused verified '+name, flush=True)
            continue
        engine.amplitude = amplitude
        # Establish unmodified base in a resumed process if needed.
        if engine.base is None and kind != 'clean':
            with patch.object(se,'build_voxel_volume',top_volume), patch.object(se,'compute_quadrant_signals',chunked_quadrants):
                engine.run(sample)
        plan = pore_parameters(kind, np.random.default_rng(sample.seed+310003))
        print('Rendering '+name, flush=True)
        with patch.object(se,'build_voxel_volume',top_volume), patch.object(se,'compute_quadrant_signals',chunked_quadrants):
            result = engine.run(replace(sample, defects=plan))
        assert np.isclose(result.metadata['effective_pixel_size_mm'], SPACING)
        raw = result.channels.topo
        crops, masks = crop_ring(raw, result.defect_mask)
        if kind != 'clean':
            assert masks.any(), 'Defect absent from inspection sectors'
        norm = inspector.norm
        normalized = np.sqrt(np.clip((crops-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
        predictions, score_arrays = {}, {}
        for method in ('cnn','median'):
            rows, scores = [], []
            for i, target in enumerate(normalized):
                context_ids = [(i+k)%FRAMES for k in (-2,-1,1,2)]
                assert i not in context_ids and len(set(context_ids)) == 4
                item = dict(target=target, context=normalized[context_ids])
                score, predmask, boxes = inspector.predict(item, method)
                gt = truth_box(masks[i])
                row = dict(window_id=f'{name}-sector-{i:02d}', scene_id=name, sequence_id='circular-background-000',
                    sector=i, angle_deg=float((90+360*i/FRAMES)%360), source_class=kind,
                    truth_boxes=gt, boxes_xyxy=boxes, target_positive=bool(gt), predicted_positive=bool(boxes),
                    pixel_tp=int(np.sum((predmask>0)&(masks[i]>0))), pixel_fp=int(np.sum((predmask>0)&(masks[i]==0))),
                    pixel_fn=int(np.sum((predmask==0)&(masks[i]>0))), **match_boxes(boxes,gt))
                rows.append(row); scores.append(score)
            predictions[method] = rows; score_arrays[method] = np.stack(scores)
        meta = dict(name=name, kind=kind, amplitude=amplitude, renderer=result.metadata,
            predictions=predictions, metrics={m:summarize(r) for m,r in predictions.items()},
            elapsed_total_seconds=time.monotonic()-START)
        np.savez_compressed(npz, raw=raw, mask=result.defect_mask, relief=engine.last_relief,
            crops_raw=crops, crops_mask=masks, cnn_scores=score_arrays['cnn'], median_scores=score_arrays['median'])
        meta['npz_sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
        write(meta_path.name, meta); summaries.append(meta)
        print(json.dumps({'case':name,'seconds':round(time.monotonic()-START),
            'cnn':meta['metrics']['cnn'],'median':meta['metrics']['median']}), flush=True)
        del result, raw, crops, masks, normalized, score_arrays
        gc.collect()
    write('summary.json', [{'name':m['name'],'metrics':m['metrics']} for m in summaries])


if __name__ == '__main__':
    main()
