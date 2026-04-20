from pathlib import Path

import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.geometry.voxel_volume import (
    apply_defect_bodies,
    build_section_plane,
    build_voxel_volume,
    slice_volume,
)
from bse_renderer.models import SliceRuntime


def _build_volume(seed: int = 42):
    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=seed, max_defects=0)
    volume = build_voxel_volume(
        beam=sampled.beam,
        material=sampled.material,
        image_size=96,
        pixel_size_mm=0.05,
        voxel_depth_mm=0.08,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.3,
        haz_width_mm=1.0,
        beam_focus_quality=sampled.beam.focusing_current_norm,
        asymmetry_offset_mm=0.0,
        seed=seed,
    )
    return volume


def test_each_defect_type_stamps_non_empty_3d_body():
    plans = {
        "keyhole_pore": {"diameter_mm": 1.1, "depth_mm": 0.8},
        "intrinsic_pore": {"diameter_mm": 0.9, "depth_mm": 0.8},
        "crack_solidification": {"length_mm": 2.8, "width_mm": 0.12, "depth_mm": 0.6},
        "crack_liquidation": {"length_mm": 2.2, "width_mm": 0.10, "depth_mm": 0.5},
        "undercut": {"groove_depth_mm": 0.3, "toe_angle_deg": 30.0},
        "lack_of_fusion": {"extent_mm": 2.2, "depth_mm": 0.7, "irregularity": 0.4},
        "cold_shut": {"extent_mm": 1.8, "depth_mm": 0.6},
        "spiking": {"spike_length_mm": 1.0, "spike_width_mm": 0.12},
    }
    for idx, (defect_type, params) in enumerate(plans.items()):
        volume = _build_volume(seed=100 + idx)
        apply_defect_bodies(
            volume,
            [{"type": defect_type, "params": params}],
            np.random.default_rng(500 + idx),
        )
        assert len(volume.defect_channels) == 1
        assert int(volume.defect_channels[0]["mask"].sum()) > 0
        assert volume.defect_channels[0]["type"] == defect_type


def test_slice_sampling_is_deterministic_for_fixed_plane():
    volume = _build_volume(seed=10)
    apply_defect_bodies(
        volume,
        [{"type": "keyhole_pore", "params": {"diameter_mm": 1.2, "depth_mm": 0.8}}],
        np.random.default_rng(10),
    )
    plane = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=4.0,
            slice_rotation_deg=15.0,
        )
    )
    first, first_defects = slice_volume(volume, plane, image_size=96, pixel_size_mm=0.05)
    second, second_defects = slice_volume(volume, plane, image_size=96, pixel_size_mm=0.05)
    assert np.allclose(first.relief_mm_map, second.relief_mm_map)
    assert np.array_equal(first.defect_mask, second.defect_mask)
    assert [d.bbox_xyxy for d in first_defects] == [d.bbox_xyxy for d in second_defects]


def test_default_slice_plane_is_cross_section_oriented():
    plane = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=0.0,
            slice_rotation_deg=0.0,
        )
    )
    assert np.allclose(np.asarray(plane.normal_xyz), np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(np.asarray(plane.origin_mm), np.array([0.0, 0.0, 0.0]), atol=1e-6)


def test_slice_orientation_changes_intersection_geometry():
    volume = _build_volume(seed=20)
    apply_defect_bodies(
        volume,
        [{"type": "lack_of_fusion", "params": {"extent_mm": 2.4, "depth_mm": 0.8, "irregularity": 0.5}}],
        np.random.default_rng(20),
    )
    plane_a = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=0.0,
            slice_rotation_deg=0.0,
        )
    )
    plane_b = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=8.0,
            slice_rotation_deg=55.0,
        )
    )
    section_a, defects_a = slice_volume(volume, plane_a, image_size=96, pixel_size_mm=0.05)
    section_b, defects_b = slice_volume(volume, plane_b, image_size=96, pixel_size_mm=0.05)
    assert defects_a
    assert defects_b
    assert not np.array_equal(section_a.defect_mask, section_b.defect_mask)


def test_far_slice_can_miss_defect_body_entirely():
    volume = _build_volume(seed=30)
    apply_defect_bodies(
        volume,
        [{"type": "intrinsic_pore", "params": {"diameter_mm": 0.8, "depth_mm": 0.7}}],
        np.random.default_rng(30),
    )
    plane = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=float(volume.z_mm.max()) + 0.8,
            slice_tilt_deg=0.0,
            slice_rotation_deg=0.0,
        )
    )
    section, defects = slice_volume(volume, plane, image_size=96, pixel_size_mm=0.05)
    assert int(section.defect_mask.sum()) == 0
    assert defects == []
