from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bse_renderer.comparison_metrics import compute_topography_delta_metrics
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.geometry.defect_injection import parse_defect_plan, validate_defect_plan
from bse_renderer.geometry.voxel_volume import build_section_plane, build_voxel_volume
from bse_renderer.geometry.weld_profile import (
    WELD_GEOMETRY_CLOSED,
    WELD_LAYOUT_CIRCULAR_WELD,
    WELD_LAYOUT_LOCAL_PATCH,
    WELD_LAYOUT_STRAIGHT_SEAM,
    build_weld_layout_maps,
    generate_weld_profile,
)
from bse_renderer.materials import material_from_average_atomic_number
from bse_renderer.models import BeamRuntime, SliceRuntime, WeldLayoutRuntime
from bse_renderer.pipeline.generate import generate_single_sample
from bse_renderer.pipeline.generate import render_comparison_case
from bse_renderer.renderer import (
    RENDER_MODEL_FORMULA_CASINO,
    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
    RENDER_MODEL_FORMULA_ONLY,
    load_renderer_calibration,
    parameter_only_calibration,
    resolve_renderer_calibration,
    save_renderer_calibration,
)
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.validation.comparison_summary import summarize_comparison_cases
from bse_renderer.visualisation.image_preview import _comparison_metric_lines, _connected_region_count


def _quiet_sampled_parameters(seed: int = 123):
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=seed, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    return replace(sampled, noise=quiet_noise, defects=[])


def test_manual_defect_plan_parse_and_validate():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    text = (
        "keyhole_pore(diameter_mm=1.1,depth_mm=0.8,count=2);"
        "crack_solidification(length_mm=2.4,width_mm=0.1,depth_mm=0.6)"
    )
    plan = parse_defect_plan(text)
    assert len(plan) == 3
    validate_defect_plan(cfg.defect, plan)


def test_average_z_material_estimation():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    mat = material_from_average_atomic_number(
        average_z=27.0,
        known_presets=cfg.material.presets.values(),
        name="Custom27",
    )
    assert mat.name == "Custom27"
    assert mat.z == 27.0
    assert 50.0 <= mat.a <= 65.0
    assert 6.0 <= mat.rho <= 10.0
    assert 0.1 <= mat.b <= 0.4
    assert 1.2 <= mat.n <= 2.2


def test_scan_engine_beam_parameter_sensitivity():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=123, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    sampled = replace(sampled, noise=quiet_noise, defects=[])

    low_beam = BeamRuntime(
        accelerating_voltage_kv=30.0,
        beam_current_ma=80.0,
        focusing_current_norm=sampled.beam.focusing_current_norm,
        travel_speed_mm_min=sampled.beam.travel_speed_mm_min,
        dwell_time_us=sampled.beam.dwell_time_us,
        focal_spot_mm=0.8,
        e0_kev=30.0,
        power_w=30.0 * 80.0,
    )
    high_beam = BeamRuntime(
        accelerating_voltage_kv=175.0,
        beam_current_ma=1000.0,
        focusing_current_norm=sampled.beam.focusing_current_norm,
        travel_speed_mm_min=sampled.beam.travel_speed_mm_min,
        dwell_time_us=sampled.beam.dwell_time_us,
        focal_spot_mm=0.3,
        e0_kev=175.0,
        power_w=175.0 * 1000.0,
    )

    engine = ScanEngine(image_size=128)
    low = engine.run(replace(sampled, beam=low_beam))
    high = engine.run(replace(sampled, beam=high_beam))

    mean_delta = float(abs(high.channels.compo.mean() - low.channels.compo.mean()))
    assert mean_delta > 0.02

    grad_low = np.hypot(*np.gradient(low.channels.compo))
    grad_high = np.hypot(*np.gradient(high.channels.compo))
    assert not np.allclose(low.channels.compo, high.channels.compo)
    assert abs(float(grad_high.mean()) - float(grad_low.mean())) > 2e-5


def test_single_sample_overrides_for_avg_z_and_defect_plan():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    sample = generate_single_sample(
        config_dir=cfg_dir,
        seed=11,
        image_size=64,
        average_atomic_number=27.5,
        defect_plan="keyhole_pore(diameter_mm=1.0,depth_mm=0.8,count=2)",
    )
    assert sample["channels"].compo.shape == (64, 64)
    assert sample["metadata"]["material"]["z"] == 27.5
    assert sample["metadata"]["defects"] == ["keyhole_pore", "keyhole_pore"]
    assert sample["metadata"]["render_model_mode"] == RENDER_MODEL_FORMULA_ONLY


def test_keyhole_pores_are_bounded_but_visible_in_compo_channel():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=101, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan("keyhole_pore(diameter_mm=1.2,depth_mm=0.8,count=2)")
    defected_sample = replace(sampled, noise=quiet_noise, defects=plan)
    clean_sample = replace(sampled, noise=quiet_noise, defects=[])

    engine = ScanEngine(image_size=128)
    defected = engine.run(defected_sample)
    clean = engine.run(clean_sample)
    mask = defected.defect_mask.astype(bool)

    assert mask.any()
    inside_defected = float(defected.channels.compo[mask].mean())
    inside_clean = float(clean.channels.compo[mask].mean())
    assert inside_defected < inside_clean
    assert inside_defected > 0.5 * inside_clean


def test_defects_couple_compo_and_directional_channels():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=101, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan("keyhole_pore(diameter_mm=1.2,depth_mm=0.8,count=1)")
    defected_sample = replace(sampled, noise=quiet_noise, defects=plan)
    clean_sample = replace(sampled, noise=quiet_noise, defects=[])

    engine = ScanEngine(image_size=128)
    defected = engine.run(defected_sample)
    clean = engine.run(clean_sample)
    mask = defected.defect_mask.astype(bool)

    assert mask.any()
    assert float(np.abs(defected.channels.compo - clean.channels.compo)[mask].mean()) > 1e-3
    assert float(np.abs(defected.channels.tx - clean.channels.tx)[mask].mean()) > 1e-5
    assert float(np.abs(defected.channels.ty - clean.channels.ty)[mask].mean()) > 1e-5


def test_intrinsic_pore_has_bounded_section_plane_contrast():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=202, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan("intrinsic_pore(diameter_mm=1.0,depth_mm=0.8,count=1)")
    engine = ScanEngine(image_size=128)
    defected = engine.run(replace(sampled, noise=quiet_noise, defects=plan))
    clean = engine.run(replace(sampled, noise=quiet_noise, defects=[]))
    mask = defected.defect_mask.astype(bool)

    assert mask.any()
    inside_defected = float(defected.channels.compo[mask].mean())
    inside_clean = float(clean.channels.compo[mask].mean())
    assert inside_defected < inside_clean
    assert inside_defected > 0.55 * inside_clean


def test_local_depression_produces_signed_directional_relief():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=101, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan("keyhole_pore(diameter_mm=1.2,depth_mm=0.8,count=1)")
    engine = ScanEngine(image_size=128)
    result = engine.run(replace(sampled, noise=quiet_noise, defects=plan))
    clean = engine.run(replace(sampled, noise=quiet_noise, defects=[]))
    mask = result.defect_mask.astype(bool)
    tx_residual = result.channels.tx - clean.channels.tx
    ty_residual = result.channels.ty - clean.channels.ty

    assert mask.any()
    assert (tx_residual[mask] > 0.0).any()
    assert (tx_residual[mask] < 0.0).any()
    assert (ty_residual[mask] > 0.0).any()
    assert (ty_residual[mask] < 0.0).any()
    assert float(result.channels.topo[mask].mean()) > 0.0


def test_defect_difference_is_localized_above_background_in_compo_and_topo():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=101, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan("keyhole_pore(diameter_mm=1.2,depth_mm=0.8,count=1)")
    engine = ScanEngine(image_size=128)
    clean = engine.run(replace(sampled, noise=quiet_noise, defects=[]))
    defected = engine.run(replace(sampled, noise=quiet_noise, defects=plan))
    mask = defected.defect_mask.astype(bool)

    assert mask.any()
    compo_delta = np.abs(defected.channels.compo - clean.channels.compo)
    topo_delta = np.abs(defected.channels.topo - clean.channels.topo)
    assert float(compo_delta[mask].mean()) > 20.0 * float(compo_delta[~mask].mean())
    assert float(topo_delta[mask].mean()) > 20.0 * float(topo_delta[~mask].mean())


def test_default_ebw_profile_is_recessed_not_crowned():
    profile = generate_weld_profile(
        width_px=128,
        height_px=128,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.05,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.8,
        seed=42,
    )
    cy = profile.shape[0] // 2
    center_band = profile[max(0, cy - 2): min(profile.shape[0], cy + 3), :]
    side_band = np.vstack([profile[:10, :], profile[-10:, :]])
    assert float(center_band.mean()) < float(side_band.mean())
    assert float(center_band.mean()) < 0.0
    assert abs(float(profile.min())) > float(profile.max())


def test_seam_depth_parameter_increases_recess_depth_monotonically():
    shallow = generate_weld_profile(
        width_px=128,
        height_px=128,
        crown_height_mm=0.0,
        seam_depth_mm=0.08,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.05,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.9,
        seed=10,
    )
    deep = generate_weld_profile(
        width_px=128,
        height_px=128,
        crown_height_mm=0.0,
        seam_depth_mm=0.30,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.05,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.9,
        seed=10,
    )
    assert float(deep.min()) < float(shallow.min())


def test_closed_fusion_profile_is_not_a_recessed_through_slit():
    profile = generate_weld_profile(
        width_px=128,
        height_px=128,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.05,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.8,
        weld_geometry_mode=WELD_GEOMETRY_CLOSED,
        seed=42,
    )
    cy = profile.shape[0] // 2
    center_band = profile[max(0, cy - 2): min(profile.shape[0], cy + 3), :]
    side_band = np.vstack([profile[:10, :], profile[-10:, :]])
    assert float(center_band.mean()) > float(side_band.mean())
    assert float(center_band.mean()) > -0.02


def test_clean_topo_baseline_avoids_dominant_periodic_stripng():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=123, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    result = ScanEngine(image_size=128).run(replace(sampled, noise=quiet_noise, defects=[]))
    topo = result.channels.topo
    column_mean_variation = float(topo.mean(axis=0).std())
    full_variation = float(topo.std())
    assert column_mean_variation < 0.30 * max(full_variation, 1e-9)


def test_geometry_adaptive_profiles_modulate_local_rendering(tmp_path):
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=314, max_defects=0)
    quiet_noise = replace(
        sampled.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
    )
    plan = parse_defect_plan(
        "cold_shut(extent_mm=2.0,depth_mm=0.5);"
        "spiking(spike_length_mm=1.6,spike_width_mm=0.16)"
    )
    sampled = replace(sampled, noise=quiet_noise, defects=plan)

    calibration_path = tmp_path / "geometry_adaptive_calibration.json"
    save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate={
            "source_campaign": "geometry_adaptive_test",
            "yield_model": {
                "scale": 1.0,
                "reference_energy_kev": 60.0,
                "energy_exponent": 0.0,
                "obliquity_exponent_scale": 1.0,
            },
            "blur_model": {
                "scale": 1.0,
                "reference_r50_nm": 180.0,
                "reference_r90_nm": 520.0,
            },
            "depth_model": {
                "range_scale": 1.0,
                "reference_z50_nm": 1450.0,
                "reference_z90_nm": 4300.0,
            },
            "defect_model": {
                "pore_visibility_scale": 1.0,
                "groove_topography_scale": 1.0,
                "cluster_visibility_scale": 1.0,
            },
            "geometry_profiles": {
                "flatbulk": {
                    "geometry": "flatbulk",
                    "yield_scale": 1.0,
                    "blur_scale": 1.0,
                    "depth_scale": 1.0,
                    "topo_gain": 1.0,
                    "openness_scale": 1.0,
                },
                "slit_defect_cold_shut": {
                    "geometry": "slit_defect_cold_shut",
                    "yield_scale": 0.92,
                    "blur_scale": 1.8,
                    "depth_scale": 1.4,
                    "topo_gain": 0.75,
                    "openness_scale": 1.6,
                },
                "spiking_surface_large": {
                    "geometry": "spiking_surface_large",
                    "yield_scale": 1.18,
                    "blur_scale": 0.45,
                    "depth_scale": 1.25,
                    "topo_gain": 2.2,
                    "openness_scale": 1.0,
                },
            },
        },
        metadata={"source": "geometry_adaptive_test"},
    )

    identity = ScanEngine(image_size=128).run(sampled)
    adaptive = ScanEngine(image_size=128, calibration_path=str(calibration_path)).run(sampled)

    assert adaptive.metadata["multi_weld_profile_profile_count"] >= 3
    used = {item["geometry"] for item in adaptive.metadata["multi_weld_profile_profiles_used"]}
    assert "slit_defect_cold_shut" in used
    assert "spiking_surface_large" in used
    assert adaptive.metadata["multi_weld_profile_blur_scale_range"][1] > 1.0
    assert adaptive.metadata["multi_weld_profile_topo_gain_range"][1] > 1.0
    assert not np.allclose(identity.channels.compo, adaptive.channels.compo)
    assert not np.allclose(identity.channels.topo, adaptive.channels.topo)


def test_render_comparison_case_writes_mode_outputs(tmp_path):
    calibration_path = tmp_path / "comparison_calibration.json"
    save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate={
            "source_campaign": "comparison_test",
            "yield_model": {
                "scale": 1.0,
                "reference_energy_kev": 60.0,
                "energy_exponent": 0.0,
                "obliquity_exponent_scale": 1.0,
            },
            "blur_model": {
                "scale": 1.0,
                "reference_r50_nm": 180.0,
                "reference_r90_nm": 520.0,
            },
            "depth_model": {
                "range_scale": 1.0,
                "reference_z50_nm": 1450.0,
                "reference_z90_nm": 4300.0,
            },
            "defect_model": {
                "pore_visibility_scale": 1.0,
                "groove_topography_scale": 1.0,
                "cluster_visibility_scale": 1.0,
            },
            "geometry_profiles": {
                "flatbulk": {
                    "geometry": "flatbulk",
                    "yield_scale": 1.0,
                    "blur_scale": 1.0,
                    "depth_scale": 1.0,
                    "topo_gain": 1.0,
                    "openness_scale": 1.0,
                },
                "slit_defect_crack_short": {
                    "geometry": "slit_defect_crack_short",
                    "yield_scale": 0.94,
                    "blur_scale": 1.7,
                    "depth_scale": 1.2,
                    "topo_gain": 0.8,
                    "openness_scale": 1.5,
                },
            },
        },
        metadata={"source": "comparison_test"},
    )

    out = render_comparison_case(
        config_dir=Path(__file__).resolve().parents[1] / "config",
        output_dir=tmp_path / "comparison",
        calibration_path=calibration_path,
        seed=55,
        image_size=96,
        defect_plan="crack_solidification(length_mm=1.2,width_mm=0.1,depth_mm=0.4)",
        max_defects=0,
        randomize_defect_location=False,
        stem="case55",
    )

    assert out["status"] == "ok"
    assert Path(out["summary_path"]).exists()
    assert RENDER_MODEL_FORMULA_ONLY in out["modes"]
    assert RENDER_MODEL_FORMULA_CASINO in out["modes"]
    assert RENDER_MODEL_FORMULA_CASINO_GEOMETRY in out["modes"]
    assert "comparison_definition" in out
    assert Path(out["modes"][RENDER_MODEL_FORMULA_ONLY]["images"]["compo_u8"]).exists()
    assert Path(out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]["images"]["topo_u8"]).exists()
    assert "delta_vs_formula_only" in out["modes"][RENDER_MODEL_FORMULA_CASINO]
    assert "delta_vs_formula_only" in out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]
    assert "delta_vs_formula_casino" in out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]
    assert "defect_mean_abs_topo_delta" in out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]["delta_vs_formula_only"]
    assert "background_mean_abs_topo_delta" in out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]["delta_vs_formula_only"]
    assert (
        out["modes"][RENDER_MODEL_FORMULA_ONLY]["metadata"]["multi_weld_profile_profile_count"]
        == 0
    )
    assert (
        out["modes"][RENDER_MODEL_FORMULA_CASINO_GEOMETRY]["metadata"]["multi_weld_profile_profile_count"]
        > 0
    )


def test_parameter_only_calibration_strips_geometry_profiles(tmp_path):
    calibration_path = tmp_path / "parameter_only_source.json"
    save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate={
            "yield_model": {"scale": 1.1},
            "defect_model": {
                "pore_visibility_scale": 1.4,
                "groove_topography_scale": 1.3,
                "cluster_visibility_scale": 1.2,
            },
            "geometry_profiles": {
                "flatbulk": {"geometry": "flatbulk"},
            },
        },
    )
    calibration = load_renderer_calibration(calibration_path)
    stripped = parameter_only_calibration(calibration)

    assert stripped.surrogate.geometry_profiles == {}
    assert stripped.surrogate.defect_model.pore_visibility_scale == 1.0
    assert stripped.surrogate.yield_model.scale == calibration.surrogate.yield_model.scale


def test_resolve_renderer_calibration_explicit_modes(tmp_path):
    calibration_path = tmp_path / "resolved_modes_source.json"
    save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate={
            "yield_model": {"scale": 1.1},
            "defect_model": {
                "pore_visibility_scale": 1.4,
                "groove_topography_scale": 1.3,
                "cluster_visibility_scale": 1.2,
            },
            "geometry_profiles": {
                "flatbulk": {"geometry": "flatbulk"},
            },
        },
    )

    formula_only = resolve_renderer_calibration(
        calibration_path,
        render_model_mode=RENDER_MODEL_FORMULA_ONLY,
    )
    formula_casino = resolve_renderer_calibration(
        calibration_path,
        render_model_mode=RENDER_MODEL_FORMULA_CASINO,
    )
    geometry = resolve_renderer_calibration(
        calibration_path,
        render_model_mode=RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
    )

    assert formula_only.surrogate.geometry_profiles == {}
    assert formula_only.surrogate.yield_model.scale == 1.0
    assert formula_only.surrogate.defect_model.pore_visibility_scale == 1.0
    assert formula_casino.surrogate.geometry_profiles == {}
    assert formula_casino.surrogate.yield_model.scale == 1.1
    assert geometry.surrogate.geometry_profiles["flatbulk"].geometry == "flatbulk"


def test_casino_render_mode_rejects_missing_calibration(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibration file not found"):
        resolve_renderer_calibration(
            tmp_path / "missing.json",
            render_model_mode=RENDER_MODEL_FORMULA_CASINO,
        )


def test_compute_topography_delta_metrics_tracks_defect_vs_background():
    reference = np.zeros((2, 2), dtype=np.float32)
    candidate = np.array([[0.4, 0.1], [0.0, 0.0]], dtype=np.float32)
    defect_mask = np.array([[1, 0], [0, 0]], dtype=bool)

    metrics = compute_topography_delta_metrics(reference, candidate, defect_mask=defect_mask)

    assert np.isclose(metrics["mean_abs_topo_delta"], 0.125)
    assert metrics["defect_pixels"] == 1
    assert metrics["background_pixels"] == 3
    assert np.isclose(metrics["defect_mean_abs_topo_delta"], 0.4)
    assert np.isclose(metrics["background_mean_abs_topo_delta"], 0.03333333507180214)
    assert metrics["defect_to_background_mean_ratio"] > 10.0


def test_summarize_comparison_cases_writes_table(tmp_path):
    calibration_path = tmp_path / "comparison_calibration.json"
    save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate={
            "source_campaign": "comparison_test",
            "yield_model": {"scale": 1.0},
            "blur_model": {"scale": 1.0},
            "depth_model": {"range_scale": 1.0},
            "geometry_profiles": {
                "flatbulk": {"geometry": "flatbulk"},
                "slit_defect_crack_short": {
                    "geometry": "slit_defect_crack_short",
                    "yield_scale": 0.9,
                    "blur_scale": 1.5,
                    "depth_scale": 1.2,
                    "topo_gain": 0.8,
                    "openness_scale": 1.4,
                },
            },
        },
    )
    render_comparison_case(
        config_dir=Path(__file__).resolve().parents[1] / "config",
        output_dir=tmp_path / "comparison_cases" / "surface_v4_tilt60_crack",
        calibration_path=calibration_path,
        seed=603,
        image_size=96,
        defect_plan="crack_solidification(length_mm=1.6,width_mm=0.08,depth_mm=0.45)",
        max_defects=0,
        slice_tilt_deg=60,
        slice_rotation_deg=0,
        randomize_defect_location=False,
        stem="surface_v4_tilt60_crack",
    )

    out = summarize_comparison_cases(comparison_root=tmp_path / "comparison_cases")

    assert out["status"] == "ok"
    assert out["num_cases"] == 1
    assert Path(out["csv_path"]).exists()
    assert Path(out["json_path"]).exists()


def test_connected_region_count_counts_disjoint_regions():
    mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )

    assert _connected_region_count(mask) == 3


def test_comparison_metric_lines_explain_localized_change():
    metrics = {
        "defect_pixels": 12,
        "background_pixels": 100,
        "defect_fraction": 12 / 112,
        "mean_abs_topo_delta": 6.093e-06,
        "max_abs_topo_delta": 0.0036,
        "defect_mean_abs_topo_delta": 6.746e-04,
        "background_mean_abs_topo_delta": 8.206e-08,
        "defect_to_background_mean_ratio": 8.221e03,
        "defect_max_abs_topo_delta": 0.0036,
    }
    mask = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)

    lines = _comparison_metric_lines(metrics, mask)

    assert any("Connected defect regions: 2" in line for line in lines)
    assert any("0.000006" in line for line in lines)
    assert any("localized defect response, not a global shift" in line for line in lines)


def test_circular_weld_layout_fits_the_full_ring_into_view():
    signed_distance_mm, axial_mm, meta = build_weld_layout_maps(
        width_px=65,
        height_px=65,
        pixel_size_mm=0.1,
        layout=WeldLayoutRuntime(
            mode=WELD_LAYOUT_CIRCULAR_WELD,
            circle_radius_mm=80.0,
            circle_position_deg=90.0,
        ),
    )

    cy = signed_distance_mm.shape[0] // 2
    cx = signed_distance_mm.shape[1] // 2
    ring_mask = np.abs(signed_distance_mm) <= 0.5 * float(meta["layout_pixel_size_mm"])
    assert float(signed_distance_mm[cy, cx]) < -70.0
    assert ring_mask.any()
    assert ring_mask[:12].any()
    assert ring_mask[-12:].any()
    assert ring_mask[:, :12].any()
    assert ring_mask[:, -12:].any()
    assert bool(meta["fit_full_shape"]) is True
    assert float(meta["layout_pixel_size_mm"]) > 0.1
    assert meta["mode"] == WELD_LAYOUT_CIRCULAR_WELD
    assert np.isclose(float(meta["circle_radius_mm"]), 80.0)
    assert np.isfinite(float(axial_mm[cy, cx]))


def test_circular_weld_layout_changes_the_local_profile_shape():
    kwargs = dict(
        width_px=128,
        height_px=128,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.1,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.8,
        seed=42,
    )
    local_profile = generate_weld_profile(**kwargs)
    circular_profile = generate_weld_profile(
        **kwargs,
        layout=WeldLayoutRuntime(
            mode=WELD_LAYOUT_CIRCULAR_WELD,
            circle_radius_mm=6.0,
            circle_position_deg=90.0,
        ),
    )

    assert not np.allclose(local_profile, circular_profile)


def test_local_patch_layout_is_more_localized_than_straight_seam():
    kwargs = dict(
        width_px=128,
        height_px=128,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.4,
        haz_width_mm=1.0,
        pixel_size_mm=0.1,
        travel_speed_mm_min=900.0,
        beam_power_kw=18.0,
        beam_focus_quality=0.8,
        seed=42,
    )
    local_profile = generate_weld_profile(
        **kwargs,
        layout=WeldLayoutRuntime(mode=WELD_LAYOUT_LOCAL_PATCH),
    )
    straight_profile = generate_weld_profile(
        **kwargs,
        layout=WeldLayoutRuntime(mode=WELD_LAYOUT_STRAIGHT_SEAM),
    )

    local_active = np.abs(local_profile) > 0.01
    straight_active = np.abs(straight_profile) > 0.01
    assert float(local_active.mean()) < float(straight_active.mean())


def test_single_sample_records_pixel_scale_and_weld_layout_metadata():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    sample = generate_single_sample(
        config_dir=cfg_dir,
        seed=7,
        image_size=64,
        pixel_size_mm=0.1,
        weld_layout_mode="circular_weld",
        circle_radius_mm=80.0,
        circle_position_deg=90.0,
        max_defects=0,
    )

    assert np.isclose(sample["metadata"]["pixel_size_mm"], 0.1)
    assert sample["metadata"]["effective_pixel_size_mm"] > sample["metadata"]["pixel_size_mm"]
    assert sample["metadata"]["weld_layout"]["mode"] == "circular_weld"
    assert np.isclose(sample["metadata"]["weld_layout"]["circle_radius_mm"], 80.0)


def test_screenshot_weld_layout_controls_have_expected_active_and_inactive_cases():
    base_circular = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        seam_rotation_deg=0.0,
        seam_offset_mm=0.0,
        circle_radius_mm=40.0,
        circle_position_deg=90.0,
    )
    rotated_circular = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        seam_rotation_deg=35.0,
        seam_offset_mm=0.0,
        circle_radius_mm=40.0,
        circle_position_deg=90.0,
    )
    offset_circular = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        seam_rotation_deg=0.0,
        seam_offset_mm=3.0,
        circle_radius_mm=40.0,
        circle_position_deg=90.0,
    )
    radius_circular = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        seam_rotation_deg=0.0,
        seam_offset_mm=0.0,
        circle_radius_mm=55.0,
        circle_position_deg=90.0,
    )
    angle_circular = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        seam_rotation_deg=0.0,
        seam_offset_mm=0.0,
        circle_radius_mm=40.0,
        circle_position_deg=25.0,
    )

    circular_signed, circular_axial, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=base_circular,
    )
    circular_rot_signed, circular_rot_axial, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=rotated_circular,
    )
    circular_offset_signed, _, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=offset_circular,
    )
    circular_radius_signed, _, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=radius_circular,
    )
    _, circular_angle_axial, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=angle_circular,
    )

    assert np.allclose(circular_signed, circular_rot_signed)
    assert np.allclose(circular_axial, circular_rot_axial)
    assert not np.allclose(circular_signed, circular_offset_signed)
    assert not np.allclose(circular_signed, circular_radius_signed)
    assert not np.allclose(circular_axial, circular_angle_axial)

    straight_0 = WeldLayoutRuntime(mode=WELD_LAYOUT_STRAIGHT_SEAM, seam_rotation_deg=0.0)
    straight_35 = WeldLayoutRuntime(mode=WELD_LAYOUT_STRAIGHT_SEAM, seam_rotation_deg=35.0)
    straight_signed_0, straight_axial_0, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=straight_0,
    )
    straight_signed_35, straight_axial_35, _ = build_weld_layout_maps(
        width_px=128,
        height_px=128,
        pixel_size_mm=0.05,
        layout=straight_35,
    )

    assert not np.allclose(straight_signed_0, straight_signed_35)
    assert not np.allclose(straight_axial_0, straight_axial_35)


def test_voxel_depth_changes_volume_depth_sampling():
    sampled = _quiet_sampled_parameters(seed=222)
    common = dict(
        beam=sampled.beam,
        material=sampled.material,
        image_size=64,
        pixel_size_mm=0.05,
        crown_height_mm=0.01,
        seam_depth_mm=0.18,
        bead_width_mm=1.2,
        haz_width_mm=1.0,
        beam_focus_quality=0.8,
        asymmetry_offset_mm=0.0,
        weld_layout=WeldLayoutRuntime(mode=WELD_LAYOUT_CIRCULAR_WELD, circle_radius_mm=40.0),
        seed=222,
    )
    fine = build_voxel_volume(voxel_depth_mm=0.04, **common)
    coarse = build_voxel_volume(voxel_depth_mm=0.12, **common)

    assert np.isclose(fine.spacing_mm[2], 0.04)
    assert np.isclose(coarse.spacing_mm[2], 0.12)
    assert fine.material_z.shape[0] > coarse.material_z.shape[0]
    assert not np.array_equal(fine.z_mm, coarse.z_mm)


def test_slice_controls_have_expected_active_and_inactive_cases():
    plane_base = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=0.0,
            slice_rotation_deg=0.0,
        )
    )
    plane_offset = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=1.5,
            slice_tilt_deg=0.0,
            slice_rotation_deg=0.0,
        )
    )
    plane_tilt = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=25.0,
            slice_rotation_deg=0.0,
        )
    )
    plane_rot_zero_tilt = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=0.0,
            slice_rotation_deg=90.0,
        )
    )
    plane_rot_tilted = build_section_plane(
        SliceRuntime(
            voxel_depth_mm=0.08,
            slice_offset_mm=0.0,
            slice_tilt_deg=25.0,
            slice_rotation_deg=90.0,
        )
    )

    assert np.isclose(plane_offset.origin_mm[0], 1.5)
    assert not np.allclose(plane_base.normal_xyz, plane_tilt.normal_xyz)
    assert np.allclose(plane_base.normal_xyz, plane_rot_zero_tilt.normal_xyz)
    assert not np.allclose(plane_tilt.normal_xyz, plane_rot_tilted.normal_xyz)


def test_top_surface_render_ignores_slice_plane_controls_but_slice_render_uses_them():
    sampled = _quiet_sampled_parameters(seed=333)
    circular_layout = WeldLayoutRuntime(
        mode=WELD_LAYOUT_CIRCULAR_WELD,
        circle_radius_mm=40.0,
        circle_position_deg=90.0,
    )
    top_surface_engine = ScanEngine(
        image_size=96,
        weld_layout=circular_layout,
        render_view_mode="top_surface",
    )
    top_surface_a = top_surface_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=0.0,
                slice_tilt_deg=0.0,
                slice_rotation_deg=0.0,
            ),
        )
    )
    top_surface_b = top_surface_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=1.5,
                slice_tilt_deg=30.0,
                slice_rotation_deg=90.0,
            ),
        )
    )

    assert np.allclose(top_surface_a.channels.topo, top_surface_b.channels.topo)
    assert np.array_equal(top_surface_a.defect_mask, top_surface_b.defect_mask)

    slice_engine = ScanEngine(
        image_size=96,
        weld_layout=WeldLayoutRuntime(mode=WELD_LAYOUT_LOCAL_PATCH),
        render_view_mode="slice",
    )
    base = slice_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=0.0,
                slice_tilt_deg=0.0,
                slice_rotation_deg=0.0,
            ),
        )
    )
    offset = slice_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=1.0,
                slice_tilt_deg=0.0,
                slice_rotation_deg=0.0,
            ),
        )
    )
    tilted = slice_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=0.0,
                slice_tilt_deg=25.0,
                slice_rotation_deg=0.0,
            ),
        )
    )
    rotated = slice_engine.run(
        replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=0.08,
                slice_offset_mm=0.0,
                slice_tilt_deg=25.0,
                slice_rotation_deg=90.0,
            ),
        )
    )

    assert float(np.mean(np.abs(base.channels.topo - offset.channels.topo))) > 1e-6
    assert float(np.mean(np.abs(base.channels.topo - tilted.channels.topo))) > 1e-4
    assert float(np.mean(np.abs(tilted.channels.topo - rotated.channels.topo))) > 1e-4
