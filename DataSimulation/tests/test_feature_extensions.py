from dataclasses import replace
from pathlib import Path

import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.geometry.defect_injection import parse_defect_plan, validate_defect_plan
from bse_renderer.geometry.weld_profile import generate_weld_profile
from bse_renderer.materials import material_from_average_atomic_number
from bse_renderer.models import BeamRuntime
from bse_renderer.pipeline.generate import generate_single_sample
from bse_renderer.renderer.scan_engine import ScanEngine


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
    result = ScanEngine(image_size=128).run(replace(sampled, noise=quiet_noise, defects=plan))
    mask = result.defect_mask.astype(bool)

    assert mask.any()
    assert (result.channels.tx[mask] > 0.0).any()
    assert (result.channels.tx[mask] < 0.0).any()
    assert (result.channels.ty[mask] > 0.0).any()
    assert (result.channels.ty[mask] < 0.0).any()
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
