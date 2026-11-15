from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.renderer import load_renderer_calibration
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.validation.casino_benchmarks import (
    assemble_casino_benchmark_campaign,
    evaluate_casino_benchmark_fit,
    extract_casino_run_note_values,
    fit_casino_benchmark_calibration,
    load_casino_assembly_config,
    load_casino_benchmark_campaign,
    parse_casino_export_run,
)


def _write_campaign(tmp_path: Path) -> Path:
    campaign = {
        "schema_version": 1,
        "campaign_id": "304ss_test_campaign",
        "source_archive": "external://CASINO-Validation/06_Attachments",
        "source_campaign_note": "external://CASINO-Validation/04_Analysis/304ss_test_campaign",
        "records": [
            {
                "sim_id": "flat0",
                "material": "304SS",
                "geometry": "flatbulk",
                "energy_kev": 60.0,
                "tilt_deg": 0.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.262,
                "mean_bse_energy_kev": 42.5,
                "peak_bse_angle_deg": 34.0,
                "r50_nm": 180.0,
                "r90_nm": 520.0,
                "z50_nm": 1450.0,
                "z90_nm": 4300.0,
                "notes": "baseline",
                "quality_flag": "usable",
            },
            {
                "sim_id": "flat10",
                "material": "304SS",
                "geometry": "flatbulk",
                "energy_kev": 60.0,
                "tilt_deg": 10.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.271,
                "mean_bse_energy_kev": 42.1,
                "peak_bse_angle_deg": 36.0,
                "r50_nm": 182.0,
                "r90_nm": 525.0,
                "z50_nm": 1480.0,
                "z90_nm": 4350.0,
                "notes": "tilt 10",
                "quality_flag": "usable",
            },
            {
                "sim_id": "flat25",
                "material": "304SS",
                "geometry": "flatbulk",
                "energy_kev": 60.0,
                "tilt_deg": 25.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.309,
                "mean_bse_energy_kev": 41.7,
                "peak_bse_angle_deg": 39.0,
                "r50_nm": 188.0,
                "r90_nm": 532.0,
                "z50_nm": 1530.0,
                "z90_nm": 4400.0,
                "notes": "tilt 25",
                "quality_flag": "usable",
            },
            {
                "sim_id": "pore",
                "material": "304SS",
                "geometry": "spherical_pore_single",
                "energy_kev": 60.0,
                "tilt_deg": 0.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.255,
                "mean_bse_energy_kev": 41.9,
                "peak_bse_angle_deg": 35.0,
                "r50_nm": 195.0,
                "r90_nm": 560.0,
                "z50_nm": 1380.0,
                "z90_nm": 4100.0,
                "notes": "pore",
                "quality_flag": "usable",
                "defect_signal_proxy": 1.15,
            },
            {
                "sim_id": "groove",
                "material": "304SS",
                "geometry": "groove_surface",
                "energy_kev": 60.0,
                "tilt_deg": 0.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.258,
                "mean_bse_energy_kev": 42.0,
                "peak_bse_angle_deg": 35.5,
                "r50_nm": 190.0,
                "r90_nm": 548.0,
                "z50_nm": 1420.0,
                "z90_nm": 4200.0,
                "notes": "groove",
                "quality_flag": "usable",
                "defect_signal_proxy": 1.2,
            },
            {
                "sim_id": "cluster",
                "material": "304SS",
                "geometry": "pore_cluster_multi",
                "energy_kev": 60.0,
                "tilt_deg": 0.0,
                "beam_diameter_nm": 10.0,
                "electrons": 500000,
                "eta_bse": 0.249,
                "mean_bse_energy_kev": 41.5,
                "peak_bse_angle_deg": 35.0,
                "r50_nm": 205.0,
                "r90_nm": 590.0,
                "z50_nm": 1340.0,
                "z90_nm": 3950.0,
                "notes": "cluster",
                "quality_flag": "usable",
                "defect_signal_proxy": 1.35,
            },
        ],
    }
    p = tmp_path / "campaign.json"
    p.write_text(json.dumps(campaign, indent=2), encoding="utf-8")
    return p


def _write_run_export(
    run_dir: Path,
    *,
    eta_bse: float,
    energy_kev: float,
    angle_peak_deg: int,
    angle_mean_shape: tuple[float, float, float] = (0.2, 0.6, 0.2),
    radius_density_scale: float = 1.0,
    z_hist: list[tuple[float, float]] | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = f'''"================================================================="
"WinCasino 3.3.0.4"
"================================================================="
"Sample Info : "
=================================================================
"Name"\t\tBox_0
"Type"\t\t2
"Region Name"\t\tUndefined
"Scale (x, y, z)"\t\t(40000, 40000, 25000)
=================================================================
"Regions Info : "
=================================================================
"Name"\t\t304_SS\tBox_0 Outside
"Region ID"\t\t1\t2
"Rho"\t\t8.03\t0
=================================================================
"initial energy"\t{energy_kev}
"-----------------------------------------------------------------"
"Scan Point"\t0
"================================================================="
"X"\t0
"Y"\t0
"Initial Energy"\t{energy_kev}
"Nb Simulated Trajectories"\t500000
"-----------------------------------------------------------------"
"Scan Point 0 Results : "

"BackScattering Coefficient"\t{eta_bse}
"nb backscattered"\t52047
"nb transmitted"\t0
"-----------------------------------------------------------------"

"##################################################################"
"Scan Point"\t0\t" Trajectories : "
"##################################################################"
"-----------------------------------------------------------------"
"Trajectory"\t0

"NbCollisions"\t4
"Zmax"\t0

"Collisions"

"X"\t"Y"\t"Z"\t"Rx"\t"Ry"\t"Rz"\t"Energy"\t"CollisionType"\t
0\t0\t-12500\t0\t0\t1\t{energy_kev}\tRegion\t
0\t0\t-12000\t0\t0\t1\t{energy_kev - 1}\tAtom\t
0\t0\t-11800\t0\t0\t-1\t{energy_kev - 2}\tAtom\t
0\t0\t-12490\t0\t0\t-1\t{energy_kev - 3}\tRegion\t
"Trajectory"\t1

"NbCollisions"\t4
"Zmax"\t0

"Collisions"

"X"\t"Y"\t"Z"\t"Rx"\t"Ry"\t"Rz"\t"Energy"\t"CollisionType"\t
0\t0\t-12500\t0\t0\t1\t{energy_kev}\tRegion\t
0\t0\t-11000\t0\t0\t1\t{energy_kev - 1}\tAtom\t
0\t0\t-10800\t0\t0\t-1\t{energy_kev - 2}\tAtom\t
0\t0\t-12480\t0\t0\t-1\t{energy_kev - 3}\tRegion\t
'''
    (run_dir / "summary.dat").write_text(summary, encoding="utf-8")

    angle_rows = [
        (angle_peak_deg - 1, angle_mean_shape[0]),
        (angle_peak_deg, angle_mean_shape[1]),
        (angle_peak_deg + 1, angle_mean_shape[2]),
    ]
    angle_text = "Backscattered Angle\nAngle (degree)\tHits (Normalized)\n" + "\n".join(
        f"{x}\t{y}" for x, y in angle_rows
    )
    (run_dir / "BSAngle.dat").write_text(angle_text, encoding="utf-8")

    energy_rows = [
        (38.0, 0.2),
        (40.0, 0.5),
        (42.0, 0.3),
    ]
    energy_text = "Backscattered Energy\nEnergy (KeV)\tHits (Normalized)\n" + "\n".join(
        f"{x}\t{y}" for x, y in energy_rows
    )
    (run_dir / "BSEnergy.dat").write_text(energy_text, encoding="utf-8")

    radius_rows = [
        (0.0, 0.0),
        (100.0, 0.004 * radius_density_scale),
        (200.0, 0.003 * radius_density_scale),
        (400.0, 0.002 * radius_density_scale),
        (800.0, 0.001 * radius_density_scale),
    ]
    radius_text = "Surface Radius of BE\nRadius (nm)\tHits (Normalized) / nm\n" + "\n".join(
        f"{x}\t{y}" for x, y in radius_rows
    )
    (run_dir / "SurfaceRadiusofBE.dat").write_text(radius_text, encoding="utf-8")

    z_rows = z_hist or [
        (0.0, 0.0),
        (500.0, 0.1),
        (1000.0, 0.4),
        (1500.0, 0.3),
        (2000.0, 0.2),
    ]
    z_text = "Z Max Backscattered\nDepth (nm)\tHits (Normalized)\n" + "\n".join(
        f"{x}\t{y}" for x, y in z_rows
    )
    (run_dir / "zmaxBS.dat").write_text(z_text, encoding="utf-8")


def test_load_casino_benchmark_campaign(tmp_path):
    p = _write_campaign(tmp_path)
    campaign = load_casino_benchmark_campaign(p)
    assert campaign.campaign_id == "304ss_test_campaign"
    assert len(campaign.records) == 6
    assert campaign.records[0].material == "304SS"


def test_fit_casino_benchmark_calibration(tmp_path):
    p = _write_campaign(tmp_path)
    out = fit_casino_benchmark_calibration(
        benchmark_summary_path=p,
        output_dir=tmp_path / "out",
        config_dir=Path(__file__).resolve().parents[1] / "config",
    )
    assert out["status"] == "ok"
    assert Path(out["calibration_file"]).exists()
    assert Path(out["geometry_profile_file"]).exists()
    assert Path(out["manifest_file"]).exists()
    report = out["fit_report"]
    assert report["status"] == "ok"
    assert "summary" in report

    calibration = load_renderer_calibration(out["calibration_file"])
    assert calibration.surrogate.yield_model.scale > 0.0
    assert calibration.surrogate.yield_model.tilt_deg_knots == (0.0, 10.0, 25.0)
    assert calibration.surrogate.blur_model.scale > 0.0
    assert calibration.surrogate.depth_model.range_scale > 0.0
    assert calibration.surrogate.blur_model.tilt_deg_knots == (0.0, 10.0, 25.0)
    assert calibration.surrogate.depth_model.tilt_deg_knots == (0.0, 10.0, 25.0)
    assert calibration.surrogate.source_campaign == "304ss_test_campaign"
    assert "flatbulk" in calibration.surrogate.geometry_profiles
    assert "spherical_pore_single" in calibration.surrogate.geometry_profiles
    assert calibration.surrogate.geometry_profiles["flatbulk"].yield_scale == 1.0


def test_evaluate_casino_benchmark_fit_report(tmp_path):
    p = _write_campaign(tmp_path)
    fit = fit_casino_benchmark_calibration(
        benchmark_summary_path=p,
        output_dir=tmp_path / "fit",
        config_dir=Path(__file__).resolve().parents[1] / "config",
    )
    report = evaluate_casino_benchmark_fit(
        benchmark_summary_path=p,
        calibration_path=fit["calibration_file"],
        config_dir=Path(__file__).resolve().parents[1] / "config",
        output_dir=tmp_path / "eval",
    )
    assert report["status"] == "ok"
    assert Path(report["report_path"]).exists()
    assert len(report["records"]) == 6


def test_scan_engine_uses_surrogate_calibration(tmp_path):
    p = _write_campaign(tmp_path)
    fit = fit_casino_benchmark_calibration(
        benchmark_summary_path=p,
        output_dir=tmp_path / "fit",
        config_dir=Path(__file__).resolve().parents[1] / "config",
    )

    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=77, material_name="304SS", max_defects=0)
    sampled = replace(
        sampled,
        beam=replace(sampled.beam, e0_kev=60.0, accelerating_voltage_kv=60.0),
        slice=replace(sampled.slice, slice_tilt_deg=25.0),
    )

    identity = ScanEngine(image_size=96).run(sampled)
    calibrated = ScanEngine(image_size=96, calibration_path=fit["calibration_file"]).run(sampled)

    assert identity.metadata["range_um"] > 0.0
    assert calibrated.metadata["effective_range_um"] > 0.0
    assert calibrated.metadata["calibration_campaign_source"] == "304ss_test_campaign"
    assert calibrated.metadata["yield_scale"] != 1.0
    assert calibrated.metadata["tilt_yield_scale"] != 1.0
    assert calibrated.metadata["effective_yield_scale_range"][0] > 0.0
    assert calibrated.metadata["effective_yield_scale_range"][1] >= calibrated.metadata["effective_yield_scale_range"][0]
    assert calibrated.metadata["beam_blur_scale"] != 1.0
    assert calibrated.metadata["tilt_blur_scale"] > 1.0
    assert calibrated.metadata["tilt_depth_scale"] > 1.0
    assert not (identity.channels.compo == calibrated.channels.compo).all()


def test_parse_casino_export_run(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_export(run_dir, eta_bse=0.26, energy_kev=60.0, angle_peak_deg=48)
    parsed = parse_casino_export_run(run_id="flat0", artifact_root=run_dir)
    assert parsed.eta_bse == 0.26
    assert parsed.peak_bse_angle_deg == 48.0
    assert parsed.mean_bse_energy_kev > 39.0
    assert parsed.r50_nm > 0.0
    assert parsed.z50_nm > 0.0
    assert parsed.depth_source == "zmax_bs_dat"


def test_parse_casino_export_run_falls_back_for_degenerate_zmax(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_export(
        run_dir,
        eta_bse=0.26,
        energy_kev=60.0,
        angle_peak_deg=48,
        z_hist=[(0.0, 1.0), (50.0, 0.0), (100.0, 0.0)],
    )
    parsed = parse_casino_export_run(run_id="flat0", artifact_root=run_dir)
    assert parsed.depth_source == "summary_dat_fallback"
    assert parsed.z50_nm > 0.0
    assert parsed.z90_nm >= parsed.z50_nm


def test_extract_casino_run_note_values_infers_context_and_builds_table(tmp_path):
    run_dir = tmp_path / "2026-05-25-17-09_304SS_flatbulk_tilted_60keV_tilt10"
    _write_run_export(run_dir, eta_bse=0.26455, energy_kev=60.0, angle_peak_deg=48)
    extracted = extract_casino_run_note_values(
        artifact_root=run_dir,
        beam_diameter_nm=10.0,
    )
    assert extracted["sim_id"] == run_dir.name
    assert extracted["material"] == "304SS"
    assert extracted["geometry"] == "flatbulk_tilted"
    assert extracted["energy_kev"] == 60.0
    assert extracted["tilt_deg"] == 10.0
    assert extracted["electrons"] == 500000
    assert extracted["eta_bse"] == 0.26455
    assert "obsidian_fill_table" in extracted
    assert "`eta_bse`" in extracted["obsidian_fill_table"]
    assert extracted["source_files"]["summary_dat"].endswith("summary.dat")


def test_assemble_casino_benchmark_campaign_with_median_reducer(tmp_path):
    export_root = tmp_path / "exports"
    _write_run_export(export_root / "flat0_run1", eta_bse=0.25, energy_kev=60.0, angle_peak_deg=47)
    _write_run_export(export_root / "flat0_run2", eta_bse=0.26, energy_kev=60.0, angle_peak_deg=48)
    _write_run_export(export_root / "flat0_run3", eta_bse=0.27, energy_kev=60.0, angle_peak_deg=49)
    _write_run_export(export_root / "tilt10", eta_bse=0.29, energy_kev=60.0, angle_peak_deg=50)

    assembly = {
        "schema_version": 1,
        "campaign_id": "304ss_wave1",
        "source_archive": "external://CASINO-Validation/06_Attachments/exports",
        "source_campaign_note": "external://CASINO-Validation/04_Analysis/304ss_wave1",
        "description": "Wave 1 assembled benchmark campaign.",
        "defaults": {
            "material": "304SS",
            "energy_kev": 60.0,
            "beam_diameter_nm": 10.0,
            "electrons": 500000,
            "quality_flag": "usable",
        },
        "runs": [
            {"run_id": "flat0_run1", "artifact_root": "flat0_run1", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "flat0_run2", "artifact_root": "flat0_run2", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "flat0_run3", "artifact_root": "flat0_run3", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "tilt10", "artifact_root": "tilt10", "geometry": "flatbulk", "tilt_deg": 10.0},
        ],
        "records": [
            {
                "sim_id": "baseline",
                "from_runs": ["flat0_run1", "flat0_run2", "flat0_run3"],
                "geometry": "flatbulk",
                "tilt_deg": 0.0,
                "notes": "Median baseline",
            },
            {
                "sim_id": "tilt10",
                "from_runs": ["tilt10"],
                "geometry": "flatbulk",
                "tilt_deg": 10.0,
                "notes": "Tilt record",
            },
        ],
    }
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_text(json.dumps(assembly, indent=2), encoding="utf-8")

    cfg = load_casino_assembly_config(assembly_path)
    assert cfg.campaign_id == "304ss_wave1"
    out = assemble_casino_benchmark_campaign(
        export_root=export_root,
        assembly_config_path=assembly_path,
        output_path=tmp_path / "campaign.json",
    )
    assert out["status"] == "ok"
    campaign = load_casino_benchmark_campaign(tmp_path / "campaign.json")
    assert campaign.campaign_id == "304ss_wave1"
    assert len(campaign.records) == 2
    baseline = next(record for record in campaign.records if record.sim_id == "baseline")
    assert baseline.eta_bse == 0.26
    assert baseline.peak_bse_angle_deg == 48.0
    assert baseline.notes == "Median baseline"
    assert baseline.quality_flag == "usable"
    assert Path(out["report_path"]).exists()


def test_assemble_casino_benchmark_campaign_with_mean_reducer(tmp_path):
    export_root = tmp_path / "exports"
    _write_run_export(export_root / "flat0_run1", eta_bse=0.24, energy_kev=60.0, angle_peak_deg=45)
    _write_run_export(export_root / "flat0_run2", eta_bse=0.27, energy_kev=60.0, angle_peak_deg=48)
    _write_run_export(export_root / "flat0_run3", eta_bse=0.30, energy_kev=60.0, angle_peak_deg=51)

    assembly = {
        "schema_version": 1,
        "campaign_id": "304ss_wave1_mean",
        "defaults": {
            "material": "304SS",
            "energy_kev": 60.0,
            "beam_diameter_nm": 10.0,
            "electrons": 500000,
            "quality_flag": "usable",
        },
        "runs": [
            {"run_id": "flat0_run1", "artifact_root": "flat0_run1", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "flat0_run2", "artifact_root": "flat0_run2", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "flat0_run3", "artifact_root": "flat0_run3", "geometry": "flatbulk", "tilt_deg": 0.0},
        ],
        "records": [
            {
                "sim_id": "baseline_mean",
                "from_runs": ["flat0_run1", "flat0_run2", "flat0_run3"],
                "geometry": "flatbulk",
                "tilt_deg": 0.0,
                "notes": "Mean baseline",
                "reducer": "mean",
            }
        ],
    }
    assembly_path = tmp_path / "assembly_mean.json"
    assembly_path.write_text(json.dumps(assembly, indent=2), encoding="utf-8")

    out = assemble_casino_benchmark_campaign(
        export_root=export_root,
        assembly_config_path=assembly_path,
        output_path=tmp_path / "campaign_mean.json",
    )
    assert out["status"] == "ok"
    campaign = load_casino_benchmark_campaign(tmp_path / "campaign_mean.json")
    baseline = campaign.records[0]
    assert baseline.sim_id == "baseline_mean"
    assert abs(baseline.eta_bse - 0.27) < 1e-9
    assert abs(baseline.peak_bse_angle_deg - 48.0) < 1e-9


def test_assemble_casino_benchmark_campaign_auto_single_run_records(tmp_path):
    export_root = tmp_path / "exports"
    _write_run_export(export_root / "baseline1", eta_bse=0.25, energy_kev=60.0, angle_peak_deg=47)
    _write_run_export(export_root / "baseline2", eta_bse=0.27, energy_kev=60.0, angle_peak_deg=49)
    _write_run_export(export_root / "tilt10", eta_bse=0.29, energy_kev=60.0, angle_peak_deg=50)
    _write_run_export(export_root / "pore", eta_bse=0.24, energy_kev=60.0, angle_peak_deg=44)

    assembly = {
        "schema_version": 1,
        "campaign_id": "304ss_wave1_auto",
        "auto_single_run_records": True,
        "defaults": {
            "material": "304SS",
            "energy_kev": 60.0,
            "beam_diameter_nm": 10.0,
            "electrons": 500000,
            "quality_flag": "usable",
        },
        "runs": [
            {"run_id": "baseline1", "artifact_root": "baseline1", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "baseline2", "artifact_root": "baseline2", "geometry": "flatbulk", "tilt_deg": 0.0},
            {"run_id": "tilt10", "artifact_root": "tilt10", "geometry": "flatbulk", "tilt_deg": 10.0},
            {"run_id": "pore", "artifact_root": "pore", "geometry": "spherical_pore_single", "tilt_deg": 0.0},
        ],
        "records": [
            {
                "sim_id": "baseline_median",
                "from_runs": ["baseline1", "baseline2"],
                "geometry": "flatbulk",
                "tilt_deg": 0.0,
                "reducer": "median",
                "notes": "Grouped baseline",
            }
        ],
    }
    assembly_path = tmp_path / "assembly_auto.json"
    assembly_path.write_text(json.dumps(assembly, indent=2), encoding="utf-8")

    out = assemble_casino_benchmark_campaign(
        export_root=export_root,
        assembly_config_path=assembly_path,
        output_path=tmp_path / "campaign_auto.json",
    )
    assert out["status"] == "ok"
    campaign = load_casino_benchmark_campaign(tmp_path / "campaign_auto.json")
    sim_ids = {record.sim_id for record in campaign.records}
    assert "baseline_median" in sim_ids
    assert "tilt10" in sim_ids
    assert "pore" in sim_ids
    assert len(campaign.records) == 3
