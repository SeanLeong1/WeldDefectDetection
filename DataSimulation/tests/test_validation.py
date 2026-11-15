from pathlib import Path

from bse_renderer.validation.casino_comparison import run_casino_comparison
from bse_renderer.validation.casino_adapter import (
    parse_casino_case_output,
    run_casino_case,
    run_casino_probe,
)
from bse_renderer.validation.casino_calibration import run_casino_calibration
from bse_renderer.validation.signal_tests import run_signal_tests
from bse_renderer.validation.standards import load_standards_defaults
from bse_renderer.validation.visibility_pilot import run_visibility_pilot


def test_signal_tests_contract():
    out = run_signal_tests()
    assert "eta_bounds_ok" in out
    assert "eta_void" in out
    assert out["eta_void"] == 0.0


def test_casino_comparison_contract():
    out = run_casino_comparison(run_budget="fast", regime_cases=2)
    assert "status" in out
    assert out["status"] in {"ok", "partial", "skipped"}
    assert "casino_probe" in out
    assert "regime_validation" in out
    assert "standards_threshold_slices" in out
    threshold_settings = out["standards_threshold_slices"]["settings"]
    assert threshold_settings["randomize_defect_location"] is False
    assert threshold_settings["slice"]["slice_offset_mm"] == 0.0
    assert threshold_settings["slice"]["slice_tilt_deg"] == 0.0
    assert threshold_settings["slice"]["slice_rotation_deg"] == 0.0
    assert "artifacts" in out
    report_path = out["artifacts"].get("validation_report")
    assert report_path is not None
    assert Path(report_path).exists()

    all_labels = []
    for case in out["regime_validation"]["cases"]:
        for bins in case["standards"]["pod_bins"].values():
            all_labels.extend([b["bin"] for b in bins])
    assert all("[0.000,0.000)" not in label for label in all_labels)


def test_casino_probe_contract():
    out = run_casino_probe(casino_exe="missing_casino_binary_for_test")
    assert "status" in out
    assert out["status"] in {"ok", "partial", "skipped"}
    assert "available" in out


def test_casino_case_unavailable_contract():
    out = run_casino_case(
        case={"beam": {"e0_kev": 60.0}},
        casino_exe="missing_casino_binary_for_test",
    )
    assert "status" in out
    assert out["status"] in {"ok", "failed", "skipped"}


def test_casino_calibration_unavailable_contract(tmp_path):
    out = run_casino_calibration(
        config_dir="config",
        casino_exe="missing_casino_binary_for_test",
        run_root=tmp_path / "calib",
        num_cases=2,
    )
    assert "status" in out
    assert out["status"] in {"ok", "partial", "skipped"}
    assert "calibration_file" in out


def test_casino_case_parser_contract(tmp_path):
    p = tmp_path / "casino_quadrants.csv"
    p.write_text("A,B,C,D\n1.0,2.0,3.0,4.0\n", encoding="utf-8")
    parsed = parse_casino_case_output(tmp_path)
    assert parsed["quadrants"]["A"] == 1.0
    assert parsed["quadrants"]["D"] == 4.0
    assert set(parsed["derived"].keys()) == {"compo", "topo", "tx", "ty", "rx"}


def test_visibility_pilot_balances_surface_classes_and_derives_channel_thresholds(tmp_path):
    out = run_visibility_pilot(
        config_dir="config",
        output_dir=tmp_path / "pilot",
        repeats_per_class_regime=1,
        image_size=64,
        save_images=False,
    )
    assert out["counts"]["planned_cases"] == 18
    assert set(out["class_summary"]) == {
        "keyhole_pore",
        "intrinsic_pore",
        "crack_solidification",
        "crack_liquidation",
        "lack_of_fusion",
        "undercut",
    }
    assert out["threshold_recommendation"]["compo"]["threshold"] > 0.0
    assert out["threshold_recommendation"]["topo"]["threshold"] > 0.0
    assert Path(out["artifacts"]["csv"]).exists()
    assert Path(out["artifacts"]["report"]).exists()


def test_standards_use_separate_data_derived_channel_thresholds():
    standards = load_standards_defaults("config")
    assert standards.contrast_detection_threshold == 0.0005
    assert standards.topo_contrast_detection_threshold == 0.0015
