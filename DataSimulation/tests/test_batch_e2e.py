import json
import tempfile
from pathlib import Path

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.batch_generator import BatchGenerator, BatchOptions
from bse_renderer.renderer.emission_b import EMISSION_B_MODE_POOLED_CONSTANT_FIT


def test_batch_outputs():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    bundle = load_config_bundle(cfg_dir)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dataset"
        summary = BatchGenerator(
            bundle,
            BatchOptions(num_images=12, output_dir=out, image_size=64, seed=5),
        ).run()
        assert Path(summary["manifest"]).exists()
        assert Path(summary["dataset_yaml"]).exists()
        assert Path(summary["coco"]).exists()
        assert Path(summary["dataset_contract"]).exists()
        assert Path(summary["records"]).exists()
        manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
        contract_records = [
            json.loads(line)
            for line in Path(summary["records"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(manifest) == 12
        assert len(contract_records) == 12
        assert contract_records[0]["contract_version"] == "weld-preview/v1"
        assert contract_records[0]["sequence_id"].startswith("static-sample-")
        assert contract_records[0]["frame_index"] == 0
        assert "compo" not in contract_records[0]["channels"]
        assert not Path(contract_records[0]["channels"]["topo"]).is_absolute()
        assert not Path(contract_records[0]["mask_path"]).is_absolute()
        assert isinstance(contract_records[0]["emission_b_value"], float)
        topo_pngs = list((out / "images" / "topo").rglob("*.png"))
        tx_pngs = list((out / "images" / "tx").rglob("*.png"))
        ty_pngs = list((out / "images" / "ty").rglob("*.png"))
        masks = list((out / "masks").rglob("*.png"))
        yolo = list((out / "labels" / "yolo").rglob("*.txt"))
        assert not (out / "images" / "compo").exists()
        assert len(topo_pngs) == 12
        assert len(tx_pngs) == 12
        assert len(ty_pngs) == 12
        assert len(masks) == 12
        assert len(yolo) == 12
        assert "tx_u16" in manifest[0]["artifacts"]
        assert "ty_u16" in manifest[0]["artifacts"]


def test_batch_outputs_can_disable_noise():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    bundle = load_config_bundle(cfg_dir)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dataset"
        summary = BatchGenerator(
            bundle,
            BatchOptions(num_images=2, output_dir=out, image_size=64, seed=11, disable_noise=True),
        ).run()
        manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
        for entry in manifest:
            noise = entry["metadata"]["noise"]
            assert noise["poisson_enabled"] is False
            assert noise["thermal_enabled"] is False
            assert noise["contamination_enabled"] is False
            assert noise["contamination_rate"] == 0.0


def test_batch_outputs_with_pooled_constant_emission_b_fit(bind_qualification, monkeypatch):
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    bundle = load_config_bundle(cfg_dir)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dataset"
        fit_path = Path(td) / "b_fit_gui_summary.json"
        fit_path.write_text(
            json.dumps(
                {
                    "model_comparison": {
                        "constant_b": {
                            "fitted_b": 0.27,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        fixture = json.loads(fit_path.read_text())
        fixture["renderer_qualification"] = {"pooled_constant_fit": {
            "surface_selection_passed": True, "adequacy_passed": True, "repeat_holdout_passed": True,
            "tilt_holdout_passed": True, "beats_lambertian": True, "tilt_domain_deg": [0, 90],
            "source_sha256": "unit-test-fixture", "energy_domain_kev": [1, 300], "material_z": bundle.material.presets[bundle.material.default_material].z}}
        fit_path.write_text(json.dumps(fixture))
        bind_qualification(fit_path, fixture)
        # This batch test covers a homogeneous-alloy fixture. Production local
        # material variations are covered by the campaign domain rejection test.
        from bse_renderer.renderer.scan_engine import ScanEngine
        original = ScanEngine.run
        def homogeneous_run(self, sample, **kwargs):
            from dataclasses import replace
            self.emission_b_fit = replace(self.emission_b_fit, qualified_material_z=None, qualified_energy_domain=None)
            return original(self, sample, **kwargs)
        monkeypatch.setattr(ScanEngine, 'run', homogeneous_run)
        summary = BatchGenerator(
            bundle,
            BatchOptions(
                num_images=2,
                output_dir=out,
                image_size=64,
                seed=17,
                disable_noise=True,
                emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT,
                emission_b_path=str(fit_path),
            ),
        ).run()
        manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
        for entry in manifest:
            emission_b = entry["metadata"]["emission_b"]
            assert emission_b["mode"] == EMISSION_B_MODE_POOLED_CONSTANT_FIT
            assert emission_b["source_type"] == "b_fit_gui_summary"
            assert abs(float(emission_b["b_stats"]["mean"]) - 0.27) <= 1e-6
            noise = entry["metadata"]["noise"]
            assert noise["poisson_enabled"] is False
            assert noise["thermal_enabled"] is False
            assert noise["contamination_enabled"] is False


def test_top_surface_batch_records_surface_visible_label_scope():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    bundle = load_config_bundle(cfg_dir)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "surface_dataset"
        summary = BatchGenerator(
            bundle,
            BatchOptions(
                num_images=3,
                output_dir=out,
                image_size=64,
                seed=23,
                disable_noise=True,
                render_view_mode="top_surface",
            ),
        ).run()
        manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
        assert summary["render_view_mode"] == "top_surface"
        assert "label_scope: surface_visible_intersections_only" in Path(
            summary["dataset_yaml"]
        ).read_text(encoding="utf-8")
        for entry in manifest:
            assert entry["defects"] == entry["visible_defects"]
            assert entry["metadata"]["surface_visible_label_policy"] is True
            assert entry["metadata"]["defects_anchored_to_surface"] is True
            assert len(entry["planned_defects"]) == (
                len(entry["visible_defects"]) + len(entry["excluded_nonvisible_defects"])
            )
