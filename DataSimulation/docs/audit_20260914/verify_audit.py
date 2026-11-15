"""Read-only project checks; writes evidence only to the requested output directory.

Run with a working Python 3.12 and the project dependencies. The optional
--workspace-libraries flag reuses this checkout's .venv site-packages when its
launcher is broken. It neither repairs the environment nor changes source code.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--workspace-libraries", action="store_true")
parser.add_argument("--inventory-only", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.workspace_libraries:
    sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages"))
sys.path[:0] = [str(ROOT / "DataSimulation"), str(ROOT / "WeldDefectDetection")]
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import cv2
import torch
torch.set_num_threads(2)

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.domain_randomisation.dataset_forms import _disable_noise
from bse_renderer.physics.emission_model import emission_distribution, solid_angle_integral
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.renderer.calibration import RendererCalibration, ChannelCalibration
from bse_renderer.validation.casino_adapter import parse_casino_case_output
from src.algorithms.classical.cnn_detector import CNNDetector
from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.evaluation.completion_verifier import verify_completion_manifest


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


inventory = []
for project in ("DataSimulation", "WeldDefectDetection"):
    paths = []
    for directory, subdirs, files in os.walk(ROOT / project):
        subdirs[:] = [name for name in subdirs if name not in {
            "data", "experiments", "__pycache__", ".git", "audit_20260914",
            ".venv", "venv", "env", "node_modules", ".git_nested_backup",
        }]
        paths.extend(Path(directory) / name for name in files if Path(name).suffix.lower() in {".py", ".cc", ".h", ".cpp", ".cmd", ".ahk"})
    for path in sorted(paths):
        raw = path.read_bytes()
        source = raw.decode("utf-8-sig")
        record = {"path": path.relative_to(ROOT).as_posix(), "lines": len(source.splitlines()),
                  "sha256": hashlib.sha256(raw).hexdigest(), "inspection": "source inventory and syntax scan"}
        if path.suffix == ".py":
            tree = ast.parse(source, filename=str(path))
            record["definitions"] = [{"name": n.name, "line": n.lineno} for n in tree.body
                                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        inventory.append(record)
write("source_inventory.json", inventory)
print("Inventory:", len(inventory), "source files;", sum(x["lines"] for x in inventory), "lines", flush=True)
if args.inventory_only:
    raise SystemExit(0)

cfg = load_config_bundle(ROOT / "DataSimulation/config")
sample = _disable_noise(sample_parameters(cfg, seed=731, max_defects=0))
sample = replace(sample, defects=[{"type": "keyhole_pore", "params": {"diameter_mm": 0.7, "depth_mm": 0.4}}])
engine = ScanEngine(image_size=48, render_view_mode="top_surface", render_model_mode="formula_only", emission_b_mode="lambertian")
normal_result = engine.run(sample)
with patch("bse_renderer.renderer.scan_engine.kanaya_okayama_range", return_value=999999.0):
    altered_range_result = engine.run(sample)

alpha = np.radians(30.0)
theta = np.radians(40.0)
normal = np.array([np.sin(alpha), 0.0, np.cos(alpha)])
direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
current_emission = float(emission_distribution(theta, 0.0, alpha, 0.0))
local_emission = float(np.dot(normal, direction))
cal = replace(RendererCalibration.identity(), topo=ChannelCalibration(scale=3.0))
injected = ScanEngine(calibration=cal, render_model_mode="formula_only")
probes = {
    "python": sys.version,
    "libraries": {"numpy": np.__version__, "opencv": cv2.__version__, "torch": torch.__version__},
    "angular_frame": {"surface_tilt_deg": 30, "detector_polar_deg": 40, "b": 0,
                      "renderer_emission": current_emission, "local_frame_emission": local_emission,
                      "relative_difference": current_emission / local_emission - 1},
    "range_only_change": {"original_range_um": normal_result.metadata["range_um"],
                          "replacement_range_um": altered_range_result.metadata["range_um"],
                          "max_abs_topo_difference": float(np.max(np.abs(normal_result.channels.topo-altered_range_result.channels.topo)))},
    "global_density_guard": {"b": 2.0, "alpha_deg": 60.0,
                             "domain_min_bracket": float(1-2*np.sin(np.radians(60))),
                             "accepted_sample_value": float(emission_distribution(np.radians(10), 0.0, np.radians(60), 2.0))},
    "unused_solid_angle_helper": {"hemisphere_integral_b0_default_grid": float(solid_angle_integral(0, np.pi/2, -np.pi, np.pi, 0.0, 0.0).item()), "analytic_value": float(np.pi)},
    "calibration_object_mode": {"declared_mode": injected.render_model_mode, "active_topo_scale": injected.calibration.topo.scale},
}
with tempfile.TemporaryDirectory(dir=OUT) as temporary:
    p = Path(temporary) / "casino_output.json"
    p.write_text(json.dumps({"quadrants": {"A": .1, "B": .2, "C": .3, "D": .4}, "meta": {"backend": "mock"}}))
    probes["mock_backend_parser"] = parse_casino_case_output(temporary)
    p.write_text("{}")
    probes["missing_quadrants_parser"] = parse_casino_case_output(temporary)
try:
    BidirectionalSurfacePredictor(context_channels=4)(torch.zeros(1, 2, 32, 32))
except RuntimeError as exc:
    probes["context_radius_one"] = str(exc)
detector = CNNDetector({"cnn": {"input_size": 32, "min_defect_area": 8}})
single = np.zeros((32, 32), np.float32)
single[16, 16] = 1
with patch.object(detector, "predict_mask", return_value=single):
    probes["cnn_component_filter"] = {"binary_mask_pixels": int(detector.predict_binary_mask(single).sum()), "detection_boxes": len(detector.detect(single))}

# Quantify the preprocessing discrepancy using the existing frozen checkpoint.
data_root = ROOT / "DataSimulation/data/output/completion_20260907/confirmation_static_lambertian_dev_v2"
checkpoint = ROOT / "WeldDefectDetection/experiments/completion_20260907/cnn_final/cnn_train_20260907-222736/best.pt"
if data_root.exists() and checkpoint.exists():
    dataset = DataSimulationSegmentationDataset(data_root, "all", input_size=128)
    detector = CNNDetector({"cnn": {"input_size": 128, "base_channels": 16, "checkpoint_path": str(checkpoint)}})
    max_input_delta = 0.0
    max_probability_delta = 0.0
    changed_pixels = 0
    changed_images = 0
    for index in range(len(dataset)):
        image = dataset[index]["image"][0].numpy()
        u8 = (image * 255).astype(np.uint8)
        max_input_delta = max(max_input_delta, float(np.max(np.abs(image-u8/255.0))))
        p_float = detector.predict_mask(image)
        p_u8 = detector.predict_mask(u8)
        changed = int(np.count_nonzero((p_float >= detector.threshold) != (p_u8 >= detector.threshold)))
        changed_pixels += changed
        changed_images += bool(changed)
        max_probability_delta = max(max_probability_delta, float(np.max(np.abs(p_float-p_u8))))
    probes["cnn_quantization_confirmation"] = {"images": len(dataset), "max_input_difference": max_input_delta,
        "max_probability_difference": max_probability_delta, "changed_binary_pixels": changed_pixels, "images_with_changed_mask": changed_images,
        "interpretation": "Diagnostic only; neither preprocessing policy was selected using these confirmation outcomes."}
write("numerical_probes.json", probes)
print("Numerical probes complete", flush=True)

manifest = ROOT / "WeldDefectDetection/completion_manifest_20260908.yaml"
verification = verify_completion_manifest(manifest)
write("completion_verification.json", verification)
print("Completion manifest:", verification["status"], verification["errors"], flush=True)
