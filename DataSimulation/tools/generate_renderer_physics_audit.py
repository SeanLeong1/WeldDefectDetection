from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bse_renderer.models import DetectorRuntime, MaterialPreset
from bse_renderer.renderer.quadrant_detector import compute_quadrant_signals
from bse_renderer.renderer.scan_engine import ScanEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the fixed analytical renderer physics panel.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size_mm = 0.05
    axis = (np.arange(129, dtype=np.float64) - 64.0) * pixel_size_mm
    x, y = np.meshgrid(axis, axis)
    sigma_mm = 0.75
    height = 0.15 * np.exp(-(x * x + y * y) / (2.0 * sigma_mm * sigma_mm))
    normals = ScanEngine._normals_from_heightmap(height, pixel_size_mm)
    detector = DetectorRuntime(
        height_mm=12.0,
        radius_mm=12.0,
        inner_radius_mm=4.0,
        outer_radius_mm=16.0,
        quadrants_deg={"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0},
        sensitivity={"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
        psi0_rad=float(np.arctan2(12.0, 12.0)),
    )
    material = MaterialPreset(name="304SS", z=25.4147, a=54.5465, rho=8.03, b=0.19, n=1.76)
    quadrants = compute_quadrant_signals(
        normals,
        detector,
        material,
        np.full(height.shape, material.z, dtype=np.float32),
        b_coeff=0.0,
    )
    x_difference = quadrants[..., 0] - quadrants[..., 2]
    y_difference = quadrants[..., 1] - quadrants[..., 3]
    topo = np.hypot(x_difference, y_difference)
    x_correlation = float(np.corrcoef(normals[..., 0].ravel(), x_difference.ravel())[0, 1])
    y_correlation = float(np.corrcoef(normals[..., 1].ravel(), y_difference.ravel())[0, 1])

    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = (
        (height, "Analytical Gaussian height (mm)", "viridis"),
        (normals[..., 0], "Physical normal x", "coolwarm"),
        (normals[..., 1], "Physical normal y", "coolwarm"),
        (x_difference, "Integrated quadrant A minus C", "coolwarm"),
        (y_difference, "Integrated quadrant B minus D", "coolwarm"),
        (topo, "Directional magnitude", "magma"),
    )
    for axis_plot, (image, title, cmap) in zip(axes.ravel(), panels):
        shown = axis_plot.imshow(image, cmap=cmap, origin="lower")
        axis_plot.set_title(title)
        axis_plot.set_xticks([])
        axis_plot.set_yticks([])
        figure.colorbar(shown, ax=axis_plot, shrink=0.76)
    figure.suptitle("Renderer physics gate: spacing-aware normals and annular quadrant integration")
    panel_path = output_dir / "renderer_physics_panel.png"
    figure.savefig(panel_path, dpi=180)
    plt.close(figure)

    report = {
        "status": "ok" if x_correlation > 0.99 and y_correlation > 0.99 else "failed",
        "pixel_size_mm": pixel_size_mm,
        "height_model": "z=0.15 exp(-(x^2+y^2)/(2*0.75^2)) mm",
        "emission_model": "explicit Lambertian fallback, B=0",
        "x_normal_vs_A_minus_C_correlation": x_correlation,
        "y_normal_vs_B_minus_D_correlation": y_correlation,
        "quadrature_order_phi": 6,
        "quadrature_order_varphi": 8,
        "panel": str(panel_path),
    }
    report_path = output_dir / "renderer_physics_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
