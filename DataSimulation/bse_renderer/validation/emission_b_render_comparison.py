from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..comparison_metrics import compute_topography_delta_metrics
from ..pipeline.generate import _build_sampled_parameters, _build_weld_layout
from ..renderer import (
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
    ScanEngine,
    load_emission_b_fit,
)
from ..renderer.calibration import RENDER_MODEL_FORMULA_ONLY


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_b_fit_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if "fit" not in report or "per_tilt" not in report["fit"]:
        raise ValueError(f"B-fit report does not contain fit.per_tilt rows: {path}")
    return report


def _per_tilt_b_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in report["fit"]["per_tilt"]:
        nominal_tilt = int(round(float(row["nominal_tilt_deg"])))
        rows[nominal_tilt] = row
    return rows


def _winner_from_combined_delta(pooled_constant_delta: float, quadratic_delta: float) -> str:
    if quadratic_delta + 1e-12 < pooled_constant_delta:
        return "quadratic"
    if pooled_constant_delta + 1e-12 < quadratic_delta:
        return "pooled_constant"
    return "tie"


def _delta_vs_target(reference, candidate) -> dict[str, Any]:
    compo_delta = np.abs(candidate.channels.compo - reference.channels.compo)
    topo_metrics = compute_topography_delta_metrics(
        reference.channels.topo,
        candidate.channels.topo,
        defect_mask=candidate.defect_mask,
    )
    combined = float(compo_delta.mean()) + float(topo_metrics["mean_abs_topo_delta"])
    return {
        "mean_abs_compo_delta": float(compo_delta.mean()),
        "max_abs_compo_delta": float(compo_delta.max()),
        "combined_mean_abs_delta": combined,
        **topo_metrics,
    }


def _bbox_from_mask(mask: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        height, width = mask_bool.shape
        return 0, height, 0, width
    ys, xs = np.where(mask_bool)
    height, width = mask_bool.shape
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, height)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    return y0, y1, x0, x1


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return np.asarray(arr[y0:y1, x0:x1])


def _profile_rows(
    bbox: tuple[int, int, int, int],
    pooled_constant_result,
    pooled_constant_clean,
    quadratic_result,
    quadratic_clean,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    y0, y1, x0, x1 = bbox
    cy = (y1 - y0) // 2
    cx = (x1 - x0) // 2
    pooled_roi = _crop(pooled_constant_result.channels.topo, bbox)
    pooled_clean_roi = _crop(pooled_constant_clean.channels.topo, bbox)
    quadratic_roi = _crop(quadratic_result.channels.topo, bbox)
    quadratic_clean_roi = _crop(quadratic_clean.channels.topo, bbox)

    horizontal_rows: list[dict[str, float]] = []
    for idx in range(pooled_roi.shape[1]):
        horizontal_rows.append(
            {
                "axis_index": float(idx),
                "pooled_constant_topo": float(pooled_roi[cy, idx]),
                "pooled_constant_clean_topo": float(pooled_clean_roi[cy, idx]),
                "quadratic_topo": float(quadratic_roi[cy, idx]),
                "quadratic_clean_topo": float(quadratic_clean_roi[cy, idx]),
            }
        )

    vertical_rows: list[dict[str, float]] = []
    for idx in range(pooled_roi.shape[0]):
        vertical_rows.append(
            {
                "axis_index": float(idx),
                "pooled_constant_topo": float(pooled_roi[idx, cx]),
                "pooled_constant_clean_topo": float(pooled_clean_roi[idx, cx]),
                "quadratic_topo": float(quadratic_roi[idx, cx]),
                "quadratic_clean_topo": float(quadratic_clean_roi[idx, cx]),
            }
        )
    return horizontal_rows, vertical_rows


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    tilts = np.asarray([float(row["nominal_tilt_deg"]) for row in rows], dtype=np.float64)
    pooled_constant_delta = np.asarray(
        [float(row["pooled_constant_combined_mean_abs_delta_vs_reference"]) for row in rows],
        dtype=np.float64,
    )
    quadratic_delta = np.asarray(
        [float(row["quadratic_combined_mean_abs_delta_vs_reference"]) for row in rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(tilts, pooled_constant_delta, "o--", label="Pooled fitted constant B", linewidth=1.8)
    ax.plot(tilts, quadratic_delta, "s-", label="Quadratic fitted B(alpha)", linewidth=2.0)
    ax.set_xlabel("Tilt (deg)")
    ax.set_ylabel("Combined mean abs delta vs per-tilt reference")
    ax.set_title("Renderer comparison: pooled constant B vs quadratic B(alpha)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_full_frame_figure(
    path: Path,
    *,
    nominal_tilt_deg: float,
    pooled_constant_result,
    pooled_constant_clean,
    quadratic_result,
    quadratic_clean,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    pooled_response = pooled_constant_result.channels.topo - pooled_constant_clean.channels.topo
    quadratic_response = quadratic_result.channels.topo - quadratic_clean.channels.topo
    response_span = max(
        float(np.max(np.abs(pooled_response))),
        float(np.max(np.abs(quadratic_response))),
        1e-6,
    )

    fig, axes = plt.subplots(4, 2, figsize=(10.5, 14.5))
    layouts = [
        ("TOPO", "topo", "gray", None),
        ("Tx", "tx", "gray", None),
        ("Ty", "ty", "gray", None),
    ]
    mode_specs = [
        ("Pooled fitted constant B", pooled_constant_result, pooled_constant_clean),
        ("Quadratic fitted B(alpha)", quadratic_result, quadratic_clean),
    ]
    for col_idx, (label, defected, clean) in enumerate(mode_specs):
        for row_idx, (title, channel_name, cmap, limits) in enumerate(layouts):
            image = getattr(defected.channels, channel_name)
            axes[row_idx, col_idx].imshow(image, cmap=cmap)
            axes[row_idx, col_idx].set_title(f"{label} {title}")
            axes[row_idx, col_idx].axis("off")
        response = defected.channels.topo - clean.channels.topo
        axes[3, col_idx].imshow(response, cmap="RdBu_r", vmin=-response_span, vmax=response_span)
        axes[3, col_idx].imshow(
            np.ma.masked_where(~defected.defect_mask.astype(bool), defected.defect_mask.astype(np.float32)),
            cmap="autumn",
            alpha=0.35,
        )
        axes[3, col_idx].set_title(f"{label} TOPO defected - clean")
        axes[3, col_idx].axis("off")
    fig.suptitle(f"Emission B comparison at tilt {nominal_tilt_deg:.0f} deg", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_roi_figure(
    path: Path,
    *,
    nominal_tilt_deg: float,
    bbox: tuple[int, int, int, int],
    pooled_constant_result,
    pooled_constant_clean,
    quadratic_result,
    quadratic_clean,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    pooled_topo = _crop(pooled_constant_result.channels.topo, bbox)
    quadratic_topo = _crop(quadratic_result.channels.topo, bbox)
    pooled_response = _crop(pooled_constant_result.channels.topo - pooled_constant_clean.channels.topo, bbox)
    quadratic_response = _crop(quadratic_result.channels.topo - quadratic_clean.channels.topo, bbox)
    mask_roi = _crop(pooled_constant_result.defect_mask.astype(np.float32), bbox)
    response_span = max(
        float(np.max(np.abs(pooled_response))),
        float(np.max(np.abs(quadratic_response))),
        1e-6,
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    axes[0, 0].imshow(pooled_topo, cmap="gray")
    axes[0, 0].imshow(np.ma.masked_where(mask_roi <= 0, mask_roi), cmap="autumn", alpha=0.35)
    axes[0, 0].set_title("Pooled constant ROI TOPO")
    axes[0, 1].imshow(quadratic_topo, cmap="gray")
    axes[0, 1].imshow(np.ma.masked_where(mask_roi <= 0, mask_roi), cmap="autumn", alpha=0.35)
    axes[0, 1].set_title("Quadratic ROI TOPO")
    axes[1, 0].imshow(pooled_response, cmap="RdBu_r", vmin=-response_span, vmax=response_span)
    axes[1, 0].set_title("Pooled constant ROI defected - clean")
    axes[1, 1].imshow(quadratic_response, cmap="RdBu_r", vmin=-response_span, vmax=response_span)
    axes[1, 1].set_title("Quadratic ROI defected - clean")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"ROI zoom at tilt {nominal_tilt_deg:.0f} deg", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_profile_figure(
    path: Path,
    *,
    nominal_tilt_deg: float,
    horizontal_rows: list[dict[str, float]],
    vertical_rows: list[dict[str, float]],
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, rows, title in (
        (axes[0], horizontal_rows, "Horizontal ROI center profile"),
        (axes[1], vertical_rows, "Vertical ROI center profile"),
    ):
        axis = np.asarray([float(row["axis_index"]) for row in rows], dtype=np.float64)
        ax.plot(axis, [float(row["pooled_constant_topo"]) for row in rows], label="Pooled constant")
        ax.plot(
            axis,
            [float(row["pooled_constant_clean_topo"]) for row in rows],
            "--",
            label="Pooled clean",
        )
        ax.plot(axis, [float(row["quadratic_topo"]) for row in rows], label="Quadratic")
        ax.plot(
            axis,
            [float(row["quadratic_clean_topo"]) for row in rows],
            "--",
            label="Quadratic clean",
        )
        ax.set_title(title)
        ax.set_xlabel("Pixel index")
        ax.set_ylabel("TOPO")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"TOPO line profiles at tilt {nominal_tilt_deg:.0f} deg", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Emission B Renderer Comparison",
        "",
        f"- Render model: `{summary['render_model_mode']}`",
        f"- Pooled fitted constant B: {summary['pooled_constant_b']:.6f}",
        f"- Reference source: per-tilt observed B from `{summary['b_fit_report_path']}`",
        f"- Comparison metric: {summary['winner_metric']}",
        f"- Quadratic wins: {summary['quadratic_win_count']}",
        f"- Pooled constant wins: {summary['pooled_constant_win_count']}",
        f"- Ties: {summary['tie_count']}",
        "",
        "## Per-Tilt Results",
        "",
        "| Tilt (deg) | Reference B | Pooled constant B | Quadratic mean B | Pooled constant combined delta | Quadratic combined delta | Winner |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['nominal_tilt_deg'])} | {row['reference_observed_b']:.6f} | {row['pooled_constant_b']:.6f} | {row['quadratic_b_mean']:.6f} | {row['pooled_constant_combined_mean_abs_delta_vs_reference']:.6f} | {row['quadratic_combined_mean_abs_delta_vs_reference']:.6f} | {row['winner']} |"
        )
        lines.append(
            f"|  |  |  |  | defect/background ratio pooled={row['pooled_constant_defect_to_background_ratio']} | defect/background ratio quadratic={row['quadratic_defect_to_background_ratio']} | artifacts: {row['artifact_dir_name']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Each tilt uses the same seed and sampled beam, detector, geometry, slice controls, and defect placement; only the emission-B mode changes between pooled constant and quadratic.",
            "- Noise is disabled for this comparison so the result reflects the emission-B law rather than stochastic rendering noise.",
            "- Each tilt directory contains a full-frame TOPO/Tx/Ty panel, ROI zoom, and TOPO line-profile figure plus CSVs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_emission_b_render_modes(
    *,
    config_dir: str | Path,
    b_fit_report_path: str | Path,
    emission_b_fit_path: str | Path,
    output_dir: str | Path,
    tilts_deg: list[float] | None = None,
    seed: int = 42,
    image_size: int = 256,
    pixel_size_mm: float = 0.05,
    material_name: str = "304SS",
    average_atomic_number: float | None = None,
    defect_plan: str | None = None,
    max_defects: int = 1,
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
    randomize_defect_location: bool = True,
    weld_layout_mode: str = "local_patch",
    seam_rotation_deg: float = 0.0,
    seam_offset_mm: float = 0.0,
    circle_radius_mm: float = 80.0,
    circle_position_deg: float = 90.0,
    render_model_mode: str = RENDER_MODEL_FORMULA_ONLY,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    report = _load_b_fit_report(b_fit_report_path)
    per_tilt = _per_tilt_b_rows(report)
    if tilts_deg is None:
        tilt_values = sorted(per_tilt.keys())
    else:
        tilt_values = [int(round(float(v))) for v in tilts_deg]

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_sample = _build_sampled_parameters(
        config_dir=config_dir,
        seed=seed,
        image_size=image_size,
        material_name=material_name,
        average_atomic_number=average_atomic_number,
        defect_plan=defect_plan,
        max_defects=max_defects,
        voxel_depth_mm=voxel_depth_mm,
        slice_offset_mm=slice_offset_mm,
        slice_tilt_deg=slice_tilt_deg,
        slice_rotation_deg=slice_rotation_deg,
        randomize_defect_location=randomize_defect_location,
    )
    noise_free = replace(
        base_sample.noise,
        poisson_enabled=False,
        thermal_enabled=False,
        contamination_enabled=False,
        contamination_rate=0.0,
        quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
    )
    defected_base_sample = replace(
        base_sample,
        noise=noise_free,
        randomize_defect_location=False,
    )
    clean_base_sample = replace(defected_base_sample, defects=[])

    pooled_constant_fit = load_emission_b_fit(
        emission_b_fit_path,
        EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    )
    pooled_constant_b = float(pooled_constant_fit.fitted_b if pooled_constant_fit.fitted_b is not None else np.nan)
    emission_b_fit_path_resolved = str(Path(emission_b_fit_path).resolve())
    engine_kwargs = {
        "image_size": image_size,
        "pixel_size_mm": pixel_size_mm,
        "calibration_path": None if calibration_path is None else str(Path(calibration_path).resolve()),
        "render_model_mode": render_model_mode,
        "weld_layout": _build_weld_layout(
            weld_layout_mode=weld_layout_mode,
            seam_rotation_deg=seam_rotation_deg,
            seam_offset_mm=seam_offset_mm,
            circle_radius_mm=circle_radius_mm,
            circle_position_deg=circle_position_deg,
        ),
    }
    reference_engine = ScanEngine(allow_unvalidated_emission_b=True, 
        emission_b_mode=EMISSION_B_MODE_CONSTANT,
        emission_b_path=None,
        **engine_kwargs,
    )
    pooled_constant_engine = ScanEngine(allow_unvalidated_emission_b=True, 
        emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT,
        emission_b_path=emission_b_fit_path_resolved,
        **engine_kwargs,
    )
    quadratic_engine = ScanEngine(allow_unvalidated_emission_b=True, 
        emission_b_mode=EMISSION_B_MODE_QUADRATIC_FIT,
        emission_b_path=emission_b_fit_path_resolved,
        **engine_kwargs,
    )

    rows: list[dict[str, Any]] = []
    quadratic_win_count = 0
    pooled_constant_win_count = 0
    tie_count = 0

    for nominal_tilt_deg in tilt_values:
        if nominal_tilt_deg not in per_tilt:
            raise ValueError(
                f"Tilt {nominal_tilt_deg} deg is not available in the B-fit report per_tilt rows."
            )
        reference_observed_b = float(per_tilt[nominal_tilt_deg]["fitted_b"])
        defected_sample = replace(
            defected_base_sample,
            slice=replace(defected_base_sample.slice, slice_tilt_deg=float(nominal_tilt_deg)),
        )
        clean_sample = replace(
            clean_base_sample,
            slice=replace(clean_base_sample.slice, slice_tilt_deg=float(nominal_tilt_deg)),
        )

        reference_sample = replace(
            defected_sample,
            material=replace(defected_sample.material, b=reference_observed_b),
        )

        reference_result = reference_engine.run(reference_sample)
        pooled_constant_result = pooled_constant_engine.run(defected_sample)
        pooled_constant_clean = pooled_constant_engine.run(clean_sample)
        quadratic_result = quadratic_engine.run(defected_sample)
        quadratic_clean = quadratic_engine.run(clean_sample)

        pooled_constant_delta = _delta_vs_target(reference_result, pooled_constant_result)
        quadratic_delta = _delta_vs_target(reference_result, quadratic_result)
        winner = _winner_from_combined_delta(
            float(pooled_constant_delta["combined_mean_abs_delta"]),
            float(quadratic_delta["combined_mean_abs_delta"]),
        )
        if winner == "quadratic":
            quadratic_win_count += 1
        elif winner == "pooled_constant":
            pooled_constant_win_count += 1
        else:
            tie_count += 1

        pooled_constant_defect_metrics = compute_topography_delta_metrics(
            pooled_constant_clean.channels.topo,
            pooled_constant_result.channels.topo,
            defect_mask=pooled_constant_result.defect_mask,
        )
        quadratic_defect_metrics = compute_topography_delta_metrics(
            quadratic_clean.channels.topo,
            quadratic_result.channels.topo,
            defect_mask=quadratic_result.defect_mask,
        )

        roi_mask = pooled_constant_result.defect_mask.astype(bool) | quadratic_result.defect_mask.astype(bool)
        bbox = _bbox_from_mask(roi_mask)
        horizontal_rows, vertical_rows = _profile_rows(
            bbox,
            pooled_constant_result,
            pooled_constant_clean,
            quadratic_result,
            quadratic_clean,
        )
        artifact_dir = output_root / f"tilt_{int(nominal_tilt_deg):03d}"
        full_frame_info = _write_full_frame_figure(
            artifact_dir / "full_frame_topo_tx_ty.png",
            nominal_tilt_deg=float(nominal_tilt_deg),
            pooled_constant_result=pooled_constant_result,
            pooled_constant_clean=pooled_constant_clean,
            quadratic_result=quadratic_result,
            quadratic_clean=quadratic_clean,
        )
        roi_info = _write_roi_figure(
            artifact_dir / "roi_zoom.png",
            nominal_tilt_deg=float(nominal_tilt_deg),
            bbox=bbox,
            pooled_constant_result=pooled_constant_result,
            pooled_constant_clean=pooled_constant_clean,
            quadratic_result=quadratic_result,
            quadratic_clean=quadratic_clean,
        )
        profile_plot_info = _write_profile_figure(
            artifact_dir / "topo_profiles.png",
            nominal_tilt_deg=float(nominal_tilt_deg),
            horizontal_rows=horizontal_rows,
            vertical_rows=vertical_rows,
        )
        _write_csv(artifact_dir / "topo_profile_horizontal.csv", horizontal_rows)
        _write_csv(artifact_dir / "topo_profile_vertical.csv", vertical_rows)

        quadratic_meta = quadratic_result.metadata.get("emission_b") or {}
        quadratic_stats = quadratic_meta.get("b_stats") or {}
        rows.append(
            {
                "nominal_tilt_deg": float(nominal_tilt_deg),
                "reference_observed_b": reference_observed_b,
                "pooled_constant_b": pooled_constant_b,
                "quadratic_b_mean": float(quadratic_stats.get("mean", np.nan)),
                "quadratic_b_min": float(quadratic_stats.get("min", np.nan)),
                "quadratic_b_max": float(quadratic_stats.get("max", np.nan)),
                "pooled_constant_mean_abs_compo_delta_vs_reference": float(
                    pooled_constant_delta["mean_abs_compo_delta"]
                ),
                "pooled_constant_mean_abs_topo_delta_vs_reference": float(
                    pooled_constant_delta["mean_abs_topo_delta"]
                ),
                "pooled_constant_combined_mean_abs_delta_vs_reference": float(
                    pooled_constant_delta["combined_mean_abs_delta"]
                ),
                "quadratic_mean_abs_compo_delta_vs_reference": float(
                    quadratic_delta["mean_abs_compo_delta"]
                ),
                "quadratic_mean_abs_topo_delta_vs_reference": float(
                    quadratic_delta["mean_abs_topo_delta"]
                ),
                "quadratic_combined_mean_abs_delta_vs_reference": float(
                    quadratic_delta["combined_mean_abs_delta"]
                ),
                "quadratic_minus_pooled_constant_combined_delta": float(
                    quadratic_delta["combined_mean_abs_delta"] - pooled_constant_delta["combined_mean_abs_delta"]
                ),
                "pooled_constant_defect_to_background_ratio": pooled_constant_defect_metrics[
                    "defect_to_background_mean_ratio"
                ],
                "quadratic_defect_to_background_ratio": quadratic_defect_metrics[
                    "defect_to_background_mean_ratio"
                ],
                "winner": winner,
                "artifact_dir_name": artifact_dir.name,
                "full_frame_plot_path": full_frame_info["path"],
                "roi_plot_path": roi_info["path"],
                "profile_plot_path": profile_plot_info["path"],
                "reference_metadata": _json_ready(reference_result.metadata),
                "pooled_constant_metadata": _json_ready(pooled_constant_result.metadata),
                "quadratic_metadata": _json_ready(quadratic_result.metadata),
            }
        )

    plot_info = _write_plot(output_root / "emission_b_render_comparison.png", rows)
    summary = {
        "status": "ok",
        "render_model_mode": render_model_mode,
        "config_dir": str(Path(config_dir).resolve()),
        "b_fit_report_path": str(Path(b_fit_report_path).resolve()),
        "emission_b_fit_path": emission_b_fit_path_resolved,
        "seed": int(seed),
        "image_size": int(image_size),
        "pixel_size_mm": float(pixel_size_mm),
        "material_name": str(material_name),
        "pooled_constant_b": pooled_constant_b,
        "winner_metric": "combined_mean_abs_delta = mean_abs_compo_delta + mean_abs_topo_delta versus the per-tilt CASINO-derived observed-B reference render",
        "tilt_count": len(rows),
        "quadratic_win_count": quadratic_win_count,
        "pooled_constant_win_count": pooled_constant_win_count,
        "tie_count": tie_count,
        "artifacts": {
            "plot_status": plot_info["status"],
            "plot_path": plot_info["path"],
            "plot_reason": plot_info["reason"],
            "csv_path": str(output_root / "emission_b_render_comparison.csv"),
            "json_path": str(output_root / "emission_b_render_comparison.json"),
            "markdown_path": str(output_root / "emission_b_render_comparison.md"),
        },
    }
    payload = {
        "summary": summary,
        "rows": rows,
    }
    _write_csv(output_root / "emission_b_render_comparison.csv", rows)
    (output_root / "emission_b_render_comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(output_root / "emission_b_render_comparison.md", payload)
    return payload
