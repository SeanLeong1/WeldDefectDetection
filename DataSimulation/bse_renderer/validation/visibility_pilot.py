from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import default_slice_runtime, load_config_bundle
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..output.image_writer import save_channel_images, save_mask
from ..renderer.scan_engine import ScanEngine
from .standards import defect_detected_proxy


SURFACE_PILOT_PLANS: dict[str, str] = {
    "keyhole_pore": "keyhole_pore(diameter_mm=0.50,depth_mm=0.35)",
    "intrinsic_pore": "intrinsic_pore(diameter_mm=0.50,depth_mm=0.35)",
    "crack_solidification": (
        "crack_solidification(length_mm=0.80,width_mm=0.08,depth_mm=0.30)"
    ),
    "crack_liquidation": (
        "crack_liquidation(length_mm=0.80,width_mm=0.08,depth_mm=0.22)"
    ),
    "lack_of_fusion": "lack_of_fusion(extent_mm=0.80,depth_mm=0.40,irregularity=0.35)",
    "undercut": "undercut(groove_depth_mm=0.50,toe_angle_deg=34.0)",
}


def _beam_overrides(regime: str) -> dict[str, float]:
    if regime == "low":
        return {
            "accelerating_voltage_kv": 45.0,
            "beam_current_ma": 90.0,
            "focusing_current_norm": 0.35,
            "travel_speed_mm_min": 1600.0,
        }
    if regime == "high":
        return {
            "accelerating_voltage_kv": 120.0,
            "beam_current_ma": 420.0,
            "focusing_current_norm": 0.85,
            "travel_speed_mm_min": 500.0,
        }
    return {
        "accelerating_voltage_kv": 70.0,
        "beam_current_ma": 220.0,
        "focusing_current_norm": 0.60,
        "travel_speed_mm_min": 900.0,
    }


def _quiet_sample(sample):
    return replace(
        sample,
        noise=replace(
            sample.noise,
            poisson_enabled=False,
            thermal_enabled=False,
            contamination_enabled=False,
            contamination_rate=0.0,
            quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
        ),
    )


def _set_regime(sample, regime: str):
    beam = replace(sample.beam, **_beam_overrides(regime))
    beam = replace(
        beam,
        e0_kev=beam.accelerating_voltage_kv,
        power_w=(beam.accelerating_voltage_kv * 1000.0)
        * (beam.beam_current_ma / 1000.0),
    )
    return replace(sample, beam=beam)


def _robust_background_threshold(values: list[float], *, quantile: float = 0.99) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot derive a visibility threshold from an empty background sample.")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_limit = median + 4.0 * 1.4826 * mad
    empirical_limit = float(np.quantile(arr, quantile, method="higher"))
    observed_max = float(np.max(arr))
    threshold = float(max(robust_limit, empirical_limit, 1.05 * observed_max, 1e-8))
    return {
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "robust_limit": float(robust_limit),
        "empirical_quantile": float(empirical_limit),
        "observed_max": observed_max,
        "quantile": float(quantile),
        "sample_count": int(arr.size),
        "observed_false_positive_rate": float(np.mean(arr >= threshold)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _save_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    compo_threshold: float,
    topo_threshold: float,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"status": "skipped", "reason": str(exc), "path": None}

    classes = list(SURFACE_PILOT_PLANS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for ax, channel, threshold in (
        (axes[0], "compo", compo_threshold),
        (axes[1], "topo", topo_threshold),
    ):
        clean = [float(row[f"clean_{channel}_contrast"]) for row in rows]
        grouped = [
            [
                float(row[f"defect_{channel}_contrast"])
                for row in rows
                if row["defect_type"] == defect_type and row["surface_visible"]
            ]
            for defect_type in classes
        ]
        positions = np.arange(1, len(classes) + 1, dtype=float)
        ax.boxplot(grouped, positions=positions, widths=0.55, showfliers=True)
        ax.scatter(
            np.full(len(clean), len(classes) + 1.0),
            clean,
            s=12,
            alpha=0.45,
            color="tab:gray",
            label="clean-location null",
        )
        ax.axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.6g}")
        ax.set_xticks(
            list(positions) + [len(classes) + 1.0],
            [name.replace("_", "\n") for name in classes] + ["clean\nnull"],
        )
        ax.set_ylabel(f"{channel.upper()} local contrast")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(loc="upper right")
    axes[0].set_title("Balanced noiseless top-surface visibility pilot")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def run_visibility_pilot(
    *,
    config_dir: str | Path = "config",
    output_dir: str | Path = "data/output/visibility_pilot",
    repeats_per_class_regime: int = 6,
    image_size: int = 128,
    pixel_size_mm: float = 0.05,
    base_seed: int = 731,
    calibration_path: str | None = None,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | None = None,
    save_images: bool = True,
) -> dict[str, Any]:
    if repeats_per_class_regime < 1:
        raise ValueError("repeats_per_class_regime must be at least 1.")
    cfg = load_config_bundle(config_dir)
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    parsed_plans = {
        name: parse_defect_plan(plan_text)
        for name, plan_text in SURFACE_PILOT_PLANS.items()
    }
    for defects in parsed_plans.values():
        validate_defect_plan(cfg.defect, defects)

    engine = ScanEngine(
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        calibration_path=calibration_path,
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=emission_b_path,
        render_view_mode="top_surface",
    )
    controlled_slice = default_slice_runtime(cfg.slice)
    rows: list[dict[str, Any]] = []
    regimes = ("low", "mid", "high")

    for regime_index, regime in enumerate(regimes):
        for repeat in range(repeats_per_class_regime):
            seed = int(base_seed + 1000 * regime_index + repeat)
            sampled = sample_parameters(
                cfg,
                seed=seed,
                max_defects=0,
                randomize_defect_location=False,
            )
            sampled = _set_regime(_quiet_sample(sampled), regime)
            sampled = replace(sampled, slice=controlled_slice, defects=[])
            baseline = engine.run(sampled)
            baseline_artifacts: dict[str, str] = {}
            if save_images:
                baseline_artifacts = save_channel_images(
                    output_root=out_root,
                    split=f"{regime}/baseline",
                    stem=f"repeat_{repeat:02d}",
                    compo=baseline.channels.compo,
                    topo=baseline.channels.topo,
                    display_mapping=baseline.metadata.get("display_mapping"),
                )

            for defect_type, defects in parsed_plans.items():
                candidate = engine.run(replace(sampled, defects=defects))
                matching = [d for d in candidate.defects if d.defect_type == defect_type]
                visible = bool(matching)
                visible_area_px = int(sum(int(d.mask.sum()) for d in matching))
                artifacts: dict[str, str] = {}
                if save_images:
                    split = f"{regime}/{defect_type}"
                    stem = f"repeat_{repeat:02d}"
                    artifacts = save_channel_images(
                        output_root=out_root,
                        split=split,
                        stem=stem,
                        compo=candidate.channels.compo,
                        topo=candidate.channels.topo,
                        display_mapping=baseline.metadata.get("display_mapping"),
                    )
                    artifacts["mask"] = save_mask(
                        out_root,
                        split,
                        stem,
                        candidate.defect_mask,
                    )

                if visible:
                    defect = matching[0]
                    _, defect_compo, defect_topo = defect_detected_proxy(
                        defect,
                        candidate.channels.compo,
                        candidate.channels.topo,
                        threshold=np.inf,
                    )
                    _, clean_compo, clean_topo = defect_detected_proxy(
                        defect,
                        baseline.channels.compo,
                        baseline.channels.topo,
                        threshold=np.inf,
                    )
                    mask = defect.mask.astype(bool)
                    compo_delta = float(
                        np.mean(np.abs(candidate.channels.compo[mask] - baseline.channels.compo[mask]))
                    )
                    topo_delta = float(
                        np.mean(np.abs(candidate.channels.topo[mask] - baseline.channels.topo[mask]))
                    )
                else:
                    defect_compo = defect_topo = clean_compo = clean_topo = 0.0
                    compo_delta = topo_delta = 0.0

                rows.append(
                    {
                        "regime": regime,
                        "repeat": repeat,
                        "seed": seed,
                        "defect_type": defect_type,
                        "surface_visible": visible,
                        "visible_area_px": visible_area_px,
                        "clean_compo_contrast": float(clean_compo),
                        "clean_topo_contrast": float(clean_topo),
                        "defect_compo_contrast": float(defect_compo),
                        "defect_topo_contrast": float(defect_topo),
                        "paired_compo_delta": compo_delta,
                        "paired_topo_delta": topo_delta,
                        "baseline_compo_path": baseline_artifacts.get("compo_u8", ""),
                        "baseline_topo_path": baseline_artifacts.get("topo_u8", ""),
                        "defect_compo_path": artifacts.get("compo_u8", ""),
                        "defect_topo_path": artifacts.get("topo_u8", ""),
                        "mask_path": artifacts.get("mask", ""),
                    }
                )

    visible_rows = [row for row in rows if row["surface_visible"]]
    compo_stats = _robust_background_threshold(
        [float(row["clean_compo_contrast"]) for row in visible_rows]
    )
    topo_stats = _robust_background_threshold(
        [float(row["clean_topo_contrast"]) for row in visible_rows]
    )
    compo_threshold = float(compo_stats["threshold"])
    topo_threshold = float(topo_stats["threshold"])

    for row in rows:
        row["detected"] = bool(
            row["surface_visible"]
            and (
                float(row["defect_compo_contrast"]) >= compo_threshold
                or float(row["defect_topo_contrast"]) >= topo_threshold
            )
        )

    class_summary: dict[str, dict[str, Any]] = {}
    for defect_type in SURFACE_PILOT_PLANS:
        class_rows = [row for row in rows if row["defect_type"] == defect_type]
        visible_class_rows = [row for row in class_rows if row["surface_visible"]]
        class_summary[defect_type] = {
            "planned_count": len(class_rows),
            "surface_visible_count": len(visible_class_rows),
            "excluded_nonvisible_count": len(class_rows) - len(visible_class_rows),
            "detected_count": int(sum(bool(row["detected"]) for row in visible_class_rows)),
            "detection_rate_visible": (
                float(np.mean([bool(row["detected"]) for row in visible_class_rows]))
                if visible_class_rows
                else 0.0
            ),
            "median_visible_area_px": (
                float(np.median([int(row["visible_area_px"]) for row in visible_class_rows]))
                if visible_class_rows
                else 0.0
            ),
            "median_compo_contrast": (
                float(np.median([float(row["defect_compo_contrast"]) for row in visible_class_rows]))
                if visible_class_rows
                else 0.0
            ),
            "median_topo_contrast": (
                float(np.median([float(row["defect_topo_contrast"]) for row in visible_class_rows]))
                if visible_class_rows
                else 0.0
            ),
        }

    csv_path = _write_csv(out_root / "visibility_pilot.csv", rows)
    plot = _save_plot(
        out_root / "visibility_pilot.png",
        rows,
        compo_threshold=compo_threshold,
        topo_threshold=topo_threshold,
    )
    policy = {
        "task_scope": "surface_visible_defect_identification",
        "render_view_mode": "top_surface",
        "label_rule": "Only defect intersections with at least 6 visible pixels are labelled.",
        "nonvisible_planned_defects": "Excluded from positive labels and recorded in metadata.",
        "noise_enabled": False,
        "classes": list(SURFACE_PILOT_PLANS),
    }
    recommendation = {
        "method": (
            "max(clean-location 99th percentile, clean median + 4 scaled MAD, "
            "1.05 times observed clean maximum)"
        ),
        "compo": compo_stats,
        "topo": topo_stats,
    }
    all_visible_detected = bool(
        visible_rows and all(bool(row["detected"]) for row in visible_rows)
    )
    renderer_strength_decision = {
        "adjustment_applied": False,
        "all_surface_visible_cases_detected": all_visible_detected,
        "reason": (
            "No defect-specific amplification is justified: every surface-visible pilot case "
            "separates from the paired clean-background distribution using the derived "
            "channel-specific thresholds."
            if all_visible_detected
            else "One or more surface-visible cases remain below the derived thresholds; "
            "renderer changes require defect-specific physical or experimental evidence."
        ),
    }
    report = {
        "status": "ok",
        "settings": {
            "config_dir": str(Path(config_dir).resolve()),
            "repeats_per_class_regime": int(repeats_per_class_regime),
            "regimes": list(regimes),
            "image_size": int(image_size),
            "pixel_size_mm": float(pixel_size_mm),
            "base_seed": int(base_seed),
            "noise_enabled": False,
            "render_view_mode": "top_surface",
            "render_model_mode": engine.render_model_mode,
            "emission_b_mode": engine.emission_b_mode,
            "emission_b_path": engine.emission_b_path,
            "randomize_defect_location": False,
            "slice": {
                "voxel_depth_mm": float(controlled_slice.voxel_depth_mm),
                "slice_offset_mm": float(controlled_slice.slice_offset_mm),
                "slice_tilt_deg": float(controlled_slice.slice_tilt_deg),
                "slice_rotation_deg": float(controlled_slice.slice_rotation_deg),
            },
        },
        "counts": {
            "planned_cases": len(rows),
            "surface_visible_cases": len(visible_rows),
            "excluded_nonvisible_cases": len(rows) - len(visible_rows),
        },
        "threshold_recommendation": recommendation,
        "renderer_strength_decision": renderer_strength_decision,
        "surface_visibility_policy": policy,
        "class_summary": class_summary,
        "artifacts": {
            "csv": csv_path,
            "plot": plot,
            "report": str(out_root / "visibility_pilot_report.json"),
            "thresholds": str(out_root / "recommended_visibility_thresholds.json"),
            "policy": str(out_root / "surface_visibility_policy.json"),
        },
    }
    (out_root / "recommended_visibility_thresholds.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    (out_root / "surface_visibility_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    (out_root / "visibility_pilot_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
