from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config_loader import default_slice_runtime, load_config_bundle
from ..domain_randomisation.dataset_forms import (
    DATASET_CLASSES,
    _disable_noise,
    _regime_for_sample,
    _sample_defect,
    _set_regime,
)
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..models import WeldLayoutRuntime
from ..renderer.emission_b import (
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
)
from ..renderer.scan_engine import ScanEngine


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _panel(array: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.clip((np.asarray(array, dtype=np.float32) - low) / max(high - low, 1e-12), 0.0, 1.0)
    return Image.fromarray((np.sqrt(scaled) * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _save_triptych(path: Path, pooled: np.ndarray, sine: np.ndarray, title: str) -> None:
    both = np.concatenate([pooled.reshape(-1), sine.reshape(-1)])
    low = float(np.quantile(both, 0.01))
    high = float(np.quantile(both, 0.99))
    delta = np.abs(sine - pooled)
    delta_high = max(float(np.quantile(delta, 0.99)), 1e-12)
    panels = [_panel(pooled, low, high), _panel(sine, low, high), _panel(delta, 0.0, delta_high)]
    width, height = panels[0].size
    canvas = Image.new("RGB", (3 * width, height + 26), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), f"{title}: pooled constant | B(alpha) sine | absolute delta", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_emission_b_topo_ablation(
    *,
    config_dir: str | Path,
    b_fit_path: str | Path,
    output_dir: str | Path,
    repeats_per_class: int = 3,
    image_size: int = 128,
    pixel_size_mm: float = 0.05,
    seed: int = 3101,
) -> dict[str, Any]:
    if repeats_per_class < 1:
        raise ValueError("repeats_per_class must be positive.")
    cfg = load_config_bundle(config_dir)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fit_path = Path(b_fit_path).expanduser().resolve()
    fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
    engine_kwargs = {
        "image_size": image_size,
        "pixel_size_mm": pixel_size_mm,
        "render_model_mode": "formula_only",
        "emission_b_path": str(fit_path),
        "render_view_mode": "top_surface",
        "weld_layout": WeldLayoutRuntime(mode="local_patch"),
    }
    pooled_engine = ScanEngine(allow_unvalidated_emission_b=True, 
        **engine_kwargs, emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT
    )
    sine_engine = ScanEngine(allow_unvalidated_emission_b=True, **engine_kwargs, emission_b_mode=EMISSION_B_MODE_SIN_FIT)
    controlled_slice = default_slice_runtime(cfg.slice)
    rows: list[dict[str, Any]] = []

    for class_index, defect_type in enumerate(DATASET_CLASSES):
        for repetition in range(repeats_per_class):
            sample_seed = seed + class_index * 10000 + repetition
            rng = np.random.default_rng(sample_seed)
            regime = _regime_for_sample(repetition, class_index)
            sampled = sample_parameters(
                cfg, seed=sample_seed, max_defects=0, randomize_defect_location=True
            )
            sampled = replace(
                _set_regime(_disable_noise(sampled), regime),
                slice=controlled_slice,
                defects=_sample_defect(defect_type, rng),
            )
            pooled = pooled_engine.run(sampled)
            sine = sine_engine.run(sampled)
            delta = np.abs(sine.channels.topo - pooled.channels.topo)
            pooled_scale = max(
                float(np.quantile(np.abs(pooled.channels.topo), 0.99)), 1e-12
            )
            mask = np.asarray(sine.defect_mask, dtype=bool)
            row = {
                "sample_id": f"{defect_type}-{repetition:03d}",
                "defect_type": defect_type,
                "beam_regime": regime,
                "seed": sample_seed,
                "pooled_b_mean": float(pooled.metadata["emission_b"]["b_stats"]["mean"]),
                "sine_b_min": float(sine.metadata["emission_b"]["b_stats"]["min"]),
                "sine_b_mean": float(sine.metadata["emission_b"]["b_stats"]["mean"]),
                "sine_b_max": float(sine.metadata["emission_b"]["b_stats"]["max"]),
                "topo_mean_abs_delta": float(delta.mean()),
                "topo_p99_abs_delta": float(np.quantile(delta, 0.99)),
                "topo_normalized_mean_abs_delta": float(delta.mean() / pooled_scale),
                "defect_roi_mean_abs_delta": float(delta[mask].mean()) if np.any(mask) else 0.0,
            }
            rows.append(row)
            if repetition == 0:
                _save_triptych(
                    root / "triptychs" / f"{defect_type}.png",
                    pooled.channels.topo,
                    sine.channels.topo,
                    defect_type,
                )

    csv_path = root / "paired_topo_ablation.csv"
    _write_csv(csv_path, rows)
    summary = {
        "status": "ok",
        "case_count": len(rows),
        "classes": list(DATASET_CLASSES),
        "repeats_per_class": repeats_per_class,
        "paired_seed_policy": "identical sampled parameters for both B modes",
        "pooled_mode": EMISSION_B_MODE_POOLED_CONSTANT_FIT,
        "candidate_mode": EMISSION_B_MODE_SIN_FIT,
        "fit_recommended_model": fit_payload.get("recommended_model"),
        "selected_mode": (
            EMISSION_B_MODE_SIN_FIT
            if fit_payload.get("recommended_model") == "b_alpha_sin"
            else None
        ),
        "mean_normalized_topo_delta": float(
            np.mean([row["topo_normalized_mean_abs_delta"] for row in rows])
        ),
        "mean_defect_roi_abs_delta": float(
            np.mean([row["defect_roi_mean_abs_delta"] for row in rows if row["defect_type"] != "clean"])
        ),
        "csv": str(csv_path),
        "triptych_dir": str(root / "triptychs"),
        "decision_basis": (
            "The fit bundle's held-out likelihood selects b_alpha_sin; paired TOPO deltas quantify "
            "the downstream image impact but are not themselves a CASINO ground-truth score."
        ),
    }
    (root / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
