from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..renderer.calibration import (
    RENDER_MODEL_FORMULA_CASINO,
    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
    RENDER_MODEL_FORMULA_ONLY,
)


_CASE_RE = re.compile(r"(?P<calibration>.+)_tilt(?P<tilt>\d+)_(?P<defect>.+)$")


def _parse_case_stem(case_stem: str) -> dict[str, Any]:
    match = _CASE_RE.match(case_stem)
    if match is None:
        return {
            "calibration_label": case_stem,
            "tilt_deg": None,
            "defect_label": None,
        }
    groups = match.groupdict()
    return {
        "calibration_label": groups["calibration"],
        "tilt_deg": int(groups["tilt"]),
        "defect_label": groups["defect"],
    }


def _profile_names(items: list[dict[str, Any]] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        geometry = item.get("geometry")
        if geometry is None:
            continue
        out.append(str(geometry))
    return out


def _load_row(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    case_stem = str(raw.get("case_stem", path.stem.removesuffix("_comparison_summary")))
    parsed = _parse_case_stem(case_stem)
    modes = raw.get("modes", {})
    formula_only = modes.get(RENDER_MODEL_FORMULA_ONLY, {})
    formula_casino = modes.get(RENDER_MODEL_FORMULA_CASINO, {})
    geometry = modes.get(RENDER_MODEL_FORMULA_CASINO_GEOMETRY, {})
    casino_delta = formula_casino.get("delta_vs_formula_only", {})
    geometry_delta = geometry.get("delta_vs_formula_only", {})
    geometry_vs_casino_delta = geometry.get("delta_vs_formula_casino", {})
    formula_meta = formula_only.get("metadata", {}) or {}
    casino_meta = formula_casino.get("metadata", {}) or {}
    geometry_meta = geometry.get("metadata", {}) or {}
    profiles_used = _profile_names(geometry_meta.get("multi_weld_profile_profiles_used"))
    return {
        "case_stem": case_stem,
        "calibration_label": parsed["calibration_label"],
        "tilt_deg": parsed["tilt_deg"],
        "defect_label": parsed["defect_label"],
        "calibration_path": raw.get("calibration_path"),
        "calibration_campaign_source": geometry_meta.get("calibration_campaign_source"),
        "formula_only_profile_count": formula_meta.get("multi_weld_profile_profile_count"),
        "formula_casino_profile_count": casino_meta.get("multi_weld_profile_profile_count"),
        "formula_casino_geometry_profile_count": geometry_meta.get("multi_weld_profile_profile_count"),
        "profiles_used": profiles_used,
        "profiles_used_joined": ";".join(profiles_used),
        "formula_casino_mean_abs_topo_delta": casino_delta.get("mean_abs_topo_delta"),
        "formula_casino_geometry_mean_abs_topo_delta": geometry_delta.get("mean_abs_topo_delta"),
        "formula_casino_geometry_vs_formula_casino_mean_abs_topo_delta": geometry_vs_casino_delta.get("mean_abs_topo_delta"),
        "max_abs_topo_delta": geometry_delta.get("max_abs_topo_delta"),
        "mean_topo_delta": geometry_delta.get("mean_topo_delta"),
        "defect_pixels": geometry_delta.get("defect_pixels"),
        "background_pixels": geometry_delta.get("background_pixels"),
        "defect_fraction": geometry_delta.get("defect_fraction"),
        "defect_mean_abs_topo_delta": geometry_delta.get("defect_mean_abs_topo_delta"),
        "background_mean_abs_topo_delta": geometry_delta.get("background_mean_abs_topo_delta"),
        "defect_to_background_mean_ratio": geometry_delta.get("defect_to_background_mean_ratio"),
        "tilt_yield_scale": geometry_meta.get("tilt_yield_scale"),
        "tilt_blur_scale": geometry_meta.get("tilt_blur_scale"),
        "tilt_depth_scale": geometry_meta.get("tilt_depth_scale"),
        "summary_path": str(path),
    }


def summarize_comparison_cases(
    *,
    comparison_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(comparison_root).resolve()
    if not root.exists():
        raise ValueError(f"Comparison root does not exist: {root}")
    files = sorted(root.rglob("*_comparison_summary.json"))
    if not files:
        raise ValueError(f"No comparison summary files found under: {root}")

    rows = [_load_row(path) for path in files]
    rows.sort(
        key=lambda item: (
            str(item["calibration_label"]),
            -1 if item["tilt_deg"] is None else int(item["tilt_deg"]),
            str(item["defect_label"]),
            str(item["case_stem"]),
        )
    )

    out_dir = root if output_dir is None else Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_summary_table.csv"
    json_path = out_dir / "comparison_summary_table.json"

    fieldnames = [
        "case_stem",
        "calibration_label",
        "tilt_deg",
        "defect_label",
        "calibration_campaign_source",
        "formula_only_profile_count",
        "formula_casino_profile_count",
        "formula_casino_geometry_profile_count",
        "profiles_used_joined",
        "formula_casino_mean_abs_topo_delta",
        "formula_casino_geometry_mean_abs_topo_delta",
        "formula_casino_geometry_vs_formula_casino_mean_abs_topo_delta",
        "max_abs_topo_delta",
        "mean_topo_delta",
        "defect_pixels",
        "background_pixels",
        "defect_fraction",
        "defect_mean_abs_topo_delta",
        "background_mean_abs_topo_delta",
        "defect_to_background_mean_ratio",
        "tilt_yield_scale",
        "tilt_blur_scale",
        "tilt_depth_scale",
        "summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    best_by_ratio = max(
        (row for row in rows if row.get("defect_to_background_mean_ratio") is not None),
        key=lambda row: float(row["defect_to_background_mean_ratio"]),
        default=None,
    )
    best_by_mean_abs = max(
        (row for row in rows if row.get("formula_casino_geometry_mean_abs_topo_delta") is not None),
        key=lambda row: float(row["formula_casino_geometry_mean_abs_topo_delta"]),
        default=None,
    )

    payload = {
        "status": "ok",
        "comparison_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(rows),
        "best_by_ratio_case": None if best_by_ratio is None else best_by_ratio["case_stem"],
        "best_by_mean_abs_topo_case": None if best_by_mean_abs is None else best_by_mean_abs["case_stem"],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "comparison_root": str(root),
        "num_cases": len(rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "best_by_ratio_case": payload["best_by_ratio_case"],
        "best_by_mean_abs_topo_case": payload["best_by_mean_abs_topo_case"],
    }
