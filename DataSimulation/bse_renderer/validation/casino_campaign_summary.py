from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _group_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in fields)


def summarize_casino_benchmark_campaign(
    benchmark_summary_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    group_by: tuple[str, ...] = ("geometry",),
    usable_only: bool = True,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_summary_path).resolve()
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    records = list(raw.get("records", []))
    if usable_only:
        records = [record for record in records if str(record.get("quality_flag", "usable")) == "usable"]
    if not records:
        raise ValueError("No benchmark records available for summary.")

    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (benchmark_path.parent / f"{benchmark_path.stem}_summary").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    baseline_fields = ("material", "energy_kev", "tilt_deg", "beam_diameter_nm", "electrons")
    for record in records:
        if str(record.get("geometry", "")) != "flatbulk":
            continue
        eta = record.get("eta_bse")
        if eta is None:
            continue
        baseline_groups[_group_key(record, baseline_fields)].append(float(eta))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record, group_by)].append(record)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: tuple("" if value is None else str(value) for value in item)):
        group_records = grouped[key]
        eta_values = [float(record["eta_bse"]) for record in group_records if record.get("eta_bse") is not None]
        mean_bse_energy = [
            float(record["mean_bse_energy_kev"])
            for record in group_records
            if record.get("mean_bse_energy_kev") is not None
        ]
        r90_values = [float(record["r90_nm"]) for record in group_records if record.get("r90_nm") is not None]
        z90_values = [float(record["z90_nm"]) for record in group_records if record.get("z90_nm") is not None]

        baseline_key = _group_key(group_records[0], baseline_fields)
        baseline_eta = _safe_mean(baseline_groups.get(baseline_key, []))
        mean_eta = _safe_mean(eta_values)
        row: dict[str, Any] = {field: value for field, value in zip(group_by, key)}
        row.update(
            {
                "count": len(group_records),
                "sim_ids": [record.get("sim_id") for record in group_records],
                "mean_eta_bse": mean_eta,
                "std_eta_bse": _safe_std(eta_values),
                "min_eta_bse": min(eta_values) if eta_values else None,
                "max_eta_bse": max(eta_values) if eta_values else None,
                "flatbulk_baseline_eta_bse": baseline_eta,
                "delta_eta_bse_vs_flatbulk": (
                    None if mean_eta is None or baseline_eta is None else float(mean_eta - baseline_eta)
                ),
                "mean_bse_energy_kev": _safe_mean(mean_bse_energy),
                "mean_r90_nm": _safe_mean(r90_values),
                "mean_z90_nm": _safe_mean(z90_values),
                "quality_flags": sorted({str(record.get("quality_flag", "")) for record in group_records}),
            }
        )
        summary_rows.append(row)

    csv_fields = list(group_by) + [
        "count",
        "mean_eta_bse",
        "std_eta_bse",
        "min_eta_bse",
        "max_eta_bse",
        "flatbulk_baseline_eta_bse",
        "delta_eta_bse_vs_flatbulk",
        "mean_bse_energy_kev",
        "mean_r90_nm",
        "mean_z90_nm",
        "quality_flags",
        "sim_ids",
    ]
    csv_path = output_root / "campaign_group_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    **row,
                    "quality_flags": ",".join(row["quality_flags"]),
                    "sim_ids": ",".join(str(value) for value in row["sim_ids"] if value is not None),
                }
            )

    json_path = output_root / "campaign_group_summary.json"
    payload = {
        "benchmark_summary_path": str(benchmark_path),
        "usable_only": usable_only,
        "group_by": list(group_by),
        "num_input_records": len(records),
        "num_groups": len(summary_rows),
        "groups": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "benchmark_summary_path": str(benchmark_path),
        "output_dir": str(output_root),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "num_input_records": len(records),
        "num_groups": len(summary_rows),
        "group_by": list(group_by),
        "usable_only": usable_only,
    }
