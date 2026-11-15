from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def create_run_dir(root: str | Path, prefix: str) -> Path:
    base = ensure_dir(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path, filename: str = "config_snapshot.yaml") -> Path:
    out_dir = ensure_dir(output_dir)
    snapshot_path = out_dir / filename
    snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return snapshot_path
