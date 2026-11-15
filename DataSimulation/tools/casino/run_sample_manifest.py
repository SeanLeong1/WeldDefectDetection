from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest at {path} must be a JSON object.")
    return raw


def _resolve_adapter(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    env = os.getenv("BSE_CASINO_EXE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).with_name("casino_adapter.cmd")).resolve()


def _build_case(defaults: dict[str, Any], sample: dict[str, Any], index: int) -> dict[str, Any]:
    beam = dict(defaults.get("beam", {}))
    beam.update(sample.get("beam", {}))

    material = dict(defaults.get("material", {}))
    material.update(sample.get("material", {}))

    case = {
        "seed": int(sample.get("seed", defaults.get("seed", 1000 + index))),
        "beam": beam,
        "material": material,
        "geometry": sample.get("geometry", "unknown"),
        "tilt_deg": float(sample.get("tilt_deg", defaults.get("tilt_deg", 0.0))),
        "sample": {
            "id": str(sample["sample_id"]),
            "name": str(sample.get("name", sample["sample_id"])),
            "path": str(Path(sample["sample_path"]).expanduser().resolve()),
        },
        "notes": str(sample.get("notes", "")),
    }
    return case


def run_manifest(
    *,
    manifest_path: str | Path,
    adapter_exe: str | None,
    output_root: str | Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    defaults = manifest.get("defaults", {})
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a non-empty 'samples' list.")

    adapter = _resolve_adapter(adapter_exe)
    output_base = Path(output_root).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{idx}] must be an object.")
        sample_id = str(sample["sample_id"])
        case = _build_case(defaults, sample, idx)
        run_dir = output_base / sample_id
        run_dir.mkdir(parents=True, exist_ok=True)
        case_path = run_dir / "case.json"
        out_path = run_dir / "casino_output.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

        proc = subprocess.run(
            [str(adapter), "--input", str(case_path), "--output", str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "sample_path": case["sample"]["path"],
            "case_path": str(case_path),
            "output_path": str(out_path),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "status": "ok" if proc.returncode == 0 and out_path.exists() else "failed",
        }
        if out_path.exists():
            try:
                row["adapter_output"] = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                row["adapter_output"] = None
        rows.append(row)

    summary = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "adapter_exe": str(adapter),
        "output_root": str(output_base),
        "num_samples": len(rows),
        "num_ok": int(sum(1 for row in rows if row["status"] == "ok")),
        "runs": rows,
    }
    (output_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch of existing CASINO .sam samples through the adapter.")
    parser.add_argument("--manifest", required=True, help="Path to the sample manifest JSON.")
    parser.add_argument(
        "--adapter-exe",
        default=None,
        help="Path to casino_adapter.cmd/.exe. Defaults to BSE_CASINO_EXE or tools/casino/casino_adapter.cmd.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where per-sample case/output folders and summary.json will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_manifest(
        manifest_path=args.manifest,
        adapter_exe=args.adapter_exe,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
