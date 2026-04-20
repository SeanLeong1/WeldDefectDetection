from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_executable(path_or_name: str | None) -> str | None:
    if not path_or_name:
        return None
    p = Path(path_or_name)
    if p.exists():
        return str(p.resolve())
    return shutil.which(path_or_name)


def run_casino_probe(casino_exe: str | None = None) -> dict[str, Any]:
    """
    Probe CASINO executable availability.

    Expected environment fallback:
      - BSE_CASINO_EXE
    """
    exe = _resolve_executable(casino_exe or os.getenv("BSE_CASINO_EXE"))
    if exe is None:
        return {
            "status": "skipped",
            "available": False,
            "casino_exe": None,
            "message": (
                "CASINO executable not found. "
                "Set --casino-exe or BSE_CASINO_EXE to a runnable CASINO adapter."
            ),
        }
    try:
        # Some CASINO wrappers return non-zero for --version; do not hard-fail.
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        version = (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return {
            "status": "partial",
            "available": True,
            "casino_exe": exe,
            "message": f"CASINO executable found, version probe failed: {exc}",
        }

    return {
        "status": "ok",
        "available": True,
        "casino_exe": exe,
        "message": "CASINO executable found",
        "version_text": version,
    }


def _parse_quadrants_from_json(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    quad = raw.get("quadrants", {})
    return {
        "A": float(quad.get("A", 0.0)),
        "B": float(quad.get("B", 0.0)),
        "C": float(quad.get("C", 0.0)),
        "D": float(quad.get("D", 0.0)),
    }


def _parse_quadrants_from_csv(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise ValueError("casino_quadrants.csv has no rows")
    rec = rows[-1]
    return {
        "A": float(rec.get("A", "0") or 0.0),
        "B": float(rec.get("B", "0") or 0.0),
        "C": float(rec.get("C", "0") or 0.0),
        "D": float(rec.get("D", "0") or 0.0),
    }


def parse_casino_case_output(work_dir: str | Path) -> dict[str, Any]:
    wd = Path(work_dir)
    json_path = wd / "casino_output.json"
    csv_path = wd / "casino_quadrants.csv"

    parser = None
    quadrants: dict[str, float] | None = None
    if json_path.exists():
        quadrants = _parse_quadrants_from_json(json_path)
        parser = "json"
    elif csv_path.exists():
        quadrants = _parse_quadrants_from_csv(csv_path)
        parser = "csv"
    else:
        raise FileNotFoundError(
            "Expected casino_output.json or casino_quadrants.csv in run directory."
        )

    a = quadrants["A"]
    b = quadrants["B"]
    c = quadrants["C"]
    d = quadrants["D"]
    compo = float(a + b + c + d)
    tx = float((a + d) - (b + c))
    ty = float((a + b) - (c + d))
    topo = float(np.sqrt(tx * tx + ty * ty))
    rx = float((a - c) / max(a + c, 1e-8))
    return {
        "quadrants": quadrants,
        "derived": {
            "compo": compo,
            "topo": topo,
            "tx": tx,
            "ty": ty,
            "rx": rx,
        },
        "parser": parser,
    }


def run_casino_case(
    *,
    case: dict[str, Any],
    casino_exe: str | None = None,
    run_dir: str | Path | None = None,
    timeout_s: float = 240.0,
) -> dict[str, Any]:
    """
    Execute one CASINO case through a command-line adapter and parse quadrants.

    Contract expected from adapter executable:
      casino_exe --input <case.json> --output <casino_output.json>
    Output may be JSON or CSV (see parse_casino_case_output).
    """
    probe = run_casino_probe(casino_exe=casino_exe)
    if not probe.get("available"):
        return {
            "status": "skipped",
            "message": probe.get("message", "CASINO unavailable"),
            "probe": probe,
        }

    temp_ctx = None
    if run_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="casino_case_")
        work_dir = Path(temp_ctx.name)
    else:
        work_dir = Path(run_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        case_path = work_dir / "case.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        output_path = work_dir / "casino_output.json"
        t0 = time.time()
        proc = subprocess.run(
            [str(probe["casino_exe"]), "--input", str(case_path), "--output", str(output_path)],
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_s),
            check=False,
            cwd=str(work_dir),
        )
        runtime_s = float(time.time() - t0)
        if proc.returncode != 0:
            return {
                "status": "failed",
                "message": (
                    "CASINO adapter execution failed. "
                    "Provide a wrapper that accepts --input/--output and writes "
                    "casino_output.json (or casino_quadrants.csv)."
                ),
                "probe": probe,
                "work_dir": str(work_dir),
                "stdout_tail": (proc.stdout or "")[-1500:],
                "stderr_tail": (proc.stderr or "")[-1500:],
                "runtime_s": runtime_s,
            }

        parsed = parse_casino_case_output(work_dir)
        return {
            "status": "ok",
            "message": "CASINO case parsed successfully",
            "probe": probe,
            "work_dir": str(work_dir),
            "runtime_s": runtime_s,
            "quadrants": parsed["quadrants"],
            "derived": parsed["derived"],
            "parser": parsed["parser"],
            "stdout_tail": (proc.stdout or "")[-1000:],
            "stderr_tail": (proc.stderr or "")[-1000:],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"CASINO case failed: {exc}",
            "probe": probe,
            "work_dir": str(work_dir),
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
