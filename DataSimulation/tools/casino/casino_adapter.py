from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import shlex
import subprocess
from typing import Any


ADAPTER_VERSION = "0.2.0-external-runner"


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_case(path: str) -> dict[str, Any]:
    p = pathlib.Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Input case JSON must be an object.")
    return raw


def _defect_severity(case: dict[str, Any]) -> float:
    defects = case.get("defects", [])
    if not isinstance(defects, list) or not defects:
        return 0.0
    sev = 0.0
    for d in defects:
        if not isinstance(d, dict):
            continue
        d_type = str(d.get("type", "")).lower()
        params = d.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if "pore" in d_type:
            sev += float(params.get("diameter_mm", 0.4)) * 0.25
            sev += float(params.get("depth_mm", 0.3)) * 0.15
        elif "crack" in d_type:
            sev += float(params.get("length_mm", 0.8)) * 0.18
            sev += float(params.get("depth_mm", 0.25)) * 0.20
        elif "fusion" in d_type:
            sev += float(params.get("extent_mm", 0.8)) * 0.16
            sev += float(params.get("depth_mm", 0.3)) * 0.20
        elif "undercut" in d_type:
            sev += float(params.get("groove_depth_mm", 0.3)) * 0.30
        else:
            sev += 0.1
    return float(sev)


def _mock_quadrants(case: dict[str, Any]) -> dict[str, float]:
    beam = case.get("beam", {})
    material = case.get("material", {})
    seed = int(case.get("seed", 0))
    if not isinstance(beam, dict):
        beam = {}
    if not isinstance(material, dict):
        material = {}

    e0 = float(beam.get("e0_kev", 60.0))
    z = float(material.get("z", 26.0))
    focus = float(beam.get("focusing_current_norm", 0.6))

    rng = random.Random(seed)
    sev = _defect_severity(case)

    # Approximate trend only for wiring checks; replace with real CASINO mapping.
    eta = 0.025 + 0.0048 * math.sqrt(max(z, 0.0)) + 0.00035 * math.sqrt(max(e0, 0.0))
    eta = _clip(eta - 0.012 * sev, 0.03, 0.85)

    tx = _clip(0.06 * (focus - 0.5) + rng.uniform(-0.01, 0.01), -0.2, 0.2)
    ty = _clip(0.03 * sev + rng.uniform(-0.01, 0.01), -0.2, 0.2)
    base = eta / 4.0

    a = max(1e-8, base + 0.5 * (tx + ty))
    b = max(1e-8, base + 0.5 * (-tx + ty))
    c = max(1e-8, base + 0.5 * (-tx - ty))
    d = max(1e-8, base + 0.5 * (tx - ty))

    # Preserve eta sum after positivity clipping.
    total = a + b + c + d
    if total > 0:
        scale = eta / total
        a *= scale
        b *= scale
        c *= scale
        d *= scale

    return {"A": float(a), "B": float(b), "C": float(c), "D": float(d)}


def _extract_quadrants_from_json(path: pathlib.Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON at {path} must be an object.")
    quad = raw.get("quadrants", raw)
    if not isinstance(quad, dict):
        raise ValueError(f"JSON at {path} missing 'quadrants' mapping.")
    out = {}
    for key in ("A", "B", "C", "D"):
        if key not in quad:
            raise ValueError(f"JSON at {path} missing key '{key}'.")
        out[key] = float(quad[key])
    return out


def _extract_quadrants_from_csv(path: pathlib.Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"CSV at {path} has no rows.")
    rec = rows[-1]
    out = {}
    for key in ("A", "B", "C", "D"):
        if key not in rec:
            raise ValueError(f"CSV at {path} missing column '{key}'.")
        out[key] = float(rec[key])
    return out


def _discover_quadrants(
    output_file: pathlib.Path,
    work_dir: pathlib.Path,
) -> tuple[dict[str, float], str]:
    if output_file.exists():
        return _extract_quadrants_from_json(output_file), str(output_file)

    candidates = [
        work_dir / "casino_output.json",
        work_dir / "quadrants.json",
        work_dir / "bse_quadrants.json",
        work_dir / "casino_quadrants.csv",
        work_dir / "quadrants.csv",
        work_dir / "bse_quadrants.csv",
    ]
    for c in candidates:
        if not c.exists():
            continue
        if c.suffix.lower() == ".json":
            return _extract_quadrants_from_json(c), str(c)
        if c.suffix.lower() == ".csv":
            return _extract_quadrants_from_csv(c), str(c)
    raise FileNotFoundError(
        "No quadrant output found. Expected one of: "
        f"{output_file.name}, casino_output.json, quadrants.json, bse_quadrants.json, "
        "casino_quadrants.csv, quadrants.csv, bse_quadrants.csv"
    )


def _write_output(output_file: pathlib.Path, quadrants: dict[str, float], meta: dict[str, Any]) -> None:
    payload = {
        "quadrants": {
            "A": float(quadrants["A"]),
            "B": float(quadrants["B"]),
            "C": float(quadrants["C"]),
            "D": float(quadrants["D"]),
        },
        "meta": meta,
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_runner_cmd(
    runner_cmd: str,
    *,
    input_path: str,
    output_path: str,
    work_dir: str,
    casino_exe: str,
) -> str:
    return runner_cmd.format(
        input=input_path,
        output=output_path,
        work_dir=work_dir,
        casino_exe=casino_exe,
    )


def _run_external(
    *,
    case_path: str,
    output_path: str,
    casino_exe: str | None,
    runner_script: str | None,
    runner_cmd: str | None,
    timeout_s: float,
) -> None:
    work_dir = pathlib.Path(output_path).resolve().parent
    output_file = pathlib.Path(output_path).resolve()

    exe = casino_exe or os.getenv("BSE_CASINO_WIN_EXE")
    if not exe:
        raise RuntimeError(
            "External mode requires a CASINO executable path via --casino-exe or BSE_CASINO_WIN_EXE."
        )

    runner_script_path = runner_script or os.getenv("BSE_CASINO_EXTERNAL_RUNNER_SCRIPT", "").strip()
    if runner_script_path:
        py_exe = os.getenv("BSE_CASINO_PYTHON", os.getenv("PYTHON", "python"))
        runner_extra = os.getenv("BSE_CASINO_EXTERNAL_RUNNER_ARGS", "").strip()
        cmd = [
            py_exe,
            runner_script_path,
            "--case",
            case_path,
            "--out",
            output_path,
            "--work",
            str(work_dir),
            "--casino",
            exe,
        ]
        if runner_extra:
            cmd.extend(shlex.split(runner_extra, posix=False))
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    else:
        runner = runner_cmd or os.getenv("BSE_CASINO_EXTERNAL_RUNNER", "").strip()
        if runner:
            formatted = _format_runner_cmd(
                runner,
                input_path=case_path,
                output_path=output_path,
                work_dir=str(work_dir),
                casino_exe=exe,
            )
            proc = subprocess.run(
                formatted,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                shell=True,
                timeout=timeout_s,
                check=False,
            )
        else:
            extra = os.getenv("BSE_CASINO_WIN_ARGS", "").strip()
            cmd = [exe]
            if extra:
                cmd.extend(shlex.split(extra, posix=False))
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )

    if proc.returncode != 0:
        raise RuntimeError(
            f"External CASINO run returned non-zero ({proc.returncode}). "
            f"stdout_tail={proc.stdout[-400:]} stderr_tail={proc.stderr[-400:]}"
        )

    quadrants, discovered_from = _discover_quadrants(output_file=output_file, work_dir=work_dir)
    _write_output(
        output_file,
        quadrants,
        meta={
            "backend": "external",
            "adapter_version": ADAPTER_VERSION,
            "runner_script_used": runner_script_path if runner_script_path else None,
            "runner_cmd_used": (runner_cmd or os.getenv("BSE_CASINO_EXTERNAL_RUNNER", "").strip()) or None,
            "casino_exe": exe,
            "discovered_from": discovered_from,
            "stdout_tail": (proc.stdout or "")[-1000:],
            "stderr_tail": (proc.stderr or "")[-1000:],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino_adapter",
        description="CASINO adapter scaffold for DataSimulation validation pipeline.",
    )
    parser.add_argument("--version", action="store_true", help="Print adapter version and exit.")
    parser.add_argument("--input", dest="input_path", default=None, help="Path to case.json")
    parser.add_argument("--output", dest="output_path", default=None, help="Path to casino_output.json")
    parser.add_argument(
        "--mode",
        choices=["mock", "external"],
        default=os.getenv("BSE_CASINO_ADAPTER_MODE", "mock"),
        help="mock = synthetic quadrants; external = run CASINO via executable/runner and parse quadrant outputs.",
    )
    parser.add_argument(
        "--casino-exe",
        default=None,
        help="Path to CASINO executable for external mode (or set BSE_CASINO_WIN_EXE).",
    )
    parser.add_argument(
        "--runner-cmd",
        default=None,
        help=(
            "Optional command template for external mode. "
            "Supports {input}, {output}, {work_dir}, {casino_exe}. "
            "Or set BSE_CASINO_EXTERNAL_RUNNER."
        ),
    )
    parser.add_argument(
        "--runner-script",
        default=None,
        help=(
            "Optional Python runner script path for external mode. "
            "This is easiest on Windows paths with spaces. "
            "Or set BSE_CASINO_EXTERNAL_RUNNER_SCRIPT."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("BSE_CASINO_TIMEOUT_S", "600")),
        help="Timeout for external runner/CASINO process in seconds.",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(f"DataSimulation CASINO adapter {ADAPTER_VERSION}")
        return 0

    if not args.input_path or not args.output_path:
        parser.error("--input and --output are required unless --version is used.")

    case = _load_case(args.input_path)
    output_file = pathlib.Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "mock":
        quads = _mock_quadrants(case)
        payload = {
            "quadrants": quads,
            "meta": {
                "backend": "mock",
                "adapter_version": ADAPTER_VERSION,
                "note": "Scaffold output only. Replace with real CASINO external mode for physics-valid MC comparison.",
            },
        }
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    _run_external(
        case_path=args.input_path,
        output_path=args.output_path,
        casino_exe=args.casino_exe,
        runner_script=args.runner_script,
        runner_cmd=args.runner_cmd,
        timeout_s=max(1.0, float(args.timeout_s)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
