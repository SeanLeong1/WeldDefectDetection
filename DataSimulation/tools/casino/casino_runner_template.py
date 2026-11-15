from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import re
import subprocess
import tempfile
from typing import Any


RUNNER_VERSION = "0.3.0-ahk"


def _load_case(path: str) -> dict[str, Any]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("case must be a JSON object")
    return raw


def _emit_mock_from_case(case: dict[str, Any]) -> dict[str, float]:
    seed = int(case.get("seed", 0))
    beam = case.get("beam", {}) if isinstance(case.get("beam", {}), dict) else {}
    z = float(case.get("material", {}).get("z", 26.0)) if isinstance(case.get("material", {}), dict) else 26.0
    e0 = float(beam.get("e0_kev", 60.0))
    rng = random.Random(seed)
    eta = max(0.05, min(0.8, 0.02 + 0.004 * (z**0.5) + 0.0004 * (e0**0.5)))
    tx = rng.uniform(-0.01, 0.01)
    ty = rng.uniform(-0.01, 0.01)
    base = eta / 4.0
    return {
        "A": float(base + 0.5 * (tx + ty)),
        "B": float(base + 0.5 * (-tx + ty)),
        "C": float(base + 0.5 * (-tx - ty)),
        "D": float(base + 0.5 * (tx - ty)),
    }


def _default_profile_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("casino_ahk_profile.example.json")


def _default_ahk_candidates() -> list[pathlib.Path]:
    return [
        pathlib.Path(r"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"),
        pathlib.Path(r"C:\Program Files\AutoHotkey\AutoHotkey64.exe"),
        pathlib.Path(r"C:\Program Files\AutoHotkey\AutoHotkey.exe"),
    ]


def _resolve_ahk_exe(user_value: str | None) -> pathlib.Path:
    if user_value:
        p = pathlib.Path(user_value).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"AutoHotkey executable not found: {p}")
    env = os.getenv("BSE_AUTOHOTKEY_EXE", "").strip()
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"BSE_AUTOHOTKEY_EXE points to missing file: {p}")
    for cand in _default_ahk_candidates():
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "AutoHotkey executable not found. Install AutoHotkey and set BSE_AUTOHOTKEY_EXE "
        "or pass --ahk-exe."
    )


def _load_profile(path: str | None) -> dict[str, Any]:
    p = pathlib.Path(path).expanduser().resolve() if path else _default_profile_path().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"CASINO AHK profile not found: {p}. Copy and customize casino_ahk_profile.example.json."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("profile JSON must be an object")
    return raw


def _lookup_path(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


_SUBST_RE = re.compile(r"\$\{([^}]+)\}")


def _flatten_case_vars(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            out.update(_flatten_case_vars(value, child_prefix))
        return out
    if isinstance(data, list):
        for idx, value in enumerate(data):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(_flatten_case_vars(value, child_prefix))
        return out
    if prefix:
        out[prefix] = data
    return out


def _substitute(text: str, variables: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            raise KeyError(key)
        return str(variables[key])

    return _SUBST_RE.sub(repl, text)


def _ahk_escape_text(text: str) -> str:
    out = text.replace("`", "``")
    out = out.replace("%", "`%")
    out = out.replace(",", "`,")
    out = out.replace(";", "`;")
    out = out.replace("\r\n", "`n").replace("\n", "`n")
    return out


def _render_ahk_script(
    profile: dict[str, Any],
    *,
    case: dict[str, Any],
    casino_exe: pathlib.Path,
    work_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> str:
    title = str(profile.get("window_title", "CASINO"))
    launch = bool(profile.get("launch_casino", True))
    launch_wait_ms = int(profile.get("launch_wait_ms", 1500))
    default_timeout_s = float(profile.get("default_timeout_s", 60.0))
    actions = profile.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise ValueError("profile.actions must be a non-empty list")

    vars_map = _flatten_case_vars(case)
    vars_map.update(
        {
            "seed": case.get("seed", 0),
            "beam.e0_kev": _lookup_path(case, "beam.e0_kev"),
            "beam.beam_current_ma": _lookup_path(case, "beam.beam_current_ma"),
            "beam.focusing_current_norm": _lookup_path(case, "beam.focusing_current_norm"),
            "beam.travel_speed_mm_min": _lookup_path(case, "beam.travel_speed_mm_min"),
            "material.name": _lookup_path(case, "material.name"),
            "material.z": _lookup_path(case, "material.z"),
            "material.rho": _lookup_path(case, "material.rho"),
            "work_dir": str(work_dir),
            "output_path": str(out_path),
            "casino_exe": str(casino_exe),
        }
    )

    lines = [
        "#NoEnv",
        "#SingleInstance Force",
        "SendMode Input",
        "SetTitleMatchMode, 2",
        "CoordMode, Mouse, Screen",
        "SetWorkingDir, %A_ScriptDir%",
    ]
    if launch:
        lines.append(f'Run, "{casino_exe}"')
        lines.append(f"Sleep, {max(0, launch_wait_ms)}")

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{idx}] must be an object")
        a_type = str(action.get("type", "")).strip().lower()
        if a_type == "sleep_ms":
            lines.append(f"Sleep, {int(action.get('ms', 250))}")
            continue
        if a_type == "win_wait_active":
            a_title = _substitute(str(action.get("title", title)), vars_map)
            timeout = float(action.get("timeout_s", default_timeout_s))
            lines.append(f"WinWaitActive, {a_title},, {timeout:.3f}")
            lines.append("if ErrorLevel")
            lines.append("{")
            lines.append("  ExitApp, 10")
            lines.append("}")
            continue
        if a_type == "win_activate":
            a_title = _substitute(str(action.get("title", title)), vars_map)
            lines.append(f"WinActivate, {a_title}")
            lines.append("Sleep, 120")
            continue
        if a_type == "click":
            x = int(action["x"])
            y = int(action["y"])
            clicks = int(action.get("clicks", 1))
            button = str(action.get("button", "Left"))
            lines.append(f"Click, {x}, {y}, {clicks}, {button}")
            delay_ms = int(action.get("post_delay_ms", 120))
            if delay_ms > 0:
                lines.append(f"Sleep, {delay_ms}")
            continue
        if a_type == "send_text":
            raw_text = _substitute(str(action.get("text", "")), vars_map)
            lines.append(f"SendInput, {{Text}}{_ahk_escape_text(raw_text)}")
            delay_ms = int(action.get("post_delay_ms", 80))
            if delay_ms > 0:
                lines.append(f"Sleep, {delay_ms}")
            continue
        if a_type == "send_key":
            keys = _substitute(str(action.get("keys", "{Enter}")), vars_map)
            lines.append(f"SendInput, {keys}")
            delay_ms = int(action.get("post_delay_ms", 80))
            if delay_ms > 0:
                lines.append(f"Sleep, {delay_ms}")
            continue
        raise ValueError(f"Unsupported action type: {a_type}")

    lines.append("ExitApp, 0")
    return "\n".join(lines) + "\n"


def _run_ahk_script(ahk_exe: pathlib.Path, script_path: pathlib.Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ahk_exe), str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1.0, float(timeout_s)),
    )


def _resolve_path(base_dir: pathlib.Path, maybe_relative: str) -> pathlib.Path:
    p = pathlib.Path(maybe_relative)
    if p.is_absolute():
        return p
    return base_dir / p


def _parse_quadrants_from_json(path: pathlib.Path, keys: dict[str, str] | None = None) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = keys or {"A": "quadrants.A", "B": "quadrants.B", "C": "quadrants.C", "D": "quadrants.D"}
    return {k: float(_lookup_path(raw, v)) for k, v in keys.items()}


def _parse_quadrants_from_csv(path: pathlib.Path, columns: dict[str, str] | None = None) -> dict[str, float]:
    columns = columns or {"A": "A", "B": "B", "C": "C", "D": "D"}
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"No rows in {path}")
    rec = rows[-1]
    return {k: float(rec[v]) for k, v in columns.items()}


def _parse_quadrants_from_regex(path: pathlib.Path, patterns: dict[str, str]) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for key in ("A", "B", "C", "D"):
        pat = patterns.get(key)
        if not pat:
            raise ValueError(f"Missing regex pattern for {key}")
        m = re.search(pat, text, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Pattern for {key} not found in {path}")
        out[key] = float(m.group(1))
    return out


def _parse_quadrants_from_numbers(path: pathlib.Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) < 4:
        raise ValueError(f"Could not find at least 4 numbers in {path}")
    a, b, c, d = [float(x) for x in nums[-4:]]
    return {"A": a, "B": b, "C": c, "D": d}


def _load_quadrants_from_profile(profile: dict[str, Any], work_dir: pathlib.Path, out_path: pathlib.Path) -> tuple[dict[str, float], str]:
    out_cfg = profile.get("output", {})
    if not isinstance(out_cfg, dict):
        out_cfg = {}
    fmt = str(out_cfg.get("format", "json")).lower()
    path_raw = str(out_cfg.get("path", str(out_path)))
    parsed_path = _resolve_path(work_dir, path_raw)
    if not parsed_path.exists():
        raise FileNotFoundError(f"Configured output file not found: {parsed_path}")

    if fmt == "json":
        return _parse_quadrants_from_json(parsed_path, out_cfg.get("keys")), str(parsed_path)
    if fmt == "csv_columns":
        return _parse_quadrants_from_csv(parsed_path, out_cfg.get("columns")), str(parsed_path)
    if fmt == "regex":
        patterns = out_cfg.get("patterns")
        if not isinstance(patterns, dict):
            raise ValueError("output.patterns must be a mapping for regex format")
        return _parse_quadrants_from_regex(parsed_path, patterns), str(parsed_path)
    if fmt == "last4numbers":
        return _parse_quadrants_from_numbers(parsed_path), str(parsed_path)
    raise ValueError(f"Unsupported output.format: {fmt}")


def _write_adapter_output(out: pathlib.Path, quadrants: dict[str, float], meta: dict[str, Any]) -> None:
    out.write_text(
        json.dumps(
            {
                "quadrants": {
                    "A": float(quadrants["A"]),
                    "B": float(quadrants["B"]),
                    "C": float(quadrants["C"]),
                    "D": float(quadrants["D"]),
                },
                "meta": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AutoHotkey-driven CASINO runner used by tools/casino/casino_adapter.py in external mode."
        )
    )
    parser.add_argument("--case", required=True, help="Path to case JSON ({input})")
    parser.add_argument("--out", required=True, help="Path to output JSON ({output})")
    parser.add_argument("--work", required=True, help="Working directory ({work_dir})")
    parser.add_argument("--casino", required=True, help="CASINO executable path ({casino_exe})")
    parser.add_argument("--profile", default=None, help="Path to CASINO AHK profile JSON.")
    parser.add_argument("--ahk-exe", default=None, help="Path to AutoHotkey executable.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Timeout for AHK process.")
    parser.add_argument("--keep-script", action="store_true", help="Keep generated .ahk script in work dir.")
    parser.add_argument(
        "--emit-mock",
        action="store_true",
        help="Write mock quadrants (for external-mode plumbing checks).",
    )
    args = parser.parse_args(argv)

    case = _load_case(args.case)
    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = pathlib.Path(args.work).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.emit_mock:
        _write_adapter_output(
            out,
            _emit_mock_from_case(case),
            {
                "runner": "ahk",
                "runner_version": RUNNER_VERSION,
                "mode": "emit-mock",
                "note": "Mock mode only. Remove --emit-mock for real CASINO automation.",
            },
        )
        return 0

    profile = _load_profile(args.profile)
    if bool(profile.get("template_profile", False)):
        raise SystemExit(
            "Profile is still marked as template_profile=true. "
            "Copy and customize the profile before running real automation."
        )

    ahk_exe = _resolve_ahk_exe(args.ahk_exe)
    casino_exe = pathlib.Path(args.casino).resolve()
    if not casino_exe.exists():
        raise SystemExit(f"CASINO executable not found: {casino_exe}")

    script_text = _render_ahk_script(
        profile,
        case=case,
        casino_exe=casino_exe,
        work_dir=work_dir,
        out_path=out,
    )
    script_path = work_dir / "casino_auto_case.ahk"
    if not args.keep_script:
        temp = tempfile.NamedTemporaryFile(prefix="casino_auto_", suffix=".ahk", dir=str(work_dir), delete=False)
        temp.close()
        script_path = pathlib.Path(temp.name)
    script_path.write_text(script_text, encoding="utf-8")

    proc = _run_ahk_script(ahk_exe, script_path, timeout_s=args.timeout_s)
    if proc.returncode != 0:
        raise SystemExit(
            f"AutoHotkey automation failed (code {proc.returncode}). "
            f"stdout_tail={proc.stdout[-500:]} stderr_tail={proc.stderr[-500:]}"
        )

    quadrants, parsed_from = _load_quadrants_from_profile(profile, work_dir=work_dir, out_path=out)
    _write_adapter_output(
        out,
        quadrants,
        {
            "runner": "ahk",
            "runner_version": RUNNER_VERSION,
            "mode": "external",
            "ahk_exe": str(ahk_exe),
            "casino_exe": str(casino_exe),
            "profile_path": str(pathlib.Path(args.profile).resolve()) if args.profile else str(_default_profile_path().resolve()),
            "generated_script": str(script_path),
            "parsed_from": parsed_from,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
