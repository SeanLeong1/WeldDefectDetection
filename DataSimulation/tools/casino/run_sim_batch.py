from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _discover_sim_entries(sim_root: Path, cas_root: Path | None = None) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for sim_path in sorted(sim_root.rglob("*.sim")):
        if cas_root is None:
            cas_path = sim_path.with_suffix(".cas")
        else:
            cas_path = cas_root / sim_path.relative_to(sim_root).with_suffix(".cas")
            cas_path.parent.mkdir(parents=True, exist_ok=True)
        entries.append(
            {
                "sim_path": str(sim_path.resolve()),
                "cas_path": str(cas_path.resolve()),
            }
        )
    return entries


def _load_manifest(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        entries = raw.get("entries", [])
    else:
        entries = raw
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a list or an object with an 'entries' list.")
    out: list[dict[str, str]] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {idx} must be an object.")
        if "sim_path" not in item or "cas_path" not in item:
            raise ValueError(f"Manifest entry {idx} must contain sim_path and cas_path.")
        out.append(
            {
                "sim_path": str(Path(item["sim_path"]).expanduser().resolve()),
                "cas_path": str(Path(item["cas_path"]).expanduser().resolve()),
            }
        )
    return out


def _write_load_sim_script(entries: list[dict[str, str]], script_path: Path) -> None:
    lines: list[str] = []
    for entry in entries:
        # CASINO v3 sample scripts use normal Windows paths with single
        # backslashes, not escaped backslashes.
        sim_path = entry["sim_path"]
        cas_path = entry["cas_path"]
        lines.append(f'LOAD "{sim_path}";')
        lines.append(f'SIM "{cas_path}";')
        lines.append("")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_per_entry_scripts(entries: list[dict[str, str]], script_dir: Path) -> list[dict[str, str]]:
    script_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, str]] = []
    for idx, entry in enumerate(entries, start=1):
        sim_name = Path(entry["sim_path"]).stem
        entry_script = script_dir / f"{idx:02d}_{sim_name}.txt"
        _write_load_sim_script([entry], entry_script)
        out.append(
            {
                "name": sim_name,
                "script_path": str(entry_script.resolve()),
                "sim_path": entry["sim_path"],
                "cas_path": entry["cas_path"],
            }
        )
    return out


def _write_console_runner_ps1(
    *,
    console_exe: Path,
    per_entry_scripts: list[dict[str, str]],
    runner_path: Path,
) -> None:
    lines: list[str] = [
        "$ErrorActionPreference = 'Stop'",
        f"$consoleExe = {json.dumps(str(console_exe.resolve()))}",
        "",
        "$runs = @(",
    ]
    for item in per_entry_scripts:
        lines.append("  @{")
        lines.append(f"    Name = {json.dumps(item['name'])}")
        lines.append(f"    Script = {json.dumps(item['script_path'])}")
        lines.append(f"    Sim = {json.dumps(item['sim_path'])}")
        lines.append(f"    Cas = {json.dumps(item['cas_path'])}")
        lines.append("  }")
    lines.append(")")
    lines.append("")
    lines.append("foreach ($run in $runs) {")
    lines.append('  Write-Host ("Running {0}" -f $run.Name)')
    lines.append("  & $consoleExe $run.Script")
    lines.append("  if ($LASTEXITCODE -ne 0) {")
    lines.append('    throw ("CASINO console runner failed for {0} with exit code {1}" -f $run.Name, $LASTEXITCODE)')
    lines.append("  }")
    lines.append('  if (-not (Test-Path -LiteralPath $run.Cas)) {')
    lines.append('    Write-Warning ("No .cas file detected after run: {0}" -f $run.Cas)')
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("Write-Host 'All CASINO runs finished.'")
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_casino_exe(user_value: str | None) -> Path | None:
    if user_value:
        return Path(user_value).expanduser().resolve()
    env = os.getenv("BSE_CASINO_WIN_EXE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return None


def _launch_casino(
    *,
    casino_exe: Path,
    script_path: Path,
    launch_template: str | None,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    if launch_template:
        cmd = launch_template.format(
            casino_exe=str(casino_exe),
            script=str(script_path),
        )
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            shell=True,
            check=False,
        )
    return subprocess.run(
        [str(casino_exe), str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def run_batch(
    *,
    manifest_path: str | None,
    sim_root: str | None,
    cas_root: str | None,
    output_script: str | Path,
    output_summary: str | Path | None,
    output_runner_ps1: str | Path | None,
    casino_exe: str | None,
    launch_template: str | None,
    run_now: bool,
) -> dict[str, Any]:
    if bool(manifest_path) == bool(sim_root):
        raise ValueError("Provide exactly one of --manifest or --sim-root.")

    if manifest_path:
        entries = _load_manifest(Path(manifest_path).resolve())
    else:
        root = Path(sim_root).expanduser().resolve()
        cas_root_path = Path(cas_root).expanduser().resolve() if cas_root else None
        entries = _discover_sim_entries(root, cas_root_path)

    if not entries:
        raise ValueError("No .sim entries found.")

    script_path = Path(output_script).expanduser().resolve()
    _write_load_sim_script(entries, script_path)

    summary: dict[str, Any] = {
        "script_path": str(script_path),
        "num_entries": len(entries),
        "entries": entries,
        "run_attempted": False,
        "run_status": "not_run",
    }

    if output_runner_ps1:
        exe = _resolve_casino_exe(casino_exe)
        if exe is None:
            raise ValueError("CASINO executable not provided. Pass --casino-exe or set BSE_CASINO_WIN_EXE.")
        runner_path = Path(output_runner_ps1).expanduser().resolve()
        per_entry_dir = runner_path.with_suffix("")
        per_entry_scripts = _write_per_entry_scripts(entries, per_entry_dir)
        _write_console_runner_ps1(
            console_exe=exe,
            per_entry_scripts=per_entry_scripts,
            runner_path=runner_path,
        )
        summary["runner_ps1_path"] = str(runner_path)
        summary["per_entry_script_dir"] = str(per_entry_dir)
        summary["per_entry_scripts"] = per_entry_scripts

    if run_now:
        exe = _resolve_casino_exe(casino_exe)
        if exe is None:
            raise ValueError("CASINO executable not provided. Pass --casino-exe or set BSE_CASINO_WIN_EXE.")
        proc = _launch_casino(
            casino_exe=exe,
            script_path=script_path,
            launch_template=launch_template,
            cwd=script_path.parent,
        )
        summary.update(
            {
                "run_attempted": True,
                "run_status": "ok" if proc.returncode == 0 else "failed",
                "casino_exe": str(exe),
                "launch_template": launch_template,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )

    if output_summary:
        summary_path = Path(output_summary).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a direct CASINO LOAD/SIM batch script from .sim files and optionally launch CASINO."
    )
    parser.add_argument("--manifest", default=None, help="JSON manifest with entries containing sim_path and cas_path.")
    parser.add_argument("--sim-root", default=None, help="Root folder to recursively discover .sim files.")
    parser.add_argument("--cas-root", default=None, help="Optional separate output root for generated .cas paths.")
    parser.add_argument("--output-script", required=True, help="Path to write the LOAD/SIM command script.")
    parser.add_argument("--output-summary", default=None, help="Optional path to write a JSON summary.")
    parser.add_argument(
        "--output-runner-ps1",
        default=None,
        help=(
            "Optional path to write a PowerShell runner for CASINO v3 console mode. "
            "When set, per-entry one-case script files are also generated in a sibling folder."
        ),
    )
    parser.add_argument("--casino-exe", default=None, help="Path to wincasino2_64.exe. Defaults to BSE_CASINO_WIN_EXE.")
    parser.add_argument(
        "--launch-template",
        default=None,
        help=(
            "Optional command template used when --run-now is set. "
            "Placeholders: {casino_exe}, {script}. "
            "Example: '\"{casino_exe}\" /run \"{script}\"'"
        ),
    )
    parser.add_argument("--run-now", action="store_true", help="Attempt to launch CASINO after writing the script.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_batch(
        manifest_path=args.manifest,
        sim_root=args.sim_root,
        cas_root=args.cas_root,
        output_script=args.output_script,
        output_summary=args.output_summary,
        output_runner_ps1=args.output_runner_ps1,
        casino_exe=args.casino_exe,
        launch_template=args.launch_template,
        run_now=args.run_now,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
