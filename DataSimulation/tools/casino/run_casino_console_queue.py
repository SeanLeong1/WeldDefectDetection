from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _load_summary(summary_path: Path) -> list[dict[str, str]]:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    per_entry_scripts = raw.get("per_entry_scripts", [])
    if not isinstance(per_entry_scripts, list) or not per_entry_scripts:
        raise ValueError(
            "Summary file does not contain per_entry_scripts. "
            "Regenerate with run_sim_batch.py using --output-runner-ps1."
        )
    out: list[dict[str, str]] = []
    for idx, item in enumerate(per_entry_scripts):
        if not isinstance(item, dict):
            raise ValueError(f"per_entry_scripts[{idx}] must be an object.")
        script_path = item.get("script_path")
        if not script_path:
            raise ValueError(f"per_entry_scripts[{idx}] is missing script_path.")
        out.append(
            {
                "name": str(item.get("name", Path(script_path).stem)),
                "script_path": str(Path(script_path).expanduser().resolve()),
                "sim_path": str(item.get("sim_path", "")),
                "cas_path": str(item.get("cas_path", "")),
            }
        )
    return out


def _discover_script_dir(script_dir: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for path in sorted(script_dir.glob("*.txt")):
        out.append(
            {
                "name": path.stem,
                "script_path": str(path.resolve()),
                "sim_path": "",
                "cas_path": "",
            }
        )
    if not out:
        raise ValueError(f"No .txt scripts found in '{script_dir}'.")
    return out


def _launch(console_exe: Path, script_path: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(console_exe), str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _terminate_running_jobs(running: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interrupted: list[dict[str, Any]] = []
    for job in running:
        proc: subprocess.Popen[str] = job["proc"]
        try:
            proc.terminate()
        except Exception:
            pass
    for job in running:
        proc = job["proc"]
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        interrupted.append(
            {
                "name": job["name"],
                "script_path": job["script_path"],
                "sim_path": job["sim_path"],
                "cas_path": job["cas_path"],
                "pid": job["pid"],
                "returncode": (
                    int(proc.returncode)
                    if proc.returncode is not None
                    else -1
                ),
                "duration_s": round(time.time() - job["started_at"], 3),
                "stdout": stdout,
                "stderr": stderr,
                "status": "interrupted",
            }
        )
    return interrupted


def run_queue(
    *,
    console_exe: str,
    summary_path: str | None,
    script_dir: str | None,
    max_parallel: int,
    poll_interval_s: float,
    output_report: str | None,
    stop_on_failure: bool,
) -> dict[str, Any]:
    if bool(summary_path) == bool(script_dir):
        raise ValueError("Provide exactly one of --summary or --script-dir.")
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")

    exe = Path(console_exe).expanduser().resolve()
    if summary_path:
        queue = _load_summary(Path(summary_path).expanduser().resolve())
    else:
        queue = _discover_script_dir(Path(script_dir).expanduser().resolve())

    pending = list(queue)
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    failures = 0
    interrupted = False

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                job = pending.pop(0)
                script_path = Path(job["script_path"])
                proc = _launch(exe, script_path)
                running.append(
                    {
                        **job,
                        "pid": proc.pid,
                        "proc": proc,
                        "started_at": time.time(),
                    }
                )
                print(f"START {job['name']} pid={proc.pid}")

            next_running: list[dict[str, Any]] = []
            for job in running:
                proc = job["proc"]
                ret = proc.poll()
                if ret is None:
                    next_running.append(job)
                    continue

                stdout, stderr = proc.communicate()
                result = {
                    "name": job["name"],
                    "script_path": job["script_path"],
                    "sim_path": job["sim_path"],
                    "cas_path": job["cas_path"],
                    "pid": job["pid"],
                    "returncode": int(ret),
                    "duration_s": round(time.time() - job["started_at"], 3),
                    "stdout": stdout,
                    "stderr": stderr,
                    "status": "ok" if ret == 0 else "failed",
                }
                finished.append(result)
                print(f"END   {job['name']} returncode={ret} duration_s={result['duration_s']}")
                if ret != 0:
                    failures += 1
                    if stop_on_failure:
                        finished.extend(_terminate_running_jobs(next_running))
                        next_running = []
                        pending = []
                        break

            running = next_running
            if running:
                time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        interrupted = True
        print("INTERRUPT received; terminating active CASINO jobs...")
        finished.extend(_terminate_running_jobs(running))
        running = []

    report: dict[str, Any] = {
        "console_exe": str(exe),
        "max_parallel": max_parallel,
        "num_jobs": len(queue),
        "num_failures": failures,
        "interrupted": interrupted,
        "num_completed": sum(1 for r in finished if r["status"] == "ok"),
        "num_interrupted": sum(1 for r in finished if r["status"] == "interrupted"),
        "num_pending": len(pending),
        "results": finished,
    }

    if output_report:
        report_path = Path(output_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CASINO v3 console scripts with a fixed-width queue of parallel processes."
    )
    parser.add_argument("--console-exe", required=True, help="Path to console_casino3_script_x64.exe.")
    parser.add_argument("--summary", default=None, help="Summary JSON from run_sim_batch.py containing per_entry_scripts.")
    parser.add_argument("--script-dir", default=None, help="Folder containing one-case CASINO script .txt files.")
    parser.add_argument("--max-parallel", type=int, default=2, help="Maximum CASINO console instances to run at once.")
    parser.add_argument("--poll-interval-s", type=float, default=1.0, help="Queue polling interval in seconds.")
    parser.add_argument("--output-report", default=None, help="Optional path to write a JSON execution report.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop launching new work after the first non-zero exit code.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_queue(
        console_exe=args.console_exe,
        summary_path=args.summary,
        script_dir=args.script_dir,
        max_parallel=args.max_parallel,
        poll_interval_s=args.poll_interval_s,
        output_report=args.output_report,
        stop_on_failure=args.stop_on_failure,
    )
    print(json.dumps(report, indent=2))
    return 130 if report.get("interrupted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
