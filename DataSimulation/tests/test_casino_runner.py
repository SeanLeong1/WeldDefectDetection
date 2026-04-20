from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_casino_runner_emit_mock(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runner = repo / "tools" / "casino" / "casino_runner_template.py"
    case_path = tmp_path / "case.json"
    out_path = tmp_path / "casino_output.json"

    case_path.write_text(
        json.dumps(
            {
                "seed": 123,
                "beam": {"e0_kev": 60.0, "beam_current_ma": 300.0, "focusing_current_norm": 0.6},
                "material": {"name": "S275JR", "z": 26.0, "rho": 7.87},
                "defects": [],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--case",
            str(case_path),
            "--out",
            str(out_path),
            "--work",
            str(tmp_path),
            "--casino",
            "dummy_casino.exe",
            "--emit-mock",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    quads = payload["quadrants"]
    assert set(quads.keys()) == {"A", "B", "C", "D"}
