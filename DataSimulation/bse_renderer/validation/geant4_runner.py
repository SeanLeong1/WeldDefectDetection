from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def geant4_available() -> bool:
    return shutil.which("geant4-config") is not None


def run_geant4_probe() -> dict:
    if not geant4_available():
        return {"available": False, "message": "geant4-config not found in PATH"}
    try:
        proc = subprocess.run(
            ["geant4-config", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return {"available": True, "ok": False, "message": str(exc)}
    return {"available": True, "ok": True, "version": proc.stdout.strip()}


def geant4_sources_dir() -> Path:
    return Path(__file__).resolve().parent / "geant4"

