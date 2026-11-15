"""Run both first-party suites independently of the shell working directory."""
from pathlib import Path
import sys
import os
import tempfile
os.environ.setdefault("MPLBACKEND", "Agg")

root = Path(__file__).resolve().parents[2]
test_temp=(root/'.tmp/project_test_temp').resolve()
assert root in test_temp.parents
test_temp.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(test_temp)
sys.path[:0] = [str(root / "DataSimulation"), str(root / "WeldDefectDetection")]

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([
        str(root / "DataSimulation" / "tests"),
        str(root / "WeldDefectDetection" / "tests"),
        *sys.argv[1:],
    ]))
