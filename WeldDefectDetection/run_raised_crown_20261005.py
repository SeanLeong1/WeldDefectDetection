"""Reproduce the separately versioned raised-crown study."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
if __name__=='__main__':
    from raised_crown.campaign import main
    from raised_crown import assessment as a
    from raised_crown.execution import evaluate
    a.evaluate=evaluate
    main()
