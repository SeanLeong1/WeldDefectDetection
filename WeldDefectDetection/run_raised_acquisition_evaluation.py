"""Isolated evaluation worker; model settings are already frozen."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
if __name__=='__main__':
    from raised_acquisition.assessment import evaluate_method,stress
    if sys.argv[1]=='stress':
        import torch
        from raised_acquisition.learning import verify_freeze
        torch.set_num_threads(4)
        stress(verify_freeze(),(int(sys.argv[2]),int(sys.argv[3])))
    else: evaluate_method(sys.argv[1])
