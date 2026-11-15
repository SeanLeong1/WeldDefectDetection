"""Run all count-study stages with bounded parallelism and durable status."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from count_study.runner import main
if __name__=='__main__':main()
