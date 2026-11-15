"""Run the new experiment without modifying historical campaigns."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'DataSimulation'))
from variability_campaign.campaign import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare','pilot','development','train','confirmation','evaluate','report','all'], default='all')
    p.add_argument('--root', type=Path, default=REPO/'DataSimulation/data/output/variability_campaign_20260926_v1')
    args = p.parse_args()
    run(args.root, args.stage)
