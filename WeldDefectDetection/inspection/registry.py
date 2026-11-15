"""One explicit current-model choice, with labelled historical workflows."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
CURRENT=ROOT/'DataSimulation/data/output/circular_study_20261004_v4'
LEGACY=ROOT/'DataSimulation/data/output/variability_campaign_20260926_v1'
RASTER=ROOT/'DataSimulation/data/output/acquisition_campaign_20260928_v2'

def campaigns():
    paths=[('Historical | straight crops | 26 Sep',LEGACY),('Historical | raster v2 crops | 28 Sep',RASTER),
           ('Current | circular full ring | 4 Oct',CURRENT)]
    return {label:p for label,p in paths if (p/'frozen_models.json').exists()}

def available_methods(campaign):
    data=json.loads((Path(campaign)/'frozen_models.json').read_text())
    return ('cnn','median') if 'cnn_candidates' in data else tuple(data['models'])

def example_path(campaign):
    campaign=Path(campaign)
    return campaign/('datasets/test/test-000-keyhole_pore.npz' if campaign==CURRENT else 'inference_smoke/keyhole_pore_0.00.npz')
