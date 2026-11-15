"""Independent regression checks for the new experimental plumbing."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import numpy as np
import torch
from count_study import campaign as c, learning as learn
from circular_study import metrics
from raised_crown import learning as base
from src.algorithms.classical.cnn_detector import TinyDefectSegNet


def test_circular_selection_matches_existing_metric():
    rng=np.random.default_rng(9927)
    meta=dict(scene_id='a',background_id='a',source_class='open_intrinsic',count_multiplier=4,
              diameter_mm=.3,repeat=0,dwell_bin=0,amplitude=.05)
    for trial in range(12):
        score=np.zeros((128,5024),'f4');truth=np.zeros_like(score,dtype='u1')
        for j in range(4):
            y=int(rng.integers(5,110));x=int(rng.integers(0,5024))
            ids=(np.arange(int(rng.integers(2,15)))+x)%5024
            score[y:y+int(rng.integers(2,12)),ids]=rng.uniform(.3,1.)
            truth[y:y+7,ids]=1
        if trial%2==0:score[50:56,:5]=.9;score[50:56,-5:]=.9;truth[50:56,:5]=1;truth[50:56,-5:]=1
        boxes=metrics.regions(truth)[1]
        for close in (1,3):
            actual=learn.component_rows(score,truth,boxes,meta,.5,close,c.AREAS)
            for area,row in zip(c.AREAS,actual):
                expected=metrics.evaluate(dict(mask=truth[None],meta=[meta]),score[None],
                    dict(threshold=.5,min_area=area,closing=close),retain=True)[0]
                for key in ('tp','fp','fn','intersection','union','predicted_pixels','truth_pixels','predicted_regions'):
                    assert row[key]==expected[key]


def test_lazy_patch_pipeline_and_training_step():
    rows=c.read(c.OUT/'datasets/train/000/index.json')
    norm=c.read(c.old.OUT/'normalization.json');data=learn.Data(rows,norm)
    ids=np.array([[0,32,0],[1,64,5020],[6,96,200]])
    x,y=data.batch(ids)
    for sample,(i,r,a) in zip(x,ids):
        assert np.array_equal(sample[0],base.tile(data.image(int(i)),int(r),int(a)))
    model=TinyDefectSegNet(in_channels=1,base_channels=24)
    assert sum(p.numel() for p in model.parameters())==262777
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    logits=model(torch.from_numpy(x));target=torch.from_numpy(y)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()


def test_matched_count_variants_keep_geometry_and_dwell():
    rows=c.read(c.OUT/'datasets/train/000/index.json')
    assert len(rows)==18
    for kind in c.KINDS:
        group=[r for r in rows if r['source_class']==kind]
        assert len({r['geometry_sha256'] for r in group})==1
        assert len({r['dwell_time_us'] for r in group})==1
        rates=[r['profile']['effective_count_rate']/r['count_multiplier'] for r in group]
        assert np.allclose(rates,rates[0])
        assert len({r['acquisition_seed'] for r in group})==2
    assert c.read(c.OUT/'noise_checks.json')['status']=='passed'


def test_acquisition_streams_are_disjoint_across_splits():
    seeds={split:{c.BASES[split]+i+repeat*100000000 for i in range(n) for repeat in (0,1)}
           for split,n in c.COUNTS.items()}
    assert not seeds['train']&seeds['val']
    assert not seeds['train']&seeds['test']
    assert not seeds['val']&seeds['test']
    assert all(len(seeds[s])==2*n for s,n in c.COUNTS.items())
