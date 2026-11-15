"""Check that follow-up scoring preserves the original circular metric."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import median_localisation_20261110 as s

def test_periodic_components_match_reference():
    rng=np.random.default_rng(1110)
    for trial in range(12):
        score=np.zeros((128,5024),'f4')
        for _ in range(15):
            y=int(rng.integers(0,115));x=int(rng.integers(0,5024))
            score[y:y+int(rng.integers(1,13)),(x+np.arange(int(rng.integers(1,20))))%5024]=1
        score[40:47,:5]=1;score[40:47,-4:]=1
        if trial%2:score[46:51,-7:]=1
        for closing in (1,3):
            compact=s.compact_components(score,.5,closing)
            for area in s.AREAS:
                actual=sorted(z[:4] for z in compact if z[4]>=area)
                _,expected=s.metrics.binary(score,.5,area,closing)
                assert actual==sorted(expected)

def test_box_transform_wrap_clip_and_identity():
    boxes=[[5020,0,5032,7],[100,121,110,128],[10,30,17,37]]
    assert s.transform(boxes,s.correction_identity())==boxes
    corrected=s.transform(boxes,dict(radial_rule='constant',dy=-2,dx=8,scale=1,pad=0))
    assert corrected[0]==[4,0,16,5]
    assert s.box_mask(boxes).sum()==84+70+49
    corrected=s.transform([[10,30,20,40],[10,90,20,100]],dict(radial_rule='toward_crown',dy=4,dx=0,scale=1,pad=0))
    assert corrected==[[10,34,20,44],[10,86,20,96]]

def test_box_objective_matches_original_match():
    pred=[[[5020,40,5032,50]],[[10,0,20,10]],[]]
    truth=[[[5022,40,5030,50]],[],[[10,0,20,10]]]
    rows=[dict(count_multiplier=m) for m in (1,4,64)]
    actual=s.box_objective(rows,truth,pred)
    assert actual['by_count']['1']['tp']==1
    assert actual['by_count']['4']['fp']==1
    assert actual['by_count']['64']['fn']==1
    assert actual['macro_f1']==1/3
