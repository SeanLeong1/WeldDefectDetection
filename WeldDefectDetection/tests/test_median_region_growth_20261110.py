"""Behavioural checks for bounded, periodic, label-free region reconstruction."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import median_region_growth_20261110 as s


def test_weak_pixels_need_connected_seed_and_bounded_reach():
    x=np.zeros((20,40),'f4');x[10,10]=20;x[10,11:20]=8;x[2:4,2:4]=8
    cc=s.grow_components(x,16,6,3)
    assert len(cc)==1 and cc[0]['box']==[10,10,14,11] and cc[0]['area']==4


def test_both_polarities_are_accepted_and_recorded():
    x=np.zeros((20,40),'f4');x[10,10]=-20;x[10,11:14]=8
    a=s.grow_components(x,16,6,3);b=s.grow_components(-x,16,6,3)
    assert a[0]['box']==b[0]['box'] and a[0]['area']==4
    assert a[0]['positive']==3 and a[0]['negative']==1
    assert b[0]['positive']==1 and b[0]['negative']==3


def test_growth_crosses_circular_seam_without_radial_wrap():
    x=np.zeros((20,40),'f4');x[0,-1]=20;x[0,:3]=8;x[-1,-1]=8
    cc=s.grow_components(x,16,6,3)
    assert len(cc)==1 and cc[0]['box']==[39,0,43,1] and cc[0]['area']==4


def component(box,area=4):
    return dict(box=box,area=area,seeds=1,positive=area,negative=0,fragments=1)


def test_merging_joins_small_aligned_fragments_before_area_cutoff():
    cc=[component([10,4,14,5]),component([10,7,14,8])]
    assert s.form_regions(cc,8,0,40)==[]
    got=s.form_regions(cc,8,2,40)
    assert len(got)==1 and got[0]['box']==[10,4,14,8] and got[0]['fragments']==2


def test_merging_rejects_diagonal_or_distant_regions():
    a=component([10,4,14,5]);b=component([15,6,19,7]);c=component([10,12,14,13])
    assert len(s.form_regions([a,b,c],1,2,100))==3


def test_merging_handles_seam_and_caps_group_extent():
    got=s.form_regions([component([97,5,100,8]),component([1,5,4,8])],1,2,100)
    assert len(got)==1 and got[0]['box']==[97,5,104,8]
    cc=[component([0,5,20,8]),component([22,5,42,8])]
    assert len(s.form_regions(cc,1,2,100))==2


def test_empty_and_zero_radius():
    x=np.zeros((20,40),'f4');assert s.grow_components(x,16,8,3)==[]
    x[5:8,10:14]=20
    cc=s.grow_components(x,16,16,0)
    assert len(cc)==1 and cc[0]['box']==[10,5,14,8] and cc[0]['area']==12


def test_spatial_heap_matches_exhaustive_greedy_reference():
    def reference(cc,gap,period):
        work=sorted([dict(v) for v in cc],key=lambda v:tuple(v['box']))
        while True:
            pairs=[]
            for i,a in enumerate(work):
                for j in range(i+1,len(work)):
                    rank=s.merge_pair(a,work[j],gap,period)
                    if rank is not None:pairs.append((rank,i,j))
            if not pairs:break
            (_,_,box),i,j=min(pairs);a,b=work[i],work[j]
            width=box[2]-box[0];box[0]%=period;box[2]=box[0]+width
            merged=dict(box=box,**{k:a[k]+b[k] for k in ('area','seeds','positive','negative','fragments')})
            work=sorted([v for k,v in enumerate(work) if k not in (i,j)]+[merged],key=lambda v:tuple(v['box']))
        return work
    rng=np.random.default_rng(1110)
    for period in (40,64,100):
        for _ in range(20):
            cc=[]
            for j in range(12):
                x=int(rng.integers(period));y=int(rng.integers(25))
                cc.append(component([x,y,x+int(rng.integers(2,8)),y+int(rng.integers(2,8))]))
            for gap in (2,4):
                assert s.form_regions(cc,1,gap,period)==reference(cc,gap,period)
