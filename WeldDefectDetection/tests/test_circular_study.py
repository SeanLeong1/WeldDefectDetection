"""Scientific-contract checks for the continuous ring study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pytest
from circular_study import geometry as g,metrics,median,training
from circular_study import generation


def kwargs():
    return dict(seed=12,depth=.14,crown=.012,width=1.1,haz=1.,focus=.35,speed=800.,power=18.,amplitude=.1)


def test_surface_has_periodic_value_and_derivative():
    u=np.linspace(-1.6,1.6,129)
    a=g.surface(u,0.,**kwargs());b=g.surface(u,2*np.pi,**kwargs())
    np.testing.assert_allclose(a,b,atol=1e-8)
    eps=1e-6
    left=(g.surface(u,0.,**kwargs())-g.surface(u,-eps,**kwargs()))/eps
    right=(g.surface(u,eps,**kwargs())-g.surface(u,0.,**kwargs()))/eps
    np.testing.assert_allclose(left,right,atol=.012)


def test_same_physical_points_do_not_depend_on_requested_grid():
    u=np.linspace(-1.6,1.6,129);t=np.linspace(0,2*np.pi,129)
    fine=g.surface(u,t,**kwargs());coarse=g.surface(u[::2],t[::2],**kwargs())
    np.testing.assert_array_equal(fine[::2],coarse)


def test_periodic_components_join_diagonally_and_use_short_box():
    m=np.zeros((12,64),'u1');m[3:6,:3]=1;m[4:7,-2:]=1
    kept,boxes=metrics.regions(m)
    assert len(boxes)==1 and boxes[0]==[62,3,67,7]
    assert kept.sum()==m.sum()
    assert metrics.match(boxes,boxes,64)['tp']==1
    shifted=[[b[0]-64,b[1],b[2]-64,b[3]] for b in boxes]
    assert metrics.match(shifted,boxes,64)['tp']==1


def test_area_filter_is_applied_after_seam_merge():
    m=np.zeros((12,64),'u1');m[4:6,:2]=1;m[4:6,-2:]=1
    kept,boxes=metrics.regions(m,min_area=7)
    assert kept.sum()==8 and len(boxes)==1


def test_context_is_disjoint_despite_overlapping_targets():
    image=np.arange(g.COLS)[None,:]
    target=set(median.tile(image,16).ravel())
    contexts=[set(median.tile(image,16+k).ravel()) for k in (-128,-64,64,128)]
    assert all(not target.intersection(c) for c in contexts)
    assert all(not a.intersection(b) for i,a in enumerate(contexts) for b in contexts[i+1:])


def test_stitch_covers_each_angle_once_and_context_crop_is_correct():
    image=np.broadcast_to(np.arange(128,dtype='f4'),(1,1,96,128)).copy()
    data=dict(x=image,mask=np.zeros((1,64,128),'u1'),meta=[dict(split='train')])
    class IdentityLogit(torch.nn.Module):
        def forward(self,x):return x
    for variant in ('fixed_negatives','larger_context'):
        config=training.configuration(variant)
        prediction=training.predict(data,IdentityLogit(),config)
        np.testing.assert_allclose(prediction[0,0],torch.sigmoid(torch.arange(128,dtype=torch.float32)).numpy())


def test_metric_counts_regions_and_clean_rings_separately():
    rows=[dict(tp=1,fp=2,fn=1,intersection=10,union=20,source_class='keyhole_pore',predicted_regions=3),
          dict(tp=0,fp=3,fn=0,intersection=0,union=30,source_class='clean',predicted_regions=3)]
    m=metrics.summary(rows)
    assert m['f1']==.25 and m['clean_ring_fa']==1 and m['clean_false_regions']==3


def test_every_split_covers_all_angle_size_and_offset_strata():
    for split,count in generation.COUNTS.items():
        for kind in ('keyhole_pore','intrinsic_pore'):
            plans=[generation.plan(kind,generation.BASES[split]+i,i)[0]['params'] for i in range(count)]
            bins=np.bincount([int(p['angle_rad']/(np.pi/2)) for p in plans],minlength=4)
            assert bins.min()>0 and bins.max()-bins.min()<=1
            assert len({p['radial_offset_mm'] for p in plans})==3
            lo,hi=(.5,1.2) if kind=='keyhole_pore' else (.5,1.)
            assert {int(3*(p['diameter_mm']-lo)/(hi-lo)) for p in plans}=={0,1,2}


def test_operating_point_selection_cannot_use_test_labels():
    data=dict(meta=[dict(split='test')],mask=np.zeros((1,64,128),'u1'))
    with pytest.raises(ValueError,match='validation'):
        metrics.select(data,np.zeros((1,64,128),'f4'),[.5])


def test_evaluation_cannot_silently_drop_scenes():
    data=dict(meta=[{},{}],mask=np.zeros((2,64,128),'u1'))
    with pytest.raises(ValueError,match='every scene'):
        metrics.evaluate(data,np.zeros((1,64,128),'f4'),dict(threshold=.5,min_area=1,closing=1))
