"""Numerical checks for exact quadrature reuse and circular context geometry."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import run_circular_campaign_20261004 as campaign


def test_cached_quadrature_matches_uncached_after_local_and_global_changes():
    cfg=campaign.old.load_config_bundle(ROOT/'DataSimulation/config')
    sample,_=campaign.c.sampled(cfg,2130400001,0)
    n=48
    normals=np.zeros((n,n,3),np.float32);normals[...,2]=1
    kw=dict(normals=normals,detector=sample.detector,material=sample.material,
            material_z_map=np.full((n,n),sample.material.z,np.float32),b_coeff=.92)
    cache={}
    with patch.object(campaign.old,'RADIUS',.5):
        baseline=campaign.annular_quadrants(cache,**kw)
        normals=normals.copy();normals[10:14,12:18]=[.3,.4,np.sqrt(.75)]
        kw['normals']=normals
        cached=campaign.annular_quadrants(cache,**kw)
        direct=campaign.annular_quadrants(**kw)
        np.testing.assert_array_equal(cached,direct)
        assert not np.array_equal(cached,baseline)
        kw['directional_gain']=.8
        np.testing.assert_array_equal(campaign.annular_quadrants(cache,**kw),campaign.annular_quadrants(**kw))


def test_ring_rectification_preserves_radial_coordinate_and_wrap_neighbours():
    old=campaign.old
    axis=(np.arange(old.N)-(old.N-1)/2)*old.SPACING
    radial=np.hypot(axis[:,None],axis[None,:]).astype(np.float32)
    images,masks=old.crop_ring(radial,np.zeros(radial.shape,np.uint8))
    expected=old.RADIUS+(np.arange(64)-31.5)*old.SPACING
    np.testing.assert_allclose(images,np.broadcast_to(expected[None,:,None],images.shape),atol=2e-5)
    assert not masks.any()
    for i in range(old.FRAMES):
        context=[(i+k)%old.FRAMES for k in (-2,-1,1,2)]
        assert len(set(context))==4 and i not in context
