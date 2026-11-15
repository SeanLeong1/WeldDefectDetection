"""Independent numerical contracts for the new surface-variability experiment."""
from pathlib import Path
import sys
import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'DataSimulation'))
sys.path.insert(0,str(REPO/'WeldDefectDetection'))
from variability_campaign.renderer import gradients,normals,smooth_variations


@pytest.mark.parametrize('method',['finite_difference','cubic_spline'])
def test_physical_plane_gradient(method):
    x=np.arange(37)*.05;y=np.arange(23)*.05
    xx,yy=np.meshgrid(x,y);h=.3*xx-.2*yy+.4
    gy,gx=gradients(h,.05,method)
    np.testing.assert_allclose(gx,.3,atol=1e-12)
    np.testing.assert_allclose(gy,-.2,atol=1e-12)
    expected=np.array([-.3,.2,1]);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(normals(h,.05,method),np.broadcast_to(expected,(*h.shape,3)),atol=1e-7)


def test_cubic_spline_recovers_quadratic_at_edges():
    x=np.arange(20)*.07;y=np.arange(15)*.07;xx,yy=np.meshgrid(x,y)
    gy,gx=gradients(.1*xx**2+.3*yy**2+.2*xx*yy,.07,'cubic_spline')
    np.testing.assert_allclose(gx,.2*xx+.2*yy,atol=1e-12)
    np.testing.assert_allclose(gy,.6*yy+.2*xx,atol=1e-12)


def test_variations_bounded_reproducible_and_distinct():
    a=smooth_variations(576,123);b=smooth_variations(576,123);c=smooth_variations(576,124)
    for aa,bb,cc in zip(a,b,c):
        np.testing.assert_array_equal(aa,bb)
        assert np.max(np.abs(aa))<=1.000001
        assert not np.array_equal(aa,cc)
        assert np.max(np.abs(np.diff(aa)))<.1


def test_invalid_gradient_inputs_rejected():
    with pytest.raises(ValueError):gradients(np.zeros((8,8)),0)
    with pytest.raises(ValueError):gradients(np.full((8,8),np.nan),.05)
    with pytest.raises(ValueError):gradients(np.zeros((8,8)),.05,'unknown')
