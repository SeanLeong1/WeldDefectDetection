"""Physical coordinate, label and numerical invariants of the new study."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
import numpy as np
import torch
from raised_crown import geometry as g,learning as learn
from raised_crown.assessment import map_back
from raised_crown.validation import geometry_checks


def test_inspector_rejects_same_dimensions_with_incompatible_renderer(tmp_path):
    import json
    import pytest
    from raised_crown.gui import infer
    base=dict(geometry=g.specification(),profile='raised',visibility_resolution=dict(radial_mm=.1,angular_stride=4,ray_step_mm=.1),
              renderer_metadata=dict(emission_b=dict(mode='lambertian')),acquisition=dict(version='raster_v2'))
    for key,value in [('profile','recessed'),('visibility_resolution',{}),('acquisition',dict(version='legacy'))]:
        path=tmp_path/'incompatible.npz'
        path.with_suffix('.json').write_text(json.dumps(dict(metadata=dict(base,**{key:value}))))
        with pytest.raises(ValueError,match='renderer contract'):infer(path)

def test_geometry_dimensions_volume_periodicity_and_hidden_control():
    report=geometry_checks()
    assert report['no_electron_depth_cutoff']
    assert 74<report['max_inclination_deg']<76

def test_fusion_is_solid_and_buried_void_does_not_move_surface():
    u=np.array([0.,.59,.61,1.5]);z=np.full(4,-5.)
    np.testing.assert_array_equal(g.fusion(u,z,3.),[True,True,False,False])
    p=dict(kind='intrinsic_pore',effect='hidden',diameter_mm=.6,depth_mm=.2,
        angle_rad=0.,radial_offset_mm=0.,roof_mm=.1)
    assert g.pore_contains(0.,0.,2.7,3.,p)
    assert not g.pore_contains(0.,0.,3.,3.,p)
    change,label,_=g.surface_effect(np.array([0.]),np.array([0.]),p,.0025)
    assert change[0]==0 and not label[0]

def test_full_band_tile_stitching_has_no_missing_stripe_or_seam():
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.zeros((len(x),1,64,64))
    out=learn.predict_image(np.zeros((160,5024),'f4'),Constant())
    assert out.shape==(128,5024)
    np.testing.assert_array_equal(out,.5)

def test_nominal_coordinate_map_is_identity_and_missing_coverage_stays_zero():
    y=np.arange(128)[:,None];a=np.arange(5024)[None,:]*2*np.pi/5024
    image=(y/128+.2*np.sin(a)).astype('f4')
    mapped=map_back(image,dict(dx=0.,dy=0.,dr=0.))
    # Boundary roundoff must not erase otherwise covered edge rows.
    np.testing.assert_allclose(mapped,image,atol=1e-5)
    moved=map_back(np.ones_like(image),dict(dx=0.,dy=0.,dr=.5))
    assert np.all(moved[:9]==0)

def test_input_contract_rejects_historical_shape():
    import pytest
    with pytest.raises(ValueError):learn.predict_image(np.zeros((96,5024),'f4'),None)
