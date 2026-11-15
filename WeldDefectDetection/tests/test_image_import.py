from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
from inspection.image_import import ImportSettings,load_raster,prepare,import_image,coordinates
from inspection.final_methods import FinalInspector,example_path,load_raw

NORM={'low':2.,'high':12.}

@pytest.mark.parametrize('ext',['png','tif'])
def test_16bit_grayscale_values_preserved(tmp_path,ext):
    values=np.array([[0,256,1000],[65535,32768,500]],dtype='uint16')
    p=tmp_path/f'image.{ext}';Image.fromarray(values).save(p)
    actual,alpha,info=load_raster(p)
    np.testing.assert_array_equal(actual,values)
    assert alpha is None

def test_tiff_frame_selection(tmp_path):
    p=tmp_path/'stack.tiff'
    first=Image.fromarray(np.arange(16,dtype='f4').reshape(4,4))
    second=Image.fromarray(np.arange(16,dtype='f4').reshape(4,4)+100)
    first.save(p,save_all=True,append_images=[second])
    a,_,info=load_raster(p,1);assert info['frames']==2
    np.testing.assert_array_equal(a,np.asarray(second))
    with pytest.raises(ValueError,match='frame'):load_raster(p,2)

def test_rgb_jpeg_luminance(tmp_path):
    p=tmp_path/'color.jpeg';Image.new('RGB',(20,20),(200,100,50)).save(p,quality=100)
    a,_,info=load_raster(p)
    assert a.shape==(20,20) and abs(float(a.mean())-124.2)<2
    assert 'luminance' in info['conversion']

def test_ring_sampling_matches_linear_coordinate_field():
    yy,xx=np.mgrid[:300,:300];values=(xx+2*yy).astype('f4')
    s=ImportSettings(centre_x=150,centre_y=150,radius_px=100,intensity='raw')
    raw,_=prepare(values,None,s,NORM);x,y=coordinates(s)
    np.testing.assert_allclose(raw,x+2*y,atol=4e-5,rtol=0)
    # The opposite radial orientation reverses only the row dimension.
    reverse=ImportSettings(**{**s.__dict__,'reverse_rows':True})
    np.testing.assert_array_equal(prepare(values,None,reverse,NORM)[0],raw[::-1])

def test_display_mapping_and_provenance(tmp_path):
    p=tmp_path/'unwrapped.png'
    values=np.tile(np.linspace(0,255,5024,dtype='uint8'),(160,1));Image.fromarray(values).save(p)
    out,record=import_image(p,ImportSettings(layout='unwrapped'),NORM,tmp_path)
    raw=np.load(out)
    assert raw.shape==(160,5024) and raw.min()==2 and raw.max()==12
    assert record['external_image'] and not record['physical_validation']
    assert json.loads(out.with_suffix('.import.json').read_text())==record

def test_rejects_clipped_or_transparent_annulus():
    yy,xx=np.mgrid[:300,:300];a=(xx+yy).astype('f4')
    with pytest.raises(ValueError,match='fit inside'):
        prepare(a,None,ImportSettings(centre_x=0,centre_y=0,radius_px=100),NORM)
    alpha=np.ones(a.shape,bool);alpha[150,250]=False
    with pytest.raises(ValueError,match='transparent'):
        prepare(a,alpha,ImportSettings(centre_x=150,centre_y=150,radius_px=100),NORM)

def test_raw_float_tiff_replays_original_inference(tmp_path):
    raw=load_raw(example_path());p=tmp_path/'raw.tiff';Image.fromarray(raw).save(p)
    inspector=FinalInspector()
    out,_=import_image(p,ImportSettings(layout='unwrapped',intensity='raw'),inspector.norm,tmp_path)
    np.testing.assert_array_equal(load_raw(out),raw)
    old=inspector.infer(example_path());new=inspector.infer(out)
    assert new['boxes']==old['boxes']
    for m in ('cnn','median'):np.testing.assert_array_equal(old['scores'][m],new['scores'][m])
    assert new['import_metadata']['image']['mode']=='F'

def test_no_pdf_support(tmp_path):
    with pytest.raises(ValueError,match='PDF'):load_raster(tmp_path/'page.pdf')
