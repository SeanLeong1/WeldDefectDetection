"""Explicit raster-to-annulus preparation; never infers calibration from DPI."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,uuid
import numpy as np
from PIL import Image,ImageOps
from scipy.ndimage import map_coordinates

FORMATS={'.tif','.tiff','.png','.jpg','.jpeg'}
MAX_PIXELS=20_000_000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_raster(path,frame=0):
    path=Path(path)
    if path.suffix.lower() not in FORMATS:raise ValueError('Choose TIFF, PNG or JPEG. PDF is not supported.')
    if path.stat().st_size>256*1024*1024:raise ValueError('Image exceeds 256 MiB.')
    with Image.open(path) as im:
        frames=getattr(im,'n_frames',1)
        if not 0<=frame<frames:raise ValueError(f'Choose a TIFF frame between 1 and {frames}.')
        im.seek(frame)
        if im.width*im.height>MAX_PIXELS:raise ValueError('Image exceeds 20 million pixels.')
        im=ImageOps.exif_transpose(im)
        mode=im.mode;alpha=None
        if mode in ('RGBA','LA') or 'transparency' in im.info:
            rgba=np.asarray(im.convert('RGBA'));alpha=rgba[...,3]==255
        if mode=='P' or mode in ('RGB','RGBA','CMYK','YCbCr','LA'):
            rgb=np.asarray(im.convert('RGB'),dtype='f4')
            values=rgb[...,0]*.299+rgb[...,1]*.587+rgb[...,2]*.114
            conversion='RGB luminance: 0.299 R + 0.587 G + 0.114 B'
        else:
            values=np.asarray(im).astype('f4');conversion='native grayscale values preserved'
        if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Expected finite grayscale or RGB image values.')
        info=dict(frame=frame,frames=frames,mode=mode,width=im.width,height=im.height,conversion=conversion)
    return values,alpha,info

@dataclass(frozen=True)
class ImportSettings:
    layout:str='circular'
    centre_x:float=0
    centre_y:float=0
    radius_px:float=1
    start_angle_deg:float=0
    clockwise:bool=False
    reverse_rows:bool=False
    intensity:str='display'
    gain:float=1
    offset:float=0
    frame:int=0

def coordinates(settings):
    radii=settings.radius_px*(40+(np.arange(160)-79.5)*.05)/40
    direction=-1 if settings.clockwise else 1
    angles=np.deg2rad(settings.start_angle_deg)+direction*np.arange(5024)*2*np.pi/5024
    x=settings.centre_x+radii[:,None]*np.cos(angles)
    y=settings.centre_y-radii[:,None]*np.sin(angles)
    return x,y

def prepare(values,alpha,settings,norm):
    nums=[settings.centre_x,settings.centre_y,settings.radius_px,settings.start_angle_deg,settings.gain,settings.offset]
    if not np.isfinite(nums).all() or settings.radius_px<=0:raise ValueError('Use finite values and a positive ring radius.')
    if settings.layout=='circular':
        x,y=coordinates(settings)
        if x.min()<0 or y.min()<0 or x.max()>values.shape[1]-1 or y.max()>values.shape[0]-1:
            raise ValueError('The full 8 mm annulus must fit inside the image. Adjust the centre/radius or use an uncropped image.')
        if alpha is not None and np.any(map_coordinates(alpha.astype('f4'),[y,x],order=1,mode='constant',cval=0)<.999):
            raise ValueError('The selected annulus crosses transparent pixels. Adjust the ring or use a complete image.')
        annulus=map_coordinates(values,[y,x],order=1,mode='nearest')
    elif settings.layout=='unwrapped':
        if values.shape!=(160,5024):raise ValueError('An unwrapped image must be exactly 160 x 5024, including the radial halo.')
        if alpha is not None and not alpha.all():raise ValueError('The annulus contains transparent pixels.')
        annulus=values.copy()
    else:raise ValueError('Unknown image layout.')
    if settings.reverse_rows:annulus=annulus[::-1].copy()
    mapping={}
    if settings.intensity=='display':
        low,high=np.percentile(annulus,[1,99])
        if high<=low:raise ValueError('The selected ring has insufficient intensity variation.')
        unit=np.clip((annulus-low)/(high-low),0,1)
        # Explicit display approximation, never claimed to recover original raw counts.
        raw=norm['low']+unit**2*(norm['high']-norm['low'])
        mapping=dict(kind='exploratory percentile display mapping',display_low=float(low),display_high=float(high),
                     formula='raw = training_low + clipped_display_fraction^2 * (training_high - training_low)')
    elif settings.intensity=='raw':
        raw=annulus*settings.gain+settings.offset
        mapping=dict(kind='user-declared raw TOPO units',gain=settings.gain,offset=settings.offset)
    else:raise ValueError('Unknown intensity interpretation.')
    if not np.isfinite(raw).all() or np.ptp(raw)<=0:raise ValueError('Prepared TOPO must contain finite, varying intensities.')
    return raw.astype('f4'),mapping

def import_image(path,settings,norm,output_root):
    values,alpha,info=load_raster(path,settings.frame)
    raw,mapping=prepare(values,alpha,settings,norm)
    folder=Path(output_root)/'imports'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    target=folder/'topo.npy';np.save(target,raw)
    record=dict(schema='weld-raster-import/v1',source=str(Path(path).resolve()),source_sha256=sha(path),
                prepared_sha256=sha(target),settings=asdict(settings),image=info,intensity_mapping=mapping,
                normalization=norm,ring_radius_mm=40,radial_pitch_mm=.05,external_image=True,
                interpretation='Exploratory external image; preprocessing does not establish model applicability.',physical_validation=False)
    target.with_suffix('.import.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    return target,record
