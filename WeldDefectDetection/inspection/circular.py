"""Portable inference adapter for the frozen 4 October circular study.

Only raw/normalized complete annuli have the required spatial context. This
adapter never reads labels, guesses a circle, resizes physical scale or trains.
"""
import json,zipfile
from pathlib import Path
import numpy as np
import torch
from circular_study import metrics,median
from src.algorithms.classical.cnn_detector import TinyDefectSegNet

def load_annulus(path,options,norm):
    from .engine import MAX_BYTES,_array_ok,digest
    options.check();path=Path(path)
    if options.mode=='display':raise ValueError('Current circular model needs a complete raw or normalized annulus. Choose a historical model for exploratory display-image import.')
    if path.stat().st_size>MAX_BYTES:raise ValueError('File exceeds import limit.')
    spacing=options.pixel_size_mm;metadata={};sidecar=path.with_name(path.name+'.json')
    if sidecar.exists():
        metadata=json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if metadata.get('channel','topo_magnitude')!='topo_magnitude':raise ValueError('Circular input requires TOPO magnitude.')
        if 'radius_mm' in metadata and not np.isclose(float(metadata['radius_mm']),40.,rtol=0,atol=1e-9):
            raise ValueError('Circular input radius must be 40 mm.')
        if 'pixel_size_mm' in metadata:
            declared=float(metadata['pixel_size_mm'])
            if spacing is not None and not np.isclose(spacing,declared):raise ValueError('Pixel size disagrees with sidecar.')
            spacing=declared
    if path.suffix.lower()=='.npz':
        if options.mode!='raw':raise ValueError('Raw NPZ requires Raw TOPO mode.')
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist())>MAX_BYTES:raise ValueError('Expanded NPZ exceeds import limit.')
        with np.load(path,allow_pickle=False) as a:
            key=next((k for k in ('raw','topo') if k in a),None)
            if key is None:raise ValueError('Annulus NPZ requires raw or topo; masks are never input.')
            array=a[key].copy()
            if 'pixel_size_mm' in a:
                value=float(a['pixel_size_mm'])
                if spacing is not None and not np.isclose(value,spacing):raise ValueError('Pixel size disagrees with NPZ.')
                spacing=value
    elif path.suffix.lower()=='.npy':array=np.load(path,allow_pickle=False)
    else:raise ValueError('Circular model accepts NPZ/NPY annuli. Cartesian pictures need an explicit ring extraction first.')
    array=_array_ok(array,'Annulus')
    if array.ndim==3 and array.shape[0]==3:array=array[0]
    if array.shape not in ((96,5024),(128,5024)):raise ValueError('Expected a complete 96x5024 annulus, or a 128x5024 annulus with extra radial halo.')
    if spacing is None or not np.isclose(spacing,.05,rtol=0,atol=1e-9):raise ValueError('Circular radial pixel size must be declared as 0.05 mm.')
    if array.shape[0]==128:array=array[16:112]
    if np.ptp(array)==0:raise ValueError('Annulus has no intensity variation.')
    warnings=['Synthetic model; assumed 40 mm ring radius and known unwrapping. Angular columns are periodic.']
    if options.mode=='raw':
        clipped=float(np.mean((array<norm['low'])|(array>norm['high'])))
        x=np.sqrt(np.clip((array-norm['low'])/(norm['high']-norm['low']),0,1)).astype('f4')
        if clipped>.01:warnings.append(f'{clipped:.1%} of raw pixels clip under frozen normalization.')
    else:
        if array.min()<0 or array.max()>1:raise ValueError('Normalized input must be in [0,1].')
        x=array.astype('f4');warnings.append('Caller declares the current circular training normalization.')
    return dict(target=x[16:80],annulus=x[None],context='complete_periodic_ring',pixel_size_mm=.05,
        arc_pixel_size_mm=2*np.pi*40/5024,warnings=warnings,original_dtype=str(array.dtype),shape=[64,5024],mode=options.mode,
        metadata_sha256=digest(sidecar) if sidecar.exists() else None)

class CircularInspector:
    def __init__(self,campaign):
        from .engine import digest
        self.campaign=Path(campaign).resolve();self.record=json.loads((self.campaign/'frozen_models.json').read_text())
        normpath=self.campaign/'normalization.json'
        expected=next(h for p,h in self.record['files'].items() if Path(p.replace('\\','/')).name=='normalization.json')
        if digest(normpath)!=expected:raise ValueError('Frozen circular normalization hash mismatch.')
        self.norm=dict(json.loads(normpath.read_text()),_layout='circular_annulus')
        self.frozen=dict(models={m:dict(parameters=self.record[m]['selection']['best']) for m in ('cnn','median')})
        self.models={};self.model_hashes={}

    def source_hashes(self):
        from .engine import digest
        import inspect
        paths=[Path(__file__),Path(metrics.__file__),Path(median.__file__),Path(inspect.getfile(TinyDefectSegNet))]
        return {str(p):digest(p) for p in paths}

    def _network(self,arm):
        from .engine import digest
        if arm!='cnn':raise ValueError('Circular adapter supports CNN and median only.')
        if arm not in self.models:
            spec=self.record['cnn'];path=self.campaign/'training'/Path(spec['checkpoint'].replace('\\','/')).name
            if digest(path)!=spec['checkpoint_sha256']:raise ValueError('Frozen circular checkpoint hash mismatch.')
            model=TinyDefectSegNet(in_channels=1,base_channels=spec['config']['base_channels'])
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);model.eval()
            self.models[arm]=model;self.model_hashes[arm]=digest(path)
        return self.models[arm]

    def score(self,item,arm,cancel=None):
        if arm=='median':return median.scores(dict(x=item['annulus'][None]),self.record['median']['spec'])[0]
        if arm!='cnn':raise ValueError('This circular experiment has no threshold-baseline model.')
        model=self._network(arm);chunks=[];size=self.record['cnn']['config']['input_size']
        image=item['annulus'] if size==96 else item['annulus'][:,16:80]
        centres=list(range(16,5024,32))
        with torch.inference_mode():
            for start in range(0,len(centres),32):
                if cancel is not None and cancel.is_set():raise InterruptedError('Cancelled before next ring segment.')
                x=np.stack([median.tile(image,c,size) for c in centres[start:start+32]]).astype('f4')
                score=torch.sigmoid(model(torch.from_numpy(x)))[:,0].numpy()
                if size==96:score=score[:,16:80,16:80]
                chunks.extend(score[:,:,16:48])
        return np.concatenate(chunks,axis=1)

    def predict(self,item,arm,cancel=None):
        score=self.score(item,arm,cancel);p=self.record[arm]['selection']['best']
        mask,boxes=metrics.binary(score,p['threshold'],p['min_area'],p['closing'])
        return score,mask,boxes

    def coordinates(self,boxes):
        arc=2*np.pi*40/5024
        return dict(coordinate_policy='Pixel boxes use half-open angular columns and central64 rows. Physical box edges are half a sample from pixel centres; column0 is centred at angle0. Seam-crossing edges may exceed360 degrees or be slightly negative.',
            boxes_mm=[[(a-.5)*arc,(b-32)*.05,(c-.5)*arc,(d-32)*.05] for a,b,c,d in boxes],
            boxes_mm_axes=['arc_length_from_zero_angle_mm','radial_offset_from_40mm_mm'],
            boxes_angle_radial=[[(a-.5)*360/5024,(b-32)*.05,(c-.5)*360/5024,(d-32)*.05] for a,b,c,d in boxes])
