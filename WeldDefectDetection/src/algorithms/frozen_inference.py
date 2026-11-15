"""Inference from raw TOPO arrays, with no dataset labels or training access."""
from pathlib import Path
import json
import hashlib
import numpy as np
from src.algorithms.classical.cnn_detector import CNNDetector, torch
from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor
from src.utils.image_preprocessing import filter_small_components, PREPROCESSING_VERSION

def prepare_raw(raw, normalization):
    lo,hi=normalization['raw_low'],normalization['raw_high']
    if not np.isfinite([lo,hi]).all() or hi <= lo or normalization.get('source_split')!='train' or normalization.get('tone_curve')!='sem_sqrt':
        raise ValueError('A valid frozen training normalization is required.')
    raw=np.asarray(raw,dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('Raw TOPO input must be finite.')
    toned=.14+.72*np.power(np.clip((raw-lo)/(hi-lo),0,1),1.05)
    # Match the scientific dataset's saved 16-bit TOPO representation exactly.
    return (np.clip(toned,0,1)*65535).astype(np.uint16).astype(np.float32)/65535

def infer_frozen(kind, input_path, checkpoint_path, normalization_path):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    norm_path=Path(normalization_path)
    if checkpoint.get('normalization_sha256') != hashlib.sha256(norm_path.read_bytes()).hexdigest():
        raise ValueError('Checkpoint normalization identity does not match.')
    if checkpoint.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Checkpoint preprocessing identity is unsupported.')
    norm=json.loads(norm_path.read_text())
    if kind=='static':
        raw=np.load(input_path,allow_pickle=False)
        if not isinstance(raw,np.ndarray) or raw.ndim!=2:
            raise ValueError('Static input must be one raw two-dimensional .npy array.')
        cfg=checkpoint['config']
        cfg['cnn']['checkpoint_path']=str(checkpoint_path)
        cfg['cnn']['device']='cpu'
        detector=CNNDetector(cfg)
        _,debug=detector.detect_with_debug(prepare_raw(raw,norm))
        return {'probability':debug.probability_map,'mask':debug.binary_mask}
    with np.load(input_path,allow_pickle=False) as inputs:
        context=prepare_raw(inputs['context'],norm)
        target=prepare_raw(inputs['target'],norm)
        offsets=np.asarray(inputs['context_offsets_px'])
    radius=int(checkpoint['context_radius'])
    stride=int(checkpoint['stride_px'])
    if context.ndim!=3 or target.ndim!=2 or context.shape!=(2*radius,*target.shape) or stride<target.shape[1]:
        raise ValueError('Spatial input violates the checkpoint channel/stride contract.')
    expected=np.array(list(range(-radius,0))+list(range(1,radius+1)))*stride
    if not np.array_equal(offsets,expected) or checkpoint.get('context_preparation')!='non_overlapping_native_coordinates':
        raise ValueError('Spatial context must have the frozen order and disjoint crop intervals.')
    model=BidirectionalSurfacePredictor(context_channels=2*radius,base_channels=checkpoint['base_channels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        predicted=model(torch.from_numpy(context[None]))[0,0].numpy()
    residual=np.abs(target-predicted)
    mask=filter_small_components(residual>=checkpoint['residual_threshold'],checkpoint['min_area'])
    return {'reconstruction':predicted,'residual':residual,'mask':mask}
