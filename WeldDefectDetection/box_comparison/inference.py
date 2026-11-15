"""Label-free raw TOPO inference. Only boxes are exported as detections."""
from pathlib import Path
import cv2
import numpy as np
from bse_renderer.utils import apply_sem_tone_curve
from .core import median_reference, boxes_from_mask, threshold_mask, write
from .training import scores_for
from .campaign import read, check_protocol, require, verify_files


def normalize_raw(raw, normalization):
    raw=np.asarray(raw,dtype=np.float32)
    low,high=normalization['raw_low'],normalization['raw_high']
    if not np.isfinite(raw).all() or high<=low or normalization['tone_curve']!='sem_sqrt':
        raise ValueError('Invalid raw input or frozen normalization.')
    # Match the dataset's tone function and uint16 truncation exactly. The
    # historical "sem_sqrt" label is a legacy name, not a sqrt implementation.
    toned=apply_sem_tone_curve(np.clip((raw-low)/(high-low),0,1))
    return (toned*65535).astype(np.uint16).astype(np.float32)/65535


def infer(root,input_path,output,arm):
    root=Path(root);check_protocol(root);require(root,'freeze')
    frozen=read(root/'frozen_models.json')
    norm=frozen['normalization'];verify_files({norm['path']:norm['sha256']})
    with np.load(input_path,allow_pickle=False) as bundle:
        if 'mask' in bundle.files or 'labels' in bundle.files:
            raise ValueError('Inference input must not contain labels.')
        if float(bundle['energy_kev'])!=60 or not np.isclose(float(bundle['pixel_size_mm']),.05):
            raise ValueError('Unsupported energy or physical pixel spacing.')
        if str(bundle['layout'])!='straight_seam' or float(bundle['rotation_deg'])!=0:
            raise ValueError('This model requires unrotated straight seam inputs.')
        target=normalize_raw(bundle['target_raw'],read(norm['path']))
        if target.shape!=(64,64): raise ValueError('Expected a 64x64 raw target.')
        reference=np.zeros_like(target)
        if arm in ('median','cnn_median'):
            context=bundle['context_raw'];offsets=bundle['context_offsets_xy']
            if context.shape!=(4,64,64) or not np.array_equal(offsets,np.array([[-128,0],[-64,0],[64,0],[128,0]])):
                raise ValueError('Four target-excluded contiguous neighbour crops at the documented offsets are required.')
            reference=median_reference(normalize_raw(context,read(norm['path'])))
        origin=np.asarray(bundle['target_origin_xy'],dtype=int) if 'target_origin_xy' in bundle.files else np.zeros(2,dtype=int)
        if origin.shape!=(2,): raise ValueError('Invalid target origin.')
    spec=frozen['models'][arm]
    score=scores_for(dict(target=target[None],reference=reference[None]),spec)[0]
    mask=threshold_mask(score,**{k:spec['parameters'][k] for k in ('threshold','min_area','opening')})
    boxes=boxes_from_mask(mask)
    payload=dict(approach=arm,coordinates='half-open xyxy in pixels',boxes_xyxy=boxes,
        scene_boxes_xyxy=[[x0+int(origin[0]),y0+int(origin[1]),x1+int(origin[0]),y1+int(origin[1])] for x0,y0,x1,y1 in boxes])
    write(output,payload)
    canvas=cv2.cvtColor((target*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for x0,y0,x1,y1 in boxes: cv2.rectangle(canvas,(x0,y0),(x1-1,y1-1),(0,180,255),1)
    cv2.imwrite(str(Path(output).with_suffix('.png')),canvas)
    return payload
