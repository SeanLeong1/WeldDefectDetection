from pathlib import Path
import cv2
import numpy as np
from box_comparison.core import median_reference,threshold_mask,boxes_from_mask,write
from box_comparison.inference import normalize_raw
from box_comparison.campaign import read,verify_files
from src.data.temporal_window_dataset import TemporalWindowDataset
from .methods import predicted_regions
from .training import scores_for


def infer(root,input_path,output,arm):
    from .campaign import check,require
    root=Path(root);check(root);require(root,'freeze');frozen=read(root/'frozen_models.json')
    n=frozen['normalization'];verify_files({n['path']:n['sha256']});norm=read(n['path'])
    with np.load(input_path,allow_pickle=False) as item:
        if {'mask','labels'} & set(item.files):raise ValueError('Labels are not inference inputs.')
        if float(item['energy_kev'])!=60 or not np.isclose(float(item['pixel_size_mm']),.05):raise ValueError('Unsupported physical settings.')
        if str(item['layout'])!='straight_seam' or float(item['rotation_deg'])!=0:raise ValueError('Unsupported weld geometry.')
        target=normalize_raw(item['target_raw'],norm)
        if target.shape!=(64,64):raise ValueError('Expected 64x64 target.')
        reference=np.zeros_like(target)
        if arm in ('median','cnn_median'):
            if item['context_raw'].shape!=(4,64,64) or not np.array_equal(item['context_offsets_xy'],[[-128,0],[-64,0],[64,0],[128,0]]):
                raise ValueError('Expected four correctly positioned non-overlapping neighbours.')
            reference=median_reference(normalize_raw(item['context_raw'],norm))
        origin=np.asarray(item['target_origin_xy'],dtype=int) if 'target_origin_xy' in item.files else np.zeros(2,dtype=int)
        if origin.shape!=(2,):raise ValueError('Invalid origin.')
    spec=frozen['models'][arm];score=scores_for(dict(target=target[None],reference=reference[None]),spec)[0]
    if spec['implementation']=='previous':
        mask=threshold_mask(score,**{k:spec['parameters'][k] for k in ('threshold','min_area','opening')});boxes=boxes_from_mask(mask)
    else:mask,boxes=predicted_regions(score,**spec['parameters'])
    result=dict(approach=arm,coordinates='half-open xyxy pixels',boxes_xyxy=boxes,
        scene_boxes_xyxy=[[x0+int(origin[0]),y0+int(origin[1]),x1+int(origin[0]),y1+int(origin[1])] for x0,y0,x1,y1 in boxes])
    write(output,result)
    image=cv2.cvtColor((target*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for x0,y0,x1,y1 in boxes:cv2.rectangle(image,(x0,y0),(x1-1,y1-1),(0,180,255),1)
    cv2.imwrite(str(Path(output).with_suffix('.png')),image)
    return result


def smoke(root):
    from .campaign import ARMS
    ds=TemporalWindowDataset(root/'confirmation','all')
    window=next(w for w in ds.windows if w['target_is_defective'])
    record=ds.record_map[window['target_sample_id']];neighbours=window['previous_sample_ids']+window['future_sample_ids']
    folder=root/'inference_smoke';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'input.npz',target_raw=np.load(root/'confirmation'/record['raw_channels']['topo']),
        context_raw=np.stack([np.load(root/'confirmation'/ds.record_map[s]['raw_channels']['topo']) for s in neighbours]),
        context_offsets_xy=np.array([[-128,0],[-64,0],[64,0],[128,0]]),
        target_origin_xy=np.array([record['crop_x0_px'],record['crop_y0_px']]),energy_kev=60.,pixel_size_mm=.05,layout='straight_seam',rotation_deg=0.)
    for arm in ARMS:
        got=infer(root,folder/'input.npz',folder/f'{arm}.json',arm)
        expected=next(r for r in read(root/'predictions'/f'{arm}.json') if r['window_id']==window['window_id'])
        if got['boxes_xyxy']!=expected['boxes_xyxy']:raise ValueError('Label-free inference and evaluation differ.')
    write(folder/'verification.json',dict(status='ok',all_four_match=True,input_has_no_labels=True,window_id=window['window_id']))
