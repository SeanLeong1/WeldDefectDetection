from pathlib import Path
import cv2
import numpy as np
from box_comparison.campaign import read,verify_files
from box_comparison.core import write,threshold_mask,boxes_from_mask
from box_comparison.inference import normalize_raw
from box_refinement.methods import predicted_regions
from .methods import scores_for


def load_input(path,norm,arm):
    with np.load(path,allow_pickle=False) as item:
        if {'mask','labels','truth_boxes'} & set(item.files):raise ValueError('Labels are not inference inputs.')
        if float(item['energy_kev'])!=60 or not np.isclose(float(item['pixel_size_mm']),.05):raise ValueError('Unsupported physical domain.')
        if str(item['layout'])!='straight_seam' or float(item['rotation_deg'])!=0:raise ValueError('Unsupported geometry.')
        target=normalize_raw(item['target_raw'],norm)
        if target.shape!=(64,64):raise ValueError('Expected 64x64 target.')
        context=np.zeros((4,64,64),np.float32)
        if arm in ('median','cnn_median'):
            if item['context_raw'].shape!=(4,64,64) or not np.array_equal(item['context_offsets_xy'],[[-128,0],[-64,0],[64,0],[128,0]]):
                raise ValueError('Expected target-excluded non-overlapping context at fixed offsets.')
            context=normalize_raw(item['context_raw'],norm)
        origin=np.asarray(item['target_origin_xy'],dtype=int) if 'target_origin_xy' in item.files else np.zeros(2,dtype=int)
        if origin.shape!=(2,):raise ValueError('Invalid origin.')
    return dict(target=target[None],context=context[None],reference=np.median(context,axis=0)[None]),origin


def infer(root,input_path,output,arm):
    from .campaign import check,require
    root=Path(root);check(root);require(root,'freeze');frozen=read(root/'frozen_models.json')
    norm=frozen['normalization'];verify_files({norm['path']:norm['sha256']})
    data,origin=load_input(input_path,read(norm['path']),arm)
    spec=frozen['models'][arm];score=scores_for(data,spec)[0]
    if spec['implementation']=='historical':
        old=spec['spec']
        if old['implementation']=='previous':
            p=old['parameters'];mask=threshold_mask(score[0],p['threshold'],p['min_area'],p.get('opening',1));boxes=boxes_from_mask(mask)
        else:_,boxes=predicted_regions(score[0],**old['parameters'])
    else:_,boxes=predicted_regions(score[spec.get('box_channel',0)],**spec['parameters'])
    result=dict(approach=arm,coordinates='half-open xyxy pixels',boxes_xyxy=boxes,
        scene_boxes_xyxy=[[x0+int(origin[0]),y0+int(origin[1]),x1+int(origin[0]),y1+int(origin[1])] for x0,y0,x1,y1 in boxes])
    write(output,result)
    image=cv2.cvtColor((data['target'][0]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for x0,y0,x1,y1 in boxes:cv2.rectangle(image,(x0,y0),(x1-1,y1-1),(0,180,255),1)
    cv2.imwrite(str(Path(output).with_suffix('.png')),image)
    return result


def smoke(root):
    from src.data.temporal_window_dataset import TemporalWindowDataset
    from .campaign import ARMS
    dataset=root/'datasets/confirmation';ds=TemporalWindowDataset(dataset,'all')
    window=next(w for w in ds.windows if w['target_is_defective'])
    record=ds.record_map[window['target_sample_id']];ids=window['previous_sample_ids']+window['future_sample_ids']
    folder=root/'inference_smoke';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'input.npz',target_raw=np.load(dataset/record['raw_channels']['topo']),
        context_raw=np.stack([np.load(dataset/ds.record_map[s]['raw_channels']['topo']) for s in ids]),
        context_offsets_xy=np.array([[-128,0],[-64,0],[64,0],[128,0]]),target_origin_xy=np.array([record['crop_x0_px'],record['crop_y0_px']]),
        energy_kev=60.,pixel_size_mm=.05,layout='straight_seam',rotation_deg=0.)
    for arm in ARMS:
        result=infer(root,folder/'input.npz',folder/f'{arm}.json',arm)
        expected=next(r for r in read(root/'predictions'/f'{arm}.json') if r['window_id']=='confirmation:'+window['window_id'])
        if result['boxes_xyxy']!=expected['boxes_xyxy']:raise ValueError('Inference/evaluation mismatch.')
    write(folder/'verification.json',dict(status='ok',all_four_match=True,input_has_no_labels=True))
