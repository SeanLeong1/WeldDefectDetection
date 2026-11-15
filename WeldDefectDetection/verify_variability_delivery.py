"""Independent final checks of paired data, split isolation and label-free inference."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'DataSimulation'))
from variability_campaign.campaign import read,write,sha,load,ARMS,require
from infer_variability_boxes import infer


def verify(root):
    root=Path(root);torch.set_num_threads(4)
    for stage in ['prepare','pilot','development','train','confirmation','evaluate','report']:require(root,stage)
    norm=read(root/'normalization.json');dev=read(root/'datasets/development/index.json');confirmation=read(root/'datasets/confirmation/index.json')
    assert set(norm['scene_ids'])=={r['scene_id'] for r in dev if r['split']=='train'}
    assert not ({r['seed'] for r in dev}&{r['seed'] for r in confirmation})
    masks={};background_parameters={};class_counts={};hist=np.zeros(91,dtype=np.int64);clipping={};variants=0
    for r in dev+confirmation:
        assert sha(r['npz_path'])==r['npz_sha256']
        with np.load(r['npz_path']) as a:
            raw,mask=a['raw'],a['mask']
            assert raw.shape==mask.shape==(64,576)
            assert np.isfinite(raw).all() and np.isfinite(a['quiet']).all() and np.isfinite(a['relief']).all()
            assert np.max(np.abs(a['width_fraction']-1))<=r['amplitude']+1e-6
            assert np.max(np.abs(a['depth_fraction']-1))<=r['amplitude']+1e-6
            key=(r['background_id'],r['source_class'])
            if key in masks:np.testing.assert_array_equal(masks[key],mask);variants+=1
            else:masks[key]=mask.copy()
            if r['source_class']=='clean':assert not mask.any()
            else:assert mask[:,r['target_frame']*64:(r['target_frame']+1)*64].any()
            c=clipping.setdefault(r['split'],dict(pixels=0,below=0,above=0))
            c['pixels']+=raw.size;c['below']+=int((raw<norm['raw_low']).sum());c['above']+=int((raw>norm['raw_high']).sum())
        params={k:v for k,v in r['parameters'].items() if k!='defects'}
        key=r['background_id']
        if key in background_parameters:assert params==background_parameters[key]
        else:background_parameters[key]=params
        key=f'{r["split"]}:{r["amplitude"]}:{r["source_class"]}'
        class_counts[key]=class_counts.get(key,0)+1
    # The label-free entry point must match the evaluated boxes for every method
    # on clean, keyhole and intrinsic scenes at both nominal/stress settings.
    checks=[];folder=root/'inference_smoke';folder.mkdir(exist_ok=True)
    for cls in ('clean','keyhole_pore','intrinsic_pore'):
        for level in (0.,.2):
            r=next(r for r in confirmation if r['source_class']==cls and r['amplitude']==level)
            t=r['target_frame'];ix=[t-2,t-1,t+1,t+2]
            with np.load(r['npz_path']) as a:frames=np.stack([a['raw'][:,k*64:(k+1)*64] for k in range(9)])
            path=folder/f'{cls}_{level:.2f}.npz'
            np.savez_compressed(path,target_raw=frames[t],context_raw=frames[ix],pixel_size_mm=np.array(.05))
            for arm in ARMS:
                result=infer(root,path,arm)
                expected=next(p for p in read(root/'predictions'/f'noisy_{arm}.json') if p['window_id']==f'{r["scene_id"]}-w{t}')
                assert result['boxes_xyxy']==expected['boxes_xyxy']
                write(folder/f'{cls}_{level:.2f}_{arm}.json',result)
                checks.append(dict(scene=r['scene_id'],arm=arm,exact_box_match=True))
    result=dict(status='ok',scene_count=len(dev)+len(confirmation),independent_background_count=len(background_parameters),
                class_counts=class_counts,paired_level_mask_checks=variants,clipping=clipping,
                identical_acquisition_parameters_within_background=True,train_only_normalization=True,
                train_confirmation_seed_disjoint=True,label_free_checks=checks,
                scripts={str(Path(__file__).resolve()):sha(__file__),str(REPO/'WeldDefectDetection/infer_variability_boxes.py'):sha(REPO/'WeldDefectDetection/infer_variability_boxes.py')})
    write(root/'delivery_verification.json',result)
    # Re-seal the inventory after adding independent verification artifacts.
    inventory={str(p.resolve()):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json'}
    write(root/'FINAL_MANIFEST.json',dict(files=inventory,physical_validation=False,scope='Completed campaign and independent verification artifacts'))
    print(json.dumps(dict(status='ok',scenes=result['scene_count'],paired_mask_checks=variants,label_free_checks=len(checks),files=len(inventory))))


if __name__=='__main__':
    verify(REPO/'DataSimulation/data/output/variability_campaign_20260926_v1')
