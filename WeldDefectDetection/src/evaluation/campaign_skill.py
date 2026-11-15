"""Independent static-task skill assessment against prespecified simple masks."""
from pathlib import Path
import csv
import numpy as np
import cv2

def skill_intervals(counts, baseline_counts, truth, prediction, *, seed, iterations=2000):
    counts=np.asarray(counts,dtype=float)
    baseline_counts=np.asarray(baseline_counts,dtype=float)
    truth=np.asarray(truth,dtype=bool)
    prediction=np.asarray(prediction,dtype=bool)
    clean,defective=np.flatnonzero(~truth),np.flatnonzero(truth)
    if not len(clean) or not len(defective):
        raise ValueError('Skill validation requires clean and defective independent images.')
    rng=np.random.default_rng(seed)
    draws=[]
    def iou(x):
        total=x.sum(axis=0)
        return float(total[0]/max(total.sum(),1))
    for _ in range(iterations):
        indices=np.concatenate([rng.choice(clean,len(clean)),rng.choice(defective,len(defective))])
        ba=.5*(prediction[indices][truth[indices]].mean()+(~prediction[indices][~truth[indices]]).mean())
        score=iou(counts[indices])
        draws.append([ba,score,score-iou(baseline_counts[indices])])
    limits=np.quantile(draws,[.025,.975],axis=0)
    return {'balanced_accuracy_ci95':limits[:,0].tolist(),
            'iou_improvement_over_background_ci95':limits[:,1].tolist(),
            'iou_improvement_over_weld_region_ci95':limits[:,2].tolist(),
            'demonstrated_skill':bool(limits[0,0]>.5 and limits[0,1]>0 and limits[0,2]>0),
            'bootstrap_unit':'independent image, stratified by clean/defective', 'iterations':iterations}

def evaluate_campaign_skill(evaluation_dir: Path, *, seed: int):
    rows=list(csv.DictReader((evaluation_dir/'metrics.csv').open(newline='',encoding='utf-8')))
    counts,baselines,truth,prediction=[],[],[],[]
    for row in rows:
        mask=cv2.imread(row['mask_path'],cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read confirmation mask {row['mask_path']}")
        mask=mask>0
        if int(mask.sum()) != int(row['target_positive_pixels']):
            raise ValueError('Skill baseline requires the campaign native-resolution mask contract.')
        weld=np.zeros_like(mask)
        weld[3*mask.shape[0]//8:5*mask.shape[0]//8,:]=True
        counts.append([float(row[k]) for k in ('tp','fp','fn')])
        baselines.append([np.sum(weld & mask),np.sum(weld & ~mask),np.sum(~weld & mask)])
        truth.append(bool(mask.any()))
        prediction.append(row['predicted_is_defective'].lower()=='true')
    result=skill_intervals(counts,baselines,truth,prediction,seed=seed)
    return {'status':'ok','evaluation_dir':str(evaluation_dir),'primary_task':'static',
            'baseline':'all-background and central horizontal quarter-height weld region',**result}
