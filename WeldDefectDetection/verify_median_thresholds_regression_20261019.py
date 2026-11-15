"""Reproduce saved historical cases with the new comparison evaluator."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'WeldDefectDetection'))
import run_median_thresholds_20261019 as study


def main():
    frozen=study.verify();historical=study.read(study.original.OUT/'results.json')
    norm=study.read(study.original.OUT/'normalization.json')
    models={k:study.base.model_for(v) for k,v in frozen['cnns'].items()}
    wanted={f'test-{i:03d}-{kind}' for i in (0,8) for kind in ('clean','open_intrinsic')}
    index=[r for r in study.read(study.original.OUT/'datasets/test/index.json') if r['scene_id'] in wanted]
    assert len(index)==4
    records=[]
    for meta in index:
        assert study.sha(meta['npz_path'])==meta['npz_sha256']
        with np.load(meta['npz_path']) as z:image=study.base.normalize(z['raw'][0],norm);mask=z['mask'][16:144]
        scores={'median':study.base.median_image(image,frozen['median']['spec'])}
        scores.update({k:study.base.predict_image(image,m) for k,m in models.items()})
        for method,score in scores.items():
            oldkey='median' if method=='median' else historical['selected'][method]
            old=next(r for r in historical['results'][oldkey]['rows'] if r['scene_id']==meta['scene_id'])
            spec=frozen['median'] if method=='median' else frozen['cnns'][method]
            new=study.rows_for(score,mask,meta,spec['selection']['best'],True)['0.50']
            for key in ('tp','fp','fn','intersection','union','predicted_pixels','truth_pixels','predicted_regions'):
                assert new[key]==old[key],(meta['scene_id'],method,key,new[key],old[key])
            for key in ('predicted_boxes','truth_boxes'):assert sorted(new[key])==sorted(old[key])
            records.append(dict(scene_id=meta['scene_id'],method=method,status='exact_match',tp=new['tp'],fp=new['fp'],fn=new['fn']))
            print('MATCHED',meta['scene_id'],method,flush=True)
    study.write(study.OUT/'historical_regression_check.json',dict(status='passed',scenes=4,method_scene_checks=12,
        purpose='Check evaluator/inference consistency after observing lower fresh CNN scores; no parameter changes',records=records))


if __name__=='__main__':main()
