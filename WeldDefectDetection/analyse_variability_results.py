"""Paired diagnostic contrasts from frozen predictions; no retuning."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'DataSimulation/data/output/variability_campaign_20260926_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def counts(rows,ids):
    return np.array([[sum(r[k] for r in rows if r['sequence_id']==sid) for k in ('tp','fp','fn')]+
        [sum(r['predicted_positive'] and not r['target_positive'] for r in rows if r['sequence_id']==sid),
         sum(not r['target_positive'] for r in rows if r['sequence_id']==sid)] for sid in ids])


def score(z):
    tp,fp,fn,fa,negative=np.moveaxis(z,-1,0)
    return np.stack([2*tp/np.maximum(2*tp+fp+fn,1),fa/np.maximum(negative,1)],axis=-1)


def analyse():
    noisy=read(ROOT/'predictions/noisy_cnn.json');quiet=read(ROOT/'predictions/quiet_cnn.json')
    ids=sorted({r['sequence_id'] for r in noisy});rng=np.random.default_rng(926990004)
    ix=rng.integers(len(ids),size=(10000,len(ids)));results={}
    pairs=[('noisy_10_minus_0',[r for r in noisy if r['amplitude']==.1],[r for r in noisy if r['amplitude']==0]),
           ('noisy_20_minus_0_exploratory',[r for r in noisy if r['amplitude']==.2],[r for r in noisy if r['amplitude']==0]),
           ('quiet_minus_noisy_at_10',[r for r in quiet if r['amplitude']==.1],[r for r in noisy if r['amplitude']==.1])]
    for name,left,right in pairs:
        a,b=counts(left,ids),counts(right,ids)
        difference=score(a[ix].sum(1))-score(b[ix].sum(1));point=score(a.sum(0))-score(b.sum(0))
        results[name]={key:dict(difference=float(point[j]),ci95=np.quantile(difference[:,j],[.025,.975]).tolist())
                       for j,key in enumerate(['box_f1','negative_crop_false_alarm_rate'])}
    (ROOT/'paired_robustness_changes.json').write_text(json.dumps(dict(draws=10000,clusters='latent_background',comparisons=results,
        scope='Post-evaluation descriptive contrasts of fixed predictions, no model or threshold changes; exploratory intervals.'),indent=2))
    a=results['noisy_10_minus_0']['box_f1'];n=results['quiet_minus_noisy_at_10']['box_f1']
    text=f'''# Interpretation of the completed experiment

The primary objective remains computer-vision detection of surface-expressed defects. The controlled geometry experiment measures tolerance to nuisance variation; it does not establish manufacturing acceptance limits.

1. **CNN detection persists under the trained variation range.** Noisy box F1 is 0.529 at zero added variation and 0.543 at 10%. Their paired difference is {a['difference']:.3f}, with a descriptive 95% background-bootstrap interval [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]. This does not establish an improvement or statistical equivalence. False alarms on entirely clean weld crops rise from 6.7% to 11.3%, so stable overall F1 should not be described as complete invariance.

2. **Noise remains a substantial limitation.** At 10% variation, applying the same frozen CNN and thresholds to matched noise-free images increases F1 from 0.543 to 0.800. The paired difference is {n['difference']:.3f}, interval [{n['ci95'][0]:.3f}, {n['ci95'][1]:.3f}]. Noise-free evaluation retains detector sensitivity mismatch and all geometric labels. It is not a newly trained model or a physical experiment.

3. **The pore families differ strongly.** At 10%, the CNN has keyhole-source box F1 0.853 and intrinsic-source box F1 0.327. These source-stratified scores include the negative windows within each source sequence; they exclude the separate clean-source sequences. They show that weak intrinsic pore response remains difficult.

4. **The implemented classical recipes fail on the noisy benchmark.** At 10%, median box F1 is 0.012, with false alarms on 95.1% of all negative target crops and 93.3% of clean-source crops. Its noise-free F1 is 0.374. The median implementation is a neighbouring median with row-profile alignment and a training-derived per-pixel residual MAD scale. The threshold implementation uses a row-profile-centred local high-pass residual with a training-derived scale. These are specific recipes and validation searches, not all possible median or threshold algorithms. The historical campaign included additional row-standardised candidates; its scores are not a directly comparable same-data control here. No baseline redesign or threshold tuning was performed after viewing confirmation results.

5. **The 20% condition is exploratory.** It exceeds the training variation range and six scenes in two backgrounds required an explicit constant-B extrapolation above 60 degrees. Maximum incidence was 61.545 degrees. Report it separately from the unchanged 0-10% primary-domain results.

6. **Spline interpolation merits consideration but does not resolve physical validity.** The optional spline gradients substantially reduce error on the known smooth analytic surface. On procedural welds they change normals by roughly one degree on average, while pilot image differences are only about 5-11% of the existing noise RMS. There is no measured normal-field reference to establish that spline-rendered welds are physically closer to a real specimen. The main CV experiment therefore retains finite differences and reports the spline experiment separately.

Given the remaining time, retain the trained CNN as the main result and the tested median method as a documented baseline. The defensible finding is moderate synthetic localisation performance, tolerance to the tested normal variation, and remaining sensitivity to noise, false alarms and weak intrinsic-pore signals. A claim that every acceptable weld is correctly accepted would contradict the false-alarm measurements.

The point results are in `results.json`; the paired contrasts are in `paired_robustness_changes.json`; data and inference integrity are recorded in `delivery_verification.json`.
'''
    (ROOT/'INTERPRETATION.md').write_text(text,encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__':analyse()
