"""Paired background uncertainty for every declared localization condition."""
import numpy as np
from scipy.stats import beta
from . import campaign as c


def run():
    records=[r for r in c.read(c.OUT/'index.json') if r['pore_count']<=1]
    groups={}
    for meta in records:
        groups.setdefault(meta['background_id'],[]).append(c.read(c.OUT/'predictions_native'/(meta['scene_id']+'.json')))
    keys=sorted(groups);rng=np.random.default_rng(2240400099)
    draws=rng.integers(0,len(keys),size=(2000,len(keys)));results={}
    for method in ('cnn','median'):
        def counts(condition):
            return np.array([[sum(p[f'{condition}/{method}/best'][field] for p in groups[k]) for field in ('tp','fp','fn')] for k in keys])
        def f1(x):return 2*x[...,0]/np.maximum(2*x[...,0]+x[...,1]+x[...,2],1)
        nominal=counts('nominal');base=f1(nominal[draws].sum(axis=1))
        for error in c.PERTURBATIONS:
            current=counts(error['name']);delta=f1(current[draws].sum(axis=1))-base
            results[f'{error["name"]}/{method}']=dict(f1_difference=float(f1(current.sum(axis=0))-f1(nominal.sum(axis=0))),
                paired_percentile_95=np.quantile(delta,[.025,.975]).tolist())
            clean=[p[f'{error["name"]}/{method}/best'] for k in keys for p in groups[k] if p[f'{error["name"]}/{method}/best']['variant']=='clean']
            alarms=sum(r['predicted_regions']>0 for r in clean);n=len(clean)
            results[f'{error["name"]}/{method}']['clean_alarm_exact_95']=[
                0. if alarms==0 else float(beta.ppf(.025,alarms,n-alarms+1)),
                1. if alarms==n else float(beta.ppf(.975,alarms+1,n-alarms))]
    c.write(c.OUT/'paired_localisation.json',dict(backgrounds=len(keys),bootstrap_repeats=2000,
        scope='All declared conditions; primary frozen operating point; paired clean and single-pore views grouped by background. F1 differences use paired percentile bootstrap; clean-ring alarm rates use exact two-sided Clopper-Pearson intervals to avoid degenerate bootstrap bounds at zero/all alarms. Pointwise intervals, not simultaneous confidence bounds or a new model-selection rule.',results=results))


if __name__=='__main__':run()
