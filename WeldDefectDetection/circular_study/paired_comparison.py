"""Descriptive paired-background uncertainty for the frozen primary comparisons."""
import numpy as np
from . import generation as g,metrics,evaluation


def values(rows,keys):
    out=[]
    for key in keys:
        m=metrics.summary([r for r in rows if r['background_id']==key])
        out.append([m[k] for k in ('tp','fp','fn','clean_rings_flagged','clean_rings')])
    return np.asarray(out,float)


def rates(totals):
    tp,fp,fn,flagged,clean=totals.T
    return np.stack([2*tp/np.maximum(2*tp+fp+fn,1),tp/np.maximum(tp+fn,1),flagged/np.maximum(clean,1)],axis=1)


def run():
    frozen=evaluation.verify_freeze();r=g.read(g.OUT/'results.json')
    base=max([c for c in frozen['cnn_candidates'] if c['variant']=='fixed_negatives'],key=lambda c:metrics.key(c['selection']['best']['metrics']))
    baseline=f'cnn_fixed_negatives_{base["seed"]}'
    pairs=[('selected_cnn_minus_baseline',r['selected_cnn'],baseline),
           ('selected_median_minus_baseline',r['selected_median'],'median_legacy_scale'),
           ('selected_cnn_minus_selected_median',r['selected_cnn'],r['selected_median'])]
    output={}
    for name,a,b in pairs:
        aa=g.read(g.OUT/f'predictions/{a}_best.json');bb=g.read(g.OUT/f'predictions/{b}_best.json')
        assert {r['scene_id'] for r in aa}=={r['scene_id'] for r in bb}
        keys=sorted({r['background_id'] for r in aa});av=values(aa,keys);bv=values(bb,keys)
        indices=np.random.default_rng(100485001).integers(len(keys),size=(5000,len(keys)))
        deltas=rates(av[indices].sum(axis=1))-rates(bv[indices].sum(axis=1))
        observed=(rates(av.sum(axis=0)[None])-rates(bv.sum(axis=0)[None]))[0]
        output[name]=dict(a=a,b=b,backgrounds=len(keys),
            metrics={k:dict(difference=float(observed[i]),interval=np.quantile(deltas[:,i],[.025,.975]).tolist())
                     for i,k in enumerate(('f1','recall','clean_ring_fa'))})
    g.write(g.OUT/'paired_comparisons.json',dict(comparisons=output,
        interpretation='5000 paired whole-background bootstrap draws. Descriptive intervals conditional on frozen models and simulator; not multiplicity-adjusted tests, not seed variability, and not measured-weld generalisation evidence.'))
    print(output,flush=True)


if __name__=='__main__':run()
