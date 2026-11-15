"""Unwrapping-origin sensitivity of fixed frozen CNNs, without label scoring.

Shifts are multiples of the pooling stride four, so they preserve pooling-grid
phase while changing the position of the inference-tile boundaries.
"""
import numpy as np,torch
from . import generation as g,training,evaluation


def run():
    torch.set_num_threads(4);frozen=evaluation.verify_freeze();full=g.load('test')
    data={k:v[:3] for k,v in full.items()};results=[];arrays={}
    specs=[next(c for c in frozen['cnn_candidates'] if c['variant']==v and c['seed']==g.SEEDS[0]) for v in ('fixed_negatives','larger_context')]
    for spec in specs:
        model=training.load_model(spec);reference=training.predict(data,model,spec['config'])
        for shift in (8,16,24):
            shifted=dict(x=np.roll(data['x'],shift,axis=-1),mask=np.roll(data['mask'],shift,axis=-1),meta=data['meta'])
            restored=np.roll(training.predict(shifted,model,spec['config']),-shift,axis=-1)
            difference=abs(restored-reference)
            results.append(dict(variant=spec['variant'],shift_columns=shift,mean_absolute_score_change=float(difference.mean()),
                maximum_absolute_score_change=float(difference.max()),p99_absolute_score_change=float(np.quantile(difference,.99))))
            arrays[f'{spec["variant"]}_{shift}']=difference.mean(axis=(0,1))
    np.savez_compressed(g.OUT/'diagnostics/tile_origin_arrays.npz',**arrays)
    g.write(g.OUT/'diagnostics/tile_origin_sensitivity.json',dict(rows=results,
        scenes=[r['scene_id'] for r in data['meta']],
        interpretation='Fixed first test background, three class variants, no threshold tuning and no ground-truth scoring. Same physical pixels with circular origin changed by pooling-stride multiples. Measures tiling sensitivity, not detection improvement by itself.'))
    print(results,flush=True)


if __name__=='__main__':run()
