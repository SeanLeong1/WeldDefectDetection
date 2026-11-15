"""Explain frozen validation selection without selecting again on test outcomes."""
import numpy as np
import torch
from . import campaign as c,learning as learn
from circular_study import metrics


def run():
    output=c.OUT/'selection_diagnostic.json'
    if output.exists():return c.read(output)
    torch.set_num_threads(4);frozen=learn.verify_freeze();data=learn.load('val')
    scores=learn.predict(data,learn.model_for(frozen['cnn']));op=frozen['cnn']['selection']['best']
    rows=metrics.evaluate(data,scores,op,retain=True)
    measured=metrics.summary(rows)
    assert measured==op['metrics'],'Frozen validation replay differed'
    backgrounds=sorted([dict(background_id=bg,metrics=metrics.summary([x for x in rows if x['background_id']==bg]))
                        for bg in sorted({x['background_id'] for x in rows})],key=lambda x:-x['metrics']['fp'])
    worst=backgrounds[0]['background_id'];index=next(i for i,m in enumerate(data['meta']) if m['background_id']==worst and m['source_class']=='clean')
    meta=data['meta'][index];norm=c.read(c.OUT/'normalization.json')
    with np.load(meta['npz_path']) as a:raw=a['raw'][0,16:144].copy()
    train=c.read(c.OUT/'datasets/train/index.json');features={}
    for group in ('beam','detector'):
        for key,value in meta['parameters'][group].items():
            if not isinstance(value,(int,float)):continue
            values=[x['parameters'][group][key] for x in train if x['source_class']=='clean']
            features[group+'.'+key]=dict(value=value,training_min=min(values),training_max=max(values),outside_training_range=value<min(values) or value>max(values))
    result=dict(selected_seed=frozen['cnn']['seed'],validation_replay_exact=True,metrics=measured,backgrounds=backgrounds,
        worst_clean_scene=meta['scene_id'],high_clip_fraction=float(np.mean(raw>norm['high'])),features=features,rows=rows,
        selection_effect='None; original validation choice remains frozen despite confirmation differences',
        limit='Post-freeze descriptive diagnosis. Extreme features or clipping associations do not identify a causal mechanism.')
    c.write(output,result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask=metrics.binary(scores[index],op['threshold'],op['min_area'],op['closing'])[0]
    fig,axes=plt.subplots(4,1,figsize=(14,6),layout='constrained',sharex=True)
    for ax,title,value in zip(axes,('Raw TOPO','Normalized TOPO','CNN score','False predicted regions'),(raw,data['x'][index,0,16:144],scores[index],mask)):
        im=ax.imshow(value,origin='lower',aspect='auto',extent=(0,360,-3.2,3.2));ax.set_ylabel(title);fig.colorbar(im,ax=ax,shrink=.8)
    axes[-1].set_xlabel('Angle (degrees)');fig.suptitle(f'Post-freeze validation diagnosis: {meta["scene_id"]}, a clean ring')
    fig.savefig(c.ROOT/'docs/figures/raised_crown_results_20261005/validation_false_alarms.png',dpi=150);plt.close(fig)
    print('Frozen validation replay:',measured,'worst background:',worst,flush=True)
    return result
