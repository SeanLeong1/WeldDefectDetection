"""Descriptive post-freeze error analysis; no model or data changes."""
from . import campaign as c
from circular_study import metrics


def run():
    results=c.read(c.OUT/'results.json')
    quality={x['scene_id']:x for x in c.read(c.OUT/'image_quality.json')['records']}
    metadata={x['scene_id']:x for x in c.read(c.OUT/'datasets/test/index.json')}
    output={}
    for method in (results['selected'],'median'):
        rows=results['results'][method]['rows']
        backgrounds=[]
        for bg in sorted({x['background_id'] for x in rows}):
            group=[x for x in rows if x['background_id']==bg]
            clean=next(x for x in group if x['source_class']=='clean')
            q=quality[clean['scene_id']];meta=metadata[clean['scene_id']]
            backgrounds.append(dict(background_id=bg,metrics=metrics.summary(group),clean_false_regions=clean['fp'],
                high_clip_fraction=q['high_clip_fraction'],noise_rms=q['noise_rms'],
                amplitude=meta['amplitude'],beam=meta['parameters']['beam'],detector=meta['parameters']['detector']))
        positive=[dict(x,**{k:quality[x['scene_id']][k] for k in ('diameter_mm','radial_offset_mm','signal_in_mask','noise_rms')})
                  for x in rows if x['truth_pixels']>0]
        by_offset={str(off):metrics.summary([x for x in positive if x['radial_offset_mm']==off])
                   for off in sorted({x['radial_offset_mm'] for x in positive})}
        by_size={f'{lo:.3f}-{hi:.3f}':metrics.summary([x for x in positive if lo<=x['diameter_mm']<hi])
                 for lo,hi in ((.5,2/3),(2/3,5/6),(5/6,1.000001))}
        output[method]=dict(backgrounds=sorted(backgrounds,key=lambda x:-x['metrics']['fp']),by_offset=by_offset,
            by_size=by_size,
            without_hidden_control=metrics.summary([x for x in rows if x['source_class']!='hidden']),
            missed_surface_scenes=[x for x in positive if x['fn']>0],
            hidden_rings_flagged=results['results'][method]['hidden_rings_flagged'])
    c.write(c.OUT/'failure_analysis.json',dict(models=output,
        selection_effect='None; descriptive confirmation analysis only',
        limits='Associations are not causal estimates. Geometry, acquisition and pore factors are not a fully crossed design. A geometric surface label need not be observable under the declared detector response.'))
    return output
