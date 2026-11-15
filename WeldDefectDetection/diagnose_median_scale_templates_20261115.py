"""Check the user's numerical premises without using confirmation for selection."""
from pathlib import Path
import numpy as np
import median_scale_templates_20261115 as st
ex=st.ex

def historical():
    records=[]
    for campaign,name in [('median_localisation_20261110_v1','combined'),('median_region_growth_20261110_v1','previous')]:
        root=st.ROOT/'DataSimulation/data/output'/campaign
        index={r['scene_id']:r for r in st.read(root/'datasets/test/index.json')}
        for r in st.read(root/'confirmation'/f'{name}.json')['rows']:
            if r['count_multiplier']!=64:continue
            boxes=r['predicted_boxes'];truth=r['truth_boxes']
            ious=[ex.metrics.circular_iou(p,truth[0],5024) for p in boxes] if truth else [0.]*len(boxes)
            best=max(ious,default=0.);bestindex=int(np.argmax(ious)) if ious else -1
            # Strict one-to-one assignment consumes the highest-IoU region only.
            unmatched=[v for i,v in enumerate(ious) if not (best>=.5 and i==bestindex)]
            records.append(dict(scene_id=campaign+':'+r['scene_id'],amplitude=index[r['scene_id']]['amplitude'],
                truth=len(truth),predicted=len(boxes),tp=r['tp'],fp=r['fp'],fn=r['fn'],
                any_hit=int(best>0),overlap_regions=sum(v>0 for v in ious),
                strict_unmatched_with_overlap=sum(v>0 for v in unmatched),strict_unmatched_without_overlap=sum(v==0 for v in unmatched)))
    counts={k:sum(r[k] for r in records) for k in records[0] if k not in ('scene_id','amplitude')}
    counts['any_overlap_fp']=counts['predicted']-counts['any_hit']
    counts['any_overlap_f1']=2*counts['any_hit']/(counts['predicted']+counts['truth'])
    counts['any_overlap_recall']=counts['any_hit']/counts['truth']
    by={}
    for a in (0,.05,.1):
        rr=[r for r in records if r['amplitude']==a];tp,fp,fn=[sum(r[k] for r in rr) for k in ('tp','fp','fn')]
        by[str(a)]=dict(pore_acquisitions=sum(r['truth'] for r in rr),tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1))
    return dict(counts=counts,by_amplitude=by,rows=records,
      interpretation='Overlap establishes contact, not a correct pore boundary. Any-overlap F1 is a conditional attainable value with the existing prediction count, not a universal ceiling after merging or recovering missed pores.')

def robust_rows(d):
    med=np.median(d,axis=1,keepdims=True)
    return 1.4826*np.median(abs(d-med),axis=1)

def noise_and_structure():
    rr=[r for r in st.read(ex.OLD/'datasets/val/index.json') if r['source_class']=='clean']
    lookup={(r['background_id'],r['count_multiplier'],r['repeat']):r for r in rr};records=[]
    for i,r in enumerate(v for v in rr if v['repeat']==0):
        x0=ex.normalized(np.load(r['raw_path']));r1=lookup[(r['background_id'],r['count_multiplier'],1)]
        x1=ex.normalized(np.load(r1['raw_path']));noise=(x0-x1)/np.sqrt(2)
        with np.load(r['geometry_path']) as z:quiet=ex.normalized(z['quiet'])
        rec=dict(background_id=r['background_id'],count=r['count_multiplier'],amplitude=r['amplitude'],
            repeat_noise_MAD=float(np.median(robust_rows(noise[16:144]))),repeat_noise_RMS=float(np.sqrt(np.mean(noise[16:144]**2))))
        for ref in st.REFS:
            d=st.residual(x0,ref)[16:144];q=st.residual(quiet,ref)[16:144];spread=robust_rows(d)
            rec[ref]=dict(residual_MAD=float(np.median(spread)),residual_RMS=float(np.sqrt(np.mean(d*d))),
                quiet_residual_RMS=float(np.sqrt(np.mean(q*q))),floor_active_row_fraction=float(np.mean(spread<.003)),
                quiet_to_repeat_RMS=float(np.sqrt(np.mean(q*q))/max(rec['repeat_noise_RMS'],1e-12)))
        records.append(rec)
        if (i+1)%16==0:print('REFERENCE DIAGNOSTIC',i+1,'/96',flush=True)
    by={}
    for c in (1,4,64):
        rows=[r for r in records if r['count']==c]
        by[str(c)]={key:float(np.median([r[key] for r in rows])) for key in ('repeat_noise_MAD','repeat_noise_RMS')}
        for ref in st.REFS:by[str(c)][ref]={key:float(np.median([r[ref][key] for r in rows])) for key in rows[0][ref]}
    scale=np.asarray(ex.CAL_OLD['scale'])[16:144]
    return dict(scope='32 clean validation backgrounds, all counts. Shared sqrt normalization and input sigma1. Repeat difference divided by sqrt2; acquisition noise may be correlated. RMS and row-MAD are distinct diagnostics, not interchangeable variance estimates.',by_count=by,rows=records,
        original_static_floor_active_fraction=float(np.mean(scale<=.003+1e-9)),
        original_scale_min_median_max=[float(v) for v in (scale.min(),np.median(scale),scale.max())],
        original_threshold16_residual_min_median_max=[float(v) for v in (16*scale.min(),16*np.median(scale),16*scale.max())])

if __name__=='__main__':
    st.verify();h=historical();print('HISTORICAL COUNTS',h['counts'],flush=True)
    st.write(st.OUT/'premise_diagnostics.json',dict(historical=h,noise_structure=noise_and_structure()))
