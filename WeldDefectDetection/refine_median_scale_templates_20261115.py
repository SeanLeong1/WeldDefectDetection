"""Development-only amendment: revisit box settings after changing references.

The initial sequential search selected area8/no-merge on the original score.
This bounded refinement checks whether that choice suppresses interactions.
It is completed before freezing or creating any new confirmation image.
"""
from pathlib import Path
import itertools,json
import numpy as np
import median_scale_templates_20261115 as st

def main():
    st.verify();assert not (st.OUT/'frozen.json').exists();assert not (st.OUT/'datasets/test').exists()
    target=st.OUT/'refinement.json'
    if target.exists():return
    s=st.read(st.OUT/'selection.json');st.write(st.OUT/'initial_selection.json',s)
    protocol=st.read(st.OUT/'protocol.json')
    protocol['development_amendment']=dict(created_utc=st.ex.utc(),reason='Original-score box optimum uses area8/no merge; reassess interactions after reference/calibration changes.',
       search='Best initial score operating point within each of five reference/calibration branches; also short/near adaptive floor.003 threshold16. Re-evaluate all 72 box settings per distinct operating point.',
       confirmation_state='Not frozen; no fresh confirmation images exist',source_sha=st.sha(Path(__file__)))
    protocol['sources'][str(Path(__file__))]=st.sha(Path(__file__))
    st.write(st.OUT/'protocol.json',protocol)
    caches=[st.read(st.OUT/'validation_cache'/f"{r['scene_id']}.json") for r in st.read(st.ex.OLD/'datasets/val/index.json')]
    start=[]
    for prefix in ('long_adaptive','long_count','short_adaptive','near_adaptive','periodic_adaptive'):
        rr=[r for r in s['trials'] if r['op']['family']=='components' and r['op']['score'].startswith(prefix)]
        start.append(st.choose(rr)['op'])
    start.extend({**s['policies']['box_only']['op'],'score':f'{ref}_adaptive_0.003','threshold':16.} for ref in ('short','near'))
    starts={(p['score'],p['threshold']):p for p in start};trials=[]
    for (score,t),seed in starts.items():
        key=f'{score}:{t:g}'
        for gap,area in itertools.product((0,4,10),(3,8)):
            merged=[st.merge(c['components'][key],gap,area) for c in caches]
            for shift,pad,shape in itertools.product(('fixed','row'),(0,.1,.25),('union','width1.6')):
                op={**seed,'gap':gap,'area':area,'shift':shift,'pad_fraction':pad,'shape':shape}
                rows=[]
                for cache,boxes in zip(caches,merged):
                    # Area/gap already applied; transform without repeating unions.
                    pred=st.boxes_from_components([b+[1] for b in boxes],{**op,'gap':0,'area':1})
                    rows.append(st.ex.result_row(cache['meta'],cache['truth'],pred))
                trials.append(dict(op=op,**st.ex.summary(rows)))
        print('REFINED',score,t,'best',st.choose(trials)['macro_f1'],flush=True)
    alltrials=[r for r in s['trials'] if r['op']['family'] not in ('baseline','previous')]+trials
    p=s['policies']
    p['adaptive']=st.choose([r for r in alltrials if r['op']['family']=='components' and r['op']['score'].startswith('long_adaptive')])
    p['count_calibrated']=st.choose([r for r in alltrials if r['op']['family']=='components' and r['op']['score'].startswith('long_count')])
    p['reference']=st.choose([r for r in alltrials if r['op']['family']=='components' and r['op']['score'].split('_')[0] in ('short','near','periodic')])
    p['selected']=st.choose(alltrials)
    p['per_count']={str(c):st.choose(alltrials,c) for c in (1,4,64)}
    ref=p['reference']['op']
    for name,change in [('no_merge',dict(gap=0)),('merge10',dict(gap=10)),('fixed_shift',dict(shift='fixed')),('fixed_padding',dict(pad_fraction=0)),('area8',dict(area=8)),('long_reference',dict(score=ref['score'].replace(ref['score'].split('_')[0],'long',1)))]:
        op={**ref,**change};r=dict(op=op,**st.ex.summary(st.evaluate(caches,op)));p['ablation_'+name]=r;trials.append(r)
    st.write(target,dict(candidates=len(trials),trials=trials,completed_utc=st.ex.utc()))
    s['trials']+=trials;s['candidate_count']=len(s['trials']);s['policies']=p;s['amended_before_confirmation']=True
    st.write(st.OUT/'selection.json',s)
    print('FINAL DEVELOPMENT POLICIES',json.dumps(p),flush=True)

if __name__=='__main__':main()
