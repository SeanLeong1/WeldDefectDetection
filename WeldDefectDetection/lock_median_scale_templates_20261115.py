"""Lock component controls and all policies before any fresh data generation."""
from pathlib import Path
import median_scale_templates_20261115 as st

def main():
    st.verify();assert not (st.OUT/'datasets/test').exists()
    if (st.OUT/'frozen.json').exists():return st.verify(True)
    s=st.read(st.OUT/'selection.json');op=s['policies']['reference']['op']
    caches=[st.read(st.OUT/'validation_cache'/f"{r['scene_id']}.json") for r in st.read(st.ex.OLD/'datasets/val/index.json')]
    changes=dict(row_shift=dict(shift='row'),proportional_padding=dict(pad_fraction=.1),
        lower_floor=dict(score=op['score'].rsplit('_',1)[0]+'_0.001'),
        previous_reference=dict(score='short_adaptive_0.003'))
    for name,change in changes.items():
        newop={**op,**change};v=dict(op=newop,**st.ex.summary(st.evaluate(caches,newop)))
        s['policies']['control_'+name]=v;s['trials'].append(v)
    s['candidate_count']=len(s['trials']);st.write(st.OUT/'selection.json',s)
    p=st.read(st.OUT/'protocol.json');p['sources'][str(Path(__file__))]=st.sha(Path(__file__))
    p['locked_controls']=dict(utc=st.ex.utc(),changes=changes,purpose='Single-setting controls, not additional selection candidates')
    st.write(st.OUT/'protocol.json',p);st.freeze()

if __name__=='__main__':main()
