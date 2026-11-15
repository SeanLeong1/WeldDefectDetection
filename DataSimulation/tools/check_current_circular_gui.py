"""Hidden event-loop integration check; historical sources remain untouched."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from inspection.workbench import create_app
from inspection.registry import CURRENT,RASTER,LEGACY

def main():
    folder=ROOT/'data/audit_circular_current_20261004';folder.mkdir(parents=True,exist_ok=True)
    app=create_app(ROOT/'DataSimulation/config');app.root.withdraw();checks={}
    try:
        app.root.update();p=app.inspection_panel
        assert p.campaigns[p.campaign.get()]==CURRENT
        assert app.notebook.select()==str(app.current_panel)
        checks['current_default']=str(CURRENT)
        app.current_panel.load_reference();p.output.set(str(folder/'gui'));p.start()
        deadline=time.monotonic()+180
        while p.busy:
            app.root.update();time.sleep(.02)
            if time.monotonic()>deadline:raise TimeoutError('GUI inference timeout')
        assert p.result and len(p.result['records'])==2
        assert all(r['status']=='complete' for r in p.result['records']),p.result
        assert p.preview.cget('image');checks['current_inference']=p.result['output']
        for campaign in (RASTER,LEGACY):
            label=next(k for k,v in p.campaigns.items() if v==campaign);p.campaign.set(label);p._campaign_changed();p.example()
            assert Path(p.source.get()).exists()
        checks['historical_selections_preserved']=True
        app.current_panel.load_reference()
        app.current_panel.variant.set('pair_overlap');app.current_panel.generate()
        deadline=time.monotonic()+180
        while app.current_panel.busy:
            app.root.update();time.sleep(.02)
            if time.monotonic()>deadline:raise TimeoutError('Current GUI generation timeout')
        generated=Path(p.source.get());assert 'current_circular_examples' in generated.parts
        import numpy as np
        with np.load(generated) as a:
            assert a['raw'].shape==(3,128,5024) and len(a['instances'])==2
            assert np.array_equal(a['mask'],a['instances'].any(axis=0))
        checks['current_generation']=str(generated)
        p.start();deadline=time.monotonic()+180
        while p.busy:
            app.root.update();time.sleep(.02)
            if time.monotonic()>deadline:raise TimeoutError('Generated-example inference timeout')
        assert all(r['status']=='complete' for r in p.result['records']),p.result
        checks['generated_example_inference']=p.result['output']
        checks['current_contract']=p.contract_text.get()
        checks['tabs']=[app.notebook.tab(t,'text') for t in app.notebook.tabs()]
        checks['visual_signoff']='Hidden event-loop/widget and inference check; no native screenshot or user visual sign-off.'
        (folder/'gui_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks))
    finally:app.root.destroy()

if __name__=='__main__':main()
