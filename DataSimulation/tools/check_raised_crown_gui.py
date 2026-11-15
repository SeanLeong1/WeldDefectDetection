"""Hidden event-loop and actual renderer/inference integration checks."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from raised_crown.gui import create_app
from raised_crown import campaign as c

def main():
    app=create_app(ROOT/'DataSimulation/config');app.root.withdraw();p=app.raised_panel;checks={}
    try:
        app.root.update();assert app.notebook.select()==str(p)
        p.start(False);deadline=time.monotonic()+300
        while p.busy:
            app.root.update();time.sleep(.03)
            if time.monotonic()>deadline:raise TimeoutError('Raised example inspection timeout')
        assert p.status.get().startswith('Saved'),p.status.get();checks['example_inference']=p.status.get()
        p.kind.set('hidden');p.seed.set('2360500044');p.start(True);deadline=time.monotonic()+900
        while p.busy:
            app.root.update();time.sleep(.03)
            if time.monotonic()>deadline:raise TimeoutError('Raised generation/inference timeout')
        assert p.status.get().startswith('Saved'),p.status.get();checks['generated_hidden_control']=p.status.get()
        checks['historical_tabs']=[app.notebook.tab(tab,'text') for tab in app.notebook.tabs()]
        assert 'Historical recessed | 4 Oct' in checks['historical_tabs']
        checks['native_screenshot']=False
        checks['scope']='Hidden widget/event-loop and real render/inference/export checks, not manual visual signoff'
        c.write(c.OUT/'gui_checks.json',checks);print(json.dumps(checks))
    finally:app.root.destroy()

if __name__=='__main__':main()
