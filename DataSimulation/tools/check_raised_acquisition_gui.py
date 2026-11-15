"""Hidden event-loop, label-free inference, and generation/export checks."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from raised_acquisition import campaign as c,gui


def wait(app,panel):
    deadline=time.monotonic()+900
    while panel.busy:
        app.root.update(); time.sleep(.03)
        if time.monotonic()>deadline: raise TimeoutError('Acquisition GUI check timed out')
    assert panel.status.get().startswith('Saved'),panel.status.get()


def main():
    app=gui.create_app(ROOT/'DataSimulation/config'); app.root.withdraw(); panel=app.acquisition_panel
    checks={}
    try:
        app.root.update(); assert app.notebook.select()==str(panel)
        for arm in c.ARMS:
            panel.arm.set(arm); panel.start(False); wait(app,panel); checks[arm]=panel.status.get()
        panel.kind.set('hidden'); panel.seed.set('2450600044'); panel.dwell.set('0.1'); panel.index.set('7')
        panel.start(True); wait(app,panel); checks['generation_at_low_dwell']=panel.status.get()
        checks['historical_tabs']=[app.notebook.tab(tab,'text') for tab in app.notebook.tabs()]
        assert 'Historical crown | 5 Oct' in checks['historical_tabs']
        assert 'Historical recessed | 4 Oct' in checks['historical_tabs']
        source=c.OUT/'datasets/test/test-000-open_intrinsic.npz'
        expected=gui.infer(source); meta=c.read(source.with_suffix('.json'))
        folder=c.OUT/'gui_label_free'; folder.mkdir(exist_ok=True); stripped=folder/'scene.npz'
        with np.load(source) as a: np.savez_compressed(stripped,raw=a['raw'],height=a['height'])
        meta['npz_sha256']=c.sha(stripped); c.write(stripped.with_suffix('.json'),meta)
        actual=gui.infer(stripped)
        assert actual['boxes']==expected['boxes']
        assert all(np.array_equal(actual['scores'][m],expected['scores'][m]) for m in ('cnn','median'))
        checks.update(labels_absent_identical_inference=True,native_screenshot=False,
            scope='Real render/inference/export and hidden widget event loop; exported figures inspected separately')
        c.write(c.OUT/'gui_checks.json',checks); print(checks)
    finally: app.root.destroy()


if __name__=='__main__': main()
