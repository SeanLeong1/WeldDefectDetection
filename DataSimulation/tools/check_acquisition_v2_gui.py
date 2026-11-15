"""Hidden event-loop smoke test of the v2 GUI selection and inference."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from bse_renderer.gui import App
from inspection.gui import V2_CAMPAIGN


def main():
    out=ROOT/'data/audit_circular_current_20261004/historical_v2'; out.mkdir(parents=True,exist_ok=True)
    app=App(ROOT/'DataSimulation/config'); app.root.withdraw()
    checks={}
    try:
        app.root.update()
        panel=app.inspection_panel
        panel.campaign.set(next(k for k,v in panel.campaigns.items() if v==V2_CAMPAIGN));panel._campaign_changed()
        assert panel.campaigns[panel.campaign.get()]==V2_CAMPAIGN
        checks['explicit_historical_campaign']=str(V2_CAMPAIGN)
        assert 'Threshold baseline' not in panel.method_box.cget('values')
        panel.example(); panel.output.set(str(out/'gui_inspection')); panel.start()
        deadline=time.monotonic()+90
        while panel.busy:
            app.root.update(); time.sleep(.02)
            if time.monotonic()>deadline: raise TimeoutError('GUI inference timed out')
        app.root.update(); result=panel.result
        assert result and len(result['records'])==2
        assert all(r['status']=='complete' for r in result['records'])
        assert result['campaign']==str(V2_CAMPAIGN.resolve())
        checks['v2_methods']=[r['method'] for r in result['records']]
        checks['v2_results']=result['output']
        checks['preview_has_image']=bool(panel.preview.cget('image')); assert checks['preview_has_image']
        panel.campaign.set('Historical | straight crops | 26 Sep'); panel._campaign_changed()
        assert 'Threshold baseline' in panel.method_box.cget('values')
        checks['legacy_methods_preserved']=True
        app.v_preview_noise.set(True); noise=app._manual_sample()[3]
        assert noise.poisson_enabled and noise.thermal_enabled and noise.contamination_enabled
        app.v_preview_noise.set(False); noise=app._manual_sample()[3]
        assert not noise.poisson_enabled and not noise.thermal_enabled and not noise.contamination_enabled
        checks['preview_noise_switch_controls_flags']=True
        checks['visual_signoff']='Hidden event-loop and widget checks only; not a native screenshot or user visual sign-off.'
        (out/'gui_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        print(json.dumps(checks),flush=True)
    finally: app.root.destroy()


if __name__=='__main__': main()
