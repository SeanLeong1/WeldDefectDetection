"""Package the completed report, settings, predictions and reproducibility scripts."""
from pathlib import Path
import csv,json,zipfile,hashlib
import median_exploration_20261115 as ex


def main():
    ex.verify(True)
    assert ex.read(ex.OUT/'final_audit.json')['status']=='passed'
    summary=ex.read(ex.REPORT/'results.json')
    with (ex.REPORT/'confirmation_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','count_multiplier','tp','fp','fn','precision','recall','f1','clean_alarms','clean_scenes'])
        w.writeheader()
        for method,result in summary['results'].items():
            for count,row in result['by_count'].items():w.writerow(dict(method=method,count_multiplier=count,**row))
    readme='''Median improvement experiment, 15 November 2026

Open REPORT.html for the complete illustrated report. Images are embedded in the HTML.
REPORT.md is the editable text version. results.json and confirmation_results.csv contain
the aggregate results; audit.json records the integrity checks.

The evidence/ folder contains frozen settings, the protocol, full confirmation
predictions and selected diagnostics. scripts/ contains the experiment and analysis
entry points. These scripts require the existing thesis Codebase2 checkout and its
.venv Python environment, historical datasets, frozen CNN checkpoint and dependencies.
The bundle is an evidence handoff, not a self-contained training environment.

Large raw image arrays and complete validation component caches remain in:
DataSimulation/data/output/median_exploration_20261115_v1

Use the frozen revised median on a compatible raw registered TOPO array:
.venv\\Scripts\\python.exe WeldDefectDetection\\apply_revised_median_20261115.py INPUT.npy --output PREDICTIONS.json

All experiments are synthetic. No measured-image validity or industrial deployment
claim is established. Renderer changes are sensitivity assumptions, not calibrated
physical corrections. Do not tune on the fresh confirmation outcomes.
'''
    (ex.REPORT/'README.txt').write_text(readme,encoding='utf-8')
    files={p.name:p for p in ex.REPORT.iterdir() if p.is_file() and p.suffix not in ('.zip',) and p.name!='bundle_manifest.json'}
    for name in ('protocol.json','frozen.json','calibration.json','selection.json','diagnostic.json','validation_ablations.json','reference_bias_diagnostic.json','reference_ghost_diagnostic.json','confirmation_median.json','confirmation_cnn.json','final_audit.json','historical_cnn_replay.json'):
        p=ex.OUT/name
        if p.exists():files['evidence/'+name]=p
    for name in ('protocol.json','results.json'):
        files['evidence/renderer_sensitivity/'+name]=ex.OUT/'renderer_sensitivity'/name
    for name in ('median_exploration_20261115.py','median_renderer_sensitivity_20261115.py','stream_median_confirmation_20261115.py','report_median_exploration_20261115.py','ablate_median_exploration_20261115.py','diagnose_median_reference_20261115.py','audit_median_exploration_20261115.py','apply_revised_median_20261115.py','package_median_exploration_20261115.py'):
        files['scripts/'+name]=ex.ROOT/'WeldDefectDetection'/name
    manifest={name:dict(bytes=p.stat().st_size,sha256=ex.sha(p)) for name,p in sorted(files.items())}
    ex.write(ex.REPORT/'bundle_manifest.json',manifest)
    target=ex.REPORT/'Median_Improvement_Experiment_20261115.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.write(ex.REPORT/'bundle_manifest.json','bundle_manifest.json')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,v in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==v['sha256']
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=ex.sha(target),files=len(files)+1)))


if __name__=='__main__':main()
