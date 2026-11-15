"""Verify, document and package the completed study without rewriting history."""
from pathlib import Path
import shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from raised_acquisition import campaign as c, learning as learn


def publish():
    result=c.read(c.OUT/'results.json'); frozen=learn.verify_freeze(); archive=ROOT/'docs/archive'; archive.mkdir(exist_ok=True)
    stress=c.read(c.OUT/'robustness_results.json')['groups'][frozen['selected_arm']]
    targets={'PROJECT_CURRENT':ROOT/'docs/PROJECT_CURRENT.md','README':ROOT/'README.md',
        'DataSimulation_README':ROOT/'DataSimulation/README.md','WeldDefectDetection_README':ROOT/'WeldDefectDetection/README.md',
        'PROJECT_PIPELINE_CURRENT':ROOT/'DataSimulation/PROJECT_PIPELINE_CURRENT.md','launch_weld_studio':ROOT/'launch_weld_studio.py'}
    for label,path in targets.items():
        backup=archive/f'{label}_before_acquisition_20261006{path.suffix}'
        if path.exists() and not backup.exists(): shutil.copy2(path,backup)
    rows=[]
    for name,label in [(result['selected']['coverage'],'Coverage CNN'),(result['selected']['hard_negative'],'Hard-negative CNN'),
        ('median','New median'),('frozen_v5_cnn','Frozen 5 Oct CNN on the same new data'),('frozen_v5_median','Frozen 5 Oct median on the same new data')]:
        m=result['results'][name]['metrics']; rows.append(f'| {label} | {m["f1"]:.3f} | {m["recall"]:.3f} | {m["clean_rings_flagged"]}/{m["clean_rings"]} |')
    current='''# Current project: circular crown acquisition study, 6 October 2026

The latest completed experiment is `raised_acquisition_20261006_v1`. It addresses the low-dwell acquisition gap and tests training-only hard negatives, using the unchanged circular crown, surface renderer, CNN architecture and neighbouring-median comparator.

The model remains an 80 mm circular weld with a single-peaked 3 mm crown, a solid fusion region reaching 7 mm below the original surface, a 2.8 mm fusion width at that surface and 1.2 mm width at 5 mm depth. The crown footprint remains an assumption. This is a synthetic surface-response study, not a calibrated electron-transport or welding-fluid simulation. No researched electron-depth cutoff was implemented.

## Completed evidence

There are 396 new scenes across 84 independent backgrounds: 24 training, 24 validation, 24 confirmation and 12 stress backgrounds. Each split covers four dwell strata from 0.1 to 20 microseconds. Every paired scene condition remains with its background. Three CNN seeds per arm use matching initialization and positive samples, with the same optimizer-update budget. Confirmation was generated after both arms and median settings were frozen.

Fresh balanced-dwell confirmation:

| Method | Box F1 | Recall | Clean rings flagged |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

The GUI default is **{frozen['selected_arm']}**, selected using validation before confirmation. Read paired intervals, seed variability and recall alongside the point scores. The challenge deliberately oversamples low dwell and does not estimate natural deployment prevalence.

**Localisation remains a limitation.** On the same separate 84-scene robustness set, this model scores {stress['nominal']['metrics']['f1']:.3f} F1 with nominal coordinates and {stress['image_estimated']['metrics']['f1']:.3f} with image-estimated coordinates. The main-confirmation score is not an end-to-end unknown-image result. The next experiment is localisation-aware training and validation with fresh confirmation, keeping this campaign frozen.

- [Full results, paired uncertainty and method changes](RAISED_ACQUISITION_RESULTS_20261006.md)
- [Prospective experiment protocol](RAISED_ACQUISITION_PROTOCOL_20261006.md)
- [Unchanged geometry, CNN layers and median method](RAISED_CROWN_METHODS_20261005.md)
- [Historical 5 October raised-crown results](RAISED_CROWN_RESULTS_20261005.md)
- [SEM/BSE depth and EBW pore research](SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md)
- [Previous current-project guide](archive/PROJECT_CURRENT_before_acquisition_20261006.md)

## Run the GUI

Use `Launch Weld Vision.cmd` or `launch_weld_studio.py`. After audit and GUI checks pass, the first tab is **Acquisition study | 6 Oct**. It exposes the two frozen CNN arms, the unchanged median comparator, dwell time and layout controls. Generated examples stay outside research splits. The tab displays actual height, TOPO and full-ring predictions; inference never reads truth masks. Evaluation examples are chosen by the nearest available dwell, not by score.

The previous crown workflow remains **Historical crown | 5 Oct**. Older recessed circular and straight-weld workflows are retained on their historical tabs. The explicit launchers `launch_raised_acquisition.py` and `launch_raised_crown.py` remain available.

## Reproduce

```powershell
.venv\\Scripts\\python.exe WeldDefectDetection/run_raised_acquisition_20261006.py all
.venv\\Scripts\\python.exe WeldDefectDetection/complete_raised_acquisition_20261006.py
```

Full replay requires the research workspace and tested Python environment. Source snapshots, scene checksums, histories, mining coordinates, selection records and predictions are in `DataSimulation/data/output/raised_acquisition_20261006_v1/`. The portable inference bundle includes a limited example set, reports and frozen models, not all research data or installed dependencies.

The latest acquisition bundle is recorded in `data/releases/CURRENT_ACQUISITION_RELEASE.json`. The 5 October `CURRENT_RAISED_RELEASE.json` and 4 October `CURRENT_RELEASE.json` are deliberately preserved as historical records.

## History and limits

| Evidence stream | Status |
|---|---|
| September straight-weld campaigns | Historical, unchanged |
| 4 October recessed circular study and robustness | Historical, unchanged |
| Raised-crown development v1 to v4 | Preserved pilot/development history |
| 5 October raised-crown v5 | Frozen reference, unchanged |
| 6 October acquisition v1 | Latest completed matched training experiment |

Open surface pores, prescribed volume-displacement bulges and enclosed/no-surface-response controls remain distinct. A buried pore does not automatically imply a surface bump. One height peak may create two bright TOPO slope bands. Physical claims still require measured crown profiles, detector/acquisition calibration and independent real images. Unfamiliar weld profiles remain deferred.

Further training changes require a new campaign and fresh confirmation. Do not replace selected models or thresholds using the now-observed confirmation results.
'''
    (ROOT/'docs/PROJECT_CURRENT.md').write_text(current,encoding='utf-8')
    (ROOT/'README.md').write_text('''# Weld Vision: circular crown acquisition study

The latest completed study covers low-dwell acquisition and compares uniform versus hard-negative CNN training, alongside the neighbouring-median method. The 3 mm raised circular crown and 7 mm solid fusion geometry remain unchanged. Historical campaigns, models and releases are preserved.

Double-click **Launch Weld Vision.cmd**, or run:

```powershell
.venv\\Scripts\\python.exe launch_weld_studio.py
```

Start with [current results and version history](docs/PROJECT_CURRENT.md), [the new experiment results](docs/RAISED_ACQUISITION_RESULTS_20261006.md), [the prospective protocol](docs/RAISED_ACQUISITION_PROTOCOL_20261006.md), and [the chronological CNN/median guide](docs/RAISED_CROWN_METHODS_20261005.md).

The first tab opens the acquisition experiment after its checks pass. Earlier raised-crown and recessed workflows remain on labelled historical tabs. Inference requires the declared generated raw TOPO contract, not arbitrary microscope images: 160 x 5024 annuli, 0.05 mm radial pitch and 40 mm nominal radius. Truth labels do not enter inference.

This is synthetic evidence. TOPO is detector contrast, not surface height; the GUI shows both. No calibrated SEM penetration cutoff or welding-fluid model has been added. Reported false alarms and uncertainty matter as much as F1.

The newest bundle pointer is `data/releases/CURRENT_ACQUISITION_RELEASE.json`; earlier pointers remain historical. Reproduction commands and scientific limits are in the current-project guide. The [previous README is preserved](docs/archive/README_before_acquisition_20261006.md).
''',encoding='utf-8')
    for label in ('DataSimulation_README','WeldDefectDetection_README','PROJECT_PIPELINE_CURRENT'):
        path=targets[label]; backup=archive/f'{label}_before_acquisition_20261006.md'
        path.write_text('''# Current workflow: circular acquisition study, 6 October 2026

The newest completed workflow is the acquisition-coverage and hard-negative experiment. See [current status](../docs/PROJECT_CURRENT.md) and [results](../docs/RAISED_ACQUISITION_RESULTS_20261006.md). Run the repository-root `launch_weld_studio.py` for the new tab and preserved historical workflows.

The material below is preserved historical documentation and does not supersede the current-project guide.

---

'''+backup.read_text(encoding='utf-8'),encoding='utf-8')
    launcher='''"""Launch the newest verified workflow while preserving historical tabs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
if __name__=='__main__':
    from raised_crown.campaign import OUT as crown,read
    from raised_acquisition.campaign import OUT as acquisition
    def ready(folder):
        return ((folder/'audit.json').exists() and (folder/'gui_checks.json').exists()
            and read(folder/'audit.json').get('status')=='passed')
    if ready(acquisition):
        from raised_acquisition.gui import create_app
    elif ready(crown):
        from raised_crown.gui import create_app
    else:
        from inspection.workbench import create_app
    create_app(ROOT/'DataSimulation/config').run()
'''
    (ROOT/'launch_weld_studio.py').write_text(launcher,encoding='utf-8')
    c.write(c.OUT/'publication.json',dict(current_files={str(p):c.sha(p) for p in targets.values()},
        historical_document_copies={str(p):c.sha(p) for p in archive.glob('*before_acquisition_20261006.*')}))


def main():
    from raised_acquisition.reporting import report
    from raised_acquisition.quality import run as quality
    c.audit(); quality(); report()
    check=ROOT/'DataSimulation/tools/check_raised_acquisition_gui.py'
    if not (c.OUT/'gui_checks.json').exists(): subprocess.run([sys.executable,str(check)],cwd=ROOT,check=True)
    publish()
    sources=list((ROOT/'WeldDefectDetection/raised_acquisition').glob('*.py'))
    sources+=list((ROOT/'WeldDefectDetection').glob('*raised_acquisition*.py'))
    sources+=[ROOT/'DataSimulation/tools'/x for x in ('check_raised_acquisition_gui.py','package_raised_acquisition.py')]
    sources+=[ROOT/'WeldDefectDetection/tests/test_raised_acquisition.py',ROOT/'launch_raised_acquisition.py',ROOT/'launch_weld_studio.py']
    with zipfile.ZipFile(c.OUT/'analysis_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for source in sources: z.write(source,source.relative_to(ROOT))
    c.write(c.OUT/'analysis_sources.json',{str(p):c.sha(p) for p in sources})
    subprocess.run([sys.executable,str(ROOT/'DataSimulation/tools/package_raised_acquisition.py')],cwd=ROOT,check=True)
    c.audit()
    record=dict(complete=True,physical_validation=False,historical_artifacts_unchanged=True,
        artifacts={p.name:c.sha(p) for p in c.OUT.glob('*.json') if p.name!='delivery_manifest.json'},
        release=c.read(ROOT/'data/releases/CURRENT_ACQUISITION_RELEASE.json'))
    c.write(c.OUT/'delivery_manifest.json',record); print('ACQUISITION DELIVERY COMPLETE',flush=True)


if __name__=='__main__': main()
