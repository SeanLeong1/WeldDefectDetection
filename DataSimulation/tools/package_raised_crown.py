"""Portable inference release, preserving all three historical model choices."""
from pathlib import Path
import json,zipfile,subprocess,sys,uuid,shutil,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from raised_crown import campaign as c

def main():
    assert c.read(c.OUT/'audit.json')['status']=='passed'
    assert (c.OUT/'gui_checks.json').exists()
    previous=c.read(ROOT/'data/releases/CURRENT_RELEASE.json')
    assert c.sha(previous['archive'])==previous['archive_sha256']
    destination=ROOT/'.tmp'/('raised_release_'+uuid.uuid4().hex[:8]);destination.mkdir(parents=True)
    with zipfile.ZipFile(previous['archive']) as z:
        # The known, checksummed local archive has only workspace-relative files.
        for name in z.namelist():
            resolved=(destination/name).resolve()
            assert resolved.is_relative_to(destination.resolve())
        z.extractall(destination)
    old_manifest=destination/'RELEASE_MANIFEST.json'
    if old_manifest.exists():
        preserved=destination/'docs/archive/RELEASE_MANIFEST_20261004_historical.json'
        preserved.parent.mkdir(parents=True,exist_ok=True);old_manifest.rename(preserved)
    files=list((ROOT/'WeldDefectDetection/raised_crown').glob('*.py'))
    files+=list((ROOT/'WeldDefectDetection').glob('run_raised*.py'))
    files += [ROOT/'WeldDefectDetection/complete_raised_delivery_20261005.py']
    files += [ROOT/'launch_raised_crown.py',ROOT/'launch_weld_studio.py',ROOT/'docs/PROJECT_CURRENT.md',ROOT/'README.md']
    files += [ROOT/module/'README.md' for module in ('DataSimulation','WeldDefectDetection')]
    files += [ROOT/module/'requirements.txt' for module in ('DataSimulation','WeldDefectDetection')]
    files += [ROOT/'DataSimulation/PROJECT_PIPELINE_CURRENT.md']
    files += [ROOT/'DataSimulation/tools'/name for name in ('check_raised_crown_gui.py','package_raised_crown.py')]
    files += list((ROOT/'docs/archive').glob('*before_raised_20261005.*'))
    files += list((ROOT/'docs').glob('RAISED_CROWN*.md'))+[ROOT/'docs/SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md']
    files += list((ROOT/'docs/figures/raised_crown_results_20261005').glob('*.png'))
    files += list((ROOT/'docs/figures/raised_crown_plan_20261005').glob('*'))
    files += [c.OUT/n for n in ('normalization.json','protocol.json','cv_protocol.json','results.json','robustness_results.json','audit.json','gui_checks.json','dataset_manifest.json','sources.zip')]
    files += [c.OUT/'pilot/gates.json']
    files += [c.OUT/'traced_sample.json',c.OUT/'traced_sample.npz']
    files += [c.OUT/'small_pore_signal_sampling.json',c.OUT/'small_pore_signal_sampling.npz']
    files += [c.OUT/'selection_diagnostic.json']
    files += [c.OUT/'dwell_diagnostic.json']
    files += [c.OUT/'analysis_sources.json',c.OUT/'analysis_sources.zip']
    files += list((c.OUT/'execution_logs').glob('*.json'))
    files += [c.OUT/n for n in ('small_pore_diagnostic.json','image_quality.json','failure_analysis.json','project_test_checks.json','development_reuse.json','paired_profile/results.json')]
    for kind in ('clean','open_keyhole','open_intrinsic','bulge','hidden'):
        files += [c.OUT/f'datasets/test/test-000-{kind}.{extension}' for extension in ('npz','json')]
    frozen=c.read(c.OUT/'frozen_models.json')
    files += [Path(s['checkpoint']) for s in frozen['cnn_candidates']]
    for source in files:
        if source.is_file():
            target=destination/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    newout=destination/c.OUT.relative_to(ROOT)
    c.write(newout/'research_frozen_models.json',frozen)
    portable=copy.deepcopy(frozen);portable['portable']=True
    portable['verification_scope']='Bundled inference artifacts; original full research freeze retained separately'
    for spec in [portable['cnn'],*portable['cnn_candidates']]:spec['checkpoint']=Path(spec['checkpoint']).relative_to(ROOT).as_posix()
    required=[destination/Path(s['checkpoint']) for s in portable['cnn_candidates']]
    required += [newout/'normalization.json']
    required += list((destination/'WeldDefectDetection/raised_crown').glob('*.py'))
    required += [destination/Path(path).relative_to(ROOT) for path in frozen['files'] if Path(path).suffix=='.py']
    portable['files']={p.relative_to(destination).as_posix():c.sha(p) for p in set(required)}
    c.write(newout/'frozen_models.json',portable)
    code="import sys,json;from pathlib import Path;sys.path[:0]=['DataSimulation','WeldDefectDetection'];from raised_crown.gui import infer;from raised_crown.campaign import OUT;from inspection.engine import FrozenInspector;from inspection.registry import CURRENT,RASTER,LEGACY;x=infer(OUT/'datasets/test/test-000-open_keyhole.npz');assert all(Path(OUT).is_relative_to(Path.cwd()) for _ in [0]);[FrozenInspector(p) for p in (CURRENT,RASTER,LEGACY)];print(json.dumps(x['boxes']))"
    run=subprocess.run([sys.executable,'-c',code],cwd=destination,text=True,capture_output=True,timeout=300)
    if run.returncode:raise RuntimeError(run.stderr)
    from raised_crown.gui import infer
    expected=infer(c.OUT/'datasets/test/test-000-open_keyhole.npz')['boxes']
    actual=json.loads(run.stdout.strip().splitlines()[-1]);assert actual==expected
    # Test relocated generation and inference, with all imports under the package.
    gen="import sys,json;sys.path[:0]=['DataSimulation','WeldDefectDetection'];from raised_crown.gui import generate,infer;p=generate(2360500099,'open_intrinsic');x=infer(p);print(json.dumps({'path':str(p),'boxes':x['boxes']}))"
    run=subprocess.run([sys.executable,'-c',gen],cwd=destination,text=True,capture_output=True,timeout=900)
    if run.returncode:raise RuntimeError(run.stderr)
    relocated=json.loads(run.stdout.strip().splitlines()[-1])
    c.verify_sources()
    reference_record=c.OUT/'release_reference.json'
    if reference_record.exists():
        reference=c.read(reference_record);assert c.sha(reference['path'])==reference['npz_sha256']
        assert reference['protocol_sha256']==c.sha(c.OUT/'protocol.json')
    else:
        from raised_crown.gui import generate
        path=generate(2360500099,'open_intrinsic')
        reference=dict(path=str(path),npz_sha256=c.sha(path),protocol_sha256=c.sha(c.OUT/'protocol.json'))
        c.write(reference_record,reference)
    with np.load(reference['path']) as expected_arrays,np.load(relocated['path']) as actual_arrays:
        assert set(expected_arrays.files)==set(actual_arrays.files)
        assert all(np.array_equal(expected_arrays[k],actual_arrays[k]) for k in expected_arrays.files),'Relocated generation differed'
    (destination/'BUNDLE_README.md').write_text('''# Raised-crown inference bundle

This bundle contains the selected raised-crown models, examples, source and reports, plus the historical model choices. The complete research datasets and Python dependencies are not bundled.

Use an existing compatible Python environment and run `python launch_weld_studio.py` from this folder. The workspace tested-version snapshot is `requirements-studio-tested.txt`; dependency lists are in `requirements-studio.txt`. Relocation checks used the existing environment, not a fresh dependency installation.

Start with `docs/PROJECT_CURRENT.md`. The current file checksums are in `RAISED_RELEASE_MANIFEST.json`. The previous release manifest is retained under `docs/archive/RELEASE_MANIFEST_20261004_historical.json` as a record of the original historical archive, not as checksums for this combined bundle.

This is a synthetic research release. Consult the confirmation scores and false alarms before interpreting predictions. No measured SEM calibration or manufacturing acceptance is established.
''',encoding='utf-8')
    allfiles=[p for p in destination.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest={p.relative_to(destination).as_posix():c.sha(p) for p in allfiles}
    c.write(destination/'RAISED_RELEASE_MANIFEST.json',dict(files=manifest,physical_validation=False))
    archive=ROOT/'data/releases'/f'WeldVision_RaisedCrown_20261005_{uuid.uuid4().hex[:8]}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in allfiles+[destination/'RAISED_RELEASE_MANIFEST.json']:z.write(p,p.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in manifest.items())
    record=dict(archive=str(archive),archive_sha256=c.sha(archive),files=len(allfiles)+1,relocated_checkout=str(destination),
        relocated_inference_exact_boxes=True,relocated_generation=relocated,historical_models_load=True,
        relocated_generation_exact_arrays=True,
        archive_contents_verified=True,
        release_status='verified synthetic research release',physical_validation=False,
        electron_depth_cutoff_implemented=False,dependency_environment='Existing tested environment; clean dependency install not tested')
    c.write(ROOT/'data/releases/CURRENT_RAISED_RELEASE.json',record);c.write(c.OUT/'release_checks.json',record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
