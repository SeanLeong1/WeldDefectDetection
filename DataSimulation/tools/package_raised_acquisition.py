"""Versioned inference release, with independently verified relocation."""
from pathlib import Path
import copy,json,shutil,subprocess,sys,uuid,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from raised_acquisition import campaign as c, learning as learn, gui


def main():
    assert c.read(c.OUT/'audit.json')['status']=='passed'
    assert c.read(c.OUT/'gui_checks.json')['labels_absent_identical_inference']
    frozen=learn.verify_freeze(); c.verify(historical=True)
    previous=c.read(ROOT/'data/releases/CURRENT_RAISED_RELEASE.json')
    assert c.sha(previous['archive'])==previous['archive_sha256']
    destination=ROOT/'.tmp'/('acquisition_release_'+uuid.uuid4().hex[:8]); destination.mkdir(parents=True)
    with zipfile.ZipFile(previous['archive']) as z:
        assert all((destination/name).resolve().is_relative_to(destination.resolve()) for name in z.namelist())
        z.extractall(destination)
    old_manifest=destination/'RAISED_RELEASE_MANIFEST.json'
    if old_manifest.exists():
        archive=destination/'docs/archive/RAISED_RELEASE_MANIFEST_20261005_historical.json'; archive.parent.mkdir(exist_ok=True,parents=True)
        old_manifest.rename(archive)
    files=list((ROOT/'WeldDefectDetection/raised_acquisition').glob('*.py'))
    files+=list((ROOT/'WeldDefectDetection').glob('run_raised_acquisition*.py'))
    files+=[ROOT/'WeldDefectDetection/complete_raised_acquisition_20261006.py']
    files+=[ROOT/'WeldDefectDetection/tests/test_raised_acquisition.py']
    files+=[ROOT/name for name in ('launch_raised_acquisition.py','launch_weld_studio.py','README.md','docs/PROJECT_CURRENT.md',
        'DataSimulation/README.md','WeldDefectDetection/README.md','DataSimulation/PROJECT_PIPELINE_CURRENT.md')]
    files+=list((ROOT/'docs').glob('RAISED_ACQUISITION*.md'))+list((ROOT/'docs/figures/raised_acquisition_20261006').glob('*.png'))
    files+=list((ROOT/'docs/archive').glob('*before_acquisition_20261006.*'))
    files+=[ROOT/'DataSimulation/tools'/name for name in ('check_raised_acquisition_gui.py','package_raised_acquisition.py')]
    files+=list(c.OUT.glob('*.json'))+list(c.OUT.glob('*.zip'))
    files+=list((c.OUT/'training').rglob('*.pt'))+list((c.OUT/'training').rglob('*.json'))
    files+=list((c.OUT/'predictions').glob('*.json'))
    for path in files:
        if path.name in ('frozen_models.json','delivery_manifest.json'): continue
        if path.is_file():
            target=destination/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    output=destination/c.OUT.relative_to(ROOT)
    for split,indices in [('test',range(4)),('stress',range(1))]:
        rows=[]
        for row in c.read(c.OUT/f'datasets/{split}/index.json'):
            if row['index'] not in indices: continue
            source=Path(row['npz_path']); target=destination/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
            row=copy.deepcopy(row); row['npz_path']=source.relative_to(ROOT).as_posix()
            c.write(target.with_suffix('.json'),row); rows.append(row)
        c.write(output/f'datasets/{split}/index.json',rows)
    c.write(output/'research_frozen_models.json',frozen)
    portable=copy.deepcopy(frozen); portable['portable']=True
    portable['verification_scope']='Bundled inference files; original full research freeze preserved separately'
    specs=[portable['cnn']]
    for arm in portable['arms'].values(): specs += [arm['cnn'],*arm['cnn_candidates']]
    for spec in specs: spec['checkpoint']=Path(spec['checkpoint']).relative_to(ROOT).as_posix()
    required=[destination/spec['checkpoint'] for spec in specs]
    required += [output/'normalization.json']
    required += list((destination/'WeldDefectDetection/raised_acquisition').glob('*.py'))
    required += [destination/Path(path).relative_to(ROOT) for path in frozen['files'] if Path(path).suffix=='.py']
    portable['files']={p.relative_to(destination).as_posix():c.sha(p) for p in set(required)}
    c.write(output/'frozen_models.json',portable)
    code="import sys,json;from pathlib import Path;sys.path[:0]=['DataSimulation','WeldDefectDetection'];from raised_acquisition.gui import infer;from raised_acquisition.campaign import OUT,ARMS;from raised_crown.learning import verify_freeze;verify_freeze();print(json.dumps({a:infer(OUT/'datasets/test/test-000-open_intrinsic.npz',a)['boxes'] for a in ARMS}))"
    result=subprocess.run([sys.executable,'-c',code],cwd=destination,text=True,capture_output=True,timeout=600)
    if result.returncode: raise RuntimeError(result.stderr)
    actual=json.loads(result.stdout.strip().splitlines()[-1])
    expected={arm:gui.infer(c.OUT/'datasets/test/test-000-open_intrinsic.npz',arm)['boxes'] for arm in c.ARMS}
    assert actual==expected
    # A fresh example checks relocated rendering imports and generation as well.
    reference=c.OUT/'release_generation_reference.json'
    if reference.exists():
        local=c.read(reference); assert c.sha(local['path'])==local['sha256']
    else:
        path=gui.generate(2450600099,'open_intrinsic',.2,5)
        local=dict(path=str(path),sha256=c.sha(path)); c.write(reference,local)
    code="import sys,json;sys.path[:0]=['DataSimulation','WeldDefectDetection'];from raised_acquisition.gui import generate,infer;p=generate(2450600099,'open_intrinsic',.2,5);x=infer(p);print(json.dumps({'path':str(p),'boxes':x['boxes']}))"
    result=subprocess.run([sys.executable,'-c',code],cwd=destination,text=True,capture_output=True,timeout=900)
    if result.returncode: raise RuntimeError(result.stderr)
    relocated=json.loads(result.stdout.strip().splitlines()[-1])
    with np.load(local['path']) as a,np.load(relocated['path']) as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    checks=dict(relocated_inference_identical=True,relocated_generation_identical=True,
        historical_model_manifest_verified=True,dependency_environment='Existing workspace environment, not a fresh install',
        complete_training_data_bundled=False)
    c.write(output/'relocation_checks.json',checks)
    (destination/'BUNDLE_README.md').write_text('''# Circular acquisition study inference bundle

Run `python launch_weld_studio.py` using the tested project environment. See `docs/PROJECT_CURRENT.md` and `docs/RAISED_ACQUISITION_RESULTS_20261006.md`.

The new tab compares the frozen coverage and hard-negative CNN arms with the median method. The 5 October raised-crown and 4 October recessed workflows are preserved as historical tabs.

This archive includes six CNN checkpoints, reports, source, and a limited example set: four confirmation backgrounds and one stress background. It does not include the full training datasets or Python dependencies. Use the full research workspace to reproduce training. Bundled example lookup chooses the nearest available dwell for the requested condition.

ACQUISITION_RELEASE_MANIFEST.json verifies this bundle. Earlier manifests in docs/archive describe their original archives, not the combined bundle. Research_frozen_models.json preserves original research provenance; frozen_models.json verifies relocatable inference assets.

Synthetic evidence only. No SEM electron-depth cutoff or physical weld-fluid simulation has been added.
''',encoding='utf-8')
    paths=[p for p in destination.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest={p.relative_to(destination).as_posix():c.sha(p) for p in paths}
    c.write(destination/'ACQUISITION_RELEASE_MANIFEST.json',dict(files=manifest,physical_validation=False))
    release=ROOT/'data/releases'/f'WeldVision_Acquisition_20261006_{uuid.uuid4().hex[:8]}.zip'
    with zipfile.ZipFile(release,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths+[destination/'ACQUISITION_RELEASE_MANIFEST.json']: z.write(path,path.relative_to(destination))
    import hashlib
    with zipfile.ZipFile(release) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for name,digest in manifest.items(): assert hashlib.sha256(z.read(name)).hexdigest()==digest
    record=dict(archive=str(release),archive_sha256=c.sha(release),files=len(paths)+1,**checks)
    c.write(release.with_suffix('.verification.json'),record)
    c.write(ROOT/'data/releases/CURRENT_ACQUISITION_RELEASE.json',record)
    c.verify(historical=True); print(record)


if __name__=='__main__': main()
