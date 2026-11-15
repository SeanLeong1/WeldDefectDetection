"""Package current inference and explicit historical options; verify relocation."""
from pathlib import Path
import sys,json,zipfile,time,uuid,subprocess,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
from inspection.engine import digest
from inspection.registry import CURRENT,RASTER,LEGACY

def main():
    output=ROOT/'data/audit_circular_current_20261004';output.mkdir(parents=True,exist_ok=True)
    files=set()
    for directory in (ROOT/'DataSimulation/bse_renderer',ROOT/'DataSimulation/config',ROOT/'DataSimulation/entrypoints',ROOT/'WeldDefectDetection'):
        files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml') and not any(x in p.parts for x in ('__pycache__','.venv','experiments','.git','.tmp')))
    for name in ('README.md','launch_weld_studio.py','Launch Weld Vision.cmd','requirements-studio.txt','requirements-studio-tested.txt'):
        path=ROOT/name
        if path.exists():files.add(path)
    for pattern in ('CIRCULAR*.md','PROJECT_CURRENT.md','ACQUISITION_V2_DELIVERY_20260928.md'):
        files.update((ROOT/'docs').glob(pattern))
    files.update((ROOT/'docs/figures').glob('circular*/*.png'))
    files.update(p for p in (ROOT/'docs/archive').glob('*20261004.*') if p.is_file())
    robustness=ROOT/'DataSimulation/data/output/circular_robustness_20261004_v1'
    for name in ('README.md','protocol.json','protocol_native_v2.json','results.json','paired_localisation.json','audit.json','dataset_manifest.json','delivery_manifest.json','analysis_sources.zip','sources.zip'):
        path=robustness/name
        if path.exists():files.add(path)
    for name in ('resolution.json','postprocessing.json','fixed_failure_case.json','localisation_resampling.json'):
        path=robustness/'investigation'/name
        if path.exists():files.add(path)
    execution=robustness/'execution/parallel_protocol.json'
    if execution.exists():files.add(execution)
    for campaign in (CURRENT,RASTER,LEGACY):
        files.update(campaign/n for n in ('normalization.json','frozen_models.json'))
        record=json.loads((campaign/'frozen_models.json').read_text())
        spec=record['cnn'] if campaign==CURRENT else record['models']['cnn']
        files.add(campaign/'training'/Path(spec['checkpoint'].replace('\\','/')).name)
        if campaign==CURRENT:
            for kind in ('clean','keyhole_pore','intrinsic_pore'):
                files.add(campaign/f'datasets/test/test-000-{kind}.npz')
        else:
            files.update((campaign/'inference_smoke').glob('*.npz'));files.update((campaign/'inference_smoke').glob('*.json'))
    files.update((ROOT/'DataSimulation/data/output/assumed_constant_campaign_20260915_v2/model').glob('*.json'))
    files.update(ROOT/'DataSimulation/tools'/name for name in ('check_current_circular_gui.py','package_current_circular.py','package_studio.py','run_project_tests.py'))
    dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    manifest=dict(schema='weld-vision-current-release/v1',current_campaign=CURRENT.relative_to(ROOT).as_posix(),
        historical_campaigns=[p.relative_to(ROOT).as_posix() for p in (RASTER,LEGACY)],
        python=sys.version,dependencies=dependencies,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        scope='Current/historical inference, current circular example generation, source and guides. Full research populations and dependencies are not bundled.',
        physical_validation=False,files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)})
    releases=ROOT/'data/releases';releases.mkdir(parents=True,exist_ok=True)
    archive=releases/f'WeldVision_CurrentCircular_20261004_{uuid.uuid4().hex[:8]}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):z.write(path,path.relative_to(ROOT).as_posix())
        z.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,indent=2))
    destination=ROOT/'.tmp'/f'current_release_{uuid.uuid4().hex[:8]}';destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target=(destination/name).resolve()
            if destination.resolve() not in target.parents:raise ValueError('Unsafe archive path')
        z.extractall(destination)
    assert all(digest(destination/p)==h for p,h in manifest['files'].items())
    checks=[]
    for campaign in (CURRENT,RASTER,LEGACY):
        relative=campaign.relative_to(ROOT);sample='datasets/test/test-000-keyhole_pore.npz' if campaign==CURRENT else 'inference_smoke/keyhole_pore_0.00.npz'
        args=[sys.executable,str(destination/'WeldDefectDetection/inspect_images.py'),'--input',str(destination/relative/sample),
            '--output',str(output/'relocated_inference'),'--mode','raw','--pixel-size-mm','.05','--method','all','--campaign',str(destination/relative)]
        process=subprocess.run(args,cwd=destination/'docs',capture_output=True,text=True,timeout=180)
        if process.returncode:raise RuntimeError(process.stdout+process.stderr)
        info=json.loads(process.stdout.strip().splitlines()[-1]);results=json.loads((Path(info['output'])/'results.json').read_text())
        assert info['errors']==0
        for row in results['records']:
            if campaign==CURRENT:
                arm='cnn_capacity24_100482003' if row['method']=='cnn' else 'median_neighbours8'
                expected=next(r for r in json.loads((CURRENT/f'predictions/{arm}_best.json').read_text()) if r['scene_id']=='test-000-keyhole_pore')['predicted_boxes']
            else:
                reference=campaign/f'inference_smoke/keyhole_pore_0.00_{row["method"]}.json'
                if reference.exists():expected=json.loads(reference.read_text())['boxes_xyxy']
                else:
                    from inspection.engine import FrozenInspector,load_image,Options
                    inspector=FrozenInspector(campaign)
                    item=load_image(campaign/sample,Options(mode='raw',pixel_size_mm=.05),inspector.norm)
                    expected=inspector.predict(item,row['method'])[2]
            assert row['boxes_xyxy']==expected,(campaign,row['method'])
        checks.append(dict(campaign=relative.as_posix(),methods=[r['method'] for r in results['records']],exact_reference_boxes=True,output=info['output']))
    code="""import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'DataSimulation'),str(root/'WeldDefectDetection')]
from inspection.workbench import generate_example
from inspection.engine import FrozenInspector,Options,load_image
import numpy as np
p=generate_example(2260400001,'pair_overlap')
with np.load(p) as a:
    assert a['raw'].shape==(3,128,5024) and len(a['instances'])==2
    assert np.array_equal(a['mask'],a['instances'].any(axis=0))
inspector=FrozenInspector();item=load_image(p,Options(mode='raw',pixel_size_mm=.05),inspector.norm)
for method in ('cnn','median'):inspector.predict(item,method)
print(json.dumps(dict(path=str(p),generation_and_inference_passed=True)))
"""
    generation=subprocess.run([sys.executable,'-c',code,str(destination)],cwd=destination/'docs',capture_output=True,text=True,timeout=240)
    if generation.returncode:raise RuntimeError(generation.stdout+generation.stderr)
    generation_check=json.loads(generation.stdout.strip().splitlines()[-1])
    evidence=dict(archive=str(archive),archive_sha256=digest(archive),files=len(files),relocated_checkout=str(destination),
        source_hashes_verified=True,inference_checks=checks,relocated_generation=generation_check,
        dependency_environment='Existing tested Python environment; a fresh dependency installation was not tested.')
    evidence['release_status']='complete' if (ROOT/'docs/CIRCULAR_ROBUSTNESS_RESULTS_20261004.md').exists() and (robustness/'audit.json').exists() else 'software-preview'
    archive.with_suffix('.verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    if evidence['release_status']=='complete':
        (releases/'CURRENT_RELEASE.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (output/'release_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'portable_release.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence),flush=True)

if __name__=='__main__':main()
