"""Ordered campaign stages with explicit qualified or assumption-based evidence policies."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

from ..domain_randomisation.dataset_forms import _sha256_path
from ..renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit
from .casino_compact import _write_json

STAGES = ('pilot', 'generate', 'audit', 'train', 'freeze', 'confirmation', 'evaluate', 'verify')
REPO = Path(__file__).resolve().parents[3]

def inventory(directory):
    return {str(p.relative_to(directory)): _sha256_path(p) for p in sorted(directory.rglob('*')) if p.is_file()}

def verify_receipt(root, receipt):
    for relative, digest in receipt['artifacts'].items():
        path = root/relative
        if not path.is_file() or _sha256_path(path) != digest:
            raise ValueError(f'Completed campaign artifact changed: {path}')

def require_model(root):
    assumption_path = root/'assumption_campaign.json'
    if assumption_path.exists():
        campaign = json.loads(assumption_path.read_text())
        if (campaign.get('schema') != 'synthetic-assumption-campaign/v1'
                or campaign.get('emission_qualification') != 'not_established'
                or campaign.get('mode') != 'assumed_constant'):
            raise ValueError('Invalid assumption campaign policy.')
        path = (root/campaign['model']['path']).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256_path(path) != campaign['model']['sha256']:
            raise ValueError('Assumption model missing, outside campaign or changed.')
        load_emission_b_fit(path, 'assumed_constant')
        return 'assumed_constant', path
    report_path = root/'qualification/qualification_report.json'
    if not report_path.exists():
        raise ValueError('Run domain-pilot and qualify before production stages.')
    report = json.loads(report_path.read_text())
    if report.get('status') != 'passed' or not report.get('qualified_model'):
        raise ValueError('Emission qualification incomplete: '+' '.join(report.get('reasons', [])))
    path = (report_path.parent/report['qualified_model']['path']).resolve()
    if _sha256_path(path) != report['qualified_model']['sha256']:
        raise ValueError('Qualified model hash changed.')
    mode = report['qualified_model']['mode']
    qualify_emission_b_fit(load_emission_b_fit(path, mode))
    return mode, path

def jobs_for(stage):
    return {
        'pilot': ['pilot'], 'generate': ['static_dev','sequence_dev'],
        'audit': ['audit_dev'],
        'train': ['classical','cnn_0','cnn_1','cnn_2','mean_context','context_0','context_1','context_2'],
        'freeze': ['freeze'],
        'confirmation': ['static_confirmation','static_noiseless','sequence_confirmation','audit_confirmation'],
        'evaluate': ['evaluate_static','evaluate_context','skill'],
        'verify': ['verify'],
    }[stage]

def execute_stage(root, stage, *, max_seconds=86400):
    root=root.resolve()
    if not 0 < max_seconds <= 86400:
        raise ValueError('Batch budget must be positive and at most 24 hours.')
    mode, model = require_model(root)
    receipts=root/'receipts'
    receipts.mkdir(exist_ok=True)
    for previous in STAGES[:STAGES.index(stage)]:
        path=receipts/f'{previous}.json'
        if not path.exists():
            raise ValueError(f'Prerequisite stage is incomplete: {previous}')
        verify_receipt(root,json.loads(path.read_text()))
    stage_path=receipts/f'{stage}.json'
    if stage_path.exists():
        receipt=json.loads(stage_path.read_text())
        verify_receipt(root,receipt)
        return receipt
    started=time.monotonic()
    sources={str(p.relative_to(REPO)): _sha256_path(p) for package in ('DataSimulation/bse_renderer','WeldDefectDetection/src') for p in (REPO/package).rglob('*.py')}
    source_id={'sources':sources,'model_sha256':_sha256_path(model),
               'configuration': {p.name: _sha256_path(p) for p in (REPO/'DataSimulation/config').glob('*.yaml')}}
    if (root/'assumption_campaign.json').exists():
        source_id['assumption_campaign_sha256'] = _sha256_path(root/'assumption_campaign.json')
    session_path=receipts/'execution_identity.json'
    if session_path.exists() and json.loads(session_path.read_text()) != source_id:
        raise ValueError('Campaign implementation changed; start a new versioned campaign.')
    _write_json(session_path,source_id)
    artifacts={}
    for job in jobs_for(stage):
        receipt_path=receipts/f'job_{job}.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            verify_receipt(root,receipt)
            artifacts.update(receipt['artifacts'])
            continue
        remaining=max_seconds-(time.monotonic()-started)
        if remaining <= 0:
            return {'status':'paused','stage':stage}
        result_dir=root/'jobs'/job
        if result_dir.exists() and any(result_dir.iterdir()):
            raise ValueError(f'Incomplete job output exists: {result_dir}. Preserve it and use a new campaign version.')
        result_dir.mkdir(parents=True,exist_ok=True)
        log=result_dir/'process.log'
        with log.open('w',encoding='utf-8') as handle:
            try:
                completed=subprocess.run([sys.executable,str(REPO/'DataSimulation/qualified_campaign.py'),stage,
                    '--root',str(root),'--job',job],stdout=handle,stderr=subprocess.STDOUT,timeout=remaining,check=False)
            except subprocess.TimeoutExpired:
                return {'status':'paused','stage':stage,'incomplete_job':job,'log':str(log)}
        if completed.returncode:
            raise RuntimeError(f'Campaign job failed: {job}; inspect {log}')
        result=json.loads((result_dir/'result.json').read_text())
        if result.get('status') not in ('ok','completed','passed'):
            raise ValueError(f'Job {job} did not pass: {result}')
        covered=[result_dir]+[Path(p) for p in result.get('artifact_directories',[])]
        job_artifacts={str((directory/rel).relative_to(root)): digest for directory in covered for rel,digest in inventory(directory).items()}
        job_artifacts.update({str(Path(p).relative_to(root)): _sha256_path(Path(p)) for p in result.get('artifact_files',[])})
        receipt={'status':'completed','job':job,'artifacts':job_artifacts}
        _write_json(receipt_path,receipt)
        artifacts.update(job_artifacts)
    receipt={'status':'completed','stage':stage,'artifacts':artifacts,'seconds':time.monotonic()-started}
    _write_json(stage_path,receipt)
    return receipt

def perform_job(root, job):
    """Worker runs in its own bounded process; only completed jobs are resumable."""
    root=root.resolve()
    mode,model=require_model(root)
    owning_stage=next((stage for stage in STAGES if job in jobs_for(stage)),None)
    if owning_stage is None:
        raise ValueError(f'Unknown job: {job}')
    for previous in STAGES[:STAGES.index(owning_stage)]:
        receipt=root/'receipts'/f'{previous}.json'
        if not receipt.exists():
            raise ValueError(f'Prerequisite stage is incomplete: {previous}')
        verify_receipt(root,json.loads(receipt.read_text()))
    sys.path.insert(0,str(REPO/'WeldDefectDetection'))
    result_dir=root/'jobs'/job
    result_dir.mkdir(parents=True,exist_ok=True)
    data=root/'datasets'
    exp=root/'experiments'
    common=dict(config_dir=REPO/'DataSimulation/config',fixed_energy_kev=60.,render_model_mode='formula_only',emission_b_mode=mode,emission_b_path=str(model),pixel_size_mm=.05)
    assumption_file = root/'assumption_campaign.json'
    reuse_source = (json.loads(assumption_file.read_text()).get('reuse_source_campaign')
                    if assumption_file.exists() else None)
    if reuse_source and job in ('pilot', 'static_dev', 'sequence_dev'):
        from .campaign_resplit import reuse_completed_job
        result = reuse_completed_job(reuse_source, root, job)
    elif job=='pilot':
        from .qualified_pilot import run_domain_pilot
        result=run_domain_pilot(root/'integrated_pilot',config_dir=common['config_dir'],seed=915410001,emission_mode=mode,emission_path=str(model))
        if result['accepted_count'] != 84:
            raise ValueError('Integrated pilot contains rejected cases.')
        result['artifact_directories']=[str(root/'integrated_pilot')]
    elif job in ('static_dev','sequence_dev','static_confirmation','static_noiseless','sequence_confirmation'):
        from ..domain_randomisation.dataset_forms import generate_balanced_topo_dataset,generate_topo_sequence_dataset
        if job.startswith('static'):
            confirmation=job!='static_dev'
            result=generate_balanced_topo_dataset(**common,output_dir=data/job,samples_per_class=100 if confirmation else 200,image_size=128,
                seed=915210001 if confirmation else 915010001,
                normalization_path=data/'static_dev/topo_normalization.json' if confirmation else None,
                noise_profile='none' if job=='static_noiseless' else 'development')
        else:
            confirmation=job!='sequence_dev'
            result=generate_topo_sequence_dataset(**common,output_dir=data/job,sequences_per_class=10 if confirmation else 30,
                frames_per_sequence=9,frame_size=64,stride_px=64,context_radius=2,weld_layout_mode='straight_seam',
                seed=915310001 if confirmation else 915110001,
                normalization_path=data/'sequence_dev/topo_normalization.json' if confirmation else None)
        result['status']='completed'
        result['artifact_directories']=[str(data/job)]
    elif job.startswith('audit_'):
        from src.data.static_audit import audit_static_dataset
        from src.data.sequence_audit import audit_sequence_dataset
        from src.data.paired_static_audit import audit_paired_static_datasets
        confirm=job=='audit_confirmation'
        from .campaign_resplit import check_campaign_splits
        split_checks = {
            'static': check_campaign_splits(data/('static_confirmation' if confirm else 'static_dev'), 100 if confirm else 200),
            'sequence': check_campaign_splits(data/('sequence_confirmation' if confirm else 'sequence_dev'), 10 if confirm else 30)}
        checks={'static':audit_static_dataset(data/('static_confirmation' if confirm else 'static_dev'),compare_roots=[data/'static_dev'] if confirm else []),
                'sequence':audit_sequence_dataset(data/('sequence_confirmation' if confirm else 'sequence_dev'),compare_roots=[data/'sequence_dev'] if confirm else [])}
        if confirm:
            checks['paired']=audit_paired_static_datasets(data/'static_confirmation',data/'static_noiseless')
        result={'status':'ok' if all(c.get('status')=='ok' for c in checks.values()) else 'failed','checks':checks,
                'campaign_split_checks': split_checks}
    elif job in ('classical','cnn_0','cnn_1','cnn_2'):
        cfg=yaml.safe_load((REPO/'WeldDefectDetection/config_completion_final.yaml').read_text())
        cfg['data']['dataset_root']=str(data/'static_dev')
        cfg['train']['experiment_root']=str(exp/job)
        cfg['eval']['experiment_root']=str(exp/(job+'_eval'))
        cfg_path=result_dir/'config.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg))
        if job=='classical':
            from src.evaluation.tune_classical import tune_classical_baseline
            result=tune_classical_baseline(cfg_path,split='val',output_dir=exp/job)
        else:
            from src.training.train_cnn import train_cnn
            result=train_cnn(cfg_path,seed_override=915510001+int(job[-1]))
        result['artifact_directories']=[str(exp/job)]
    elif job=='mean_context':
        from src.evaluation.temporal_reconstruction import run_temporal_reconstruction
        result=run_temporal_reconstruction(dataset_root=data/'sequence_dev',output_dir=exp/job,selection_split='val',evaluation_split='test',seed=915610001)
        result['artifact_directories']=[str(exp/job)]
    elif job.startswith('context_'):
        from src.training.train_temporal_predictor import train_temporal_predictor
        result=train_temporal_predictor(dataset_root=data/'sequence_dev',output_dir=exp/job,seed=915710001+int(job[-1]),epochs=20,batch_size=16,device_name='cpu')
        result['artifact_directories']=[str(exp/job)]
    elif job=='freeze':
        results={name:json.loads((root/'jobs'/name/'result.json').read_text()) for name in jobs_for('train')}
        cnn=max((results[f'cnn_{i}'] for i in range(3)),key=lambda r:r['best_val_iou'])
        context=max((results[f'context_{i}'] for i in range(3)),key=lambda r:r['selection_summary']['pixel']['iou'])
        classical=results['classical']
        selected={'cnn':cnn,'classical':classical,'context':context,'mean_context':results['mean_context']}
        frozen={'status':'ok','selected':selected,'primary_static_method':'cnn' if cnn['best_val_iou']>classical['best_validation_metrics']['pixel_iou'] else 'morphology',
                'selection_policy':'development validation only; static and spatial methods selected separately',
                'baseline_policy':'all-background and central horizontal quarter-height weld region',
                'checkpoint_hashes':{cnn['best_checkpoint']:_sha256_path(Path(cnn['best_checkpoint'])),context['checkpoint']:_sha256_path(Path(context['checkpoint']))}}
        _write_json(root/'frozen_models.json',frozen)
        result=dict(frozen)
        result['artifact_files']=[str(root/'frozen_models.json')]
    elif job in ('evaluate_static','evaluate_context','skill'):
        frozen=json.loads((root/'frozen_models.json').read_text())
        for path,digest in frozen['checkpoint_hashes'].items():
            if _sha256_path(Path(path))!=digest:
                raise ValueError('Frozen checkpoint changed.')
        selected=frozen['selected']
        if job=='evaluate_static':
            from src.evaluation.evaluate_dataset import evaluate_dataset
            results={}
            for variant in ('static_confirmation','static_noiseless'):
                for algorithm in ('morphology','cnn'):
                    cfg=selected['classical']['tuned_config'] if algorithm=='morphology' else str(root/'jobs'/f"cnn_{selected['cnn']['seed']-915510001}"/'config.yaml')
                    results[f'{variant}_{algorithm}']=evaluate_dataset(config_path=cfg,algorithm=algorithm,split='all',
                        checkpoint=selected['cnn']['best_checkpoint'] if algorithm=='cnn' else None,
                        dataset_root_override=data/variant,output_dir=exp/f'{variant}_{algorithm}')
            result={'status':'ok','results':results,'artifact_directories':[str(exp/k) for k in results]}
        elif job=='evaluate_context':
            from src.evaluation.temporal_reconstruction import run_temporal_reconstruction
            from src.evaluation.evaluate_temporal_predictor import evaluate_temporal_predictor
            params=selected['mean_context']['selected_parameters']
            mean=run_temporal_reconstruction(dataset_root=data/'sequence_confirmation',output_dir=exp/'mean_confirmation',evaluation_split='all',
                development_dataset_root=data/'sequence_dev',frozen_threshold=params['threshold'],frozen_min_area=params['min_area'])
            learned=evaluate_temporal_predictor(dataset_root=data/'sequence_confirmation',checkpoint_path=selected['context']['checkpoint'],output_dir=exp/'context_confirmation',split='all')
            result={'status':'ok','mean':mean,'learned':learned,'artifact_directories':[str(exp/'mean_confirmation'),str(exp/'context_confirmation')]}
        else:
            from src.evaluation.campaign_skill import evaluate_campaign_skill
            result=evaluate_campaign_skill(exp/f"static_confirmation_{frozen['primary_static_method']}",seed=915910001)
    elif job=='verify':
        result={'status':'ok','operational_pipeline':'completed',
                'emission_qualification':'not_established' if mode == 'assumed_constant' else 'passed',
                'emission_policy':'engineering_assumption' if mode == 'assumed_constant' else 'qualified',
                'detection_skill':json.loads((root/'jobs/skill/result.json').read_text()),
                'scope':'synthetic only; no real SEM claim'}
    else:
        raise ValueError(f'Unknown campaign job: {job}')
    result['emission_policy'] = 'engineering_assumption' if mode == 'assumed_constant' else 'qualified'
    result['emission_model_sha256'] = _sha256_path(model)
    _write_json(result_dir/'result.json',result)
    return result
