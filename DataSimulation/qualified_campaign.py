"""Command line entry point for the qualified synthetic campaign."""
from pathlib import Path
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['domain-pilot', 'qualify', 'pilot', 'generate', 'audit', 'train', 'freeze', 'confirmation', 'evaluate', 'verify', 'status','ingest-chunk','release'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).parent/'config')
    parser.add_argument('--max-seconds', type=float, default=86400)
    parser.add_argument('--casino-source', type=Path, default=Path(__file__).parent/'data/output/casino_surface_reanalysis_20260914')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    parser.add_argument('--summary-export',type=Path)
    parser.add_argument('--tilt-deg',type=float)
    parser.add_argument('--repeat',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--qualification-evidence',type=Path)
    parser.add_argument('--validation-evidence',type=Path)
    parser.add_argument('--emission-mode',choices=['pooled_constant_fit','b_alpha_sin','angular_table'])
    args = parser.parse_args()
    if args.stage=='release':
        if any(v is None for v in (args.candidate,args.qualification_evidence,args.validation_evidence,args.emission_mode)):
            parser.error('release requires candidate, qualification-evidence, validation-evidence and emission-mode')
        from bse_renderer.validation.qualification_release import export_qualified_model
        result=export_qualified_model(args.candidate,args.qualification_evidence,args.validation_evidence,args.root/'qualification',mode=args.emission_mode)
    elif args.stage=='ingest-chunk':
        if any(v is None for v in (args.summary_export,args.tilt_deg,args.repeat,args.seed)):
            parser.error('ingest-chunk requires --summary-export, --tilt-deg, --repeat and --seed')
        from bse_renderer.validation.casino_chunk_ingest import ingest_chunk
        result=ingest_chunk(args.summary_export,args.root,tilt_deg=args.tilt_deg,repeat=args.repeat,seed=args.seed)
    elif args.job:
        from bse_renderer.validation.campaign_execution import perform_job
        result=perform_job(args.root,args.job)
    elif args.stage == 'domain-pilot':
        from bse_renderer.validation.qualified_pilot import run_domain_pilot
        result = run_domain_pilot(args.root/'domain_pilot', config_dir=args.config_dir, max_seconds=args.max_seconds)
    elif args.stage == 'qualify':
        from bse_renderer.validation.emission_qualification import run_qualification
        result = run_qualification(args.casino_source, args.root/'domain_pilot', args.root/'qualification')
    elif args.stage == 'status':
        from bse_renderer.validation.campaign_execution import STAGES, verify_receipt
        report=args.root/'qualification/qualification_report.json'
        qualification=json.loads(report.read_text()) if report.exists() else {}
        stages={}
        for stage in STAGES:
            path=args.root/'receipts'/f'{stage}.json'
            if path.exists():
                verify_receipt(args.root,json.loads(path.read_text()))
                stages[stage]='completed_verified'
            else:
                stages[stage]='not_run'
        result={'status':'completed','emission_qualification':qualification.get('status','not_run'),
                'stages':stages,'reasons':qualification.get('reasons',[])}
        print(json.dumps(result,indent=2))
        return 0
    else:
        from bse_renderer.validation.campaign_execution import execute_stage
        result=execute_stage(args.root,args.stage,max_seconds=args.max_seconds)
    print(json.dumps({k: result[k] for k in ('status','case_count','incidence_domain_deg','material_z_domain','reasons') if k in result}, indent=2))
    return 0 if result['status'] in ('completed','ok','passed','validated_compact_chunk') else 2

if __name__ == '__main__':
    raise SystemExit(main())
