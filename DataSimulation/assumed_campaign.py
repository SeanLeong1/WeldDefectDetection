"""Run the synthetic pipeline with an explicitly assumed constant B, without a physical qualification claim."""
import argparse
import json
from pathlib import Path
import shutil

from bse_renderer.domain_randomisation.dataset_forms import _sha256_path
from bse_renderer.renderer.emission_b import load_emission_b_fit
from bse_renderer.validation.casino_compact import _write_json
from bse_renderer.validation.campaign_execution import STAGES, execute_stage, require_model, verify_receipt


def initialize(root, selection_freeze):
    root, selection_freeze = root.resolve(), selection_freeze.resolve()
    if root.exists():
        raise FileExistsError('Use a new campaign directory; historical campaigns are preserved.')
    freeze = json.loads(selection_freeze.read_text())
    source = Path(freeze['candidate_models_path'])
    if _sha256_path(source) != freeze['candidate_models_sha256']:
        raise ValueError('Selected constant-B source hash changed.')
    raw = json.loads(source.read_text())
    b = raw['model_comparison']['constant_b']['fitted_b']
    model_dir = root/'model'
    model_dir.mkdir(parents=True)
    shutil.copy2(source, model_dir/'original_candidates.json')
    shutil.copy2(selection_freeze, model_dir/'original_selection_freeze.json')
    assumption = {'schema': 'emission-assumption/v1', 'status': 'engineering_assumption',
        'physical_qualification': False, 'fitted_b': b, 'assumed_tilt_domain_deg': [0., 60.],
        'energy_kev': 60., 'material_name': '304SS',
        'coefficient_source': {'path': 'original_candidates.json', 'sha256': _sha256_path(source)},
        'decision': 'User chose to stop further angular qualification and proceed with constant B for the synthetic pipeline.',
        'assumptions': ['The selected constant B is held fixed for every local incidence within the declared domain.',
            'Its use with effective local material modifiers is assumed, not independently measured.',
            'Existing empirical gain, blur, noise and defect-response rules remain synthetic image-formation assumptions.'],
        'limitations': ['Full angular adequacy and detector-acceptance qualification did not pass.',
                       'No claim of independent physical validation or real-SEM transfer.']}
    model = model_dir/'assumed_constant.json'
    _write_json(model, assumption)
    load_emission_b_fit(model, 'assumed_constant')
    campaign = {'schema': 'synthetic-assumption-campaign/v1', 'mode': 'assumed_constant',
        'status': 'initialized', 'emission_qualification': 'not_established',
        'model': {'path': str(model.relative_to(root)), 'sha256': _sha256_path(model)},
        'scope': 'Synthetic 304SS, 60 keV detection experiments conditional on fixed constant-B emission.',
        'stages': list(STAGES), 'physical_qualification_work': 'stopped_by_user',
        'development_plan': {'static_samples_per_class': 200, 'sequences_per_class': 30,
                             'split_fractions': [.6, .2, .2]},
        'confirmation_plan': {'static_samples_per_class': 100, 'sequences_per_class': 10},
        'confirmation_meaning': 'Independent synthetic detection evaluation, not CASINO angular qualification.'}
    _write_json(root/'assumption_campaign.json', campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'status', *STAGES])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection-freeze', type=Path, default=Path(__file__).parent/'data/output/casino_selection_review_20260915/candidate_freeze.json')
    parser.add_argument('--max-seconds', type=float, default=86400)
    parser.add_argument('--reuse-source-campaign', type=Path,
                        help='Reuse verified raw development renders and pilot; rebuild 60/20/20 splits and normalization.')
    args = parser.parse_args()
    if args.stage == 'initialize':
        result = initialize(args.root, args.selection_freeze)
        if args.reuse_source_campaign:
            require_model(args.reuse_source_campaign.resolve())
            result['reuse_source_campaign'] = str(args.reuse_source_campaign.resolve())
            result['reuse_policy'] = 'Hash-verified raw renders; source-grouped 60/20/20 resplit with newly fitted training normalization.'
            _write_json(args.root/'assumption_campaign.json', result)
    else:
        if not (args.root/'assumption_campaign.json').exists():
            parser.error('Initialize a separately versioned assumption campaign first.')
        require_model(args.root)
        if args.stage == 'status':
            stages = {}
            for stage in STAGES:
                path = args.root/'receipts'/f'{stage}.json'
                if path.exists():
                    verify_receipt(args.root, json.loads(path.read_text()))
                    stages[stage] = 'completed_verified'
                else:
                    stages[stage] = 'not_run'
            result = {'status': 'ok', 'emission_policy': 'engineering_assumption',
                      'emission_qualification': 'not_established', 'stages': stages}
        else:
            result = execute_stage(args.root, args.stage, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('initialized', 'completed', 'ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
