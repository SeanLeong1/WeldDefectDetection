import hashlib
import json
import pytest

@pytest.fixture
def bind_qualification():
    """Synthetic, hash-bound certificate for exercising loader mechanics only."""
    def bind(path, raw, mode='pooled_constant_fit'):
        from bse_renderer.renderer.emission_b import load_emission_b_fit, model_identity
        path.write_text(json.dumps(raw))
        fit = load_emission_b_fit(path, mode)
        evidence = raw['renderer_qualification'][mode]
        digest = model_identity(fit)
        report = dict(evidence, status='passed', model_sha256=digest,
                      detector_acceptance_passed=True, material_coverage_passed=True,
                      fresh_validation_passed=True, fitting_source_sha256=['test-fit-source'])
        fresh = {'status': 'passed', 'model_sha256': digest, 'source_sha256': ['test-validation-source']}
        refs = []
        for role, value in [('qualification', report), ('fresh_validation', fresh)]:
            target = path.with_name(role+'.json')
            target.write_text(json.dumps(value))
            refs.append({'role': role, 'path': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        evidence.update(model_sha256=digest, evidence_files=refs)
        path.write_text(json.dumps(raw))
        return raw
    return bind
