"""Release a model only from a complete, hash-bound qualification evidence bundle."""
from pathlib import Path
import json
import shutil
import math
from ..domain_randomisation.dataset_forms import _sha256_path
from ..renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit, model_identity
from .casino_compact import _write_json

def export_qualified_model(candidate: Path, evidence_report: Path, validation_report: Path,
                           output: Path, *, mode: str):
    fit=load_emission_b_fit(candidate,mode)
    report=json.loads(evidence_report.read_text())
    validation=json.loads(validation_report.read_text())
    digest=model_identity(fit)
    for item in (report,validation):
        if item.get('status')!='passed' or item.get('model_sha256')!=digest:
            raise ValueError('A passing report for the exact model is required.')
    # Numerical outcomes supplement flags. A hand-written passing flag is insufficient.
    improvement=validation.get('likelihood_improvement_ci95',[])
    if len(improvement)!=2 or not all(math.isfinite(x) for x in improvement) or not 0 < improvement[0] <= improvement[1]:
        raise ValueError('Fresh validation must improve held-out likelihood over B=0.')
    errors=validation.get('simultaneous_detector_error_ci95',[])
    if not errors or any(len(ci)!=2 or not -.02 <= ci[0] <= ci[1] <= .02 for ci in errors):
        raise ValueError('Fresh simultaneous detector intervals exceed the declared tolerance.')
    if validation.get('domain_coverage_passed') is not True or validation.get('selection_after_model_freeze') is not True:
        raise ValueError('Fresh validation must cover the domain and follow model freezing.')
    refs=report.get('source_files',[])+validation.get('source_files',[])
    if not refs:
        raise ValueError('Hashed source files are required.')
    for ref in refs:
        if _sha256_path(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Qualification source-file hash mismatch.')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Release output must be a new directory.')
    output.mkdir(parents=True,exist_ok=True)
    raw=json.loads(candidate.read_text())
    bound=[]
    for role,path in [('qualification',evidence_report),('fresh_validation',validation_report)]:
        dest=output/(role+'.json')
        shutil.copyfile(path,dest)
        bound.append({'role':role,'path':dest.name,'sha256':_sha256_path(dest)})
    gates=('surface_selection_passed','adequacy_passed','repeat_holdout_passed','tilt_holdout_passed','beats_lambertian')
    evidence={key:report.get(key) for key in gates}
    evidence.update({key:report[key] for key in ('tilt_domain_deg','energy_domain_kev','material_z')})
    evidence.update(model_sha256=digest,evidence_files=bound,source_sha256=_sha256_path(evidence_report))
    raw['renderer_qualification']={mode:evidence}
    temporary=output/'candidate_to_verify.json'
    _write_json(temporary,raw)
    qualify_emission_b_fit(load_emission_b_fit(temporary,mode))
    released=output/'qualified_model.json'
    temporary.replace(released)
    result={'status':'passed','qualified_model':{'mode':mode,'path':released.name,'sha256':_sha256_path(released)},
            'qualification_evidence_sha256':_sha256_path(evidence_report),
            'validation_evidence_sha256':_sha256_path(validation_report)}
    _write_json(output/'qualification_report.json',result)
    return result
