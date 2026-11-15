from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from .angular_table import AngularTable

EMISSION_B_MODE_CONSTANT = "constant"
EMISSION_B_MODE_ASSUMED_CONSTANT = "assumed_constant"
EMISSION_B_MODE_LAMBERTIAN = "lambertian"
EMISSION_B_MODE_POOLED_CONSTANT_FIT = "pooled_constant_fit"
EMISSION_B_MODE_SIN_FIT = "b_alpha_sin"
EMISSION_B_MODE_QUADRATIC_FIT = "quadratic_fit"
EMISSION_B_MODE_ANGULAR_TABLE = "angular_table"
EMISSION_B_MODES = (
    EMISSION_B_MODE_ASSUMED_CONSTANT,
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
    EMISSION_B_MODE_ANGULAR_TABLE,
)
_LOADED_EMISSION_B_MODES = (
    EMISSION_B_MODE_ASSUMED_CONSTANT,
    EMISSION_B_MODE_ANGULAR_TABLE,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
)
_DEFAULT_QUADRATIC_FORMULA = "B(alpha_deg) = b0 + b1 * sin(alpha_deg) + b2 * sin(alpha_deg)^2"


@dataclass(frozen=True)
class EmissionBFit:
    emission_b_mode: str
    formula: str
    source_path: str
    source_type: str
    fitted_b: float | None = None
    b0: float | None = None
    b1: float | None = None
    b2: float | None = None
    qualified_tilt_domain: tuple[float, float] | None = None
    qualified_energy_domain: tuple[float, float] | None = None
    qualified_material_z: float | None = None
    angular_table: AngularTable | None = None
    assumed_tilt_domain: tuple[float, float] | None = None
    assumption: dict | None = None

    def evaluate(self, tilt_deg: np.ndarray | float) -> np.ndarray:
        tilt = np.asarray(tilt_deg, dtype=np.float64)
        if not np.all(np.isfinite(tilt)) or np.any((tilt < 0) | (tilt > 90)):
            raise ValueError("Emission incidence must be finite and lie in [0, 90] degrees.")
        if self.qualified_tilt_domain is not None:
            lo, hi = self.qualified_tilt_domain
            if np.any((tilt < lo - 1e-5) | (tilt > hi + 1e-5)):
                raise ValueError(f"Emission fit is only qualified for incidence {lo} to {hi} degrees; extrapolation is forbidden.")
        if self.assumed_tilt_domain is not None:
            lo, hi = self.assumed_tilt_domain
            if np.any((tilt < lo-1e-5) | (tilt > hi+1e-5)):
                raise ValueError(f'Incidence is outside the declared assumed domain [{lo}, {hi}].')
        if self.angular_table is not None:
            self.angular_table.validate_domain(tilt)
            raise ValueError('An angular table is a density, not a B coefficient. Use density().')
        if self.emission_b_mode in (EMISSION_B_MODE_POOLED_CONSTANT_FIT, EMISSION_B_MODE_ASSUMED_CONSTANT):
            if self.fitted_b is None:
                raise ValueError("Pooled constant B fit is missing fitted_b.")
            return np.full_like(tilt, float(self.fitted_b), dtype=np.float32)
        if self.emission_b_mode not in {
            EMISSION_B_MODE_SIN_FIT,
            EMISSION_B_MODE_QUADRATIC_FIT,
        }:
            raise ValueError(f"Unsupported loaded emission B mode '{self.emission_b_mode}'.")
        if self.b0 is None or self.b1 is None:
            raise ValueError("Tilt-dependent B fit is missing b0 or b1.")
        s = np.sin(np.radians(tilt))
        b2 = 0.0 if self.emission_b_mode == EMISSION_B_MODE_SIN_FIT else self.b2
        if b2 is None:
            raise ValueError("Quadratic B fit is missing b2.")
        return (self.b0 + self.b1 * s + b2 * s * s).astype(np.float32)


# Backward-compatible alias for older imports.
EmissionBCurveFit = EmissionBFit


def model_identity(fit: EmissionBFit) -> str:
    model = {'mode': fit.emission_b_mode, 'fitted_b': fit.fitted_b, 'b0': fit.b0,
             'b1': fit.b1, 'b2': fit.b2,
             'angular_table': None if fit.angular_table is None else fit.angular_table.to_dict()}
    return hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_qualification_evidence(fit, evidence):
    references = evidence.get('evidence_files', [])
    if not references or evidence.get('model_sha256') != model_identity(fit):
        raise ValueError('Qualification must bind model coefficients and hashed evidence files.')
    roles = {}
    for ref in references:
        path = (Path(fit.source_path).parent / ref['path']).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref.get('sha256'):
            raise ValueError(f'Qualification evidence hash mismatch: {path}')
        roles[ref['role']] = json.loads(path.read_text(encoding='utf-8'))
    report = roles.get('qualification', {})
    for role in ('qualification','fresh_validation'):
        for source in roles.get(role,{}).get('source_files',[]):
            path=Path(source['path'])
            with path.open('rb') as handle:
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
            if digest!=source['sha256']:
                raise ValueError(f'Qualification source hash mismatch: {path}')
    if report.get('model_sha256') != model_identity(fit) or report.get('status') != 'passed':
        raise ValueError('Qualification report does not approve this exact model.')
    for gate in ('surface_selection_passed', 'adequacy_passed', 'repeat_holdout_passed',
                 'tilt_holdout_passed', 'beats_lambertian', 'detector_acceptance_passed',
                 'material_coverage_passed', 'fresh_validation_passed'):
        if report.get(gate) is not True:
            raise ValueError(f'Qualification report is missing a passing {gate}.')
    if report.get('tilt_domain_deg') != evidence.get('tilt_domain_deg') or report.get('energy_domain_kev') != evidence.get('energy_domain_kev') or report.get('material_z') != evidence.get('material_z'):
        raise ValueError('Qualification domain differs from its bound evidence report.')
    validation = roles.get('fresh_validation', {})
    if validation.get('status') != 'passed' or validation.get('model_sha256') != model_identity(fit):
        raise ValueError('Fresh independent validation is required for this model.')
    fitting_hashes = set(report.get('fitting_source_sha256', []))
    validation_hashes = set(validation.get('source_sha256', []))
    if not fitting_hashes or not validation_hashes or fitting_hashes & validation_hashes:
        raise ValueError('Fit and fresh-validation sources must be nonempty and disjoint.')


def qualify_emission_b_fit(fit: EmissionBFit) -> EmissionBFit:
    """Require independent model-specific evidence before renderer use.

    A likelihood ranking alone is not a qualification certificate.
    Historical reports remain readable through load_emission_b_fit for analysis.
    """
    if fit.emission_b_mode == EMISSION_B_MODE_ASSUMED_CONSTANT:
        raise ValueError('An engineering assumption is not an emission qualification.')
    raw = json.loads(Path(fit.source_path).read_text(encoding="utf-8"))
    evidence = (raw.get("renderer_qualification") or {}).get(fit.emission_b_mode, {})
    gates = ("surface_selection_passed", "adequacy_passed", "repeat_holdout_passed", "tilt_holdout_passed", "beats_lambertian")
    if any(evidence.get(key) is not True for key in gates):
        raise ValueError("Emission fit lacks passing renderer qualification. Use Lambertian emission or an explicit diagnostic comparison.")
    verify_qualification_evidence(fit, evidence)
    domain = np.asarray(evidence.get("tilt_domain_deg", []), dtype=float)
    if domain.shape != (2,) or not np.isfinite(domain).all() or not 0 <= domain[0] < domain[1] <= 90:
        raise ValueError("Qualification requires a finite tilt_domain_deg [min, max] within [0, 90].")
    if not evidence.get("source_sha256"):
        raise ValueError("Qualification requires source_sha256 provenance.")
    energy = np.asarray(evidence.get("energy_domain_kev", []), dtype=float)
    material_z = float(evidence.get("material_z", float("nan")))
    if energy.shape != (2,) or not np.isfinite(energy).all() or not 0 < energy[0] <= energy[1] or not np.isfinite(material_z) or material_z <= 0:
        raise ValueError("Qualification requires energy_domain_kev and material_z; energy/material transfer is not automatic.")
    # B(s)*s is cubic at most. Check its extrema over the full domain.
    if fit.angular_table is not None:
        fit.angular_table.validate_domain(domain)
        return replace(fit, qualified_tilt_domain=tuple(domain), qualified_energy_domain=tuple(energy), qualified_material_z=material_z)
    c = [float(fit.fitted_b), 0., 0.] if fit.fitted_b is not None else [float(fit.b0), float(fit.b1), float(fit.b2 or 0.)]
    if not np.isfinite(c).all():
        raise ValueError("B coefficients must be finite.")
    lo, hi = np.sin(np.radians(domain))
    points = [lo, hi]
    derivative = np.trim_zeros([3*c[2], 2*c[1], c[0]], 'f')
    if len(derivative) > 1:
        points += [float(x.real) for x in np.roots(derivative) if abs(x.imag) < 1e-10 and lo <= x.real <= hi]
    if any(abs(c[0]*s + c[1]*s*s + c[2]*s*s*s) > 1 + 1e-12 for s in points):
        raise ValueError("B model violates global positivity within its qualified domain.")
    return replace(fit, qualified_tilt_domain=(float(domain[0]), float(domain[1])),
                   qualified_energy_domain=(float(energy[0]), float(energy[1])), qualified_material_z=material_z)


def validate_emission_b_mode(mode: str | None) -> str:
    if mode is None:
        return EMISSION_B_MODE_LAMBERTIAN
    value = str(mode).strip().lower()
    if value not in EMISSION_B_MODES:
        raise ValueError(
            f"Unknown emission B mode '{mode}'. Expected one of: {', '.join(EMISSION_B_MODES)}."
        )
    return value


def _build_constant_fit(model: dict, source_path: Path, source_type: str) -> EmissionBFit | None:
    if "fitted_b" not in model:
        return None
    fitted_b = float(model["fitted_b"])
    return EmissionBFit(
        emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT,
        fitted_b=fitted_b,
        formula=str(model.get("formula", f"B = {fitted_b:.6f}")),
        source_path=str(source_path),
        source_type=source_type,
    )


def _build_quadratic_fit(model: dict, source_path: Path, source_type: str) -> EmissionBFit | None:
    if not all(key in model for key in ("b0", "b1", "b2")):
        return None
    return EmissionBFit(
        emission_b_mode=EMISSION_B_MODE_QUADRATIC_FIT,
        b0=float(model["b0"]),
        b1=float(model["b1"]),
        b2=float(model["b2"]),
        formula=str(model.get("formula", _DEFAULT_QUADRATIC_FORMULA)),
        source_path=str(source_path),
        source_type=source_type,
    )


def _build_sin_fit(model: dict, source_path: Path, source_type: str) -> EmissionBFit | None:
    if not all(key in model for key in ("b0", "b1")):
        return None
    return EmissionBFit(
        emission_b_mode=EMISSION_B_MODE_SIN_FIT,
        b0=float(model["b0"]),
        b1=float(model["b1"]),
        b2=0.0,
        formula=str(model.get("formula", "B(alpha_deg) = b0 + b1 * sin(alpha_deg)")),
        source_path=str(source_path),
        source_type=source_type,
    )


def _load_from_gui_summary(raw: dict, source_path: Path, emission_b_mode: str) -> EmissionBFit | None:
    model_comparison = raw.get("model_comparison") or {}
    if emission_b_mode == EMISSION_B_MODE_POOLED_CONSTANT_FIT:
        model = model_comparison.get("constant_b")
        if isinstance(model, dict):
            return _build_constant_fit(model, source_path, "b_fit_gui_summary")
        return None

    if emission_b_mode == EMISSION_B_MODE_SIN_FIT:
        sin_model = model_comparison.get("b_alpha_sin")
        if isinstance(sin_model, dict):
            model = sin_model.get("fit") if isinstance(sin_model.get("fit"), dict) else sin_model
            if isinstance(model, dict):
                return _build_sin_fit(model, source_path, "b_fit_gui_summary")
        return None

    quadratic = model_comparison.get("b_alpha_quadratic")
    if isinstance(quadratic, dict):
        model = quadratic.get("fit") if isinstance(quadratic.get("fit"), dict) else quadratic
        if isinstance(model, dict):
            return _build_quadratic_fit(model, source_path, "b_fit_gui_summary")
    return None


def _load_from_report(raw: dict, source_path: Path, emission_b_mode: str) -> EmissionBFit | None:
    fit = raw.get("fit") or {}
    if emission_b_mode == EMISSION_B_MODE_POOLED_CONSTANT_FIT:
        for key in ("global_pooled", "global_sample_weighted", "global_equal_run"):
            model = fit.get(key)
            if isinstance(model, dict):
                loaded = _build_constant_fit(model, source_path, "b_fit_report")
                if loaded is not None:
                    return loaded
        return None

    if emission_b_mode == EMISSION_B_MODE_SIN_FIT:
        for key in ("b_alpha_sin_sample_weighted", "b_alpha_sin_equal_run"):
            model = fit.get(key)
            if isinstance(model, dict):
                loaded = _build_sin_fit(model, source_path, "b_fit_report")
                if loaded is not None:
                    return loaded
        return None

    for key in ("b_alpha_quadratic_sample_weighted", "b_alpha_quadratic_equal_run"):
        model = fit.get(key)
        if isinstance(model, dict):
            loaded = _build_quadratic_fit(model, source_path, "b_fit_report")
            if loaded is not None:
                return loaded
    return None


def load_emission_b_fit(
    path: str | Path,
    emission_b_mode: str = EMISSION_B_MODE_QUADRATIC_FIT,
) -> EmissionBFit:
    mode = validate_emission_b_mode(emission_b_mode)
    if mode not in _LOADED_EMISSION_B_MODES:
        raise ValueError(
            "Emission B fit JSON files only support pooled fitted constant B or quadratic fitted B(alpha)."
        )

    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Emission B fit file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Emission B fit file '{p}' must contain a JSON object.")

    if mode == EMISSION_B_MODE_ASSUMED_CONSTANT:
        if (raw.get('schema') != 'emission-assumption/v1' or raw.get('status') != 'engineering_assumption'
                or raw.get('physical_qualification') is not False):
            raise ValueError('Assumed constant requires an explicitly unqualified assumption artifact.')
        b = float(raw['fitted_b'])
        domain = np.asarray(raw.get('assumed_tilt_domain_deg', []), dtype=float)
        if (not np.isfinite(b) or abs(b) >= 1 or domain.shape != (2,) or not np.isfinite(domain).all()
                or not 0 <= domain[0] < domain[1] <= 90):
            raise ValueError('Assumed B must have |B| < 1 and a finite increasing incidence domain within [0,90].')
        if raw.get('energy_kev') != 60. or raw.get('material_name') != '304SS':
            raise ValueError('This assumption campaign supports only 304SS at 60 keV.')
        evidence = raw.get('coefficient_source', {})
        source = (p.parent/evidence.get('path', '')).resolve()
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != evidence.get('sha256'):
            raise ValueError('Assumed coefficient source hash mismatch.')
        source_b = json.loads(source.read_text())['model_comparison']['constant_b']['fitted_b']
        if b != source_b:
            raise ValueError('Assumed B differs from its recorded constant fit.')
        return EmissionBFit(mode, f'B = {b:.12g} (engineering assumption)', str(p),
            'explicit_engineering_assumption_not_physically_qualified', fitted_b=b,
            assumed_tilt_domain=tuple(domain), assumption=raw)

    if mode == EMISSION_B_MODE_ANGULAR_TABLE:
        table = raw.get('angular_table', {})
        return EmissionBFit(mode, 'p(mu,varphi|alpha): interpolated probability masses', str(p),
                            'angular_table/v1', angular_table=AngularTable(table.get('tilts_deg', []), table.get('probability_mass', []), table.get('prior_count_per_bin', .5)))

    loaded = None
    if "model_comparison" in raw:
        loaded = _load_from_gui_summary(raw, p, mode)
    if loaded is None and "fit" in raw:
        loaded = _load_from_report(raw, p, mode)
    if loaded is not None:
        return loaded

    if mode == EMISSION_B_MODE_POOLED_CONSTANT_FIT:
        expected = "pooled constant emission B fit"
    elif mode == EMISSION_B_MODE_SIN_FIT:
        expected = "sine emission B(alpha) fit"
    else:
        expected = "quadratic emission B fit"
    raise ValueError(
        f"File '{p}' does not contain a {expected}. Expected b_fit_gui_summary.json or b_fit_report.json."
    )
