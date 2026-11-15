"""Low-dimensional, regularized probability-mass tables for emission directions.

The fitted model is a log-linear correction to Lambertian bin masses. It can
change the polar marginal, preserves positivity and exports the existing table
format. No coefficient is interpreted as a newly qualified physical B value.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..renderer.angular_table import AngularTable


def angular_grid():
    edges = np.linspace(0, 1, 13)
    mu, az = np.meshgrid((edges[:-1]+edges[1:])/2,
                         (np.arange(24)+.5)*2*np.pi/24-np.pi, indexing='ij')
    base = np.broadcast_to((edges[1:]**2-edges[:-1]**2)[:, None]/24, (12, 24)).copy()
    return mu.ravel(), az.ravel(), base.ravel()


def features(tilts_deg, size):
    if size not in (4, 8):
        raise ValueError('Choose four or eight features.')
    tilts = np.atleast_1d(np.asarray(tilts_deg, dtype=float))
    if not np.isfinite(tilts).all() or np.any((tilts<0)|(tilts>=90)):
        raise ValueError('Finite incidence in [0,90) is required.')
    mu, az, _ = angular_grid()
    s = np.sin(np.radians(tilts))[:, None]
    q = np.broadcast_to(2*mu**2-1, (len(tilts), len(mu)))
    h = s*np.sqrt(1-mu**2)*np.cos(az)
    parts = [q, s*s*q, h, s*s*h]
    if size == 8:
        p2 = (3*q*q-1)/2
        parts += [p2, s*s*p2, q*h, s*s*(1-mu**2)*np.cos(2*az)]
    return np.stack(parts, axis=-1)


@dataclass(frozen=True)
class RegularizedAngularFit:
    coefficients: np.ndarray
    feature_scale: np.ndarray
    feature_count: int
    ridge: float
    supported_knots: np.ndarray
    optimizer_diagnostics: dict

    def table(self, knots=None):
        knots = self.supported_knots if knots is None else np.asarray(knots, dtype=float)
        if np.any(knots < self.supported_knots[0]) or np.any(knots > self.supported_knots[-1]):
            raise ValueError('Angular fit extrapolation is forbidden.')
        _, _, base = angular_grid()
        f = features(knots, self.feature_count)/self.feature_scale
        logits = np.log(base)[None, :] + f@self.coefficients
        mass = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
        return AngularTable(knots, mass.reshape(-1, 12, 24), prior_count_per_bin=None)

    def to_dict(self):
        return {'method': 'regularized_log_linear_angular_mass/v1',
                'feature_count': self.feature_count, 'ridge': self.ridge,
                'coefficients': self.coefficients.tolist(),
                'feature_scale': self.feature_scale.tolist(),
                'supported_knots_deg': self.supported_knots.tolist(),
                'optimizer': self.optimizer_diagnostics,
                'normalization': 'softmax relative to integrated Lambertian bin mass',
                'weighting': 'equal simulation-file weight'}


def fit_regularized(runs, *, feature_count, ridge):
    """runs are Nx5 arrays: alpha, mu, azimuth, repeat, nominal tilt."""
    if not runs or not np.isfinite(ridge) or ridge < 0:
        raise ValueError('Nonempty runs and a finite nonnegative ridge are required.')
    for x in runs:
        if (x.ndim!=2 or x.shape[1]!=5 or not len(x) or not np.isfinite(x).all()
                or not np.all(x[:,4]==x[0,4]) or np.any((x[:,1]<=0)|(x[:,1]>1))):
            raise ValueError('Each run needs finite outgoing samples at one incidence.')
    knots = np.array(sorted({float(x[0, 4]) for x in runs}))
    if len(knots) < 2:
        raise ValueError('At least two incidence knots are required.')
    f = features(knots, feature_count)
    _, _, base = angular_grid()
    weights = np.zeros(len(knots))
    observed = np.zeros((len(knots), 288))
    for x in runs:
        if (x.ndim!=2 or x.shape[1]!=5 or not len(x) or not np.isfinite(x).all()
                or not np.all(x[:,4]==x[0,4]) or np.any((x[:,1]<=0)|(x[:,1]>1))):
            raise ValueError('Each run needs finite outgoing samples at one incidence.')
        idx = int(np.searchsorted(knots, x[0,4]))
        mi = np.clip((x[:,1]*12).astype(int),0,11)
        ai = np.clip((((x[:,2]+np.pi)%(2*np.pi))/(2*np.pi)*24).astype(int),0,23)
        observed[idx] += np.bincount(mi*24+ai, minlength=288)/len(x)/len(runs)
        weights[idx] += 1/len(runs)
    scale = np.sqrt(np.sum(f*f*base[None,:,None]*weights[:,None,None], axis=(0,1)))
    scale = np.maximum(scale, 1e-8)
    f /= scale
    empirical = np.sum(observed[:,:,None]*f,axis=(0,1))

    def objective(beta):
        logits = np.log(base)[None,:]+f@beta
        log_z = logsumexp(logits,axis=1)
        mass = np.exp(logits-log_z[:,None])
        loss = weights@log_z-empirical@beta+.5*ridge*(beta@beta)
        gradient = np.sum(weights[:,None,None]*mass[:,:,None]*f,axis=(0,1))-empirical+ridge*beta
        return float(loss), gradient

    result = minimize(objective,np.zeros(feature_count),jac=True,method='L-BFGS-B',
                      options={'ftol':1e-12,'gtol':1e-8,'maxiter':1000})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Angular fit failed: {result.message}')
    return RegularizedAngularFit(result.x,scale,feature_count,float(ridge),knots,
        {'success':True,'iterations':int(result.nit),'penalized_objective':float(result.fun),
         'gradient_max_abs':float(np.max(np.abs(result.jac)))})


def table_nll(samples, table):
    density = table.density(samples[:,1],samples[:,2],samples[:,4])
    if not np.all(density>0):
        raise ValueError('Nonpositive table density.')
    return float(-np.log(density).mean())
