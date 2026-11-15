"""Probability-mass interpolation for a surface-conditioned angular density."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AngularTable:
    tilts_deg: np.ndarray
    probability_mass: np.ndarray
    prior_count_per_bin: float | None = .5

    def __post_init__(self):
        knots = np.asarray(self.tilts_deg, dtype=float)
        mass = np.asarray(self.probability_mass, dtype=float)
        if (knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all()
                or not np.all(np.diff(knots) > 0) or knots[0] < 0 or knots[-1] > 90):
            raise ValueError('Angular knots must increase within [0,90].')
        if mass.shape != (len(knots), 12, 24) or not np.isfinite(mass).all() or np.any(mass < 0):
            raise ValueError('Angular mass must be finite, nonnegative and shaped [knots,12,24].')
        if not np.allclose(mass.sum(axis=(1, 2)), 1., atol=1e-10, rtol=0):
            raise ValueError('Each angular table must have probability mass one.')
        if self.prior_count_per_bin is not None and (not np.isfinite(self.prior_count_per_bin) or self.prior_count_per_bin < 0):
            raise ValueError('Prior count must be nonnegative or absent.')
        object.__setattr__(self, 'tilts_deg', knots)
        object.__setattr__(self, 'probability_mass', mass)

    def validate_domain(self, alpha_deg):
        alpha = np.asarray(alpha_deg)
        if not np.isfinite(alpha).all() or np.any(alpha < self.tilts_deg[0]-1e-5) or np.any(alpha > self.tilts_deg[-1]+1e-5):
            raise ValueError('Angular table extrapolation is forbidden.')

    def density(self, mu, azimuth_rad, alpha_deg):
        mu, az, alpha = np.broadcast_arrays(mu, azimuth_rad, alpha_deg)
        self.validate_domain(alpha)
        if not np.isfinite(mu).all() or not np.isfinite(az).all():
            raise ValueError('Angular coordinates must be finite.')
        alpha = np.clip(alpha, self.tilts_deg[0], self.tilts_deg[-1])
        upper = np.clip(np.searchsorted(self.tilts_deg, alpha, side='right'), 1, len(self.tilts_deg)-1)
        lower = upper-1
        w = (alpha-self.tilts_deg[lower])/(self.tilts_deg[upper]-self.tilts_deg[lower])
        mi = np.clip((mu*12).astype(int), 0, 11)
        ai = np.clip((((az+np.pi) % (2*np.pi))/(2*np.pi)*24).astype(int), 0, 23)
        mass = (1-w)*self.probability_mass[lower, mi, ai] + w*self.probability_mass[upper, mi, ai]
        return np.where((mu > 0) & (mu <= 1), mass/(2*np.pi/288), 0.)

    def to_dict(self):
        return {'tilts_deg': self.tilts_deg.tolist(), 'probability_mass': self.probability_mass.tolist(),
                'measure': 'dOmega=dmu*dvarphi', 'prior_count_per_bin': self.prior_count_per_bin}
