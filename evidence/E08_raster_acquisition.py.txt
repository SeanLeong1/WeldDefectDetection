"""Raster-timed acquisition v2. Mechanistic structure; unmeasured defaults.

Raw quadrant units are not calibrated electron counts. The effective count
rate explicitly encodes that unknown conversion. No temperature-to-intensity
conversion or alternating detector scan axes is used.
"""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class AcquisitionProfile:
    effective_count_rate: float = 1e9  # counts / (raw quadrant unit * second)
    readout_rms: float = .00045       # raw quadrant units, at reference dwell
    reference_dwell_us: float = 8.
    baseline_rms: float = .00020      # stationary independent electronic offsets
    common_gain_rms: float = .002     # dimensionless shared beam/electronics gain
    correlation_time_s: float = .25
    flyback_us: float = 20.
    contamination_per_s: float = .0002
    velocity_x_mm_s: float = 0.       # specimen/beam displacement, separate from offsets
    velocity_y_mm_s: float = 0.
    evidence: str = 'assumed_sensitivity_reference_not_measured'

    def validate(self):
        for key, value in asdict(self).items():
            if key == 'evidence':
                continue
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite acquisition parameter: {key}')
            if key not in ('velocity_x_mm_s','velocity_y_mm_s') and value < 0:
                raise ValueError(f'Negative acquisition parameter: {key}')
        if min(self.effective_count_rate,self.reference_dwell_us,self.correlation_time_s) <= 0:
            raise ValueError('Count rate, reference dwell and correlation time must be positive.')


def stream(seed, name):
    tag = int.from_bytes(hashlib.blake2s(name.encode(), digest_size=4).digest(),'little')
    return np.random.default_rng(np.random.SeedSequence([int(seed),tag]))


def stationary_ou(count, step_s, tau_s, rng):
    """Unit-variance OU process, initialized in its stationary distribution."""
    a = np.exp(-step_s/tau_s)
    innovations = rng.standard_normal(count)
    out = np.empty(count)
    out[0] = innovations[0]
    scale = np.sqrt(-np.expm1(-2*step_s/tau_s))
    for i in range(1,count):
        out[i] = a*out[i-1]+scale*innovations[i]
    return out


def pixel_times(shape, dwell_us, flyback_us):
    if not np.isfinite(dwell_us) or dwell_us <= 0:
        raise ValueError('Dwell time must be finite and positive.')
    h,w = shape
    dwell = dwell_us*1e-6
    line_time = w*dwell+flyback_us*1e-6
    return np.arange(h)[:,None]*line_time+np.arange(w)[None,:]*dwell, line_time


def electronic_field(shape,dwell_us,profile,seed,name):
    h,w = shape
    _,line_time = pixel_times(shape,dwell_us,profile.flyback_us)
    nodes = stationary_ou(h+1,line_time,profile.correlation_time_s,stream(seed,name))
    # Linear interpolation of slow drift at line boundaries. All quadrants
    # share one raster clock. Metadata reports this finite sampling approximation.
    fraction = np.arange(w)*dwell_us*1e-6/line_time
    return nodes[:-1,None]*(1-fraction)+nodes[1:,None]*fraction


def motion_coordinates(shape,dwell_us,spacing_mm,profile):
    if profile.velocity_x_mm_s == profile.velocity_y_mm_s == 0:
        return None
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError('Motion requires positive physical pixel spacing.')
    times,_ = pixel_times(shape,dwell_us,profile.flyback_us)
    y,x = np.indices(shape,dtype=float)
    return np.stack([y-profile.velocity_y_mm_s*times/spacing_mm,
                     x-profile.velocity_x_mm_s*times/spacing_mm])


def warp_image(image,coords,order=1):
    if coords is None:
        return image.copy()
    if image.ndim==3:
        return np.stack([map_coordinates(image[...,i],coords,order=order,mode='nearest')
                         for i in range(image.shape[-1])],axis=-1)
    return map_coordinates(image,coords,order=order,mode='nearest')


def acquire(quadrants,noise,*,seed,dwell_us,profile):
    profile.validate()
    q=np.asarray(quadrants,dtype=np.float64).copy()
    if q.ndim!=3 or q.shape[-1]!=4 or not np.isfinite(q).all() or np.any(q<0):
        raise ValueError('Acquisition expects finite, nonnegative four-quadrant signals.')
    shape=q.shape[:2]
    times,line_time=pixel_times(shape,dwell_us,profile.flyback_us)
    if noise.contamination_enabled:
        q *= np.exp(-profile.contamination_per_s*times)[...,None]
        for i,key in enumerate(('A','B','C','D')):
            gain=1+noise.quadrant_offsets.get(key,0.)
            if not np.isfinite(gain) or gain<=0:raise ValueError('Invalid detector gain offset.')
            q[...,i]*=gain
    if noise.thermal_enabled:
        common=electronic_field(shape,dwell_us,profile,seed,'common_gain')
        q*=np.maximum(0.,1+profile.common_gain_rms*common)[...,None]
    exposure=profile.effective_count_rate*dwell_us*1e-6
    for i in range(4):
        if noise.poisson_enabled:
            q[...,i]=stream(seed,f'shot_{i}').poisson(q[...,i]*exposure)/exposure
        if noise.thermal_enabled:
            baseline=profile.baseline_rms*electronic_field(shape,dwell_us,profile,seed,f'baseline_{i}')
            readout=profile.readout_rms*np.sqrt(profile.reference_dwell_us/dwell_us)
            q[...,i]+=baseline+readout*stream(seed,f'readout_{i}').standard_normal(shape)
    return np.maximum(q,0).astype(np.float32),dict(version='raster_v2',profile=asdict(profile),
        line_time_s=line_time,frame_time_s=float(times[-1,-1]+dwell_us*1e-6),
        drift_sampling='stationary OU at common raster line boundaries, linearly interpolated within each line',
        drift_line_interval_over_tau=line_time/profile.correlation_time_s,
        geometry_and_acquisition_rng='independent named SeedSequence streams',
        calibrated=False,legacy_temperature_bandwidth_contamination_rate_unused=True,
        legacy_session_progress_and_offsets_unused=True)
