"""Validate bounded new CASINO exports before admitting compact histories."""
from pathlib import Path
import json
import numpy as np
from .casino_compact import compact_summary, select_surface_exits, _write_json
from ..domain_randomisation.dataset_forms import _sha256_path

def ingest_chunk(source: Path, output: Path, *, tilt_deg: float, repeat: int, seed: int | None,
                 run_provenance: dict | None = None):
    source=source.resolve()
    output=output.resolve()
    if run_provenance is not None:
        if set(run_provenance) != {'sim_sha256', 'cas_sha256'} or any(
                not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                for v in run_provenance.values()):
            raise ValueError('Run provenance requires SIM and CAS SHA-256 hashes.')
    if not source.is_file() or source.stat().st_size > 2*1024**3:
        raise ValueError('Export must exist and be at most 2 GiB. This import guard does not control CASINO while exporting.')
    if not np.isfinite(tilt_deg) or not 0 <= tilt_deg < 90 or repeat<1 or (seed is not None and seed<0):
        raise ValueError('Finite tilt in [0,90), positive repeat and nonnegative seed required.')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Each imported chunk requires a new output directory.')
    # Count trajectory markers before allocating the compact cache.
    histories=0
    with source.open('rb') as handle:
        for line in handle:
            if line.lstrip().startswith((b'"Trajectory ', b'"Trajectory"')):
                histories+=1
                if histories>1000:
                    raise ValueError('Chunk exceeds 1000 trajectory histories.')
    output.mkdir(parents=True,exist_ok=True)
    cache=output/'boundaries.npz'
    metadata=compact_summary(source,cache)
    if not 0 < metadata['trajectory_count'] <= 1000:
        raise ValueError('Chunk requires between one and 1000 histories.')
    with np.load(cache,allow_pickle=False) as data:
        if not np.isfinite(data['first']).all() or not np.isfinite(data['final']).all():
            raise ValueError('Incomplete or nonfinite boundary history.')
        if not np.allclose(data['first'][:,6],60.,atol=.001,rtol=0):
            raise ValueError('Chunk is not a 60 keV campaign export.')
    samples,blocks=select_surface_exits(cache,tilt_deg,repeat,1.)
    if not len(samples):
        raise ValueError('Chunk has no eligible same-surface exits; inspect its CASINO export.')
    np.savez_compressed(output/'selected_exits.npz',samples=samples)
    result={'status':'validated_compact_chunk','declared_seed':seed,
            'seed_provenance':('CASINO-managed; explicit seed control not established' if seed is None else 'operator-supplied input setting; not inferred from trajectory directions'),
            'repeat':repeat,'tilt_deg':tilt_deg,'energy_kev':60.,'histories':metadata['trajectory_count'],
            'selected_exits':len(samples),'selection':blocks,
            'source_path':str(source),'source_sha256':metadata['sha256'],
            'compact_sha256':_sha256_path(cache),'compact_path':str(cache),
            'raw_source_preserved':True,'eligible_for_model_qualification':False,
            'remaining_checks':['alloy identity','independent source identity','geometry convergence','held-out model validation']}
    if run_provenance is not None:
        result.update(run_provenance)
        result['selected_exits_sha256']=_sha256_path(output/'selected_exits.npz')
    # Publish once, with the bindings already present. A second immediate
    # overwrite used to leave an incomplete receipt if a sync client locked it.
    _write_json(output/'chunk_receipt.json',result)
    return result
