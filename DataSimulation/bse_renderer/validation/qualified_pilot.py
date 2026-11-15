"""Resumable operating-domain survey. Diagnostic images are never training data."""
from __future__ import annotations

from dataclasses import replace, asdict
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from ..config_loader import load_config_bundle, default_slice_runtime
from ..domain_randomisation.dataset_forms import (
    DATASET_CLASSES, _set_regime, _regime_for_sample, _sample_defect,
    _planned_target_frame, _sha256_path,
)
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..models import WeldLayoutRuntime
from ..renderer.scan_engine import ScanEngine
from .casino_compact import _write_json


def run_domain_pilot(output: Path, *, config_dir: Path, seed: int = 914610001,
                     static_per_class: int = 10, sequences_per_class: int = 2,
                     max_seconds: float = 86400, emission_mode: str = 'lambertian',
                     emission_path: str | None = None) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config_bundle(config_dir)
    identity = {'seed': seed, 'static_per_class': static_per_class,
                'sequences_per_class': sequences_per_class, 'energy_kev': 60.,
                'emission_mode': emission_mode, 'emission_path': emission_path,
                'config_hashes': {p.name: _sha256_path(p) for p in cfg.config_dir.glob('*.yaml')},
                'source_hashes': {str(p.relative_to(Path(__file__).parents[1])): _sha256_path(p)
                                  for p in Path(__file__).parents[1].rglob('*.py')}}
    if emission_path:
        identity['emission_sha256'] = _sha256_path(Path(emission_path))
    identity_path = output / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Pilot inputs or source changed; use a new pilot directory.')
    else:
        _write_json(identity_path, identity)
    start = time.monotonic()
    rows = []
    for kind, count, size in [('static', static_per_class, 128), ('sequence', sequences_per_class, 576)]:
        engine = ScanEngine(image_size=size, pixel_size_mm=.05, render_model_mode='formula_only',
                            emission_b_mode=emission_mode, emission_b_path=emission_path,
                            render_view_mode='top_surface',
                            weld_layout=WeldLayoutRuntime(mode='local_patch' if kind == 'static' else 'straight_seam'))
        for ci, name in enumerate(DATASET_CLASSES):
            for rep in range(count):
                case_id = f'{kind}-{name}-{rep:02d}'
                record_path = output / f'{case_id}.json'
                if record_path.exists():
                    row = json.loads(record_path.read_text())
                    for rel, digest in row['artifacts'].items():
                        if _sha256_path(output / rel) != digest:
                            raise ValueError(f'Pilot artifact changed: {rel}')
                    rows.append(row)
                    continue
                if time.monotonic() - start >= max_seconds:
                    return {'status': 'paused', 'completed_cases': len(rows)}
                case_start = time.monotonic()
                attempts = []
                result = None
                for attempt in range(25):
                    case_seed = seed + (100000 if kind == 'sequence' else 0) + ci * 10000 + rep + attempt * 1000000
                    sampled = sample_parameters(cfg, seed=case_seed, material_name='304SS', max_defects=0)
                    target = None if kind == 'static' or name == 'clean' else _planned_target_frame(rep, ci, frames_per_sequence=9, context_radius=2)
                    section = default_slice_runtime(cfg.slice)
                    if target is not None:
                        section = replace(section, slice_offset_mm=(target - 4) * 64 * .05)
                    sampled = replace(_set_regime(sampled, _regime_for_sample(rep, ci), 60.),
                                      slice=section, defects=_sample_defect(name, np.random.default_rng(case_seed)))
                    result = engine.run(sampled)
                    topo = np.asarray(result.channels.topo)
                    mask = np.asarray(result.defect_mask)
                    crops = [topo] if kind == 'static' else [topo[256:320, f*64:(f+1)*64] for f in range(9)]
                    visible = name == 'clean' or name in [d.defect_type for d in result.defects]
                    if target is not None:
                        visible = visible and bool(mask[256:320, target*64:(target+1)*64].any())
                    signal = all(np.isfinite(c).all() and np.ptp(c) > 1e-12 for c in crops)
                    attempts.append({'seed': case_seed, 'visible': bool(visible), 'signal': bool(signal),
                                     'operating_domain': result.metadata['operating_domain']})
                    if visible and signal:
                        break
                artifact = output / f'{case_id}.npz'
                np.savez_compressed(artifact, topo=topo, mask=mask)
                lo, hi = np.quantile(topo, [.01, .99])
                preview = np.uint8(np.clip((topo-lo)/max(hi-lo, 1e-12), 0, 1)*255)
                preview_path = output / f'{case_id}.png'
                Image.fromarray(preview).save(preview_path)
                row = {'id': case_id, 'kind': kind, 'class': name, 'seed': case_seed,
                       'accepted': bool(visible and signal), 'attempts': attempts,
                       'metadata': result.metadata, 'parameters': asdict(sampled),
                       'seconds': time.monotonic()-case_start,
                       'artifacts': {p.name: _sha256_path(p) for p in [artifact, preview_path]}}
                _write_json(record_path, row)
                rows.append(row)
                print(f'{case_id}: {len(attempts)} attempts, {row["seconds"]:.1f}s, incidence {result.metadata["operating_domain"]["incidence_deg_quantiles"][-1]:.2f}', flush=True)
    domains = [attempt['operating_domain'] for row in rows for attempt in row['attempts']]
    result = {'status': 'completed', 'purpose': 'operating_domain_diagnostic_not_training',
              'case_count': len(rows), 'accepted_count': sum(r['accepted'] for r in rows),
              'incidence_domain_deg': [min(d['incidence_deg_quantiles'][0] for d in domains), max(d['incidence_deg_quantiles'][-1] for d in domains)],
              'material_z_domain': [min(d['material_z_range'][0] for d in domains), max(d['material_z_range'][1] for d in domains)],
              'energy_domain_kev': [60., 60.], 'material': asdict(cfg.material.presets['304SS']),
              'incidence_histogram_1deg': np.sum([d['incidence_histogram_1deg'] for d in domains], axis=0).tolist(),
              'seconds': sum(r['seconds'] for r in rows),
              'bytes': sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
              'cases': [{'id': r['id'], 'kind': r['kind'], 'class': r['class'], 'accepted': r['accepted'],
                         'attempts': len(r['attempts']), 'record_sha256': _sha256_path(output/f'{r["id"]}.json')} for r in rows]}
    _write_json(output / 'summary.json', result)
    return result
