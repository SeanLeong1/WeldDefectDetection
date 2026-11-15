"""Build a bounded CASINO 3.3 campaign from one hash-pinned, saved box template.

This is deliberately a narrow binary adapter, not a general .sim writer. Offsets
are valid ONLY for TEMPLATE_SHA. Cached geometry is checked against saved tilts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = PROJECT / 'DataSimulation/data/output/casino_reruns/parametric_model_tilt_304ss_60kev_lowtilt_x3/sim'
TEMPLATE_SHA = '5b6c9d8b95982402852bf4d7314bb00f9f9fc907d4f348e546a6ced481c523fb'
KNOTS = [0, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60]
MIDPOINTS = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 35, 42.5, 47.5, 55]


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def unpack(data, offset, fmt='d'):
    return struct.unpack_from('<' + fmt, data, offset)[0]


def make_sim(template: bytes, tilt: float, histories: int, width_scale=1., depth_scale=1.) -> bytes:
    if hashlib.sha256(template).hexdigest() != TEMPLATE_SHA:
        raise ValueError('Unsupported binary template. Refusing unverified offsets.')
    if not math.isfinite(tilt) or not 0 <= tilt <= 60:
        raise ValueError('This campaign supports 0 to 60 degrees.')
    if not 1 <= histories <= 1000 or not all(math.isfinite(s) and 1 <= s <= 4 for s in (width_scale, depth_scale)):
        raise ValueError('Require 1..1000 histories and geometry scales in [1,4].')
    out = bytearray(template)
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def rotate(x, y, z):
        return (c*x-s*z, y, s*x+c*z)

    for off, value in [(375, 40000*width_scale), (383, 40000*width_scale),
                       (391, 25000*depth_scale), (451, tilt)]:
        struct.pack_into('<d', out, off, value)
    vertices = []
    for t in range(12):
        start = 539 + 112*t
        for v in range(3):
            xyz = struct.unpack_from('<3d', template, start+24*v)
            xyz = rotate(xyz[0]*width_scale, xyz[1]*width_scale, xyz[2]*depth_scale)
            struct.pack_into('<3d', out, start+24*v, *xyz)
            vertices.append(xyz)
        normal = rotate(*struct.unpack_from('<3d', template, start+72))
        struct.pack_into('<3d', out, start+72, *normal)
    # Saved CASINO boxes use a cubic search volume: 20 times the largest
    # vertex coordinate, with a 5% beam-source margin. Triangle IDs are retained.
    bound = 20*max(abs(v) for xyz in vertices for v in xyz)
    struct.pack_into('<d', out, 471, 1.05*bound)
    struct.pack_into('<6d', out, 483, *([bound]*3 + [-bound]*3))
    struct.pack_into('<i', out, 5374, histories)
    struct.pack_into('<d', out, 5386, 60.)
    # One XY beam point at the centre of the incoming face (unrotated z=-d/2).
    # The template beam travels along +z; leave its energy/diameter/models intact.
    out[5626:] = struct.pack('<i3d', 1, 12500*depth_scale*s, 0., 0.)
    return bytes(out)


def check_saved_geometry(template: bytes) -> dict:
    errors = {}
    for tilt in [0, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60]:
        if tilt <= 25:
            path = TEMPLATE_DIR / f'{tilt}deg-r1.sim'
        else:
            path = PROJECT / f'DataSimulation/data/output/casino_reruns/parametric_model_tilt_304ss_60kev_x5_rerun/sim/{tilt}deg-r1/{tilt}deg-r1.sim'
        saved = path.read_bytes()
        predicted = make_sim(template, tilt, 1000,
                             unpack(saved, 375)/40000, unpack(saved, 391)/25000)
        offsets = [539+112*t+8*k for t in range(12) for k in range(12)]
        err = max(abs(unpack(saved, off)-unpack(predicted, off)) for off in offsets)
        if err > .001:
            raise ValueError(f'Saved geometry mismatch at {tilt} degrees: {err}')
        if saved[1883:5374] != template[1883:5374] or saved[5378:5626] != template[5378:5626]:
            raise ValueError(f'Material or physical settings differ at {tilt} degrees.')
        errors[str(tilt)] = {'max_coordinate_or_normal_error': err, 'source_sha256': sha(path),
                             'saved_histories_per_point': unpack(saved, 5374, 'i'),
                             'saved_depth_nm': unpack(saved, 391)}
    return errors


def build(destination: Path, histories=500):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Use a new campaign directory; existing files are preserved.')
    template = (TEMPLATE_DIR/'0deg-r1.sim').read_bytes()
    checks = check_saved_geometry(template)
    jobs = []
    # Geometry probes are an initial comparison, not a claim of convergence.
    stages = [('geometry', [60], [1, 2], [('base', 1., 1.), ('wide', 2., 1.), ('deep', 1., 2.)]),
              ('selection', KNOTS, [1, 2, 3, 4], [('base', 1., 1.)]),
              ('interpolation', MIDPOINTS, [1, 2, 3, 4], [('base', 1., 1.)]),
              ('confirmation', sorted(KNOTS+MIDPOINTS), [11, 12, 13, 14], [('base', 1., 1.)])]
    for stage, tilts, repeats, geometries in stages:
        for tilt in tilts:
            for geometry, ws, ds in geometries:
                for repeat in repeats:
                    name = f'{tilt:g}deg-r{repeat}'
                    folder = destination/stage/geometry/name
                    folder.mkdir(parents=True)
                    sim = folder/f'{name}.sim'
                    sim.write_bytes(make_sim(template, tilt, histories, ws, ds))
                    script = folder/'run.txt'
                    cas = folder/f'{name}.cas'
                    script.write_text(f'LOAD "{sim}";\nSIM "{cas}";\n', encoding='ascii')
                    load = folder/'check_load.txt'
                    load.write_text(f'LOAD "{sim}";\n', encoding='ascii')
                    jobs.append({'stage': stage, 'geometry': geometry, 'name': name,
                                 'tilt_deg': tilt, 'repeat': repeat, 'histories': histories,
                                 'width_nm': 40000*ws, 'depth_nm': 25000*ds,
                                 'beam_xy_nm': [12500*ds*math.sin(math.radians(tilt)), 0.],
                                 'sim_path': str(sim.relative_to(destination)), 'sim_sha256': sha(sim),
                                 'script_path': str(script.relative_to(destination)), 'script_sha256': sha(script),
                                 'cas_path': str(cas.relative_to(destination)),
                                 'seed': None, 'seed_provenance': 'CASINO-managed; explicit seed control not established'})
    manifest = {'schema': 1, 'status': 'prepared_not_simulated', 'energy_kev': 60,
                'material': '304 stainless steel, inherited unchanged from saved source',
                'template_path': str(TEMPLATE_DIR/'0deg-r1.sim'), 'template_sha256': TEMPLATE_SHA,
                'geometry_template_checks': checks, 'selection_tilts_deg': KNOTS,
                'interpolation_tilts_deg': MIDPOINTS, 'max_dat_bytes': 2*1024**3,
                'histories_per_file': histories, 'jobs': jobs}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--histories', type=int, default=500)
    args = parser.parse_args()
    result = build(args.output, args.histories)
    print(json.dumps({'jobs': len(result['jobs']), 'histories_per_file': args.histories, 'output': str(args.output)}))
