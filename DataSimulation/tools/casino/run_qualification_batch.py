"""Run one small CASINO wave; require compact receipts before the next wave.

Each process writes directly to its own log (no undrained subprocess pipes).
Existing .cas files are never overwritten. A successful process exit AND a
nonempty output are required. Receipts bind inputs, executable and output hashes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_qualification_sims import PROJECT, sha
sys.path.insert(0, str(PROJECT/'DataSimulation'))
from bse_renderer.atomic_io import write_json_atomic


def write_json(path, data):
    write_json_atomic(path, data, temporary=path.with_suffix('.tmp'))


def input_paths(root, job):
    paths = {}
    for key in ('sim_path', 'script_path', 'cas_path'):
        path = (root/job[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Path escapes campaign: {path}')
        paths[key] = path
    for key in ('sim', 'script'):
        if sha(paths[key+'_path']) != job[key+'_sha256']:
            raise ValueError(f'{key} changed for {job["name"]}; build a new campaign.')
    return paths


def check_previous(root, job, paths, ingest):
    """True means previously completed and compacted; false means not launched."""
    folder = paths['sim_path'].parent
    run_file = folder/'run_receipt.json'
    if not run_file.exists():
        if paths['cas_path'].exists() or (folder/'console.log').exists() or (folder/'attempt_started.json').exists():
            raise ValueError(f'Unverified prior attempt in {folder}. Preserve and inspect it before retrying.')
        return False
    run = json.loads(run_file.read_text())
    if run['status'] != 'completed' or run['sim_sha256'] != job['sim_sha256']:
        raise ValueError(f'Failed or mismatched prior run: {run_file}')
    if not paths['cas_path'].is_file() or sha(paths['cas_path']) != run['cas_sha256']:
        raise ValueError(f'CAS output missing or changed: {paths["cas_path"]}')
    compact_dir = folder/'compact'
    receipt_file = compact_dir/'chunk_receipt.json'
    source = folder/(job['name']+'_summary.dat')
    if not receipt_file.exists() and ingest and source.is_file():
        sys.path.insert(0, str(PROJECT/'DataSimulation'))
        from bse_renderer.validation.casino_chunk_ingest import ingest_chunk
        result = ingest_chunk(source, compact_dir, tilt_deg=job['tilt_deg'], repeat=job['repeat'], seed=None,
                              run_provenance={'sim_sha256': job['sim_sha256'], 'cas_sha256': run['cas_sha256']})
        if result['histories'] != job['histories']:
            raise ValueError(f'Expected {job["histories"]} histories; export contains {result["histories"]}.')
    if not receipt_file.exists():
        raise ValueError(f'Export all trajectories to {source}, then run again with -IngestReady. No new jobs started.')
    receipt = json.loads(receipt_file.read_text())
    if (receipt.get('status') != 'validated_compact_chunk'
            or receipt.get('sim_sha256') != job['sim_sha256']
            or receipt.get('cas_sha256') != run['cas_sha256']
            or receipt.get('histories') != job['histories']
            or receipt.get('compact_sha256') != sha(compact_dir/'boundaries.npz')
            or receipt.get('selected_exits_sha256') != sha(compact_dir/'selected_exits.npz')):
        raise ValueError(f'Compact receipt failed integrity verification: {receipt_file}')
    return True


def record_finished_attempt(record, exe_hash, frozen_hash, *, interrupted=False):
    """Record an observed exit; a terminated process never becomes completed."""
    proc, job, paths = record['proc'], record['job'], record['paths']
    code = proc.poll()
    cas = paths['cas_path']
    ok = not interrupted and code == 0 and cas.is_file() and cas.stat().st_size > 0
    receipt = {'status': 'completed' if ok else ('interrupted' if interrupted else 'failed'),
               'returncode': code, 'pid': proc.pid, 'started_utc': record['started_utc'],
               'duration_s': time.monotonic()-record['start'],
               'console_sha256': exe_hash, 'sim_sha256': job['sim_sha256'],
               'script_sha256': job['script_sha256'],
               'cas_sha256': sha(cas) if ok else None,
               'cas_bytes': cas.stat().st_size if cas.exists() else 0,
               'seed': None, 'seed_provenance': job['seed_provenance'],
               'frozen_model_sha256': frozen_hash}
    write_json(paths['sim_path'].parent/'run_receipt.json', receipt)
    return receipt


def run_wave(root, exe, stage, max_parallel=2, ingest=False, dry_run=False, frozen_model=None):
    root, exe = root.resolve(), exe.resolve()
    if not 1 <= max_parallel <= 4:
        raise ValueError('Choose one to four simultaneous instances.')
    if not exe.is_file():
        raise FileNotFoundError(exe)
    manifest = json.loads((root/'manifest.json').read_text())
    pending = []
    # Check unfinished exports across ALL stages to prevent accumulating raw files.
    for job in manifest['jobs']:
        paths = input_paths(root, job)
        done = check_previous(root, job, paths, ingest) if not dry_run else (paths['sim_path'].parent/'run_receipt.json').exists()
        if not done and job['stage'] == stage:
            pending.append((job, paths))
    selected = pending[:max_parallel]
    if not selected:
        print('No pending jobs in this stage.')
        return 0
    if stage == 'confirmation' and not dry_run and (frozen_model is None or not frozen_model.is_file()):
        raise ValueError('Confirmation requires --frozen-model: the saved selected candidate, frozen before validation.')
    print(json.dumps({'stage': stage, 'jobs_this_wave': [str(p['sim_path']) for _, p in selected],
                      'remaining_after_wave': len(pending)-len(selected), 'dry_run': dry_run}, indent=2))
    if dry_run:
        return 0
    if shutil.disk_usage(root).free < (4+4*len(selected))*1024**3:
        raise ValueError('Insufficient free disk headroom for this wave and subsequent exports.')
    frozen_hash = sha(frozen_model) if frozen_model else None
    if stage == 'confirmation':
        identity_path = root/'confirmation_identity.json'
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            if identity['frozen_model_sha256'] != frozen_hash:
                raise ValueError('Confirmation candidate changed. Use a new confirmation campaign.')
        else:
            write_json(identity_path, {'frozen_model_sha256': frozen_hash,
                                       'frozen_model_path': str(frozen_model.resolve())})
    exe_hash = sha(exe)
    running = []
    failure = False
    try:
        for job, paths in selected:
            folder = paths['sim_path'].parent
            log = (folder/'console.log').open('x', encoding='utf-8')
            try:
                proc = subprocess.Popen([str(exe), str(paths['script_path'])], cwd=folder,
                                        stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            record = {'job': job, 'paths': paths, 'proc': proc, 'log': log, 'start': time.monotonic(),
                      'started_utc': datetime.now(timezone.utc).isoformat()}
            running.append(record)
            write_json(folder/'attempt_started.json', {
                'status': 'started', 'pid': proc.pid, 'started_utc': record['started_utc'],
                'console_sha256': exe_hash, 'sim_sha256': job['sim_sha256'],
                'script_sha256': job['script_sha256'], 'frozen_model_sha256': frozen_hash})
            print(f'START {job["stage"]}/{job["geometry"]}/{job["name"]} pid={proc.pid}', flush=True)
            # Different process start times are recorded, but are NOT asserted to
            # prove independent RNG streams. That requires output/seed checks.
            time.sleep(2)
        while running:
            for record in list(running):
                proc, job, paths = record['proc'], record['job'], record['paths']
                if time.monotonic()-record['start'] > 86400:
                    raise TimeoutError('CASINO wave exceeded 24 hours.')
                code = proc.poll()
                if code is None:
                    continue
                record['log'].close()
                receipt = record_finished_attempt(record, exe_hash, frozen_hash)
                running.remove(record)
                print(f'END {job["name"]}: {receipt["status"]}', flush=True)
                if receipt['status'] != 'completed':
                    failure = True
                    raise RuntimeError(f'CASINO failed: inspect {paths["sim_path"].parent / "console.log"}')
            if running:
                time.sleep(1)
    finally:
        # On failure/interrupt, terminate every process still owned by this wave.
        for record in running:
            record['terminated_by_launcher'] = record['proc'].poll() is None
            if record['terminated_by_launcher']:
                record['proc'].terminate()
        for record in running:
            try:
                record['proc'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                record['proc'].kill()
                record['proc'].wait()
            record['log'].close()
            # A process may have finished during the inter-launch pause. Preserve
            # its observed successful exit instead of leaving an orphaned CAS.
            record_finished_attempt(record, exe_hash, frozen_hash,
                                    interrupted=record['terminated_by_launcher'])
    print('Wave finished. Export the new .cas files to *_summary.dat before running the next wave.')
    return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--console-exe', required=True, type=Path)
    parser.add_argument('--stage', choices=['geometry', 'selection', 'interpolation', 'confirmation'], default='geometry')
    parser.add_argument('--max-parallel', type=int, default=2)
    parser.add_argument('--ingest-ready', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--frozen-model', type=Path)
    args = parser.parse_args()
    lock = args.root/'active_wave.lock'
    # Exclusive creation prevents two terminal invocations launching the same jobs.
    try:
        with lock.open('x') as handle:
            handle.write(datetime.now(timezone.utc).isoformat())
    except FileExistsError:
        parser.exit(2, f'Another wave may be active: {lock}. Inspect its processes before removing a stale lock.\n')
    try:
        return run_wave(args.root, args.console_exe, args.stage, args.max_parallel,
                        args.ingest_ready, args.dry_run, args.frozen_model)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted. Attempt receipts and outputs were preserved; inspect incomplete runs before retrying.', file=sys.stderr)
        return 130
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
