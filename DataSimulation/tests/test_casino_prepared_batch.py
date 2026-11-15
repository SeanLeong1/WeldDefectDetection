import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]/'tools/casino'
sys.path.insert(0, str(TOOLS))
import prepare_qualification_sims as prep
import run_qualification_batch as runner


@pytest.mark.parametrize('tilt', [0, 2.5, 5, 27.5, 35, 45, 55, 60])
def test_generated_beam_hits_face_centre_and_keeps_physics(tilt):
    source = (prep.TEMPLATE_DIR/'0deg-r1.sim').read_bytes()
    output = prep.make_sim(source, tilt, 500)
    assert output[1883:5374] == source[1883:5374]
    assert output[5378:5626] == source[5378:5626]
    assert prep.unpack(output, 5374, 'i') == 500
    assert prep.unpack(output, 5386) == 60
    assert prep.unpack(output, 5626, 'i') == 1
    assert len(output) == 5654
    x, y, _ = struct.unpack_from('<3d', output, 5630)
    s, c = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
    # Intersect the +z ray with the rotated incoming plane in object coordinates.
    z = (-12500+s*x)/c
    local_x = c*x+s*z
    local_z = -s*x+c*z
    assert local_x == pytest.approx(0, abs=1e-9)
    assert local_z == pytest.approx(-12500)
    assert y == 0
    for t in range(12):
        normal = np.array(struct.unpack_from('<3d', output, 611+112*t))
        assert np.linalg.norm(normal) == pytest.approx(1)


def test_binary_adapter_matches_saved_meshes_and_rejects_unknown_template():
    source = (prep.TEMPLATE_DIR/'0deg-r1.sim').read_bytes()
    assert len(prep.check_saved_geometry(source)) == 11
    with pytest.raises(ValueError, match='Unsupported binary'):
        prep.make_sim(source[:-1]+b'x', 30, 500)
    for angle, count in [(61,500), (float('nan'),500), (30,1001), (30,0)]:
        with pytest.raises(ValueError):
            prep.make_sim(source, angle, count)


def setup_wave(tmp_path, count=2):
    root = tmp_path/'campaign'
    root.mkdir()
    exe = tmp_path/'console.exe'
    exe.write_bytes(b'fake console for tests')
    jobs=[]
    for n in range(count):
        folder=root/f'job{n}'
        folder.mkdir()
        sim=folder/'test.sim'
        sim.write_bytes(b'sim')
        script=folder/'run.txt'
        cas=folder/'test.cas'
        script.write_text(str(cas))
        jobs.append({'name':f'job{n}', 'stage':'geometry','geometry':'base', 'histories':500,
                     'sim_path':str(sim.relative_to(root)), 'sim_sha256':prep.sha(sim),
                     'script_path':str(script.relative_to(root)), 'script_sha256':prep.sha(script),
                     'cas_path':str(cas.relative_to(root)), 'seed_provenance':'unknown'})
    (root/'manifest.json').write_text(json.dumps({'jobs':jobs}))
    return root, exe, jobs


def test_wave_requires_real_output_and_preserves_previous_attempt(tmp_path, monkeypatch):
    root, exe, jobs = setup_wave(tmp_path, 1)
    class EmptySuccess:
        pid = 12
        def poll(self): return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: EmptySuccess())
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    with pytest.raises(RuntimeError, match='CASINO failed'):
        runner.run_wave(root, exe, 'geometry')
    receipt=json.loads((root/'job0/run_receipt.json').read_text())
    assert receipt['status']=='failed'
    with pytest.raises(ValueError, match='Failed or mismatched'):
        runner.run_wave(root, exe, 'geometry')


def test_success_pauses_until_compaction_and_detects_input_changes(tmp_path, monkeypatch):
    root, exe, jobs = setup_wave(tmp_path)
    class Success:
        pid = 13
        def __init__(self, args, **kwargs):
            Path(Path(args[1]).read_text()).write_bytes(b'CAS output')
        def poll(self): return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', Success)
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    assert runner.run_wave(root, exe, 'geometry') == 0
    with pytest.raises(ValueError, match='Export all trajectories'):
        runner.run_wave(root, exe, 'geometry')
    (root/'job0/test.sim').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sim changed'):
        runner.run_wave(root, exe, 'geometry')


def test_failure_terminates_other_live_process(tmp_path, monkeypatch):
    root, exe, jobs = setup_wave(tmp_path)
    processes=[]
    class Process:
        pid=14
        def __init__(self, *args, **kwargs):
            self.failed = not processes
            self.terminated=False
            processes.append(self)
        def poll(self): return 1 if self.failed else (2 if self.terminated else None)
        def terminate(self): self.terminated=True
        def wait(self, timeout=None): return 2
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    with pytest.raises(RuntimeError):
        runner.run_wave(root, exe, 'geometry')
    assert len(processes)==2 and processes[1].terminated


def test_compact_receipt_rejects_changed_selected_exits(tmp_path):
    root, exe, jobs=setup_wave(tmp_path,1)
    job=jobs[0]
    paths=runner.input_paths(root, job)
    cas=paths['cas_path']
    cas.write_bytes(b'cas')
    folder=cas.parent
    (folder/'run_receipt.json').write_text(json.dumps({'status':'completed', 'sim_sha256':job['sim_sha256'], 'cas_sha256':prep.sha(cas)}))
    compact=folder/'compact'
    compact.mkdir()
    (compact/'boundaries.npz').write_bytes(b'boundaries')
    (compact/'selected_exits.npz').write_bytes(b'exits')
    receipt={'status':'validated_compact_chunk','sim_sha256':job['sim_sha256'], 'cas_sha256':prep.sha(cas),
             'histories':500, 'compact_sha256':prep.sha(compact/'boundaries.npz'),
             'selected_exits_sha256':prep.sha(compact/'selected_exits.npz')}
    (compact/'chunk_receipt.json').write_text(json.dumps(receipt))
    assert runner.check_previous(root, job, paths, False)
    (compact/'selected_exits.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):
        runner.check_previous(root, job, paths, False)


def test_unknown_seed_stays_unknown_in_compact_receipt(tmp_path):
    from test_casino_compact import _summary
    from bse_renderer.validation.casino_chunk_ingest import ingest_chunk
    source=tmp_path/'summary.dat'
    _summary(source, tilts=(30,))
    receipt=ingest_chunk(source, tmp_path/'compact', tilt_deg=30, repeat=1, seed=None)
    assert receipt['declared_seed'] is None
    assert 'not established' in receipt['seed_provenance']
    assert receipt['eligible_for_model_qualification'] is False


def test_confirmation_refuses_candidate_change(tmp_path):
    root, exe, jobs=setup_wave(tmp_path,1)
    jobs[0]['stage']='confirmation'
    (root/'manifest.json').write_text(json.dumps({'jobs':jobs}))
    with pytest.raises(ValueError, match='requires --frozen-model'):
        runner.run_wave(root, exe, 'confirmation')
    model=tmp_path/'model.json'
    model.write_text('{"model":"changed"}')
    (root/'confirmation_identity.json').write_text(json.dumps({'frozen_model_sha256':'old'}))
    with pytest.raises(ValueError, match='candidate changed'):
        runner.run_wave(root, exe, 'confirmation', frozen_model=model)
    assert not (root/'job0/console.log').exists()


@pytest.mark.parametrize('completed_before_interrupt', [True, False])
def test_interruption_during_launch_pause_keeps_attempt_receipt(tmp_path, monkeypatch, completed_before_interrupt):
    root, exe, jobs=setup_wave(tmp_path)
    processes=[]
    class Process:
        pid=19
        def __init__(self,args,**kwargs):
            self.terminated=False
            Path(Path(args[1]).read_text()).write_bytes(b'CAS output or partial output')
            processes.append(self)
        def poll(self):
            if completed_before_interrupt:return 0
            return 1 if self.terminated else None
        def terminate(self):self.terminated=True
        def wait(self,timeout=None):return self.poll()
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    def interrupt(_):raise KeyboardInterrupt
    monkeypatch.setattr(runner.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_wave(root,exe,'geometry')
    assert len(processes)==1
    assert (root/'job0/attempt_started.json').exists()
    receipt=json.loads((root/'job0/run_receipt.json').read_text())
    assert receipt['status']==('completed' if completed_before_interrupt else 'interrupted')
    assert (receipt['cas_sha256'] is not None)==completed_before_interrupt
    assert not (root/'job1/console.log').exists()
