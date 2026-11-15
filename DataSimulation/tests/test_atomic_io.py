import json
from pathlib import Path

import pytest

from bse_renderer import atomic_io


def test_temporary_lock_retries_without_truncating_committed_receipt(tmp_path, monkeypatch):
    path=tmp_path/'receipt.json'
    path.write_text('{"old":true}')
    original=Path.replace
    attempts=[]
    def locked(self,target):
        attempts.append(1)
        assert json.loads(path.read_text()) == {'old':True}
        if len(attempts)<3:
            raise PermissionError('temporary file lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',locked)
    monkeypatch.setattr(atomic_io.time,'sleep',lambda _:None)
    atomic_io.write_json_atomic(path,{'complete':True})
    assert len(attempts)==3
    assert json.loads(path.read_text())=={'complete':True}
    assert not path.with_suffix('.json.tmp').exists()


def test_permanent_lock_preserves_old_receipt_and_pending_payload(tmp_path, monkeypatch):
    path=tmp_path/'receipt.json'
    path.write_text('{"old":true}')
    attempts=[]
    def locked(*args):
        attempts.append(1)
        raise PermissionError('permanent lock')
    monkeypatch.setattr(Path,'replace',locked)
    monkeypatch.setattr(atomic_io.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):
        atomic_io.write_json_atomic(path,{'complete':True})
    assert len(attempts)==8
    assert json.loads(path.read_text())=={'old':True}
    assert json.loads(path.with_suffix('.json.tmp').read_text())=={'complete':True}


def test_non_lock_errors_are_not_retried(tmp_path,monkeypatch):
    def fail(*args):raise OSError('disk full')
    def unexpected_sleep(_):pytest.fail('Non-lock errors must propagate immediately')
    monkeypatch.setattr(Path,'replace',fail)
    monkeypatch.setattr(atomic_io.time,'sleep',unexpected_sleep)
    with pytest.raises(OSError,match='disk full'):
        atomic_io.write_json_atomic(tmp_path/'receipt.json',{'value':1})


def test_temp_path_cannot_be_the_committed_file(tmp_path):
    path=tmp_path/'receipt.json'
    path.write_text('original')
    with pytest.raises(ValueError):
        atomic_io.write_json_atomic(path,{'value':1},temporary=path)
    assert path.read_text()=='original'


def test_chunk_receipt_published_once_with_complete_run_bindings(tmp_path,monkeypatch):
    from test_casino_compact import _summary
    from bse_renderer.validation import casino_chunk_ingest as ingest
    source=tmp_path/'summary.dat'
    _summary(source,tilts=(30,))
    writes=[]
    original=ingest._write_json
    def capture(path,result):
        writes.append(dict(result))
        return original(path,result)
    monkeypatch.setattr(ingest,'_write_json',capture)
    result=ingest.ingest_chunk(source,tmp_path/'compact',tilt_deg=30,repeat=1,seed=None,
                              run_provenance={'sim_sha256':'a'*64,'cas_sha256':'b'*64})
    assert len(writes)==1
    assert writes[0]['sim_sha256']=='a'*64 and writes[0]['cas_sha256']=='b'*64
    assert len(writes[0]['selected_exits_sha256'])==64
    assert json.loads((tmp_path/'compact/chunk_receipt.json').read_text())==result
