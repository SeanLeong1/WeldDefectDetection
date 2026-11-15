"""Atomic JSON writes with bounded retries for temporary Windows file locks."""
from pathlib import Path
import json
import time


def _retry_file_operation(operation):
    delays = (.1, .2, .4, .8, 1.6, 2., 2.)
    for attempt in range(len(delays)+1):
        try:
            return operation()
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32, 33)):
                raise
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def write_json_atomic(path: Path, value, *, temporary: Path | None = None):
    """Keep the committed file intact until replacement succeeds.

    If retries are exhausted, retain the temporary payload for inspection. Do
    not fall back to truncating the destination or deleting a locked receipt.
    Callers serialize writes to each destination with the campaign's wave lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(temporary) if temporary is not None else path.with_suffix(path.suffix+'.tmp')
    if temporary.resolve() == path.resolve() or temporary.parent.resolve() != path.parent.resolve():
        raise ValueError('Temporary receipt must be a different file in the destination directory.')
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    _retry_file_operation(lambda: temporary.write_text(payload, encoding='utf-8'))
    _retry_file_operation(lambda: temporary.replace(path))
