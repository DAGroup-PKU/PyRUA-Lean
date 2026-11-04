"""Small immutable records and cooperative single-writer transactions; stdlib only.

This filesystem store is not a security boundary against another process with the
same write permissions. Deploy evaluators separately when running untrusted code.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from datafarm.common import atomic_write_bytes
from datafarm.common import read_json as read
from pathlib import Path
import re
import tempfile
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    atomic_write_bytes(path, canonical(value) + b'\n')


def immutable(path, value):
    """Publish a complete record once, including on shared filesystems."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.record-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError(f'conflicting immutable record: {path.name}')
    finally:
        Path(name).unlink(missing_ok=True)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}', value):
        raise ValueError('invalid record identifier')
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind, name):
        return self.root / identifier(kind) / (identifier(name) + '.json')

    def get(self, kind, name):
        return read(self.path(kind, name))

    def put(self, kind, name, value):
        immutable(self.path(kind, name), value)
        return name

    def all(self, kind):
        return [read(p) for p in sorted((self.root / identifier(kind)).glob('*.json'))]

    @contextmanager
    def lock(self, name='controller'):
        with (self.root / (identifier(name) + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f'{name} already running') from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def transition(self, candidate, status, **details):
        record = {'candidate': candidate, 'status': status, 'time_ns': time.time_ns(), **details}
        self.put('transitions', str(record['time_ns']), record)
        write(self.path('states', candidate), record)
        return record
