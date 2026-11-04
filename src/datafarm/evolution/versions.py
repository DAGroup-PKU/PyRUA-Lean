"""Content-addressed source snapshots and atomic, compare-and-swap activation."""
from __future__ import annotations

from datafarm.common import sha256_file
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from datafarm.evolution.store import digest, read, write

EXCLUDED = {'.git', '.datafarm', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv',
            'dist', 'build', 'runs', 'artifacts', 'secrets', 'keys',
            'environments', 'external', 'data', '.env', 'bindings.json'}


def files(root):
    root = Path(root).resolve()
    if (root / '.git').exists():
        raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
        names = [x for x in raw.decode().split('\0') if x]
    else:
        names = []
        for base, dirs, names_here in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.endswith('.egg-info'))
            if any((Path(base) / d).is_symlink() for d in dirs):
                raise ValueError('source snapshot cannot contain directory symlinks')
            for name in names_here:
                names.append((Path(base) / name).relative_to(root).as_posix())
    result = {}
    for name in sorted(set(names)):
        parts = Path(name).parts
        if (EXCLUDED.intersection(parts) or any(p.endswith('.egg-info') or p.startswith('.env.') for p in parts)
                or name.endswith(('.pyc', '.log'))):
            continue
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'source snapshot cannot contain symlink: {name}')
        if not path.is_file():
            continue  # tracked deletion is reflected in the manifest
        result[name] = {'sha256': sha256_file(path),
                        'executable': bool(path.stat().st_mode & 0o111)}
    if not result:
        raise ValueError('empty source snapshot')
    return result


def snapshot(store, root):
    root = Path(root).resolve()
    if store.root.is_relative_to(root) and not EXCLUDED.intersection(store.root.relative_to(root).parts):
        raise ValueError('store inside source must live in an excluded runtime directory such as .datafarm')
    manifest = files(root)
    name = digest(manifest)
    target = store.root / 'sources' / name
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=target.parent))
        try:
            for relative in manifest:
                dest = temp / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / relative, dest)
            if files(temp) != manifest:
                raise ValueError('source changed while creating snapshot')
            try:
                os.rename(temp, target)
            except OSError:
                if not target.exists():
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    store.put('sources', name, {'id': name, 'files': manifest})
    verify_source(store, name)
    return name


def verify_source(store, name):
    manifest = store.get('sources', name)
    if digest(manifest['files']) != name:
        raise ValueError('source manifest changed')
    root = store.root / 'sources' / name
    if files(root) != manifest['files']:
        raise ValueError('pinned source changed')
    return root


def version(store, source, *, parent=None, lessons=(), evaluation=None):
    verify_source(store, source)
    inherited = store.get('versions', parent).get('lessons', []) if parent else []
    value = {'schema': 'datafarm.capability-version/1', 'source': source, 'parent': parent,
             'lessons': sorted(set(inherited) | set(lessons)), 'evaluation': evaluation}
    for item in value['lessons']:
        store.get('lessons', item)
    value['id'] = digest(value)
    store.put('versions', value['id'], value)
    return value


def resolve(store, name):
    value = store.get('versions', name)
    if digest({k: v for k, v in value.items() if k != 'id'}) != name:
        raise ValueError('capability version changed')
    return verify_source(store, value['source'])


def active(store):
    path = store.root / 'active.json'
    return read(path)['version'] if path.exists() else None


def activate(store, name, *, expected, reason):
    with store.lock('activation'):
        actual = active(store)
        if actual != expected:
            raise ValueError('active version changed; rebase and reevaluate candidate')
        resolve(store, name)
        record = {'version': name, 'previous': actual, 'reason': reason, 'time_ns': time.time_ns()}
        store.put('activations', str(record['time_ns']), record)
        write(store.root / 'active.json', record)
        return record


def initialize(store, root):
    current = active(store)
    if current:
        resolve(store, current)
        return store.get('versions', current)
    value = version(store, snapshot(store, root))
    try:
        activate(store, value['id'], expected=None, reason='initialize source baseline')
    except ValueError:
        if not active(store):
            raise
        value = store.get('versions', active(store))
    return value


def rollback(store, name):
    if not any(x['version'] == name for x in store.all('activations')):
        raise ValueError('rollback target was never active')
    return activate(store, name, expected=active(store), reason='rollback; existing run pins unchanged')
