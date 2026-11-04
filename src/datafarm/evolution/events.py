"""Explicit host feedback, scoped lessons, and reproducible context selection."""
from __future__ import annotations

from datafarm.common import sha256_file
from datafarm.evolution.store import digest, identifier

KINDS = {'user-requirement', 'correction', 'observation', 'code-fix', 'test-result',
         'session-result', 'task-result'}
SELECTORS = {'task_id', 'task_family', 'hardware', 'provider', 'run_id'}


def scope(value):
    if not isinstance(value, dict) or set(value) - SELECTORS:
        raise ValueError('scope must use exact task_id/task_family/hardware/provider/run_id selectors')
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError('scope selectors must be nonempty strings')
    return value


def matches(selectors, current):
    return all(current.get(k) == v for k, v in selectors.items())


def record(store, event):
    event = dict(event)
    if event.get('kind') not in KINDS:
        raise ValueError('unknown feedback kind')
    if not isinstance(event.get('statement'), str) or not event['statement'].strip():
        raise ValueError('feedback requires a statement')
    source = event.get('source', {})
    if not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k] for k in ('actor', 'ref')):
        raise ValueError('feedback requires source actor and ref; do not invent chat visibility')
    event['scope'] = scope(event.get('scope', {}))
    for prior in event.get('supersedes', []):
        old = store.get('events', prior)
        if old['scope'] != event['scope'] or old['kind'] != event['kind']:
            raise ValueError('supersession must preserve kind and scope')
    event['schema'] = 'datafarm.feedback/1'
    name = event.pop('id', None) or digest(event)
    identifier(name)
    event['id'] = name
    if len(str(event)) > 65536:
        raise ValueError('feedback too large; reference artifacts instead')
    store.put('events', name, event)
    return event


def learn(store, lesson):
    lesson = dict(lesson)
    required = ('problem', 'action', 'rationale', 'limits')
    if any(not isinstance(lesson.get(k), str) or not lesson[k].strip() for k in required):
        raise ValueError('lesson requires problem, action, rationale and limits')
    lesson['scope'] = scope(lesson.get('scope', {}))
    refs = lesson.get('events', [])
    if not refs:
        raise ValueError('lesson requires feedback provenance')
    for ref in refs:
        store.get('events', ref)
    for prior in lesson.get('supersedes', []):
        if store.get('lessons', prior)['scope'] != lesson['scope']:
            raise ValueError('lesson supersession must preserve scope')
    # A proposer cannot grant its own hypothesis verified status.
    lesson['status'] = 'hypothesis'
    lesson['schema'] = 'datafarm.lesson/1'
    name = lesson.pop('id', None) or digest(lesson)
    lesson['id'] = identifier(name)
    store.put('lessons', name, lesson)
    return lesson


def select(store, selectors, *, version=None, limit=8):
    """Requirements are never truncated; only promoted lessons enter default context."""
    scope(selectors)
    events = store.all('events')
    retired = {x for e in events for x in e.get('supersedes', [])}
    requirements = [e for e in events if e['kind'] == 'user-requirement'
                    and e['id'] not in retired and matches(e['scope'], selectors)]
    available = (store.get('versions', version).get('lessons', []) if version else [])
    lessons = [store.get('lessons', x) for x in available]
    retired_lessons = {x for e in lessons for x in e.get('supersedes', [])}
    lessons = [x for x in lessons if x['id'] not in retired_lessons and matches(x['scope'], selectors)]
    lessons.sort(key=lambda x: (-len(x['scope']), x['id']))
    selected = {'schema': 'datafarm.experience-context/1', 'scope': selectors,
                'version': version, 'requirements': requirements, 'lessons': lessons[:limit]}
    selected['sha256'] = digest(selected)
    return selected


def pending(store):
    consumed = {e for c in store.all('candidates') for e in c['events']}
    return [e for e in store.all('events') if e['id'] not in consumed]


def record_change(store, source_root, parent, *, statement, source_ref, selectors=None, evidence=()):
    """Capture actual source changes and check-log hashes after ordinary repo work."""
    from pathlib import Path
    from datafarm.evolution.versions import snapshot
    before = store.get('versions', parent)['source']
    after = snapshot(store, source_root)
    old, new = store.get('sources', before)['files'], store.get('sources', after)['files']
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if not changed:
        raise ValueError('no source changes to record')
    artifacts = [{'path': str(Path(p).resolve()), 'sha256': sha256_file(Path(p))}
                 for p in evidence]
    return record(store, {'kind': 'code-fix', 'statement': statement, 'scope': selectors or {},
        'source': {'actor': 'repository-working-agent', 'ref': source_ref},
        'version': parent, 'details': {'before_source': before, 'after_source': after,
                                     'changed_paths': changed, 'evidence': artifacts}})
