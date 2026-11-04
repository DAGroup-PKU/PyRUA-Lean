"""Bounded foreground improvement sessions, paired evaluation, and version promotion."""
from __future__ import annotations

import json
import math
import shutil
import time

from datafarm.evolution.events import learn, pending
from datafarm.evolution.experiments import check_policy, compare, evaluate, freeze_policy, run
from datafarm.evolution.store import digest, read, write
from datafarm.evolution.versions import active, activate, initialize, resolve, snapshot, version

PROTECTED = ('AGENTS.md', 'prompts/instructions/authority-and-scope.md',
             'prompts/instructions/instruction-layers.md', 'prompts/instructions/mode-and-style-contract.md',
             'prompts/instructions/task-cost.md', 'configs/standing-rules.json',
             'evaluations/',
             # Roots of trust that measure candidates: the harness's holdout, sandbox and
             # expert audit, and the admission certificate issuer. Changing them is an
             # evaluator revision, never an optimization candidate.
             'src/datafarm/expert/holdout.py',
             'src/datafarm/expert/isolation.py', 'src/datafarm/expert/audit.py',
             'tools/issue_admission_certificate.py',
             # The gate itself: a promoted candidate must not rewrite how later candidates are judged.
             'src/datafarm/evolution/', 'tools/pipeline_evolve.py')
# Evaluator checks run with the candidate's src/ on PYTHONPATH, so a new top-level module there
# (json.py, numpy/, sitecustomize.py, a .pth file) would run inside the frozen evaluator.
INTERPRETER_HOOKS = ('sitecustomize.py', 'usercustomize.py')


def changed_paths(store, parent, source):
    old = store.get('sources', store.get('versions', parent)['source'])['files']
    new = store.get('sources', source)['files']
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def allowed(name, prefixes):
    return any(name == p or (p.endswith('/') and name.startswith(p)) for p in prefixes)


def injects_into_evaluator(name):
    parts = name.split('/')
    return (name.endswith('.pth') or parts[-1] in INTERPRETER_HOOKS
            or (parts[0] == 'src' and len(parts) > 1 and not parts[1].startswith('datafarm_')))


def out_of_scope(changes, policy):
    protected = PROTECTED + tuple(policy.get('protected_paths', ()))
    return [name for name in changes
            if allowed(name, protected) or not allowed(name, policy['allowed_paths'])
            or injects_into_evaluator(name)]


def propose(store, parent, source_root, proposal, event_ids, policy_id):
    policy, _ = check_policy(store, policy_id)
    for field in ('hypothesis', 'expected_effect', 'alternative_explanation'):
        if not isinstance(proposal.get(field), str) or not proposal[field].strip():
            raise ValueError(f'proposal requires {field}')
    if not event_ids:
        raise ValueError('proposal requires feedback events')
    for event in event_ids:
        store.get('events', event)
    source = snapshot(store, source_root)
    changes = changed_paths(store, parent, source)
    for name in out_of_scope(changes, policy):
        raise ValueError(f'change outside optimization scope: {name}')
    lesson_ids = [learn(store, {**item, 'events': event_ids})['id'] for item in proposal.get('lessons', [])]
    if not changes and not lesson_ids:
        raise ValueError('empty candidate')
    value = {'schema': 'datafarm.evolution-candidate/1', 'parent': parent, 'source': source,
             'events': sorted(event_ids), 'policy': policy_id, 'proposal': proposal,
             'lessons': lesson_ids, 'changed_paths': changes}
    value['id'] = digest(value)
    store.put('candidates', value['id'], value)
    if not store.path('states', value['id']).exists():
        store.transition(value['id'], 'proposed')
    return value


def assess(store, name, *, timeout=600, promote=True):
    """A retry creates a new attempt; all previous logs remain untouched."""
    with store.lock('controller'):
        return _assess(store, name, timeout=timeout, promote=promote)


def _assess(store, name, *, timeout, promote):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('evaluation budget must be positive and finite')
    candidate = store.get('candidates', name)
    if digest({k: v for k, v in candidate.items() if k != 'id'}) != name:
        raise ValueError('candidate record changed')
    state = store.get('states', name)
    if state['status'] == 'promoted':
        return state
    # Recover an interrupted activation after the atomic pointer was written but
    # before the candidate state was updated. Do not rerun paid evaluations.
    current = active(store)
    if current:
        current_version = store.get('versions', current)
        evaluation_id = current_version.get('evaluation')
        if evaluation_id and store.get('evaluations', evaluation_id)['candidate'] == name:
            resolve(store, current)
            return store.transition(name, 'promoted', evaluation=evaluation_id, version=current,
                                    evidence_level=store.get('evaluations', evaluation_id)['evidence_level'], recovered=True)
    if active(store) != candidate['parent']:
        return store.transition(name, 'stale', reason='rebase on active version and evaluate again')
    policy, _ = check_policy(store, candidate['policy'])
    # Re-check scope from the stored sources, not only the record written at proposal.
    escaped = out_of_scope(changed_paths(store, candidate['parent'], candidate['source']), policy)
    if escaped:
        return store.transition(name, 'rejected', reason=f'change outside optimization scope: {escaped[0]}')
    if state['status'] == 'verified':
        measured = store.get('evaluations', state['evaluation'])
        if digest(measured) != state['evaluation'] or not measured['accepted']:
            raise ValueError('verified evaluation changed')
        resolve(store, state['version'])
        if promote:
            activate(store, state['version'], expected=candidate['parent'], reason=f"evaluation:{state['evaluation']}")
            return store.transition(name, 'promoted', evaluation=state['evaluation'], version=state['version'],
                                    evidence_level=measured['evidence_level'])
        return state
    attempt = str(time.time_ns())
    directory = store.root / 'evaluations' / name / attempt
    directory.mkdir(parents=True)
    store.transition(name, 'evaluating', attempt=attempt)
    deadline = time.monotonic() + timeout
    try:
        parent = store.get('versions', candidate['parent'])
        baseline = evaluate(store, parent['source'], candidate['policy'], directory / 'baseline', deadline,
                            lesson_ids=parent['lessons'])
        child = evaluate(store, candidate['source'], candidate['policy'], directory / 'candidate', deadline,
                         lesson_ids=sorted(set(parent['lessons']) | set(candidate['lessons'])))
        accepted, reason = compare(policy, baseline, child)
        evaluation = {'candidate': name, 'attempt': attempt, 'baseline': baseline, 'candidate_result': child,
                      'accepted': accepted, 'reason': reason, 'evidence_level': policy['evidence_level']}
        evaluation_id = digest(evaluation)
        store.put('evaluations', evaluation_id, evaluation)
        if not accepted:
            return store.transition(name, 'rejected', evaluation=evaluation_id, reason=reason)
        child_version = version(store, candidate['source'], parent=candidate['parent'],
                                lessons=candidate['lessons'] if policy.get('tests_lessons') else (), evaluation=evaluation_id)
        if promote:
            activate(store, child_version['id'], expected=candidate['parent'], reason=f'evaluation:{evaluation_id}')
        return store.transition(name, 'promoted' if promote else 'verified', evaluation=evaluation_id,
                                version=child_version['id'], evidence_level=policy['evidence_level'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return store.transition(name, 'evaluation-error', attempt=attempt, reason=str(exc))


def cycle(store, *, root, driver, policy_path, max_candidates=1, timeout=600, event_ids=None):
    """Invoke an actual coding driver; never synthesize a claimed agent success."""
    if max_candidates < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('candidate and wall budgets must be positive')
    if driver.get('input', 'messages') not in {'text', 'messages'}:
        raise ValueError('driver input must be messages or text')
    initialize(store, root)
    policy_id = freeze_policy(store, policy_path)
    session = store.root / 'cycles' / str(time.time_ns())
    session.mkdir(parents=True)
    results = []
    deadline = time.monotonic() + timeout
    with store.lock('controller'):
        for index in range(max_candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            feedback = [store.get('events', x) for x in event_ids] if event_ids else pending(store)
            if not feedback:
                break
            parent = active(store)
            source_root = resolve(store, parent)
            attempt = session / str(index)
            attempt.mkdir()
            work = attempt / 'source'
            shutil.copytree(source_root, work)
            proposal_path = attempt / 'proposal.json'
            from datafarm.agent.context import write_context
            context = write_context(attempt / 'context', 'pipeline-improver', root=source_root,
                user=json.dumps({'parent': parent, 'feedback': feedback,
                    'candidate_source': str(work), 'proposal_output': str(proposal_path),
                    'allowed_paths': check_policy(store, policy_id)[0]['allowed_paths'],
                    'prior_results': store.all('states'),
                    'instruction': 'Edit only candidate source; write proposal with hypothesis, expected_effect, '
                    'alternative_explanation, optional lessons. Do not invent results or edit evaluators. '
                    'Use feedback IDs as provenance. The host evaluates and activates.'}, ensure_ascii=False))
            payload = (json.dumps(context['messages'], ensure_ascii=False) if driver.get('input', 'messages') == 'messages'
                       else (attempt / 'context/context.md').read_text())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results.append({'status': 'budget-exhausted', 'session': str(attempt)})
                break
            execution = run(driver['argv'], cwd=work, directory=attempt / 'driver',
                            timeout=remaining, payload=payload)
            if execution['status'] != 'returned':
                results.append({'status': execution['status'], 'session': str(attempt)})
                break
            try:
                proposal = read(proposal_path)
                chosen = proposal.pop('event_ids', [e['id'] for e in feedback])
                if not set(chosen).issubset({e['id'] for e in feedback}):
                    raise ValueError('proposal selects unknown feedback')
                candidate = propose(store, parent, work, proposal, chosen, policy_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result = {'status': 'budget-exhausted', 'candidate': candidate['id']}
                else:
                    result = _assess(store, candidate['id'], timeout=remaining, promote=True)
                results.append(result)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                results.append({'status': 'proposal-error', 'reason': str(exc), 'session': str(attempt)})
                break
            write(session / 'result.json', {'results': results, 'active': active(store)})
    summary = {'results': results, 'active': active(store), 'remaining_events': len(pending(store)),
               'session': str(session), 'status': 'finished'}
    write(session / 'result.json', summary)
    return summary
