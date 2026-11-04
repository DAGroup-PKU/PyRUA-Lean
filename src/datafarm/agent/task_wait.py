"""Wait for a task transition without invoking a model or reacting to heartbeats."""
import json
import math
from pathlib import Path
import time


def model_wait_status(state):
    """An explicit dependency wait suppresses model launches, not job monitoring.

    The supervisor owns job status/liveness. Unknown statuses remain pending;
    a timeout or an old heartbeat is never interpreted as permission to retry.
    """
    ids = state.get('model_wait_for', [])
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('model_wait_for must contain nonempty job IDs')
    if not ids:
        return None
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate model_wait_for job ID')
    jobs = state.get('external_jobs', [])
    selected = []
    for identifier in ids:
        matches = [j for j in jobs if isinstance(j, dict) and j.get('id') == identifier]
        if len(matches) != 1 or not isinstance(matches[0].get('status'), str):
            raise ValueError('waiting dependency missing, ambiguous or lacks status: ' + identifier)
        selected.append(matches[0])
    # Failure is actionable even while another dependency is still running.
    if any(j['status'] in {'failed', 'error', 'cancelled', 'needs-input'} for j in selected):
        return None
    pending = [j['id'] for j in selected
               if j['status'] not in {'complete', 'completed', 'finished', 'succeeded'}]
    if not pending:
        return None
    return {'status': 'waiting-external-jobs', 'job_ids': pending,
            'next_action': 'Supervisor monitors existing jobs; resume drive on a task transition.',
            'liveness_verified': False, 'delivery_verified': False}


def meaningful_state(state):
    jobs = state.get('external_jobs', [])
    return {key: state.get(key) for key in ('status', 'phase', 'next_action', 'open_questions', 'model_wait_for')} | {
        'external_jobs': [({k: j.get(k) for k in ('id', 'job_id', 'task_id', 'status', 'error')}
                           if isinstance(j, dict) else j) for j in jobs]}


def wait_for_task(work, timeout=1800, interval=5):
    if not all(math.isfinite(x) and x > 0 for x in (timeout, interval)):
        raise ValueError('wait timeout and interval must be finite and positive')
    path = Path(work) / 'state.json'
    initial = meaningful_state(json.loads(path.read_text()))
    deadline = time.monotonic() + timeout
    while True:
        state = meaningful_state(json.loads(path.read_text()))
        if state.get('status') in {'complete', 'needs-input'} or state != initial:
            return {'status': 'task-transition', 'state': state, 'delivery_verified': False}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {'status': 'wait-timeout', 'state': state, 'delivery_verified': False}
        time.sleep(min(interval, remaining))
