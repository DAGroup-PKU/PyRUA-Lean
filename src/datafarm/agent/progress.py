"""Supervisor-owned status publication and source-bound external-job transitions.

No notifier, submission or independent success criteria live here. Completion
counts are verified deliveries; producer claims are labelled separately.
"""
from __future__ import annotations
from datafarm.common import sha256_file
import fcntl
import json
from pathlib import Path
import time

from datafarm.agent.native_codex import save
from datafarm.common import sha256_text


def refresh_external_jobs(work):
    work = Path(work)
    path = work / 'state.json'
    state = json.loads(path.read_text())
    changed = False
    for job in state.get('external_jobs', []):
        monitor = job.get('status_file')
        if not monitor:
            continue
        status_path = Path(monitor)
        if not status_path.is_absolute():
            status_path = work / status_path
        try:
            record = json.loads(status_path.read_text())
        except (OSError, ValueError):
            continue  # unknown never means failed, absent, or safe to resubmit
        if record.get('id') != job['id']:
            continue
        status = record.get('status')
        if status == 'exported-unqualified':
            status = 'finished'  # transport only; qualification remains unverified
        if status not in {'queued','running','complete','completed','finished','succeeded',
                          'failed','error','cancelled','needs-input','unknown'}:
            continue
        # A zero exit is not delivery success, but a nonzero cannot be job success.
        if status in {'complete','completed','finished','succeeded'} and record.get('exit_code', 0) != 0:
            status = 'failed'
        evidence = {'path': str(status_path.resolve()),
                    'sha256': sha256_file(status_path)}
        update = {'status': status, 'monitor_evidence': evidence}
        if record.get('error'):
            update['error'] = record['error']
        if any(job.get(k) != v for k, v in update.items()):
            job.update(update)
            changed = True
    if changed:
        save(path, state)
    return state


def publish(work, runtime):
    work = Path(work)
    routing = work / '.runtime/status-output.json'
    if not routing.exists():
        return
    config = json.loads(routing.read_text())
    path = Path(config['path']).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads((work / 'state.json').read_text())
    except (OSError, ValueError):
        state = {'phase':'invalid-state','status':'error'}
    completions = []
    for f in sorted((work / '.runtime/reviews').glob('*/completion.json')):
        record = json.loads(f.read_text())
        completions.append({'id':f.parent.name,'verdict':record['verdict'],
            'session':record['session'],'evidence':str(f),
            'sha256':sha256_file(f)})
    control_path = work / '.runtime/control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    result = control.get('last_result', {})
    # Only the host's verify_delivery result certifies full requested delivery.
    verified = result.get('valid') is True and result.get('status') == 'complete'
    worker = runtime.get('worker')
    row = {'work':str(work),'phase':state.get('phase'),
        'runtime_status':runtime.get('status'),'task_status':state.get('status'),
        'completed_episodes':result.get('episodes',0) if verified else 0,
        'reported_completed_episodes':state.get('completed_episodes',0),
        'delivery_verified':verified,'next_action':state.get('next_action'),
        'blockers':state.get('open_questions',[]),'review_results':completions,
        'external_jobs':state.get('external_jobs',[]),'sessions':control.get('sessions',0),
        'retries':control.get('retries',0),'worker':worker,
        'state_evidence':str(work/'state.json'),'cost_evidence':str(work/'cost-ledger.json'),
        'progress_evidence':state.get('progress',{}),
        'source_binding':str(work/'capability.json'),'last_result':result}
    meaningful = {k:v for k,v in row.items() if k != 'worker'}
    if worker:
        meaningful['worker_session'] = worker.get('session')
        meaningful['worker_role'] = worker.get('role')
    digest = sha256_text(json.dumps(meaningful,sort_keys=True))
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value = json.loads(path.read_text()) if path.exists() else {'schema':'datafarm.continuation-status/1','tasks':{}}
        old = value.setdefault('tasks',{}).get(config['task_id'],{})
        row.update(heartbeat=time.time(),substantive_sha256=digest,
            last_substantive_at=time.time() if digest != old.get('substantive_sha256') else old.get('last_substantive_at'))
        value['tasks'][config['task_id']] = row
        value['updated_at'] = time.time()
        if digest != old.get('substantive_sha256'):
            value['last_substantive_at'] = row['last_substantive_at']
        save(path,value)
