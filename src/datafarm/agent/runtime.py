"""Persistent control for pipeline_agent drive; task state remains the authority.

The run ledger owns daemon launch/liveness. This module owns bounded continuation,
not simulation or physical success. Same-UID agents are not a security boundary.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time

from datafarm.agent.native_codex import save
from datafarm.agent.task_wait import model_wait_status, wait_for_task
from datafarm.common import sha256_file
from datafarm.common import sha256_text


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'datafarm.agent-runtime/1':
        raise ValueError('invalid runtime schema')
    defaults = {'max_sessions': 16, 'timeout': 1800, 'poll_seconds': 5,
                'heartbeat_seconds': 10, 'max_retries': 3,
                'backoff_seconds': 5, 'backoff_max_seconds': 120}
    config = {**defaults, **config}
    for key in defaults:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('runtime setting must be finite: ' + key)
        if value < 0 or (key != 'max_retries' and value == 0):
            raise ValueError('invalid runtime setting: ' + key)
    for key in ('max_sessions', 'max_retries'):
        if not isinstance(config[key], int):
            raise ValueError(key + ' must be integer')
    if not isinstance(config.get('driver'), dict):
        raise ValueError('runtime driver required')
    if 'working_directory' in config:
        config['working_directory'] = str(Path(config['working_directory']).expanduser().resolve(strict=True))
    return config


def runtime_dir(work):
    path = Path(work) / '.runtime'
    path.mkdir(exist_ok=True)
    return path


def stop_requested(work):
    return (runtime_dir(work) / 'stop.json').exists()


def heartbeat(work, **fields):
    path = runtime_dir(work) / 'supervisor.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    runtime = {**old, **fields, 'heartbeat': time.time(), 'supervisor_pid': os.getpid(),
               'supervisor_identity': process_identity(os.getpid()), 'supervisor_host': socket.gethostname()}
    save(path, runtime)
    from datafarm.agent.progress import publish
    publish(work, runtime)


def host_identity():
    """Where a PID is meaningful. /proc liveness checks are only valid on the same boot."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = None
    return {'host': socket.gethostname(), 'boot_id': boot}


def foreign_host(record):
    """Name the host of a record written elsewhere; None for this host or a record without a host.

    On a shared filesystem a supervisor on another node sees no such PID and would
    otherwise treat a live worker as dead and resume its provider session twice.
    """
    host = record.get('host')
    if host and host != socket.gethostname():
        return host
    return None


def process_identity(pid):
    try:
        # starttime disambiguates recycled PIDs; zombie is not an active worker.
        parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if parts[0] == 'Z' else parts[19]
    except (OSError, IndexError):
        return None


def group_alive(pgid):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts = path.read_text().rsplit(')', 1)[1].split()
            if int(parts[2]) == pgid and parts[0] != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def terminate_group(pgid, timeout=5):
    """Stop the entire local group, including TERM-resistant orphan children."""
    if pgid == os.getpgrp():
        raise ValueError('refusing to terminate supervisor process group')
    for sig, duration in ((signal.SIGTERM, timeout), (signal.SIGKILL, 2)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + duration
        while group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(.05)
        if not group_alive(pgid):
            return True
    return False


def recover_inflight(work):
    """Reconcile persisted execution before permitting any new model call."""
    from datafarm.agent.cost_budget import budget_for_work
    for path in sorted((Path(work) / 'sessions').glob('*/execution.json')):
        record = json.loads(path.read_text())
        status = record.get('status')
        if status not in {'starting', 'running', 'returned', 'failed', 'timeout', 'stopped'}:
            continue
        pid = record.get('pid')
        if status in {'starting', 'running'}:
            host = foreign_host(record)
            if host:
                return {'status': 'recovery-required', 'session': str(path.parent), 'host': host,
                        'reason': 'Execution was launched on ' + host + '; its liveness is only observable '
                                  'there. Recover on that host or reconcile before another call.'}
            identity = process_identity(pid) if pid else None
            if identity and identity == record.get('process_identity'):
                return {'status': 'waiting-worker', 'session': str(path.parent), 'pid': pid,
                        'process_identity': identity, 'pgid': pid}
            native_path = Path(record.get('native_state', '/nonexistent'))
            native = json.loads(native_path.read_text()) if native_path.is_file() else {}
            identity = process_identity(native['pid']) if native.get('pid') else None
            if native.get('status') == 'running' and identity and identity == native.get('process_identity'):
                return {'status': 'waiting-worker', 'session': str(path.parent), 'pid': native['pid'],
                        'process_identity': identity, 'pgid': pid}
            result_path = path.parent / 'native-result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                record.update(status='returned' if result['exit_code'] == 0 else 'failed',
                              exit_code=result['exit_code'], recovered=True, ended=time.time())
                save(path, record)
            elif native.get('session_id'):
                record.update(status='interrupted', ended=time.time(), recovery_session_id=native['session_id'])
                save(path, record)
            else:
                return {'status': 'recovery-required', 'session': str(path.parent),
                        'reason': 'Interrupted launch without a saved session ID; reconcile before resubmitting.'}
        # A crash may occur after process return but before settlement/review finalization.
        budget = budget_for_work(work)
        if budget and record.get('cost_reservation_id'):
            with budget.transaction() as data:
                reserved = data['reservations'][record['cost_reservation_id']]['status'] == 'reserved'
            if reserved:
                receipt_path = path.parent / 'cost-receipt.json'
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                record['cost'] = budget.settle(record['cost_reservation_id'], receipt)
                save(path, record)
        if record['status'] == 'returned' and record.get('review_id'):
            review_folder = runtime_dir(work) / 'reviews' / record['review_id']
            if not (review_folder / 'completion.json').exists():
                if not (review_folder / 'report.json').exists():
                    return {'status': 'recovery-required', 'session': str(path.parent),
                            'reason': 'Reviewer returned without a report; reconcile before another call.'}
                request = json.loads((review_folder / 'request.json').read_text())
                register_identity(Path(work), path.parent, 'review-' + record['review_id'], 'task-verifier',
                                  native=(path.parent / 'native-result.json').exists())
                finish_review(work, request, review_folder, path.parent)
    return None


def register_identity(work, session, key, role, native=False):
    """Bind the trusted driver's provider identity to a single role namespace."""
    origin_verified = False
    if native:
        result = json.loads((session / 'native-result.json').read_text())
        cmd = json.loads((session / 'native-command.json').read_text())
        identity = {'schema': 'datafarm.agent-identity/1', 'session_id': result.get('session_id'),
                    'session_key': key, 'role': role, 'fresh': cmd['resume'] is None,
                    'evidence': [{'path': str((session / 'stdout.log').relative_to(work)),
                                  'sha256': sha256_file(session / 'stdout.log')}]}
        identity_path = session / 'identity.json'
        if identity_path.exists():
            saved = json.loads(identity_path.read_text())
            if any(saved.get(k) != identity.get(k) for k in ('session_id', 'session_key', 'role', 'fresh')):
                raise ValueError('native identity changed after execution')
            identity = saved
        else:
            save(identity_path, identity)
    else:
        identity = json.loads((session / 'identity.json').read_text())
    if (identity.get('schema') != 'datafarm.agent-identity/1' or not identity.get('session_id')
            or identity.get('session_key') != key or identity.get('role') != role
            or not identity.get('evidence')):
        raise ValueError('driver session identity/provenance missing')
    for ref in identity['evidence']:
        path = (Path(work) / ref['path']).resolve(strict=True)
        own_native_log = native and path == (session / 'stdout.log').resolve(strict=True)
        if (not path.is_relative_to(session.resolve()) and not own_native_log) or sha256_file(path) != ref['sha256']:
            raise ValueError('driver identity evidence escaped session or changed')
    path = runtime_dir(work) / 'identities.json'
    known = json.loads(path.read_text()) if path.exists() else {}
    previous = known.get(identity['session_id'])
    if previous and previous['session_key'] != key:
        raise ValueError('driver reused author/reviewer session identity across roles')
    if native and previous is None and not identity.get('fresh'):
        # An interrupted first turn may have saved thread.started before any result.
        for origin in (Path(work) / 'sessions').iterdir():
            if origin == session or not (origin / 'native-command.json').exists():
                continue
            origin_cmd = json.loads((origin / 'native-command.json').read_text())
            origin_execution = json.loads((origin / 'execution.json').read_text())
            origin_key = ('review-' + origin_execution['review_id']
                          if origin_execution.get('review_id') else 'author')
            if origin_cmd.get('resume') is not None or origin_key != key or origin_execution.get('role') != role:
                continue
            for line in (origin / 'stdout.log').read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'thread.started' and event.get('thread_id') == identity['session_id']:
                    origin_verified = True
                    break
            if origin_verified:
                identity['origin'] = {'path': str((origin / 'stdout.log').relative_to(work)),
                                      'sha256': sha256_file(origin / 'stdout.log'),
                                      'execution_sha256': sha256_file(origin / 'execution.json'),
                                      'command_sha256': sha256_file(origin / 'native-command.json')}
                save(session / 'identity.json', identity)
                break
    if previous is None and identity.get('fresh') is not True and not origin_verified:
        raise ValueError('unproven pre-existing driver session cannot certify independence')
    if identity.get('origin'):
        ref = identity['origin']
        origin_log = Path(work) / ref['path']
        origin = origin_log.parent.resolve(strict=True)
        if (origin.parent != (Path(work) / 'sessions').resolve() or origin_log.name != 'stdout.log'
                or sha256_file(origin_log) != ref['sha256']
                or sha256_file(origin / 'execution.json') != ref['execution_sha256']
                or sha256_file(origin / 'native-command.json') != ref['command_sha256']):
            raise ValueError('native session origin changed or escaped task sessions')
    known[identity['session_id']] = {'session_key': key, 'role': role, 'first_session':
        previous['first_session'] if previous else str(session)}
    save(path, known)
    return identity


def validate_completion(work, folder):
    completion = json.loads((folder / 'completion.json').read_text())
    session = Path(completion['session']).resolve(strict=True)
    if session.parent != (Path(work) / 'sessions').resolve():
        raise ValueError('review execution is outside task sessions')
    execution = json.loads((session / 'execution.json').read_text())
    receipt = json.loads((session / 'receipt.json').read_text())
    if (execution.get('role') != 'task-verifier' or execution.get('status') != 'returned'
            or execution.get('review_id') != folder.name
            or completion.get('role') != 'task-verifier'
            or receipt.get('role') != 'task-verifier'
            or receipt.get('messages_sha256') != execution.get('context_sha256')
            or completion.get('context_sha256') != execution.get('context_sha256')):
        raise ValueError('completed review lacks independent verifier provenance')
    if (sha256_file(folder / 'report.json') != completion['report_sha256']
            or sha256_file(session / 'execution.json') != completion['execution_sha256']
            or sha256_file(session / 'stdout.log') != completion['transcript_sha256']):
        raise ValueError('completed review report/execution/transcript changed')
    request = json.loads((folder / 'request.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    if report.get('verdict') not in {'pass', 'reject', 'unverified'} or completion.get('verdict') != report['verdict']:
        raise ValueError('completed review verdict differs from bound report')
    if report.get('subjects') != request['subjects'] or not report.get('evidence'):
        raise ValueError('completed review report lost source bindings')
    for ref in [*request['subjects'], *report['evidence']]:
        path = (Path(work) / ref['path']).resolve(strict=True)
        if not path.is_relative_to(Path(work).resolve()) or sha256_file(path) != ref['sha256']:
            raise ValueError('completed review subject/evidence changed')
    identity = json.loads((session / 'identity.json').read_text())
    if sha256_file(session / 'identity.json') != completion['identity_sha256'] or identity.get('role') != 'task-verifier':
        raise ValueError('completed review identity evidence changed')
    register_identity(Path(work), session, 'review-' + folder.name, 'task-verifier',
                      native=(session / 'native-result.json').exists())
    if completion.get('native_session'):
        native = json.loads((session / 'native-result.json').read_text())
        author_path = runtime_dir(work) / 'author.json'
        author = json.loads(author_path.read_text()) if author_path.exists() else {}
        if native.get('session_id') != completion['native_session'] or native['session_id'] == author.get('session_id'):
            raise ValueError('review reused author session or changed identity')
    return completion


def review_request(work):
    """Freeze each review once; producer-controlled status fields cannot certify it."""
    work = Path(work)
    state = json.loads((work / 'state.json').read_text())
    for request in state.get('review_requests', []):
        identifier = request.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier):
            raise ValueError('invalid review request ID')
        folder = runtime_dir(work) / 'reviews' / identifier
        if (folder / 'completion.json').exists():
            validate_completion(work, folder)
            frozen = json.loads((folder / 'request.json').read_text())
            if frozen != request:
                raise ValueError('review request changed; use a new ID')
            continue
        subjects = request.get('subjects', [])
        if not subjects or not request.get('instructions'):
            raise ValueError('review requires subjects and instructions')
        for ref in subjects:
            path = (work / ref['path']).resolve(strict=True)
            if not path.is_relative_to(work.resolve()) or sha256_file(path) != ref['sha256']:
                raise ValueError('review subject escaped workspace or changed')
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'request.json').exists():
            if json.loads((folder / 'request.json').read_text()) != request:
                raise ValueError('review request changed; use a new ID')
        else:
            save(folder / 'request.json', request)
        return request, folder
    return None


def finish_review(work, request, folder, session):
    report = json.loads((folder / 'report.json').read_text())
    if report.get('verdict') not in {'pass', 'reject', 'unverified'}:
        raise ValueError('review report requires explicit verdict')
    if report.get('subjects') != request['subjects'] or not report.get('evidence'):
        raise ValueError('review report requires source-bound subjects and evidence')
    for ref in [*request['subjects'], *report['evidence']]:
        path = (Path(work) / ref['path']).resolve(strict=True)
        if not path.is_relative_to(Path(work).resolve()) or sha256_file(path) != ref['sha256']:
            raise ValueError('review evidence changed or escaped workspace')
    # Host-generated execution provenance, not a reviewer_id supplied by the author.
    execution = json.loads((session / 'execution.json').read_text())
    if execution.get('role') != 'task-verifier' or execution.get('status') != 'returned':
        raise ValueError('review requires a returned independent verifier execution')
    save(folder / 'completion.json', {'verdict': report['verdict'],
         'report_sha256': sha256_file(folder / 'report.json'), 'session': str(session),
         'execution_sha256': sha256_file(session / 'execution.json'),
         'context_sha256': execution['context_sha256'], 'role': 'task-verifier',
         'transcript_sha256': sha256_file(session / 'stdout.log'),
         'identity_sha256': sha256_file(session / 'identity.json'),
         'native_session': json.loads((session / 'native-result.json').read_text()).get('session_id')
            if (session / 'native-result.json').exists() else None})


def failure_result(folder, status):
    text = ''
    for name in ('stderr.log', 'native-stderr.log', 'native-result.json'):
        path = Path(folder) / name
        if path.exists():
            text += path.read_text()[-20000:]
    transient = status == 'failed' and any(word in text.lower() for word in
        ('capacity', 'rate limit', 'rate_limit', '429', 'temporarily unavailable',
         'overloaded', 'connection reset', '502', '503'))
    return {'status': status, 'session': str(folder), 'transient': transient}


def supervise(args, drive):
    """Reuse drive, deterministic waits, persistent counters and ledger sentinel."""
    import fcntl
    from types import SimpleNamespace
    work = args.work.resolve(strict=True)
    config = load_config(args.config)
    rt = runtime_dir(work)
    digest = sha256_text(json.dumps(config, sort_keys=True))
    with (work / '.supervise.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = rt / 'control.json'
        control = json.loads(path.read_text()) if path.exists() else {
            'sessions': 0, 'retries': 0, 'config_history': [], 'status': 'ready'}
        if not control['config_history'] or control['config_history'][-1]['sha256'] != digest:
            control['config_history'].append({'sha256': digest, 'config': config, 'at': time.time()})
        if args.resume:
            (rt / 'stop.json').unlink(missing_ok=True)
        def stopped(signum, frame):
            save(rt / 'stop.json', {'reason': 'signal', 'signal': signum, 'at': time.time()})
        handlers = {s: signal.signal(s, stopped) for s in (signal.SIGTERM, signal.SIGINT)}
        if config.get('status_output'):
            output = config['status_output']
            if not isinstance(output, dict) or not output.get('path') or not output.get('task_id'):
                raise ValueError('status_output requires path and task_id')
            save(rt / 'status-output.json', output)
        driver_path = rt / 'driver.json'
        save(driver_path, config['driver'])
        try:
            while True:
                save(path, control)
                heartbeat(work, status=control['status'], worker=None, config_sha256=digest)
                if stop_requested(work):
                    pending = recover_inflight(work)
                    if pending and pending['status'] == 'waiting-worker':
                        execution_path = Path(pending['session']) / 'execution.json'
                        execution = json.loads(execution_path.read_text())
                        if not terminate_group(pending['pgid']):
                            result = {'status': 'stop-failed', 'worker': pending}
                            break
                        execution.update(status='stopped', ended=time.time())
                        save(execution_path, execution)
                    result = {'status': 'stopped'}
                    break
                if control.get('status') in {'failed', 'timeout'}:
                    prior_result = control.get('last_result', {})
                    if not prior_result.get('transient') or control['retries'] >= config['max_retries']:
                        result = prior_result or {'status': control['status']}
                        break
                pending = recover_inflight(work)
                if pending:
                    if pending['status'] == 'recovery-required':
                        result = pending
                        break
                    heartbeat(work, status='waiting-worker', worker=pending)
                    time.sleep(min(config['poll_seconds'], config['heartbeat_seconds']))
                    continue
                recovered_result = None
                if control.get('dispatch_after'):
                    attempts = [p for p in (work / 'sessions').glob('*/execution.json')
                                if p.parent.name.isdigit() and int(p.parent.name) >= control['dispatch_after']]
                    if not attempts:
                        result = {'status': 'recovery-required',
                                  'reason': 'Dispatch interrupted before execution identity was persisted.'}
                        break
                    latest = max(attempts, key=lambda p: p.parent.name)
                    execution = json.loads(latest.read_text())
                    if execution['status'] in {'failed', 'timeout', 'stopped', 'stop-failed'}:
                        recovered_result = failure_result(latest.parent, execution['status'])
                    elif execution['status'] == 'returned':
                        recovered_result = {'status': 'session-budget-exhausted', 'recovered_session': str(latest.parent)}
                    control['dispatch_after'] = None
                from datafarm.agent.progress import refresh_external_jobs
                state = refresh_external_jobs(work)
                if recovered_result is None and model_wait_status(state) and state.get('status') == 'working' and not review_request(work):
                    control['status'] = 'waiting-external-jobs'
                    # task_wait ignores heartbeat-only updates; bounded slices keep stop responsive.
                    wait_for_task(work, timeout=min(config['poll_seconds'], config['heartbeat_seconds']),
                                  interval=min(config['poll_seconds'], 1))
                    continue
                if recovered_result is None and state.get('status') in {'complete', 'needs-input'} and not review_request(work):
                    result = drive(SimpleNamespace(work=work, driver=driver_path, max_sessions=1,
                        timeout=config['timeout'], evolution_driver=None, runtime_config=config, supervised=True))
                    result.setdefault('status', 'complete' if result.get('valid') else 'error')
                    break
                if recovered_result is None and control['sessions'] >= config['max_sessions']:
                    result = {'status': 'session-budget-exhausted', 'sessions': control['sessions']}
                    break
                delay = control.get('retry_at', 0) - time.time()
                if recovered_result is None and delay > 0:
                    control['status'] = 'backoff'
                    time.sleep(min(delay, config['heartbeat_seconds'], 1))
                    continue
                if recovered_result is None:
                    # Reserve before launch: a crash cannot reset limits or hide a failed attempt.
                    control['sessions'] += 1
                    control.update(status='driving', dispatch_after=time.time_ns())
                    save(path, control)
                    result = drive(SimpleNamespace(work=work, driver=driver_path, max_sessions=1,
                        timeout=config['timeout'], evolution_driver=None, runtime_config=config, supervised=True))
                else:
                    result = recovered_result
                result.setdefault('status', 'complete' if result.get('valid') else 'error')
                control.update(last_result=result, status=result['status'], dispatch_after=None)
                if result['status'] in {'waiting-external-jobs', 'session-budget-exhausted'}:
                    control.update(retries=0, retry_at=0, status='ready')
                    save(path, control)
                    continue
                if result.get('transient') and control['retries'] < config['max_retries']:
                    control['retries'] += 1
                    delay = min(config['backoff_max_seconds'],
                                config['backoff_seconds'] * 2 ** (control['retries'] - 1))
                    control.update(status='backoff', retry_at=time.time() + delay)
                    save(path, control)
                    continue
                break
            control.update(status=result['status'], last_result=result)
            save(path, control)
            heartbeat(work, status=result['status'], worker=None)
            return result
        except Exception as exc:
            control.update(status='error', error={'type': type(exc).__name__, 'message': str(exc)})
            save(path, control)
            heartbeat(work, status='error', worker=None, error=control['error'])
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
