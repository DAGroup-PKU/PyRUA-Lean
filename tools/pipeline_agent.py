#!/usr/bin/env python3
"""Source-checkout entry: bind input, load general context, supervise a supplied agent."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from datafarm.common import atomic_write_json
from datafarm.agent.context import write_context
from datafarm.common import sha256_file
from datafarm.task.contract import verify_delivery, verify_input
from datafarm.evolution.integration import bind, binding, source_for, context_for, capture, feedback
from datafarm.agent.cost_budget import TaskBudget, validate_policy, micros, budget_for_work
from datafarm.agent.task_wait import wait_for_task, model_wait_status
from datafarm.agent.runtime import (supervise, runtime_dir, stop_requested, heartbeat,
    process_identity, host_identity, foreign_host, recover_inflight, review_request, finish_review, terminate_group, failure_result, register_identity)


def save(path, value):
    atomic_write_json(path, value, sort_keys=False, mode=0o644, fsync=False)


def prepare(args):
    cost_policy = validate_policy(json.loads(args.cost_policy.read_text()))
    episode = args.episode.resolve(strict=True)
    request = args.request_file.read_text()
    paths = [episode] if episode.is_file() else sorted(p for p in episode.rglob('*') if p.is_file())
    paths += [p.resolve(strict=True) for p in args.reference]
    paths = sorted(set(paths))
    if not paths:
        raise ValueError('episode directory is empty')
    # Episode directories are data inputs, never a root containing credentials or research.
    forbidden = {'keys', 'secrets', '.env', '.ssh', '.codex'}
    if any(forbidden.intersection(p.resolve().parts) for p in paths):
        raise ValueError('episode input includes a credential/config directory; choose the episode only')
    if args.target_count is not None and args.target_count <= 0:
        raise ValueError('target count must be positive')
    files = [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in paths]
    lock = {'schema': 'datafarm.episode-input/1', 'files': files,
            'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'request': request, 'target_count': args.target_count, 'stop_after': args.stop_after,
            'cost_policy_sha256': hashlib.sha256(json.dumps(cost_policy, sort_keys=True).encode()).hexdigest()}
    work = args.output.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError('workspace is not empty; use context/drive to resume it')
    work.mkdir(parents=True, exist_ok=True)
    save(work / 'input.json', lock)
    save(work / 'cost-policy.json', cost_policy)
    TaskBudget(work).status()
    save(work / 'state.json', {'schema': 'datafarm.agent-state/1', 'status': 'working',
        'phase': 'observe', 'next_action': 'Inspect the real episode and generate new task products.',
        'open_questions': [], 'external_jobs': []})
    for name in ['task', 'evidence', 'sessions']:
        (work / name).mkdir()
    capability = bind(work, ROOT, store_path=args.evolution_store,
                      selectors=json.loads(args.scope.read_text()) if args.scope else None)
    pinned_root = source_for(work, ROOT)
    (work / 'AGENTS.md').write_text('Read ' + str(pinned_root / 'AGENTS.md') + '\n'
        'This workspace contains one NEW task. Its adapter, criteria and prompts are agent-generated products.\n'
        'Read input.json and state.json. Keep state current. Do not self-certify independent review.\n')
    capture(work, kind='observation', statement='New episode task prepared; adapters are agent-generated.',
            ref=str(work / 'input.json'), details={'input_sha256': lock['sha256']})
    feedback(work, {'kind': 'user-requirement', 'statement': request,
                   'source': {'actor': 'user-request-file', 'ref': str(args.request_file.resolve())},
                   'scope': {'run_id': capability['scope']['run_id']}})
    return {'workspace': str(work), 'input_sha256': lock['sha256'], 'capability_version': capability['version']}


def user_context(work):
    if work is None:
        return ''
    lock = verify_input(work)
    state = json.loads((work / 'state.json').read_text())
    budget = budget_for_work(work)
    cost = budget.status() if budget else {'status': 'no enforced task cost policy'}
    return ('Continue the requested real task to data delivery. Repository: ' + str(source_for(work, ROOT))
        + '\nWorkspace: ' + str(work.resolve())
        + '\nOriginal user request:\n' + lock['request']
        + '\nInput manifest (read files as evidence):\n' + json.dumps(lock, ensure_ascii=False)
        + '\nCurrent state (verify job liveness before acting):\n' + json.dumps(state, ensure_ascii=False)
        + '\nWhole-task model cost budget (includes all descendants and retries):\n' + json.dumps(cost)
        + '\nGenerate missing task products yourself. Maintain state.json and next_action after each step. '
        'Use delivery.json only after actual independent review and data export. Never equate writing files '
        'with completing physical/video validation. Preserve external task IDs and failure denominators.')


def needs_input_result(state):
    """needs-input hands control to the user, so it must say what only the user can supply.

    Code requires concrete questions; which questions qualify (intent, device facts or access
    that cannot be derived) is a role instruction checked by review, not by this function.
    """
    questions = state.get('open_questions')
    if (not isinstance(questions, list) or not questions
            or not all((isinstance(q, str) and q.strip()) or (isinstance(q, dict) and q.get('question'))
                       for q in questions)):
        raise ValueError('needs-input requires concrete open_questions '
                         '(intent, device facts or access that cannot be derived)')
    return {'status': 'needs-input', 'next_action': state.get('next_action'), 'questions': questions}


def drive(args):
    work = args.work.resolve(strict=True)
    if args.max_sessions < 1 or args.timeout <= 0:
        raise ValueError('session/time budget must be positive')
    driver = json.loads(args.driver.read_text())
    native = driver.get('type') == 'codex'
    argv = ([sys.executable, str(ROOT / 'src/datafarm/agent/native_codex.py')]
            if native else driver.get('argv'))
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise ValueError('driver argv must be a nonempty array of strings; no shell command')
    kind = driver.get('input', 'messages')
    if kind not in {'text', 'messages'}:
        raise ValueError('driver input must be text or messages')
    budget = budget_for_work(work)
    control = driver.get('cost_control', {})
    if budget and budget.enforced:
        if native or (control.get('protocol') != 'datafarm.session-cost/1'
                or control.get('includes_descendants') is not True
                or micros(control.get('max_session_usd', 0)) <= 0):
            raise ValueError('budgeted task needs a metered driver that caps ALL descendant calls; '
                             'a plain CLI driver cannot enforce the task USD budget')
    if budget and budget.mode != 'disabled' and args.evolution_driver:
        # Advisory budgets too: evolution cost would not enter the task ledger.
        raise ValueError('post-run improvement is separate engineering work; '
                         'do not launch an unbudgeted evolution driver from a budgeted task')
    # This is a foreground controller. Lock prevents concurrent sessions for the same task.
    with ExitStack() as stack:
        if not getattr(args, 'supervised', False):
            outer = stack.enter_context((work / '.supervise.lock').open('a'))
            fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = stack.enter_context((work / '.driver.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = recover_inflight(work)
        if pending:
            return pending
        for _ in range(args.max_sessions):
            if stop_requested(work):
                return {'status': 'stopped'}
            verify_input(work)
            state = json.loads((work / 'state.json').read_text())
            if budget:
                costs = budget.status()
                if costs['stop_required'] or (budget.enforced and costs['unresolved']):
                    return {'status': 'cost-budget-blocked', 'cost': costs}
            review = review_request(work)
            if state['status'] == 'complete' and not review:
                return verify_delivery(work)
            if state['status'] == 'needs-input' and not review:
                return needs_input_result(state)
            if state['status'] not in {'working', 'complete', 'needs-input'}:
                raise ValueError('invalid task state status')
            waiting = model_wait_status(state)
            if waiting and not review:
                return waiting
            if budget:
                costs = budget.status()
                if costs['stop_required'] or (budget.enforced and costs['unresolved']):
                    return {'status': 'cost-budget-blocked', 'cost': costs,
                            'next_action': 'Preserve outputs and job IDs; reconcile cost before resuming.'}
            folder = work / 'sessions' / f'{time.time_ns()}'
            source_for(work, ROOT)  # detect changed pinned code before every session
            role = 'task-verifier' if review else 'orchestrator'
            identity_protocol = driver.get('session_identity_protocol') == 'datafarm.agent-identity/1'
            if review and not native and not identity_protocol:
                raise ValueError('generic review driver must implement datafarm.agent-identity/1; argv alone cannot prove independence')
            session_key = 'review-' + review[0]['id'] if review else 'author'
            user = user_context(work)
            if review:
                request, review_folder = review
                user += ('\nIndependent review request: ' + json.dumps(request, ensure_ascii=False)
                    + '\nInspect actual subjects independently. Do not modify task state, subjects or criteria. '
                    'Do not execute producer work. Write only your evidence and report to '
                    + str(review_folder) + '/report.json. Report JSON: '
                    '{"verdict":"pass|reject|unverified","subjects":<exact requested subjects>,'
                    '"evidence":[{"path":"workspace-relative-file","sha256":"..."}],"findings":[...]}. '
                    'A rejected review must retain evidence. Process exit is not a pass. '
                    'When you review delivered episodes, also write each episode review file with '
                    '"runtime_review": "' + request['id'] + '" and list '
                    'that file in the report evidence; delivery verification refuses reviews without it.')
            context = write_context(folder, role, root=ROOT, user=user, experience=context_for(work))
            payload = (folder / 'context.md').read_text() if kind == 'text' else json.dumps(context['messages'], ensure_ascii=False)
            started = time.time()
            record = {'status': 'starting', 'started': started, 'input_kind': kind,
                      'context_sha256': context['receipt']['messages_sha256'], 'role': role,
                      **host_identity()}
            if review:
                record['review_id'] = request['id']
            save(folder / 'execution.json', record)
            env = os.environ.copy()
            env.update(DATAFARM_SESSION_DIR=str(folder), DATAFARM_AGENT_ROLE=role,
                       DATAFARM_AGENT_SESSION_KEY=session_key,
                       DATAFARM_AGENT_IDENTITY_RECEIPT=str(folder / 'identity.json'))
            runtime_config = getattr(args, 'runtime_config', {})
            if native:
                native_state = runtime_dir(work) / ('review-' + request['id'] + '.json' if review else 'author.json')
                record['native_state'] = str(native_state)
                env.update(DATAFARM_NATIVE_CONFIG=str(args.driver.resolve()),
                           DATAFARM_NATIVE_STATE=str(native_state))
                save(folder / 'execution.json', record)
            if budget:
                reserved = budget.reserve(folder.name, control.get('max_session_usd', 0))
                env.update(DATAFARM_COST_RESERVATION_ID=folder.name,
                           DATAFARM_SESSION_MAX_USD=str(reserved),
                           DATAFARM_COST_RECEIPT=str(folder / 'cost-receipt.json'))
                record.update(cost_reservation_usd=reserved, cost_reservation_id=folder.name)
                save(folder / 'execution.json', record)
            if runtime_config.get('log_directory'):
                log_dir = Path(runtime_config['log_directory'])
                if not log_dir.is_absolute():
                    log_dir = work / log_dir
                log_dir = log_dir / folder.name
                log_dir.mkdir(parents=True, exist_ok=True)
                for name in ('stdout.log', 'stderr.log', 'native-stderr.log'):
                    (folder / name).symlink_to((log_dir / name).resolve())
            with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
                try:
                    p = subprocess.Popen(argv, cwd=runtime_config.get('working_directory', work), stdin=subprocess.PIPE, stdout=out, stderr=err,
                                         text=True, start_new_session=True, env=env)
                except OSError:
                    if budget:
                        budget.settle(folder.name, {'schema': 'datafarm.session-cost/1',
                            'reservation_id': folder.name, 'includes_descendants': True,
                            'complete': True, 'cost_usd': 0, 'cost_basis': 'provider',
                            'evidence': 'Popen failed before the driver started'})
                    raise
                record.update(status='running', pid=p.pid, process_identity=process_identity(p.pid))
                save(folder / 'execution.json', record)
                deadline = time.monotonic() + args.timeout
                first = True
                while True:
                    heartbeat(work, status='driving', worker={'pid': p.pid, 'session': str(folder),
                              'role': role, 'process_identity': record['process_identity'],
                              'alive': p.poll() is None})
                    stopped = stop_requested(work)
                    if stopped or time.monotonic() >= deadline:
                        terminated = terminate_group(p.pid)
                        p.wait(timeout=2)
                        record['status'] = ('stopped' if stopped else 'timeout') if terminated else 'stop-failed'
                        break
                    try:
                        p.communicate(payload if first else None,
                                      timeout=min(1, max(.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        first = False
                        continue
                    record['status'] = 'returned' if p.returncode == 0 else 'failed'
                    break
                # Descendants cannot keep running silently after their driver returns.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record.update(exit_code=p.returncode, ended=time.time())
                save(folder / 'execution.json', record)
            if budget:
                try:
                    receipt = json.loads((folder / 'cost-receipt.json').read_text())
                except (OSError, ValueError):
                    receipt = None
                costs = budget.settle(folder.name, receipt)
                record['cost'] = costs
                save(folder / 'execution.json', record)
            capture(work, kind='session-result', statement='Task agent session ' + record['status'],
                    ref=str(folder / 'execution.json'), details=record)
            if record['status'] != 'returned':
                return failure_result(folder, record['status'])
            if native or identity_protocol:
                register_identity(work, folder, session_key, role, native=native)
            if review:
                finish_review(work, request, review_folder, folder)
            if budget and costs['stop_required']:
                return {'status': 'cost-budget-blocked', 'session': str(folder), 'cost': costs}
        state = json.loads((work / 'state.json').read_text())
        if state['status'] == 'complete' and not review_request(work):
            return verify_delivery(work)
        if state['status'] == 'working':
            waiting = model_wait_status(state)
            if waiting:
                return waiting
        if state['status'] == 'needs-input':
            return needs_input_result(state)
        return {'status': 'session-budget-exhausted',
                'next_action': state.get('next_action'), 'questions': state.get('open_questions', [])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--episode', type=Path, required=True)
    prep.add_argument('--request-file', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--target-count', type=int)
    prep.add_argument('--stop-after', choices=['render', 'white'], default='render',
                      help='explicit delivery boundary; white does not certify rendering or training')
    prep.add_argument('--evolution-store', type=Path, help='use active evaluated capability from this store')
    prep.add_argument('--scope', type=Path, help='JSON exact selectors for task family/hardware/provider/run')
    prep.add_argument('--cost-policy', type=Path, default=ROOT / 'configs/task-cost-policy.json',
                      help='whole-task model policy; default advisory target $30, limit $50; excludes rendering/GPU')
    prep.add_argument('--reference', type=Path, action='append', default=[], help='additional real photo/device evidence; repeatable')
    ctx = sub.add_parser('context')
    ctx.add_argument('--role', default='orchestrator')
    ctx.add_argument('--work', type=Path)
    ctx.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--work', type=Path, required=True)
    cost = sub.add_parser('cost-status'); cost.add_argument('--work', type=Path, required=True)
    wait = sub.add_parser('wait'); wait.add_argument('--work', type=Path, required=True)
    wait.add_argument('--timeout', type=float, default=1800)
    wait.add_argument('--interval', type=float, default=5)
    d = sub.add_parser('drive')
    d.add_argument('--work', type=Path, required=True)
    d.add_argument('--driver', type=Path, required=True)
    d.add_argument('--max-sessions', type=int, default=1)
    d.add_argument('--timeout', type=float, default=1800)
    d.add_argument('--evolution-driver', type=Path, help='optional coding driver for one bounded post-run improvement cycle')
    d.add_argument('--evolution-policy', type=Path, help='trusted evaluator policy for post-run improvement')
    d.add_argument('--evolution-timeout', type=float, default=600)
    sup = sub.add_parser('supervise', help='persistent continuation; launch via run ledger')
    sup.add_argument('--work', type=Path, required=True)
    sup.add_argument('--config', type=Path, required=True)
    sup.add_argument('--resume', action='store_true', help='clear an explicit user stop; retain counters/IDs')
    stop = sub.add_parser('stop', help='request stop without cancelling external jobs')
    stop.add_argument('--work', type=Path, required=True)
    stop.add_argument('--reason', default='user stop')
    status = sub.add_parser('runtime-status')
    status.add_argument('--work', type=Path, required=True)
    f = sub.add_parser('feedback')
    f.add_argument('--work', type=Path, required=True)
    f.add_argument('--file', type=Path, required=True, help='visible host event JSON with kind, statement, source and optional scope')
    args = p.parse_args(argv)
    try:
        if args.cmd in {'context', 'drive', 'verify', 'cost-status', 'wait', 'supervise'} and args.work:
            pinned = source_for(args.work, ROOT)
            if pinned != ROOT.resolve():
                result = subprocess.run([sys.executable, str(pinned / 'tools/pipeline_agent.py'), *(argv or sys.argv[1:])])
                return result.returncode
        if args.cmd == 'prepare': result = prepare(args)
        elif args.cmd == 'context':
            result = write_context(args.output, args.role, root=ROOT, user=user_context(args.work),
                                   experience=context_for(args.work) if args.work else None)['receipt']
        elif args.cmd == 'verify': result = verify_delivery(args.work)
        elif args.cmd == 'cost-status': result = TaskBudget(args.work).status()
        elif args.cmd == 'wait': result = wait_for_task(args.work, args.timeout, args.interval)
        elif args.cmd == 'supervise': result = supervise(args, drive)
        elif args.cmd == 'stop':
            save(runtime_dir(args.work) / 'stop.json', {'reason': args.reason, 'at': time.time()})
            result = {'status': 'stop-requested'}
        elif args.cmd == 'runtime-status':
            rt = runtime_dir(args.work)
            result = json.loads((rt / 'supervisor.json').read_text()) if (rt / 'supervisor.json').exists() else {'status': 'not-started'}
            # Liveness is only observable on the supervisor's own host; elsewhere it is unknown.
            remote = foreign_host({'host': result.get('supervisor_host')})
            result['supervisor_alive'] = None if remote else bool(result.get('supervisor_pid') and process_identity(result['supervisor_pid']) == result.get('supervisor_identity') and process_identity(result['supervisor_pid']))
            worker = result.get('worker') or {}
            result['worker_alive'] = None if remote else bool(worker.get('pid') and process_identity(worker['pid']) == worker.get('process_identity') and process_identity(worker['pid']))
        elif args.cmd == 'feedback': result = feedback(args.work, json.loads(args.file.read_text()))
        else:
            if bool(args.evolution_driver) != bool(args.evolution_policy):
                raise ValueError('post-run evolution requires both driver and evaluator policy')
            result = drive(args)
            if budget_for_work(args.work):
                result['cost'] = TaskBudget(args.work).status()
            capture(args.work, kind='task-result', statement='Task driver returned; inspect delivery or remaining work.',
                    ref=str(args.work / 'state.json'), details=result)
            if args.evolution_driver:
                from datafarm.evolution.controller import cycle
                from datafarm.evolution.store import Store
                capability = binding(args.work)
                if not capability:
                    raise ValueError('post-run evolution requires a bound task workspace')
                try:
                    result['evolution'] = cycle(Store(capability['store']), root=ROOT,
                        driver=json.loads(args.evolution_driver.read_text()), policy_path=args.evolution_policy,
                        timeout=args.evolution_timeout)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    # A broken optional improvement policy must not erase a
                    # valid task result or silently claim evolution succeeded.
                    result['evolution'] = {'status': 'error', 'reason': str(exc)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('status') not in {'failed', 'timeout', 'needs-input', 'session-budget-exhausted', 'cost-budget-blocked', 'recovery-required', 'stopped', 'stop-failed', 'error'} else 75
    except (ValueError, OSError, KeyError, TypeError) as e:
        if getattr(args, 'work', None) and args.cmd in {'drive', 'verify'}:
            try:
                capture(args.work, kind='task-result', statement='Task validation or driver error: ' + str(e),
                        ref=str(args.work / 'state.json'), details={'status': 'error'})
            except (OSError, ValueError, KeyError, TypeError):
                pass  # preserve the original error when recording is unavailable
        print(f'pipeline_agent: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
