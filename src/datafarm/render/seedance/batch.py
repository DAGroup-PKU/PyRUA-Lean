"""One paid create per source, resume opaque task IDs.

The run directory holds batch.json: model, jobs, canary_ids, create_interval_seconds, upload_interval_seconds, optional
upload_host, and clock {transport_fps, canonical_fps, max_transport_frames}.
"""
import argparse
import concurrent.futures as cf
import fcntl
import json
import os
import time
from pathlib import Path

import requests

from datafarm.render.policy import check_seedance_payload, load_contract, submission_contract_path

from datafarm.render.seedance import client
from datafarm.render.seedance.common import finalize, now, read, sha, validate_clock, verify_file, write
from datafarm.render.seedance.references import contract as input_contract


def verify_resumed_contract(saved, expected):
    """Reuse an existing task ID only with its original recipe and verified policy."""
    base = {key: value for key, value in saved.items() if key != 'render_policy'}
    if base != expected:
        raise RuntimeError('submitted recipe changed')
    path = Path(expected['render_contract']) if expected.get('render_contract') else submission_contract_path()
    compiled = load_contract(path)
    policy = saved.get('render_policy', {})
    wanted = [ref['sha256'] for ref in compiled['intent']['real_references']]
    if (policy.get('schema') != 'datafarm.render-submission/1'
            or policy.get('contract_sha256') != sha(path)
            or policy.get('intent_sha256') != compiled['intent_sha256']
            or policy.get('prompt_sha256') != compiled['prompt_sha256']
            or policy.get('actual_image_sha256s', [])[:len(wanted)] != wanted):
        raise RuntimeError('submitted render policy missing or changed')


def run_one(run, job, config):
    folder = run / 'episodes' / job['id']; row = job['source']
    validate_clock(row, config['clock'])
    for key in ('white', 'carrier', 'action', 'source_manifest', 'structure'):
        verify_file(row[key])
    if 'objects' in job:
        verify_file(job['objects'])
    contract = input_contract(job, config)
    result = folder / 'result.json'; task = folder / 'task.json'; raw = folder / 'raw.mp4'
    if result.exists():
        d = read(result)
        if d['contract'] != contract:
            raise RuntimeError('completed recipe changed')
        for output in d['outputs'].values():
            verify_file(output)
        return d
    write(folder / 'input-contract.json', contract)
    if not task.exists():
        # Check prompt and real images against the render contract before the carrier is
        # published: a refused job must not have uploaded real-scene media anywhere.
        policy_path = Path(contract['render_contract']) if contract.get('render_contract') else submission_contract_path()
        check_seedance_payload(client.payload(config['model'], job, 'pending-upload'), policy_path)
        url = client.upload(row['carrier']['path'], run, folder, config['upload_interval_seconds'],
                            config.get('upload_host'))
        body = client.payload(config['model'], job, url)
        client.reserve(run, 'create-rate', config['create_interval_seconds'])
        record = client.create_once(task, body, contract)
    else:
        record = read(task)
        verify_resumed_contract(record['contract'], contract)
    if not record.get('task_id'):
        raise RuntimeError('one-create limit: no task ID, request will not be repeated')
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        try:
            if client.poll_download(task, raw):
                break
        except requests.RequestException as error:
            write(folder / 'poll-error.json', {'type': type(error).__name__, 'at': now(), 'task_id': record['task_id']})
        time.sleep(20)
    else:
        raise TimeoutError('poll deadline; existing task retained')
    result_data = finalize(raw, folder, row, config['clock'])
    for key in ('white', 'carrier', 'action', 'source_manifest', 'structure'):
        verify_file(row[key])
    result_data.update(id=job['id'], category=job['category'], task_id=record['task_id'],
                       completed_at=now(), contract=contract, inputs_unchanged=True,
                       status='technical_complete_pending_visual_review')
    write(result, result_data)
    return result_data


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['canary', 'all'], default='canary')
    parser.add_argument('--workers', type=int, default=8); args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError('workers must be 1..24')
    run = args.run.resolve(); config = read(run / 'batch.json')
    if not os.environ.get('ARK_API_KEY'):
        raise RuntimeError('ARK_API_KEY missing; no requests sent')
    if args.phase == 'all':
        continuation = read(run / 'canary-review.json')
        if not continuation.get('continue_exploration') or continuation['batch_sha256'] != sha(run / 'batch.json'):
            raise ValueError('canary review absent or bound to another batch')
    jobs = [j for j in config['jobs'] if args.phase == 'all' or j['id'] in config['canary_ids']]
    if len({j['source']['carrier']['sha256'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate source carrier')
    missing = [j['id'] for j in jobs if 'references' not in j]
    if missing:
        raise ValueError(f'jobs without an explicit references list: {missing}')
    lock = (run / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = now(); pending = list(jobs); active = {}; done = []; failures = []
    state_path = run / f'status-{args.phase}.json'
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            for job in list(pending):
                folder = run / 'episodes' / job['id']
                if len(active) >= args.workers:
                    break
                if (folder / 'failure.json').exists() and not (folder / 'result.json').exists():
                    failures.append(read(folder / 'failure.json')); pending.remove(job); continue
                active[pool.submit(run_one, run, job, config)] = job; pending.remove(job)
            completed, _ = cf.wait(active, timeout=5, return_when=cf.FIRST_COMPLETED) if active else (set(), set())
            for future in completed:
                job = active.pop(future)
                try:
                    result = future.result(); done.append({'id': job['id'], 'task_id': result['task_id']})
                except Exception as error:  # noqa: BLE001 - persist any worker failure and continue the fixed batch
                    failure = {'id': job['id'], 'type': type(error).__name__, 'reason': str(error), 'at': now()}
                    write(run / 'episodes' / job['id'] / 'failure.json', failure); failures.append(failure)
            tasks = []
            for job in jobs:
                task = run / 'episodes' / job['id'] / 'task.json'
                if task.exists():
                    t = read(task); tasks.append({'id': job['id'], 'task_id': t.get('task_id'), 'status': t['status']})
            state = {'status': 'running' if pending or active else ('complete' if not failures else 'complete_with_failures'),
                     'phase': args.phase, 'started_at': started, 'updated_at': now(), 'pid': os.getpid(),
                     'ppid': os.getppid(), 'sid': os.getsid(0), 'workers': args.workers, 'fixed_denominator': len(config['jobs']),
                     'phase_size': len(jobs), 'pending': [j['id'] for j in pending],
                     'active': [j['id'] for j in active.values()], 'completed': done, 'failures': failures,
                     'tasks': tasks, 'training_admitted': False}
            write(state_path, state)
    print(json.dumps({'status': state['status'], 'completed': len(done), 'failures': len(failures)}))


if __name__ == '__main__':
    main()
