import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from datafarm.agent.cost_budget import TaskBudget, micros, budget_for_work
from datafarm.agent.task_wait import wait_for_task, model_wait_status

ROOT = Path(__file__).resolve().parents[1]


def workspace(tmp_path, limit=50):
    (tmp_path / 'cost-policy.json').write_text(json.dumps({
        'schema': 'datafarm.task-cost-policy/2', 'mode': 'enforced', 'target_usd': min(30, limit),
        'limit_usd': limit, 'scope': 'all-model-work-through-white-data'}))
    (tmp_path / 'state.json').write_text(json.dumps({'status': 'working', 'phase': 'expert'}))
    (tmp_path / 'sessions').mkdir()
    return tmp_path


def receipt(identifier, cost):
    return {'schema': 'datafarm.session-cost/1', 'reservation_id': identifier,
            'includes_descendants': True, 'complete': True, 'cost_usd': cost,
            'cost_basis': 'api-equivalent', 'evidence': 'fixture usage'}


def test_resume_refunds_unused_reservation_but_never_resets_spend(tmp_path):
    workspace(tmp_path)
    b = TaskBudget(tmp_path)
    assert b.reserve('first', 40) == 40
    assert b.settle('first', receipt('first', 35))['remaining_usd'] == 15
    resumed = TaskBudget(tmp_path)
    assert resumed.reserve('second', 20) == 15
    assert resumed.settle('second', receipt('second', 15))['remaining_usd'] == 0
    with pytest.raises(ValueError, match='exhausted'):
        TaskBudget(tmp_path).reserve('third', 1)


@pytest.mark.parametrize('bad', [None, receipt('wrong', 0), {**receipt('one', 0), 'complete': False},
                               {**receipt('one', 0), 'includes_descendants': False},
                               receipt('one', float('nan')), receipt('one', -1)])
def test_missing_invalid_or_partial_usage_never_becomes_zero(tmp_path, bad):
    b = TaskBudget(workspace(tmp_path)); b.reserve('one', 5)
    s = b.settle('one', bad)
    assert s['blocked'] and s['committed_usd'] == 5 and s['unresolved'] == 1
    with pytest.raises(ValueError, match='blocked'):
        TaskBudget(tmp_path).reserve('two', 1)


def test_overrun_is_recorded_not_clipped(tmp_path):
    b = TaskBudget(workspace(tmp_path)); b.reserve('one', 5)
    s = b.settle('one', receipt('one', 9))
    assert s['known_usd'] == 9 and s['blocked']
    with pytest.raises(ValueError, match='already settled'):
        b.settle('one', receipt('one', 0))


def test_policy_mutation_cannot_reset_budget(tmp_path):
    b = TaskBudget(workspace(tmp_path)); b.reserve('one', 1)
    p = tmp_path / 'cost-policy.json'; s = json.loads(p.read_text()); s['limit_usd'] = 100
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match='policy changed'):
        TaskBudget(tmp_path).status()


def test_deleting_policy_cannot_turn_budgeted_task_into_unbudgeted(tmp_path):
    b = TaskBudget(workspace(tmp_path))
    (tmp_path / 'input.json').write_text(json.dumps({'cost_policy_sha256': b.digest}))
    (tmp_path / 'cost-policy.json').unlink()
    with pytest.raises(FileNotFoundError): budget_for_work(tmp_path)


def test_missing_ledger_after_execution_is_not_a_fresh_budget(tmp_path):
    workspace(tmp_path)
    folder = tmp_path / 'sessions/previous'; folder.mkdir()
    (folder / 'execution.json').write_text(json.dumps({'cost_reservation_id': 'previous'}))
    with pytest.raises(ValueError, match='refusing to reset'):
        TaskBudget(tmp_path).status()


def test_crashed_session_keeps_reservation_across_restart(tmp_path):
    b = TaskBudget(workspace(tmp_path)); b.reserve('crashed', 4)
    with pytest.raises(ValueError, match='unresolved'):
        TaskBudget(tmp_path).reserve('retry', 4)
    assert TaskBudget(tmp_path).status()['committed_usd'] == 4


def _reserve_worker(work, identifier, queue):
    try:
        TaskBudget(work).reserve(identifier, 40); queue.put('reserved')
    except ValueError:
        queue.put('blocked')


def test_concurrent_drivers_cannot_double_allocate(tmp_path):
    workspace(tmp_path)
    ctx = multiprocessing.get_context('spawn'); queue = ctx.Queue()
    workers = [ctx.Process(target=_reserve_worker, args=(tmp_path, str(i), queue)) for i in range(2)]
    for p in workers: p.start()
    for p in workers: p.join(5); assert p.exitcode == 0
    assert sorted(queue.get(timeout=1) for _ in workers) == ['blocked', 'reserved']


def driver_module(monkeypatch, work):
    spec = importlib.util.spec_from_file_location('budget_pipeline_cli', ROOT / 'tools/pipeline_agent.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'verify_input', lambda _: {})
    monkeypatch.setattr(module, 'source_for', lambda *_: ROOT)
    monkeypatch.setattr(module, 'context_for', lambda _: {})
    monkeypatch.setattr(module, 'user_context', lambda _: 'fixture input')
    monkeypatch.setattr(module, 'capture', lambda *a, **kw: None)
    def context(folder, *args, **kwargs):
        folder.mkdir(parents=True)
        receipt = {'messages_sha256': 'fixture', 'role': args[0]}
        (folder / 'receipt.json').write_text(json.dumps(receipt))
        return {'messages': [], 'receipt': receipt}
    monkeypatch.setattr(module, 'write_context', context)
    return module


def drive_args(work, script, *, metered=True, max_sessions=3):
    driver = {'argv': [sys.executable, str(script)]}
    if metered:
        driver['cost_control'] = {'protocol': 'datafarm.session-cost/1',
                                  'includes_descendants': True, 'max_session_usd': 1}
    path = work / 'driver.json'; path.write_text(json.dumps(driver))
    return SimpleNamespace(work=work, driver=path, timeout=3, max_sessions=max_sessions, evolution_driver=None)


def test_drive_stops_before_launch_without_driver_capability(tmp_path, monkeypatch):
    work = workspace(tmp_path); module = driver_module(monkeypatch, work)
    with pytest.raises(ValueError, match='metered driver'):
        module.drive(drive_args(work, work / 'must-not-run.py', metered=False))
    assert not list((work / 'sessions').iterdir())


def test_drive_dependency_wait_never_launches_or_reserves(tmp_path, monkeypatch):
    work = workspace(tmp_path); module = driver_module(monkeypatch, work)
    state = {'status': 'working', 'model_wait_for': ['gpu-1'],
             'external_jobs': [{'id': 'gpu-1', 'status': 'running'}]}
    (work / 'state.json').write_text(json.dumps(state))
    args = drive_args(work, work / 'must-not-run.py')
    for status in ['running', 'queued', 'unknown']:
        state['external_jobs'][0]['status'] = status
        (work / 'state.json').write_text(json.dumps(state))
        result = module.drive(args)
        assert result['status'] == 'waiting-external-jobs'
        assert result['job_ids'] == ['gpu-1'] and not result['liveness_verified']
    assert not list((work / 'sessions').iterdir())
    assert TaskBudget(work).status()['sessions'] == 0
    state['external_jobs'][0]['status'] = 'finished'
    assert model_wait_status(state) is None
    state['external_jobs'][0]['status'] = 'failed'
    assert model_wait_status(state) is None


@pytest.mark.parametrize('ids,jobs', [
    (['missing'], []), (['x', 'x'], [{'id': 'x', 'status': 'running'}]),
    (['x'], [{'id': 'x', 'status': 'running'}] * 2),
    ('x', []), (['x'], [{'id': 'x'}])])
def test_invalid_dependency_wait_does_not_authorize_a_call(ids, jobs):
    with pytest.raises(ValueError):
        model_wait_status({'model_wait_for': ids, 'external_jobs': jobs})


@pytest.mark.parametrize('max_sessions', [1, 3])
def test_model_can_submit_job_then_yield_without_another_call(tmp_path, monkeypatch, max_sessions):
    work = workspace(tmp_path); module = driver_module(monkeypatch, work)
    script = work / 'submit-once.py'
    script.write_text('''import os,json,pathlib,sys
json.load(sys.stdin)
pathlib.Path('state.json').write_text(json.dumps({'status':'working',
 'model_wait_for':['gpu-1'],'external_jobs':[{'id':'gpu-1','status':'running'}]}))
pathlib.Path(os.environ['DATAFARM_COST_RECEIPT']).write_text(json.dumps({
 'schema':'datafarm.session-cost/1','reservation_id':os.environ['DATAFARM_COST_RESERVATION_ID'],
 'includes_descendants':True,'complete':True,'cost_usd':0.25,
 'cost_basis':'provider','evidence':'fixture, no network'}))
''')
    args = drive_args(work, script, max_sessions=max_sessions)
    assert module.drive(args)['status'] == 'waiting-external-jobs'
    assert module.drive(args)['status'] == 'waiting-external-jobs'
    assert len(list((work / 'sessions').iterdir())) == 1
    assert TaskBudget(work).status()['known_usd'] == .25


def test_drive_accounts_descendants_and_caps_second_session(tmp_path, monkeypatch):
    work = workspace(tmp_path, limit=1.5); module = driver_module(monkeypatch, work)
    script = work / 'fake-metered.py'
    script.write_text('''import os,json,pathlib,sys
json.load(sys.stdin)
cost=float(os.environ['DATAFARM_SESSION_MAX_USD'])
p=pathlib.Path('seen-caps.json')
caps=json.loads(p.read_text()) if p.exists() else []
p.write_text(json.dumps(caps+[cost]))
pathlib.Path(os.environ['DATAFARM_COST_RECEIPT']).write_text(json.dumps({
 'schema':'datafarm.session-cost/1','reservation_id':os.environ['DATAFARM_COST_RESERVATION_ID'],
 'includes_descendants':True,'complete':True,'cost_usd':cost,
 'cost_basis':'provider','evidence':'fake provider (no network)'}))
''')
    result = module.drive(drive_args(work, script))
    assert result['status'] == 'cost-budget-blocked'
    assert json.loads((work / 'seen-caps.json').read_text()) == [1, .5]
    assert TaskBudget(work).status()['known_usd'] == 1.5


def test_drive_missing_receipt_blocks_restart_even_if_claims_complete(tmp_path, monkeypatch):
    work = workspace(tmp_path); module = driver_module(monkeypatch, work)
    script = work / 'unmetered.py'
    script.write_text("import json,sys,pathlib;json.load(sys.stdin);pathlib.Path('state.json').write_text(json.dumps({'status':'complete'}))")
    args = drive_args(work, script)
    assert module.drive(args)['status'] == 'cost-budget-blocked'
    assert module.drive(args)['status'] == 'cost-budget-blocked'
    assert len(list((work / 'sessions').iterdir())) == 1


def test_wait_ignores_heartbeat_and_returns_on_job_completion(tmp_path, monkeypatch):
    import datafarm.agent.task_wait as module
    p = tmp_path / 'state.json'
    state = {'status': 'working', 'phase': 'expert', 'external_jobs': [{'id': 'one', 'status': 'running'}]}
    p.write_text(json.dumps(state)); ticks = []
    def sleep(_):
        ticks.append(1)
        state['heartbeat'] = len(ticks)
        state['external_jobs'][0]['heartbeat'] = len(ticks)
        if len(ticks) == 3: state['external_jobs'][0]['status'] = 'finished'
        p.write_text(json.dumps(state))
    monkeypatch.setattr(module.time, 'sleep', sleep)
    result = wait_for_task(tmp_path, timeout=5, interval=.01)
    assert len(ticks) == 3 and result['status'] == 'task-transition'
    assert not result['delivery_verified']


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, None])
def test_invalid_amounts(value):
    with pytest.raises(ValueError): micros(value)
