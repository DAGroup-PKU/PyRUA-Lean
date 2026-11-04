"""Deterministic subprocess fixtures: these do not certify robot/video quality."""
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from datafarm.agent.runtime import (supervise, recover_inflight, process_identity,
    runtime_dir, review_request)
from datafarm.agent.native_codex import command, run, save
from datafarm.agent.cost_budget import TaskBudget
from test_task_cost_budget import workspace, driver_module, drive_args, receipt


def policy(work, mode='advisory', on_excess='warn'):
    path = work / 'cost-policy.json'
    value = json.loads(path.read_text())
    value.update(schema='datafarm.task-cost-policy/2', mode=mode, on_excess=on_excess)
    save(path, value)


IDENTITY_SCRIPT = """import os,json,pathlib,hashlib,uuid
identity_path=pathlib.Path(os.environ['DATAFARM_AGENT_IDENTITY_RECEIPT'])
proof=identity_path.with_name('provider-identity-fixture.txt')
identifier=str(uuid.uuid4())
proof.write_text('Fixture provider session: '+identifier)
identity_path.write_text(json.dumps({'schema':'datafarm.agent-identity/1','session_id':identifier,
'session_key':os.environ['DATAFARM_AGENT_SESSION_KEY'],'role':os.environ['DATAFARM_AGENT_ROLE'],'fresh':True,
'evidence':[{'path':str(proof.relative_to(pathlib.Path.cwd())),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}]}))
"""


def setup(tmp_path, monkeypatch, script, **config):
    work = workspace(tmp_path)
    policy(work)
    module = driver_module(monkeypatch, work)
    executable = work / 'worker.py'
    executable.write_text(IDENTITY_SCRIPT + script)
    cfg = {'schema': 'datafarm.agent-runtime/1',
           'driver': {'argv': [sys.executable, str(executable)],
                      'session_identity_protocol': 'datafarm.agent-identity/1'},
           'max_sessions': 5, 'timeout': 3, 'poll_seconds': .02,
           'heartbeat_seconds': .03, 'max_retries': 2,
           'backoff_seconds': .03, 'backoff_max_seconds': .1, **config}
    path = work / 'runtime.json'; save(path, cfg)
    return work, module, SimpleNamespace(work=work, config=path, resume=False)


def test_budget_modes_and_unknown_total(tmp_path):
    workspace(tmp_path)
    policy(tmp_path)
    b = TaskBudget(tmp_path); b.reserve('one', 0)
    status = b.settle('one', {'usage': {'input_tokens': 10}, 'cost_usd': None})
    assert status['total_usd'] is None and status['unknown_sessions'] == 1
    assert not status['blocked'] and status['warning']
    b.reserve('two', 0)
    assert b.settle('two', receipt('two', 60))['known_usd'] == 60
    b.reserve('three', 0)  # advisory warn continues
    assert TaskBudget(tmp_path).status()['sessions'] == 3


@pytest.mark.parametrize('mode,excess,blocked', [('disabled','pause',False),
    ('advisory','pause',True), ('advisory','warn',False), ('enforced','warn',True)])
def test_manual_excess_action(tmp_path, mode, excess, blocked):
    workspace(tmp_path); policy(tmp_path, mode, excess)
    b = TaskBudget(tmp_path); b.reserve('one', 1)
    status = b.settle('one', receipt('one', 60))
    assert status['blocked'] == blocked
    assert status['known_usd'] == 60


def test_native_command_has_explicit_resume_no_cap_or_credentials():
    cfg = {'model':'deployment-model','reasoning_effort':'high'}
    cmd = command(cfg, 'thread-123')
    assert cmd[:4] == ['codex', 'exec', 'resume', 'thread-123']
    assert '--last' not in cmd and '--ephemeral' not in cmd
    assert '--model' in cmd and '--json' in cmd
    assert not any('credential' in v for v in cmd)


def test_native_stream_retains_session_usage_and_resume(tmp_path, capsys):
    exe = tmp_path / 'fake-codex'
    exe.write_text('#!' + sys.executable + '\nimport sys,json\nsys.stdin.read()\n'
        'print(json.dumps({"type":"thread.started","thread_id":"saved-id"}))\n'
        'print(json.dumps({"type":"turn.completed","usage":{"input_tokens":17,"output_tokens":4}}))\n')
    exe.chmod(0o755)
    cfg = {'executable':str(exe)}
    folder = tmp_path / 'attempt1'; folder.mkdir()
    assert run(cfg, 'hi', tmp_path / 'native.json', folder, tmp_path) == 0
    result = json.loads((folder / 'native-result.json').read_text())
    assert result['session_id'] == 'saved-id' and result['usage']['input_tokens'] == 17
    assert result['cost_usd'] is None and not result['includes_descendants']
    folder2 = tmp_path / 'attempt2'; folder2.mkdir()
    run(cfg, 'continue', tmp_path / 'native.json', folder2, tmp_path)
    assert json.loads((folder2 / 'native-command.json').read_text())['resume'] == 'saved-id'


def test_normal_phase_continuation_and_persistent_limit(tmp_path, monkeypatch):
    script = '''import json,pathlib,sys
json.load(sys.stdin)
p=pathlib.Path('state.json');s=json.loads(p.read_text())
s['phase']='verify' if s['phase']=='expert' else 'done';p.write_text(json.dumps(s))
'''
    work, module, args = setup(tmp_path, monkeypatch, script, max_sessions=2)
    result = supervise(args, module.drive)
    assert result['status'] == 'session-budget-exhausted'
    assert json.loads((work / 'state.json').read_text())['phase'] == 'done'
    assert len(list((work / 'sessions').iterdir())) == 2
    supervise(args, module.drive)
    assert len(list((work / 'sessions').iterdir())) == 2
    assert TaskBudget(work).status()['unknown_sessions'] == 2


def test_transient_bounded_backoff_then_resume(tmp_path, monkeypatch):
    script = '''import json,pathlib,sys
json.load(sys.stdin)
p=pathlib.Path('attempt-count');n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1))
if n==0: print('capacity temporarily unavailable',file=sys.stderr);sys.exit(1)
pathlib.Path('state.json').write_text(json.dumps({'status':'needs-input','open_questions':['fixture done']}))
'''
    work, module, args = setup(tmp_path, monkeypatch, script)
    started = time.time(); result = supervise(args, module.drive)
    assert result['status'] == 'needs-input'
    assert time.time() - started >= .03
    assert (work / 'attempt-count').read_text() == '2'
    assert len(list((work / 'sessions').iterdir())) == 2


def test_external_completion_wakes_without_model_polling(tmp_path, monkeypatch):
    script = "import json,sys,pathlib;json.load(sys.stdin);pathlib.Path('state.json').write_text(json.dumps({'status':'needs-input','open_questions':['fixture question']}))"
    work, module, args = setup(tmp_path, monkeypatch, script)
    state = {'status':'working','model_wait_for':['original-job'],
             'external_jobs':[{'id':'original-job','status':'running'}]}
    save(work / 'state.json', state)
    def complete():
        time.sleep(.1)
        assert not list((work / 'sessions').iterdir())
        state['external_jobs'][0]['status']='complete';save(work / 'state.json',state)
    worker = threading.Thread(target=complete); worker.start()
    assert supervise(args, module.drive)['status'] == 'needs-input'
    worker.join()
    assert len(list((work / 'sessions').iterdir())) == 1


def test_live_worker_recovery_never_resubmits(tmp_path):
    workspace(tmp_path)
    folder = tmp_path / 'sessions/interrupted';folder.mkdir()
    save(folder / 'execution.json', {'status':'running','pid':os.getpid(),
                                    'process_identity':process_identity(os.getpid())})
    assert recover_inflight(tmp_path)['status'] == 'waiting-worker'
    save(folder / 'execution.json', {'status':'starting'})
    assert recover_inflight(tmp_path)['status'] == 'recovery-required'


def test_saved_native_id_recovers_dead_worker(tmp_path):
    workspace(tmp_path)
    folder=tmp_path / 'sessions/interrupted';folder.mkdir()
    native=tmp_path / 'native.json';save(native,{'session_id':'original-thread'})
    save(folder / 'execution.json', {'status':'running','pid':999999999,'native_state':str(native)})
    assert recover_inflight(tmp_path) is None
    assert json.loads((folder / 'execution.json').read_text())['recovery_session_id']=='original-thread'


def test_concurrent_supervisor_refuses_lock(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'raise RuntimeError("must not launch")')
    with (work / '.supervise.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): supervise(args,module.drive)
    assert not list((work / 'sessions').iterdir())


def test_user_stop_terminates_actual_process_and_preserves_jobs(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'import sys,time;sys.stdin.read();time.sleep(30)')
    def stop():
        path=runtime_dir(work)/'supervisor.json'
        while not path.exists() or not json.loads(path.read_text()).get('worker'): time.sleep(.01)
        save(runtime_dir(work)/'stop.json',{'reason':'user'})
    thread=threading.Thread(target=stop);thread.start()
    result=supervise(args,module.drive);thread.join()
    assert result['status']=='stopped'
    record=json.loads(next((work/'sessions').glob('*/execution.json')).read_text())
    assert process_identity(record['pid']) is None
    assert supervise(args,module.drive)['status']=='stopped'
    assert len(list((work/'sessions').iterdir()))==1


def test_fake_complete_refused(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,
        "import json,sys,pathlib;json.load(sys.stdin);pathlib.Path('state.json').write_text(json.dumps({'status':'complete'}))")
    with pytest.raises((ValueError,FileNotFoundError)): supervise(args,module.drive)
    assert not (work/'delivery.json').exists()


def test_actual_independent_review_is_dispatched_before_author(tmp_path, monkeypatch):
    script = '''import json,sys,pathlib,hashlib
json.load(sys.stdin)
folder=pathlib.Path('.runtime/reviews/check')
request=json.loads((folder/'request.json').read_text())
(folder/'report.json').write_text(json.dumps({'verdict':'reject','subjects':request['subjects'],
'evidence':request['subjects'],'findings':['fixture rejection']}))
pathlib.Path('state.json').write_text(json.dumps({'status':'needs-input','open_questions':['fixture question']}))
'''
    work,module,args=setup(tmp_path,monkeypatch,script)
    subject=work/'task/subject.txt';subject.parent.mkdir();subject.write_text('actual evidence')
    from datafarm.common import sha256_file
    req={'id':'check','subjects':[{'path':'task/subject.txt','sha256':sha256_file(subject)}],
         'instructions':'Check the actual evidence.'}
    save(work/'state.json',{'status':'working','review_requests':[req]})
    assert supervise(args,module.drive)['status']=='needs-input'
    completion=json.loads((work/'.runtime/reviews/check/completion.json').read_text())
    assert completion['verdict']=='reject' and completion['role']=='task-verifier'
    record=json.loads(next((work/'sessions').glob('*/execution.json')).read_text())
    assert record['role']=='task-verifier'


def test_enforced_rejects_native_before_launch(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'')
    policy(work,'enforced')
    cfg=json.loads(args.config.read_text());cfg['driver']={'type':'codex'};save(args.config,cfg)
    with pytest.raises(ValueError,match='metered driver'):supervise(args,module.drive)
    assert not list((work/'sessions').iterdir())


def test_exhausted_transient_retries_cannot_reset_on_restart(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,
        "import sys;sys.stdin.read();print('capacity unavailable',file=sys.stderr);sys.exit(1)",
        max_sessions=10,max_retries=1)
    assert supervise(args,module.drive)['status']=='failed'
    count=len(list((work/'sessions').iterdir()))
    assert count==2
    assert supervise(args,module.drive)['status']=='failed'
    assert len(list((work/'sessions').iterdir()))==count


def test_review_runs_when_author_is_waiting_for_input(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'')
    subject=work/'subject';subject.write_text('evidence')
    from datafarm.common import sha256_file
    request={'id':'pending','instructions':'Inspect',
             'subjects':[{'path':'subject','sha256':sha256_file(subject)}]}
    save(work/'state.json',{'status':'needs-input','open_questions':['fixture question'],'review_requests':[request]})
    (work/'worker.py').write_text(IDENTITY_SCRIPT + '''import sys,json,pathlib
json.load(sys.stdin)
f=pathlib.Path('.runtime/reviews/pending')
r=json.loads((f/'request.json').read_text())
(f/'report.json').write_text(json.dumps({'verdict':'unverified','subjects':r['subjects'],'evidence':r['subjects']}))
''')
    assert supervise(args,module.drive)['status']=='needs-input'
    assert (work/'.runtime/reviews/pending/completion.json').exists()


def test_recovered_review_finalizes_without_second_paid_turn(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'')
    from datafarm.common import sha256_file
    subject=work/'subject';subject.write_text('source')
    req={'id':'recovered','instructions':'Inspect','subjects':[{'path':'subject','sha256':sha256_file(subject)}]}
    save(work/'state.json',{'status':'working','review_requests':[req]})
    (work/'worker.py').write_text(IDENTITY_SCRIPT + '''import sys,json,pathlib
json.load(sys.stdin)
f=pathlib.Path('.runtime/reviews/recovered');r=json.loads((f/'request.json').read_text())
(f/'report.json').write_text(json.dumps({'verdict':'reject','subjects':r['subjects'],'evidence':r['subjects']}))
''')
    original=module.finish_review
    monkeypatch.setattr(module,'finish_review',lambda *a: (_ for _ in ()).throw(SystemExit('crash boundary')))
    with pytest.raises(SystemExit):supervise(args,module.drive)
    assert len(list((work/'sessions').iterdir()))==1
    monkeypatch.setattr(module,'finish_review',original)
    assert recover_inflight(work) is None
    assert review_request(work) is None
    assert len(list((work/'sessions').iterdir()))==1
    completion_path=work/'.runtime/reviews/recovered/completion.json'
    completion=json.loads(completion_path.read_text())
    execution_path=Path(completion['session'])/'execution.json'
    execution=json.loads(execution_path.read_text());execution['role']='orchestrator';save(execution_path,execution)
    completion['execution_sha256']=sha256_file(execution_path);save(completion_path,completion)
    with pytest.raises(ValueError,match='provenance'):review_request(work)


def test_crash_after_failure_preserves_retry_boundary(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,
        "import sys;sys.stdin.read();print('capacity',file=sys.stderr);sys.exit(1)",max_retries=0)
    def crash(args):
        module.drive(args)
        raise SystemExit('host crash before persisting result')
    with pytest.raises(SystemExit):supervise(args,crash)
    assert len(list((work/'sessions').iterdir()))==1
    assert supervise(args,module.drive)['status']=='failed'
    assert len(list((work/'sessions').iterdir()))==1


def test_term_resistant_group_is_killed_and_verified(tmp_path):
    import subprocess
    from datafarm.agent.runtime import terminate_group,group_alive
    p=subprocess.Popen([sys.executable,'-c',
        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'],
        stdout=subprocess.PIPE,text=True,start_new_session=True)
    assert p.stdout.readline().strip()=='ready'
    assert group_alive(p.pid)
    assert terminate_group(p.pid,timeout=.05)
    p.wait(timeout=2)
    assert not group_alive(p.pid)


def test_enforced_exact_exhaustion_reports_cost_boundary(tmp_path, monkeypatch):
    work=workspace(tmp_path,limit=1)
    module=driver_module(monkeypatch,work)
    script=work/'spend.py'
    script.write_text('''import os,json,sys,pathlib
json.load(sys.stdin)
pathlib.Path(os.environ['DATAFARM_COST_RECEIPT']).write_text(json.dumps({
'schema':'datafarm.session-cost/1','reservation_id':os.environ['DATAFARM_COST_RESERVATION_ID'],
'includes_descendants':True,'complete':True,'cost_usd':1,'cost_basis':'provider','evidence':'fixture'}))
''')
    assert module.drive(drive_args(work,script,max_sessions=1))['status']=='cost-budget-blocked'


def test_direct_drive_cannot_bypass_supervisor_lock(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'raise RuntimeError("must not run")')
    with (work/'.supervise.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.drive(drive_args(work,work/'worker.py',metered=False))
    assert not list((work/'sessions').iterdir())


def completed_fixture_review(tmp_path, monkeypatch):
    script='''import json,sys,pathlib
json.load(sys.stdin)
p=pathlib.Path('state.json');s=json.loads(p.read_text());r=s['review_requests'][0]
f=pathlib.Path('.runtime/reviews')/r['id']
(f/'report.json').write_text(json.dumps({'verdict':'reject','subjects':r['subjects'],'evidence':r['subjects']}))
s['status']='needs-input';s['open_questions']=['fixture question'];p.write_text(json.dumps(s))
'''
    work,module,args=setup(tmp_path,monkeypatch,script)
    from datafarm.common import sha256_file
    subject=work/'subject';subject.write_text('original source')
    req={'id':'bound','instructions':'Inspect actual source',
         'subjects':[{'path':'subject','sha256':sha256_file(subject)}]}
    save(work/'state.json',{'status':'working','review_requests':[req]})
    assert supervise(args,module.drive)['status']=='needs-input'
    assert review_request(work) is None
    return work


def test_completed_review_rejects_stale_subject(tmp_path, monkeypatch):
    work=completed_fixture_review(tmp_path,monkeypatch)
    (work/'subject').write_text('new unreviewed source')
    with pytest.raises(ValueError,match='subject/evidence changed'):review_request(work)


def test_completion_cannot_upgrade_rejected_report(tmp_path, monkeypatch):
    work=completed_fixture_review(tmp_path,monkeypatch)
    p=work/'.runtime/reviews/bound/completion.json';s=json.loads(p.read_text());s['verdict']='pass';save(p,s)
    with pytest.raises(ValueError,match='verdict differs'):review_request(work)


def test_generic_driver_cannot_reuse_author_identity_for_review(tmp_path):
    from datafarm.agent.runtime import register_identity
    from datafarm.common import sha256_file
    workspace(tmp_path)
    for name,key,role in [('first','author','orchestrator'),('second','review-check','task-verifier')]:
        folder=tmp_path/'sessions'/name;folder.mkdir();proof=folder/'provider.txt';proof.write_text('same-provider-session')
        save(folder/'identity.json',{'schema':'datafarm.agent-identity/1','session_id':'same-provider-session',
            'session_key':key,'role':role,'fresh':True,
            'evidence':[{'path':str(proof.relative_to(tmp_path)),'sha256':sha256_file(proof)}]})
        if role=='orchestrator':register_identity(tmp_path,folder,key,role)
        else:
            with pytest.raises(ValueError,match='reused author/reviewer'):register_identity(tmp_path,folder,key,role)


def test_argv_only_review_is_refused_before_model_launch(tmp_path, monkeypatch):
    work,module,args=setup(tmp_path,monkeypatch,'raise RuntimeError("must not run")')
    cfg=json.loads(args.config.read_text());cfg['driver'].pop('session_identity_protocol');save(args.config,cfg)
    subject=work/'subject';subject.write_text('source')
    from datafarm.common import sha256_file
    save(work/'state.json',{'status':'working','review_requests':[{'id':'check','instructions':'Inspect',
        'subjects':[{'path':'subject','sha256':sha256_file(subject)}]}]})
    with pytest.raises(ValueError,match='cannot prove independence'):supervise(args,module.drive)
    assert not list((work/'sessions').iterdir())


def test_native_first_turn_crash_retains_provable_session_origin(tmp_path):
    from datafarm.agent.runtime import register_identity
    workspace(tmp_path)
    origin=tmp_path/'sessions/first';origin.mkdir()
    save(origin/'native-command.json',{'resume':None})
    save(origin/'execution.json',{'role':'orchestrator','status':'interrupted'})
    (origin/'stdout.log').write_text(json.dumps({'type':'thread.started','thread_id':'original-thread'})+'\n')
    resumed=tmp_path/'sessions/resumed';resumed.mkdir()
    save(resumed/'native-command.json',{'resume':'original-thread'})
    save(resumed/'native-result.json',{'session_id':'original-thread'})
    (resumed/'stdout.log').write_text('resumed transcript')
    identity=register_identity(tmp_path,resumed,'author','orchestrator',native=True)
    assert identity['origin']['path']=='sessions/first/stdout.log'


def test_native_origin_identity_is_stable_on_revalidation(tmp_path):
    from datafarm.agent.runtime import register_identity
    workspace(tmp_path)
    origin=tmp_path/'sessions/first';origin.mkdir()
    save(origin/'native-command.json',{'resume':None})
    save(origin/'execution.json',{'role':'orchestrator'})
    (origin/'stdout.log').write_text(json.dumps({'type':'thread.started','thread_id':'thread'})+'\n')
    resumed=tmp_path/'sessions/second';resumed.mkdir()
    save(resumed/'native-command.json',{'resume':'thread'})
    save(resumed/'native-result.json',{'session_id':'thread'})
    (resumed/'stdout.log').write_text('resumed')
    register_identity(tmp_path,resumed,'author','orchestrator',native=True)
    before=(resumed/'identity.json').read_bytes()
    register_identity(tmp_path,resumed,'author','orchestrator',native=True)
    assert before==(resumed/'identity.json').read_bytes()


def test_native_identity_accepts_own_redirected_log(tmp_path):
    from datafarm.agent.runtime import register_identity
    workspace(tmp_path)
    session=tmp_path/'sessions/one';session.mkdir()
    logs=tmp_path/'logs';logs.mkdir();(logs/'stdout.jsonl').write_text('actual native stream')
    (session/'stdout.log').symlink_to(logs/'stdout.jsonl')
    save(session/'native-command.json',{'resume':None})
    save(session/'native-result.json',{'session_id':'new-thread'})
    assert register_identity(tmp_path,session,'author','orchestrator',native=True)['session_id']=='new-thread'


def test_native_origin_changes_are_rejected_on_every_revalidation(tmp_path):
    from datafarm.agent.runtime import register_identity
    workspace(tmp_path)
    origin=tmp_path/'sessions/first';origin.mkdir()
    save(origin/'native-command.json',{'resume':None})
    save(origin/'execution.json',{'role':'orchestrator'})
    (origin/'stdout.log').write_text(json.dumps({'type':'thread.started','thread_id':'thread'})+'\n')
    resumed=tmp_path/'sessions/second';resumed.mkdir()
    save(resumed/'native-command.json',{'resume':'thread'})
    save(resumed/'native-result.json',{'session_id':'thread'})
    (resumed/'stdout.log').write_text('resumed')
    register_identity(tmp_path,resumed,'author','orchestrator',native=True)
    (origin/'stdout.log').write_text('stale origin proof')
    with pytest.raises(ValueError,match='origin changed'):register_identity(tmp_path,resumed,'author','orchestrator',native=True)


@pytest.mark.parametrize('questions', [None, [], [' '], [{'kind': 'intent'}]])
def test_needs_input_must_ask_concrete_questions(tmp_path, monkeypatch, questions):
    state = {'status': 'needs-input'}
    if questions is not None:
        state['open_questions'] = questions
    script = ('import json,sys,pathlib;json.load(sys.stdin);'
              f'pathlib.Path("state.json").write_text({json.dumps(json.dumps(state))})')
    work, module, _ = setup(tmp_path, monkeypatch, script)
    with pytest.raises(ValueError, match='concrete open_questions'):
        module.drive(drive_args(work, work / 'worker.py', metered=False, max_sessions=1))


def test_advisory_budget_refuses_unbudgeted_post_run_evolution(tmp_path, monkeypatch):
    work, module, _ = setup(tmp_path, monkeypatch, '')
    args = drive_args(work, work / 'worker.py', metered=False)
    args.evolution_driver = work / 'driver.json'
    with pytest.raises(ValueError, match='unbudgeted evolution driver'):
        module.drive(args)
    assert not list((work / 'sessions').iterdir())


def test_inflight_execution_from_another_host_is_not_resumed(tmp_path):
    session = tmp_path / "sessions/1"; session.mkdir(parents=True)
    save(session / "execution.json", {"status": "running", "pid": os.getpid(),
                                      "process_identity": "x", "host": "another-node"})
    result = recover_inflight(tmp_path)
    assert result["status"] == "recovery-required" and result["host"] == "another-node"
    assert json.loads((session / "execution.json").read_text())["status"] == "running"
