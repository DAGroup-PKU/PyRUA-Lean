"""Real subprocess/paired-evaluator tests; agents and media are local fixtures."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from datafarm.evolution.store import Store, read, write
from datafarm.evolution.events import record, select, pending
from datafarm.evolution.versions import initialize, active, resolve, rollback, snapshot
from datafarm.evolution.experiments import freeze_policy, compare, run
from datafarm.evolution.controller import propose, assess, cycle
from datafarm.evolution.integration import bind, context_for, feedback

ROOT = Path(__file__).resolve().parents[1]


def event(store, **overrides):
    return record(store, {'kind': 'correction', 'statement': 'Handle an empty input without losing other cases.',
                         'source': {'actor': 'fixture-user', 'ref': 'message-1'}, 'scope': {}, **overrides})


def fixture(tmp_path):
    root = tmp_path / 'original source'; root.mkdir()
    (root / 'solver.py').write_text('def solve(x):\n    return x[0]\n')
    store = Store(tmp_path / 'store')
    parent = initialize(store, root)
    e = event(store)
    suite = tmp_path / 'trusted suite'; suite.mkdir()
    # This evaluator and its case list live outside either agent's source.
    (suite / 'check.py').write_text('''import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); report=pathlib.Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('solver',root/'solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cases=[('empty',[],None),('positive',[5],5),('negative',[-3],-3)]
good=0
for _,x,want in cases:
 try: good+=m.solve(x)==want
 except (IndexError,TypeError): pass
report.write_text(json.dumps({'passed':good==3,'case_ids':[x[0] for x in cases],'metrics':{'correct':good}}))
''')
    policy = {'schema': 'datafarm.evolution-policy/1', 'evidence_level': 'software',
              'allowed_paths': ['solver.py'], 'assets': ['check.py'],
              'checks': [{'id': 'replay', 'argv': [sys.executable, '{suite}/check.py', '{repo}', '{report}'], 'timeout': 10}],
              'metrics': {'replay.correct': 'max'}}
    write(suite / 'policy.json', policy)
    policy_id = freeze_policy(store, suite / 'policy.json')
    return root, store, parent, e, suite, policy_id


def candidate(root, store, parent, e, policy_id, code=None, lessons=None):
    work = root.parent / ('candidate-' + str(len(store.all('candidates'))))
    shutil.copytree(root, work)
    if code is not None:
        (work / 'solver.py').write_text(code)
    return propose(store, parent['id'], work,
                   {'hypothesis': 'Missing empty-input handling causes the error.',
                    'expected_effect': 'Empty case fixed and nonempty results unchanged.',
                    'alternative_explanation': 'Caller could be passing an incorrect type.',
                    'lessons': lessons or []}, [e['id']], policy_id)


FIX = 'def solve(x):\n    return x[0] if x else None\n'


def test_feedback_idempotent_scoped_and_superseded(tmp_path):
    store = Store(tmp_path)
    first = event(store, kind='user-requirement', scope={'hardware': 'a'})
    assert event(store, kind='user-requirement', scope={'hardware': 'a'}) == first
    assert len(store.all('events')) == 1
    with pytest.raises(ValueError, match='conflicting'):
        event(store, id=first['id'], statement='different')
    assert not select(store, {'hardware': 'b'})['requirements']
    second = event(store, kind='user-requirement', scope={'hardware': 'a'},
                   statement='Updated requirement', supersedes=[first['id']])
    assert select(store, {'hardware': 'a'})['requirements'] == [second]
    with pytest.raises(ValueError, match='scope'):
        event(store, supersedes=[first['id']])
    with pytest.raises(ValueError): event(store, scope={'task_id': '*' , 'unknown': 'x'})


def test_snapshot_source_archive_and_mutation(tmp_path):
    root, store, parent, *_ = fixture(tmp_path)
    pinned = resolve(store, parent['id'])
    (root / 'solver.py').write_text(FIX)
    assert 'if x' not in (pinned / 'solver.py').read_text()
    assert snapshot(store, pinned) == parent['source']
    (pinned / 'solver.py').write_text(FIX)
    with pytest.raises(ValueError, match='pinned source changed'): resolve(store, parent['id'])


def test_paired_improvement_promotion_rollback_and_pin(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    work = tmp_path / 'task'; work.mkdir()
    pin = bind(work, root, store_path=store.root, selectors={'hardware': 'a'})
    c = candidate(root, store, parent, e, policy, FIX)
    result = assess(store, c['id'], timeout=30)
    assert result['status'] == 'promoted'
    assert result['evidence_level'] == 'software'
    assert active(store) == result['version'] != parent['id']
    assert read(work / 'capability.json')['version'] == pin['version'] == parent['id']
    measured = store.get('evaluations', result['evaluation'])
    assert measured['baseline']['metrics']['replay.correct'] == 2
    assert measured['candidate_result']['metrics']['replay.correct'] == 3
    assert assess(store, c['id']) == result  # no duplicate expensive evaluation
    rollback(store, parent['id'])
    assert active(store) == parent['id']
    assert len(store.all('evaluations')) == 1


def test_failure_retained_and_no_self_certification(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    c = candidate(root, store, parent, e, policy, 'def solve(x):\n    return None\n')
    result = assess(store, c['id'], timeout=30)
    assert result['status'] == 'rejected' and active(store) == parent['id']
    assert not pending(store)  # no automatic repeated spending on same failure
    assert store.get('evaluations', result['evaluation'])['candidate_result']['metrics']['replay.correct'] == 1
    with pytest.raises(ValueError, match='never active'):
        rollback(store, 'invented')


def test_policy_and_candidate_tampering_rejected(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    c = candidate(root, store, parent, e, policy, FIX)
    (store.root / 'policies' / policy / 'check.py').write_text('print("pass")')
    with pytest.raises(ValueError, match='evaluator changed'): assess(store, c['id'])
    assert active(store) == parent['id']


def test_stale_candidate_cannot_replace_active_version(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    first = candidate(root, store, parent, e, policy, FIX)
    second = candidate(root, store, parent, e, policy, FIX + '# alternative\n')
    assert assess(store, first['id'], timeout=30)['status'] == 'promoted'
    assert assess(store, second['id'], timeout=30)['status'] == 'stale'


def test_lessons_need_explicit_behavior_evaluation(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    lesson = {'problem': 'Empty data', 'action': 'Check emptiness', 'rationale': 'Input may be empty',
              'limits': 'Not a policy for missing real observations', 'scope': {'hardware': 'a'}}
    c = candidate(root, store, parent, e, policy, FIX, [lesson])
    result = assess(store, c['id'], timeout=30)
    assert result['status'] == 'promoted'
    assert store.all('lessons')[0]['status'] == 'hypothesis'
    assert not select(store, {'hardware': 'a'}, version=active(store))['lessons']


def test_current_requirement_enters_pinned_task_context(tmp_path):
    root, store, parent, *_ = fixture(tmp_path)
    work = tmp_path / 'task'; work.mkdir()
    bind(work, root, store_path=store.root, selectors={'hardware': 'a'})
    event_value = feedback(work, {'kind': 'user-requirement', 'statement': 'Keep new requirement',
                                 'source': {'actor': 'user', 'ref': 'message-2'}})
    ctx = context_for(work)
    assert ctx['requirements'] == [event_value] and ctx['version'] == parent['id']


def test_scope_protects_normative_files_and_escape_paths(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    (root / 'AGENTS.md').write_text('relax rules')
    with pytest.raises(ValueError, match='outside optimization scope'):
        candidate(root, store, parent, e, policy, FIX)
    with pytest.raises(ValueError): store.get('events', '../escape')
    unsafe = root / 'nested-store'
    with pytest.raises(ValueError, match='excluded runtime'):
        snapshot(Store(unsafe), root)


@pytest.mark.parametrize('path', ['tools/issue_admission_certificate.py',
                                  'src/datafarm/expert/holdout.py', 'evaluations/x/check.py',
                                  'src/datafarm/evolution/experiments.py', 'src/json.py', 'src/numpy/__init__.py',
                                  'src/sitecustomize.py', 'src/datafarm_x/usercustomize.py', 'src/datafarm_x/hook.pth'])
def test_scope_protects_evaluation_roots_even_when_parent_prefix_allowed(tmp_path, path):
    root, store, parent, e, suite, _ = fixture(tmp_path)
    policy = read(suite / 'policy.json'); policy['allowed_paths'] = ['solver.py', 'tools/', 'src/', 'evaluations/']
    write(suite / 'policy.json', policy)
    wide = freeze_policy(store, suite / 'policy.json')
    work = tmp_path / 'hack'; shutil.copytree(root, work)
    (work / path).parent.mkdir(parents=True, exist_ok=True); (work / path).write_text('judge = lambda *_: True\n')
    with pytest.raises(ValueError, match='outside optimization scope'):
        propose(store, parent['id'], work, {'hypothesis': 'h', 'expected_effect': 'e',
                'alternative_explanation': 'a'}, [e['id']], wide)


def test_policy_protected_paths_extend_scope_and_are_rechecked(tmp_path):
    root, store, parent, e, suite, _ = fixture(tmp_path)
    policy = read(suite / 'policy.json'); policy['allowed_paths'] = ['solver.py', 'bench/']
    policy['protected_paths'] = ['bench/judge.py']
    write(suite / 'policy.json', policy)
    scoped = freeze_policy(store, suite / 'policy.json')
    work = tmp_path / 'hack'; shutil.copytree(root, work)
    (work / 'bench').mkdir(); (work / 'bench/judge.py').write_text('ok = True\n')
    with pytest.raises(ValueError, match='outside optimization scope'):
        propose(store, parent['id'], work, {'hypothesis': 'h', 'expected_effect': 'e',
                'alternative_explanation': 'a'}, [e['id']], scoped)
    policy['protected_paths'] = ['/abs']; write(suite / 'policy.json', policy)
    with pytest.raises(ValueError, match='protected_paths'):
        freeze_policy(store, suite / 'policy.json')
    # A candidate record that bypassed propose is rejected when assessed, before any evaluation.
    allowed_work = tmp_path / 'ok'; shutil.copytree(root, allowed_work)
    (allowed_work / 'solver.py').write_text(FIX)
    c = propose(store, parent['id'], allowed_work, {'hypothesis': 'h', 'expected_effect': 'e',
                'alternative_explanation': 'a'}, [e['id']], scoped)
    (allowed_work / 'bench').mkdir(); (allowed_work / 'bench/judge.py').write_text('ok = True\n')
    forged = {k: v for k, v in c.items() if k != 'id'}
    forged['source'] = snapshot(store, allowed_work)
    from datafarm.evolution.store import digest
    forged['id'] = digest(forged); store.put('candidates', forged['id'], forged)
    store.transition(forged['id'], 'proposed')
    result = assess(store, forged['id'], timeout=30)
    assert result['status'] == 'rejected' and 'bench/judge.py' in result['reason']
    assert not (store.root / 'evaluations' / forged['id']).exists()


def test_noise_aware_rules_need_minimum_gain_and_allow_bounded_regression():
    policy = {'metrics': {'a.success': {'direction': 'max', 'min_improvement': 0.10},
                          'a.tokens': {'direction': 'min', 'tolerance': 0.10, 'relative': True, 'decisive': False}}}
    b = {'valid': True, 'passed': True, 'case_ids': {'a': ['x']}, 'metrics': {'a.success': 0.60, 'a.tokens': 100.0}}
    def cand(success, tokens):
        return {**b, 'metrics': {'a.success': success, 'a.tokens': tokens}}
    assert compare(policy, b, cand(0.75, 108.0))[0]                     # +15 pt, tokens within 10 %
    assert not compare(policy, b, cand(0.65, 90.0))[0]                  # +5 pt is noise; cheaper alone is not decisive
    assert compare(policy, b, cand(0.75, 115.0)) == (False, 'metric regressed: a.tokens')
    assert compare(policy, b, cand(0.50, 80.0)) == (False, 'metric regressed: a.success')
    with pytest.raises(ValueError):
        from datafarm.evolution.experiments import metric_rule
        metric_rule({'direction': 'max', 'tolerance': -1})


@pytest.mark.parametrize('change', ['denominator', 'regression', 'missing', 'no-improvement'])
def test_comparison_rejects_invalid_claims(change):
    policy = {'metrics': {'a.score': 'max', 'a.cost': 'min'}}
    b = {'valid': True, 'passed': True, 'case_ids': {'a': ['x']}, 'metrics': {'a.score': 1, 'a.cost': 10}}
    c = json.loads(json.dumps(b)); c['metrics']['a.score'] = 2
    if change == 'denominator': c['case_ids']['a'] = ['easy']
    elif change == 'regression': c['metrics']['a.cost'] = 11
    elif change == 'missing': del c['metrics']['a.cost']
    else: c['metrics']['a.score'] = 1
    assert not compare(policy, b, c)[0]


def test_timeout_preserves_execution(tmp_path):
    result = run([sys.executable, '-c', 'import time;time.sleep(10)'], cwd=tmp_path,
                 directory=tmp_path / 'execution', timeout=.05)
    assert result['status'] == 'timeout'
    assert read(tmp_path / 'execution/execution.json')['exit_code'] is not None


def test_frozen_evaluator_case_change_fails(tmp_path):
    root, store, parent, e, suite, policy = fixture(tmp_path)
    c = candidate(root, store, parent, e, policy, FIX)
    # Changes to the caller's policy assets do not change frozen execution.
    (suite / 'check.py').write_text('raise RuntimeError("changed external file")')
    assert assess(store, c['id'], timeout=30)['status'] == 'promoted'


def test_actual_driver_cycle_with_local_script_fixture(tmp_path):
    root, store, parent, e, suite, policy = fixture(tmp_path)
    # Minimal real instruction loader assets.
    for name in ('configs', 'prompts', '.agents', 'docs'):
        shutil.copytree(ROOT / name, root / name)
    shutil.copyfile(ROOT / 'AGENTS.md', root / 'AGENTS.md')
    store = Store(tmp_path / 'cycle-store')
    initialize(store, root)
    event(store)
    driver = tmp_path / 'local-test-agent.py'
    driver.write_text('''import json,pathlib,sys
messages=json.load(sys.stdin)
task=json.loads(messages[1]['content'])
assert 'pipeline-improve' in messages[0]['content']
pathlib.Path('solver.py').write_text('def solve(x):\\n    return x[0] if x else None\\n')
pathlib.Path(task['proposal_output']).write_text(json.dumps({'hypothesis':'Empty input is not handled',
'expected_effect':'Three cases pass','alternative_explanation':'Incorrect caller type'}))
''')
    result = cycle(store, root=root, driver={'argv': [sys.executable, str(driver)]},
                   policy_path=suite / 'policy.json', timeout=30)
    assert result['results'][0]['status'] == 'promoted'
    assert len(store.all('evaluations')) == 1
    assert len(list((store.root / 'cycles').glob('*/0/context/receipt.json'))) == 1


def test_shadow_promotion_reuses_measured_evidence(tmp_path):
    root, store, parent, e, _, policy = fixture(tmp_path)
    c = candidate(root, store, parent, e, policy, FIX)
    first = assess(store, c['id'], timeout=30, promote=False)
    assert first['status'] == 'verified' and active(store) == parent['id']
    second = assess(store, c['id'], timeout=30)
    assert second['status'] == 'promoted' and len(store.all('evaluations')) == 1
    assert first['evaluation'] == second['evaluation']


def test_activation_crash_recovers_without_duplicate_evaluation(tmp_path, monkeypatch):
    root, store, parent, e, _, policy = fixture(tmp_path)
    c = candidate(root, store, parent, e, policy, FIX)
    original = store.transition
    def interrupt(name, status, **details):
        if status == 'promoted': raise RuntimeError('simulated interruption after activation')
        return original(name, status, **details)
    monkeypatch.setattr(store, 'transition', interrupt)
    with pytest.raises(RuntimeError): assess(store, c['id'], timeout=30)
    assert active(store) != parent['id']
    monkeypatch.setattr(store, 'transition', original)
    assert assess(store, c['id'])['recovered']
    assert len(store.all('evaluations')) == 1


def test_lessons_explicitly_evaluated_then_scoped_in_context(tmp_path):
    root, store, parent, e, suite, _ = fixture(tmp_path)
    check = suite / 'check.py'
    # Small deterministic tool-agent fixture uses a scoped strategy as input.
    check.write_text(check.read_text().replace('root=pathlib.Path',
        "experience=json.loads(pathlib.Path(sys.argv[3]).read_text())\nroot=pathlib.Path").replace(
        'good+=m.solve(x)==want',
        "good+=(None if not x and any(l['action']=='guard-empty' for l in experience['lessons']) else m.solve(x))==want"))
    policy = read(suite / 'policy.json'); policy['tests_lessons'] = True
    policy['checks'][0]['argv'].append('{experience}')
    write(suite / 'policy.json', policy)
    policy_id = freeze_policy(store, suite / 'policy.json')
    lesson = {'problem': 'Empty list', 'action': 'guard-empty', 'rationale': 'Empty has no first item',
              'limits': 'Only this test tool-agent domain', 'scope': {'hardware': 'a'}}
    c = candidate(root, store, parent, e, policy_id, lessons=[lesson])
    assert assess(store, c['id'], timeout=30)['status'] == 'promoted'
    assert len(select(store, {'hardware': 'a'}, version=active(store))['lessons']) == 1
    assert not select(store, {'hardware': 'b'}, version=active(store))['lessons']


def test_manual_repair_records_actual_diff_and_check_evidence(tmp_path):
    from datafarm.evolution.events import record_change
    root, store, parent, *_ = fixture(tmp_path)
    (root / 'solver.py').write_text(FIX)
    output = tmp_path / 'check-output.txt'; output.write_text('3 cases passed')
    event = record_change(store, root, parent['id'], statement='Manual repair after empty-input failure',
                          source_ref='session/manual-fix', evidence=[output])
    assert event['kind'] == 'code-fix' and event['details']['changed_paths'] == ['solver.py']
    assert event['details']['evidence'][0]['sha256']
    assert active(store) == parent['id']  # observing a fix is not self-certifying it


def test_source_archive_rejects_directory_symlink(tmp_path):
    root, store, *_ = fixture(tmp_path)
    external = tmp_path / 'external'; external.mkdir()
    (external / 'hidden.py').write_text('VALUE = 2')
    (root / 'imported').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='directory symlinks'): snapshot(store, root)
