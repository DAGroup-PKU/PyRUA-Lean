"""Cold context, unknown task, continuation and false-completion behavior."""
import base64
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from render_policy_fixtures import policy_fixture

from datafarm.agent.context import build_context, write_context
from datafarm.render.policy import check_seedance_payload, compile_prompt
from datafarm.common import sha256_file
from datafarm.task.contract import verify_delivery

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/pipeline_agent.py'


def cli(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True,
                          text=True, check=False)


def workspace(tmp_path, *, budgeted=False):
    ep = tmp_path / 'never-seen-scene.mp4'; ep.write_bytes(b'test-only episode')
    request = tmp_path / 'request.txt'; request.write_text('Adapt this never-seen task and generate one episode. sim2real.')
    work = tmp_path / 'new-work'
    result = cli('prepare', '--episode', ep, '--request-file', request, '--output', work, '--target-count', '1',
                 '--evolution-store', tmp_path / 'evolution-store')
    assert result.returncode == 0, result.stderr
    # These existing driver tests also cover resume compatibility with workspaces
    # prepared before cost metering. New budgeted drivers are tested separately.
    if not budgeted:
        (work / 'cost-policy.json').unlink()
        (work / 'cost-ledger.json').unlink()
        lock = json.loads((work / 'input.json').read_text())
        lock.pop('cost_policy_sha256')
        (work / 'input.json').write_text(json.dumps(lock))
    return work


def test_new_workspace_has_whole_task_cost_policy(tmp_path):
    work = workspace(tmp_path, budgeted=True)
    policy = json.loads((work / 'cost-policy.json').read_text())
    assert policy['target_usd'] == 30 and policy['limit_usd'] == 50
    result = cli('cost-status', '--work', work)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['remaining_usd'] == 50


def ref(work, name, content):
    p = work / name; p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(content) if not isinstance(content, str) else content)
    return {'path': name, 'sha256': sha256_file(p)}


def visual_ref(work, name):
    # A decoded PNG fixture is enough to exercise the binding contract; this is
    # software evidence and makes no claim about real visual quality.
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
    path = work / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(png)
    return {'path': name, 'sha256': sha256_file(path)}


def delivery(work):
    artifacts = {k: ref(work, f'evidence/{k}.txt', f'fixture {k}')
                 for k in ['action', 'source', 'video', 'prompt', 'references']}
    policy_path, body, intent, overlay = policy_fixture(work)
    intent['real_references'][0]['provenance']['source'] = json.loads((work / 'input.json').read_text())['files'][0]
    compiled = compile_prompt(work, intent, overlay, assets=ROOT)
    policy_path.write_text(json.dumps(compiled))
    body['content'][0]['text'] = compiled['prompt']
    policy_ref = {'path': policy_path.name, 'sha256': sha256_file(policy_path)}
    submission_ref = ref(work, 'submission.json', check_seedance_payload(body, policy_path))
    artifacts['prompt'] = ref(work, 'prompt.txt', compiled['prompt'])
    artifacts['references'] = ref(work, 'references.json', intent['real_references'])
    transcript = ref(work, 'review-transcript.txt' , 'synthetic independent review execution')
    comparison = visual_ref(work, 'evidence/real-vs-generated.png')
    execution = ref(work, 'review-execution.json', {'role': 'reviewer', 'actor_id': 'review-1',
                    'producer_id': 'producer-1', 'transcript': transcript})
    coverage = {name: {'verdict': 'pass', 'comparison_basis': 'real-reference-vs-generated',
                       'scope': 'full-view',
                       'views': intent['appearance_views'] if name == 'declared_view_photography' else ['head'],
                       'generated_frames': [0],
                       'real_reference_sha256s': [intent['real_references'][0]['sha256']],
                       'evidence': [comparison]}
                for name in ['reference_baseline', 'robot_body_and_attached_hardware', 'task_objects',
                             'table_and_background', 'declared_view_photography', 'overall_photographic_realism']}
    review = {'schema': 'datafarm.task-review/2', 'producer_id': 'producer-1', 'reviewer_id': 'review-1',
        'execution': execution, 'subjects': {k: v['sha256'] for k, v in artifacts.items()},
        'verdicts': {k: 'pass' for k in ['task', 'motion', 'reference', 'style', 'alignment', 'joint_margin']},
        'evidence': [transcript, comparison],
        'visual_review': {'inspection_method': 'decoded-real-reference-and-generated-video',
                          'generated_video_sha256': artifacts['video']['sha256'],
                          'real_reference_sha256s': [intent['real_references'][0]['sha256']],
                          'coverage': coverage}}
    datafile = ref(work, 'dataset/data.bin', 'synthetic data')
    manifest = ref(work, 'dataset.json', {'schema': 'datafarm.dataset-files/1', 'episode_ids': ['new-001'], 'files': [datafile]})
    d = {'schema': 'datafarm.agent-delivery/1', 'producer_id': 'producer-1', 'target_count': 1,
         'input_sha256': json.loads((work / 'input.json').read_text())['sha256'], 'dataset': manifest,
         'attempts': ref(work, 'attempts.json', [{'id': 'failed-0', 'outcome': 'rejected'}, {'id': 'new-001', 'outcome': 'admitted'}]),
         'episodes': [{'id': 'new-001', 'artifacts': artifacts, 'render_contract': policy_ref, 'submission': submission_ref, 'review': ref(work, 'review.json', review)}],
         'grounding': ref(work, 'grounding.json', {'schema': 'datafarm.grounding/1', 'physical_quantities': [
             {'name': 'cloth_friction', 'label': 'assumed', 'unit': '', 'value': 0.6,
              'randomization': {'low': 0.4, 'high': 0.9}}]})}
    (work / 'delivery.json').write_text(json.dumps(d))
    return d, review


def test_all_roles_explicitly_load_without_historical_task_rules(tmp_path):
    roles = json.loads((ROOT / 'configs/agent-instructions.json').read_text())['roles']
    for role in roles:
        c = write_context(tmp_path / role, role)
        assert c['messages'][0]['role'] == 'system'
        assert any(x['path'] == f'prompts/roles/{role}.md' for x in c['receipt']['files'])
        # The checkout's own path may be anywhere, including under /home in CI.
        assert not re.search(r'/home/|/mnt/', c['messages'][0]['content'].replace(str(ROOT), ''))
        assert not c['receipt']['task_products_auto_loaded']


def test_unknown_task_and_input_mutation(tmp_path):
    work = workspace(tmp_path)
    assert not (work / 'task/adapter.json').exists()  # generated by agent, never prerequisite
    result = cli('context', '--work', work, '--output', tmp_path / 'context')
    assert result.returncode == 0, result.stderr
    (tmp_path / 'never-seen-scene.mp4').write_bytes(b'changed real input')
    result = cli('context', '--work', work, '--output', tmp_path / 'context2')
    assert result.returncode == 2 and 'changed' in result.stderr


def test_context_missing_asset_fails_instead_of_silently_omitting(tmp_path):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/agent-instructions.json').write_text((ROOT / 'configs/agent-instructions.json').read_text())
    with pytest.raises(FileNotFoundError): build_context('orchestrator', root=tmp_path)


def test_delivery_binds_real_files_and_independent_evidence(tmp_path):
    work = workspace(tmp_path); d, review = delivery(work)
    assert verify_delivery(work)['attempts'] == 2
    for field, value in [('reviewer_id', 'producer-1'), ('verdicts', {**review['verdicts'], 'style': 'unverified'}),
                         ('subjects', {**review['subjects'], 'video': '0'*64}), ('evidence', [])]:
        changed = copy.deepcopy(review); changed[field] = value
        d['episodes'][0]['review'] = ref(work, 'review.json', changed)
        (work / 'delivery.json').write_text(json.dumps(d))
        with pytest.raises(ValueError): verify_delivery(work)
    delivery(work)
    (work / 'dataset/data.bin').write_text('changed data')
    with pytest.raises(ValueError, match='changed'): verify_delivery(work)


def test_local_texture_cannot_claim_full_scene_real_appearance(tmp_path):
    work = workspace(tmp_path); d, review = delivery(work)
    changed = copy.deepcopy(review)
    for name in ['reference_baseline', 'table_and_background', 'declared_view_photography',
                 'overall_photographic_realism']:
        changed['visual_review']['coverage'][name]['scope'] = 'local-detail'
    d['episodes'][0]['review'] = ref(work, 'review.json', changed)
    (work / 'delivery.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='local detail cannot establish full-scene appearance'):
        verify_delivery(work)


def test_text_only_review_without_real_visual_comparison_cannot_pass_style(tmp_path):
    work = workspace(tmp_path); d, review = delivery(work)
    changed = copy.deepcopy(review)
    text_only = ref(work, 'evidence/claimed-comparison.txt', 'right wrist texture resembles reference')
    changed['evidence'].append(text_only)
    for item in changed['visual_review']['coverage'].values():
        item['evidence'] = [text_only]
    d['episodes'][0]['review'] = ref(work, 'review.json', changed)
    (work / 'delivery.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='text-only/non-image evidence'):
        verify_delivery(work)


def test_required_scene_component_cannot_be_waived_as_not_applicable(tmp_path):
    work = workspace(tmp_path); d, review = delivery(work)
    changed = copy.deepcopy(review)
    objects = changed['visual_review']['coverage']['task_objects']
    objects.update(verdict='not_applicable', rationale='claimed absent without changing task intent')
    d['episodes'][0]['review'] = ref(work, 'review.json', changed)
    (work / 'delivery.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='required appearance coverage cannot be not-applicable'):
        verify_delivery(work)


def test_drive_receives_instructions_and_continues_then_rejects_fake_done(tmp_path):
    work = workspace(tmp_path)
    script = tmp_path / 'driver.py'
    script.write_text('''import json, pathlib, sys
m=json.load(sys.stdin)
assert m[0]['role']=='system' and 'new-task-adaptation' in m[0]['content']
p=pathlib.Path('state.json'); s=json.loads(p.read_text()); n=s.get('test_sessions',0)+1
s.update(test_sessions=n, phase='generate-new-adapter', status='complete' if n==2 else 'working')
p.write_text(json.dumps(s))
''')
    config = tmp_path / 'driver.json'; config.write_text(json.dumps({'argv': [sys.executable, str(script)], 'input': 'messages'}))
    result = cli('drive', '--work', work, '--driver', config, '--max-sessions', 2)
    assert result.returncode == 2 and 'delivery.json' in result.stderr
    assert len(list((work / 'sessions').glob('*/receipt.json'))) == 2
    assert json.loads((work / 'state.json').read_text())['test_sessions'] == 2


def test_drive_honors_timeout_and_input_pause(tmp_path):
    work = workspace(tmp_path)
    script = tmp_path / 'slow.py'; script.write_text('import time; time.sleep(10)')
    config = tmp_path / 'driver.json'; config.write_text(json.dumps({'argv': [sys.executable, str(script)]}))
    result = cli('drive', '--work', work, '--driver', config, '--timeout', '.05')
    assert result.returncode == 75 and json.loads(result.stdout)['status'] == 'timeout'
    state = json.loads((work / 'state.json').read_text()); state.update(status='needs-input', open_questions=['sim2real or OOD?'])
    (work / 'state.json').write_text(json.dumps(state))
    result = cli('drive', '--work', work, '--driver', config)
    assert result.returncode == 75 and 'sim2real or OOD?' in result.stdout
    assert len(list((work / 'sessions').glob('*/receipt.json'))) == 1


def test_delivery_cannot_shrink_target_or_reuse_an_old_submission(tmp_path):
    work = workspace(tmp_path); d, _ = delivery(work)
    d['target_count'] = 2
    (work / 'delivery.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='target count changed'): verify_delivery(work)
    d, _ = delivery(work)
    submission = json.loads((work / 'submission.json').read_text()); submission['intent_sha256'] = 'f'*64
    d['episodes'][0]['submission'] = ref(work, 'submission.json', submission)
    (work / 'delivery.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='stale submission'): verify_delivery(work)


def test_successful_driver_exit_requires_a_verified_delivery(tmp_path):
    work = workspace(tmp_path); delivery(work)
    script = tmp_path / 'finish.py'
    script.write_text("import json,pathlib,sys; json.load(sys.stdin); p=pathlib.Path('state.json'); s=json.loads(p.read_text()); s['status']='complete'; p.write_text(json.dumps(s))")
    config = tmp_path / 'driver.json'; config.write_text(json.dumps({'argv': [sys.executable, str(script)]}))
    result = cli('drive', '--work', work, '--driver', config)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['valid'] is True


def test_feedback_loaded_and_optional_evolution_error_preserves_delivery(tmp_path):
    work = workspace(tmp_path); delivery(work)
    event = tmp_path / 'feedback.json'
    event.write_text(json.dumps({'kind': 'user-requirement', 'statement': 'Fixture current user clarification',
                               'source': {'actor': 'fixture-user', 'ref': 'message-2'}}))
    assert cli('feedback', '--work', work, '--file', event).returncode == 0
    script = tmp_path / 'finish-with-feedback.py'
    script.write_text("import json,pathlib,sys; m=json.load(sys.stdin); assert 'Fixture current user clarification' in m[1]['content']; p=pathlib.Path('state.json'); s=json.loads(p.read_text()); s['status']='complete'; p.write_text(json.dumps(s))")
    config = tmp_path / 'driver.json'; config.write_text(json.dumps({'argv': [sys.executable, str(script)]}))
    result = cli('drive', '--work', work, '--driver', config, '--evolution-driver', config,
                 '--evolution-policy', tmp_path / 'missing-policy.json')
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['valid'] is True and output['evolution']['status'] == 'error'
    from datafarm.evolution.store import Store
    store = Store(json.loads((work / 'capability.json').read_text())['store'])
    assert any(e['kind'] == 'session-result' and e['evidence']['sha256'] for e in store.all('events'))
