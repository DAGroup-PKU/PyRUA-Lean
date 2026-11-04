"""CPU checks of the phase runner with stub harnesses; no agent CLI, model or simulator is called."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import pytest

from datafarm.agent import phases
from datafarm.common import sha256_file
from datafarm.task.contract import runtime_managed, verify_review_independence

ROOT = Path(__file__).resolve().parents[1]
PRICES = {'model-a': {'input': 1, 'cache_read': .1, 'cache_write_5m': 2, 'output': 10}}
PI = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time, uuid
args = sys.argv[1:]
root = pathlib.Path(args[args.index('--session-dir') + 1]); root.mkdir(parents=True, exist_ok=True)
sid = args[args.index('--session') + 1] if '--session' in args else str(uuid.uuid4())
assert pathlib.Path(args[-1][1:]).read_text()
tokens = int(os.environ.get('STUB_OUTPUT', '1000'))
model = os.environ.get('STUB_MODEL', 'route/model-a')
stop = os.environ.get('STUB_STOP', 'stop')
print(json.dumps({'type': 'session', 'id': sid}, separators=(',', ':')), flush=True)
print(json.dumps({'type': 'message_update', 'delta': 'x'}, separators=(',', ':')), flush=True)
with (root / f'{uuid.uuid4()}.jsonl').open('w') as log:
    log.write(json.dumps({'type': 'session', 'id': sid}) + '\\n')
    log.write(json.dumps({'type': 'message', 'id': str(uuid.uuid4()), 'timestamp': 't', 'message': {'role': 'assistant', 'model': model,
        'usage': {'input': 0, 'cacheRead': 0, 'cacheWrite': 0, 'output': tokens}}}) + '\\n')
    log.write(json.dumps({'type': 'compaction', 'id': str(uuid.uuid4()), 'timestamp': 't', 'usage': {'input': 1000}}) + '\\n')
    log.write(json.dumps({'type': 'custom', 'id': str(uuid.uuid4()), 'timestamp': 't', 'data': {'kind': 'provider_response', 'model': model,
        'usage': {'input': 1000}}}) + '\\n')
    log.flush()
    time.sleep(float(os.environ.get('STUB_SLEEP', '0')))
print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'stopReason': stop,
    'content': [{'type': 'text', 'text': 'final words'}]}}), flush=True)
'''
CLAUDE = '''#!/usr/bin/env python3
import json, os, sys, uuid
sys.stdin.read()
print(json.dumps({'type': 'system', 'subtype': 'init', 'model': 'model-a', 'session_id': str(uuid.uuid4())}))
print(json.dumps({'type': 'result', 'is_error': False, 'result': 'claude words', 'total_cost_usd': 0.5}))
'''
CODEX = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time, uuid
sys.stdin.read()
thread = str(uuid.uuid4())
print(json.dumps({'type': 'thread.started', 'thread_id': thread}), flush=True)
folder = pathlib.Path(os.environ['CODEX_HOME']) / 'sessions/2026/01/01'; folder.mkdir(parents=True)
usage = {'input_tokens': 2000, 'cached_input_tokens': 1000, 'output_tokens': int(os.environ.get('STUB_OUTPUT', '100'))}
record = json.dumps({'type': 'token_usage_record', 'payload': {'response_id': 'r1', 'usage': usage}})
(folder / f'rollout-x-{thread}.jsonl').write_text(record + '\\n' + record + '\\n')
time.sleep(float(os.environ.get('STUB_SLEEP', '0')))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'codex words'}}), flush=True)
print(json.dumps({'type': 'turn.completed', 'usage': usage}), flush=True)
'''


def config(harness='pi', **changes):
    value = {'schema': 'datafarm.phase-run/1', 'harness': harness, 'prices_usd_per_mtok': PRICES, 'task_cap_usd': 100,
             'phase_caps_usd': {'P1': 50, 'R1': 50, 'P2': 50}, 'poll_seconds': .05}
    return {**value, **changes}


@pytest.fixture
def task(tmp_path, monkeypatch):
    folder = tmp_path / 'bin'
    folder.mkdir()
    for name, source in (('pi', PI), ('claude', CLAUDE), ('codex', CODEX)):
        path = folder / name
        path.write_text(source)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv('PATH', f'{folder}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'codex-home'))
    episode = tmp_path / 'episode.mp4'
    episode.write_bytes(b'episode')
    request = tmp_path / 'request.txt'
    request.write_text('make data')
    root = tmp_path / 'task'
    subprocess.run([sys.executable, str(ROOT / 'tools/pipeline_agent.py'), 'prepare', '--episode', str(episode),
                    '--request-file', str(request), '--target-count', '2', '--stop-after', 'white',
                    '--output', str(root / 'work')], check=True, capture_output=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    return root


def run(task, phase='P1', cfg=None, template=None, **kwargs):
    return phases.run_phase(task, cfg or config(), phase, template or phase, 'brief text', source=ROOT, **kwargs)


def test_pi_phase_is_priced_from_session_files_and_recorded(task):
    row = run(task)
    # output 1000 at $10, then compaction and reducer 1000 each at $1
    assert row['status'] == 'done' and row['phase_usd'] == pytest.approx(.012)
    assert row['session_id'] and row['role'] == 'producer'
    assert (task / 'phases/P1/final.md').read_text().strip() == 'final words'
    assert 'message_update' not in (task / 'phases/P1/stream.jsonl').read_text()
    assert 'brief text' in (task / 'phases/P1/prompt.md').read_text()
    assert 'instructions for any coding agent' in (task / 'phases/P1/prompt.md').read_text()
    assert phases.cost_status(task)['task_usd'] == pytest.approx(.012)


def test_a_response_past_the_cap_is_marked(task, monkeypatch):
    monkeypatch.setenv('STUB_OUTPUT', '1000000')
    row = run(task, cfg=config(phase_caps_usd={'P1': 1}))
    assert row['status'] == 'done' and row['over_cap'] is True


def test_phase_cap_kills_the_session_and_keeps_the_record(task, monkeypatch):
    monkeypatch.setenv('STUB_OUTPUT', '1000000')
    monkeypatch.setenv('STUB_SLEEP', '60')
    started = time.time()
    row = run(task, cfg=config(phase_caps_usd={'P1': 1}))
    assert time.time() - started < 30
    assert row['status'] == 'stopped' and 'phase cap' in row['stop_reason'] and row['phase_usd'] > 1
    assert phases.ledger(task)[-1]['phase'] == 'P1'


def test_task_cap_stops_a_running_phase_and_refuses_the_next(task, monkeypatch):
    monkeypatch.setenv('STUB_OUTPUT', '1000000')
    monkeypatch.setenv('STUB_SLEEP', '60')
    row = run(task, cfg=config(task_cap_usd=5))
    assert row['status'] == 'stopped' and 'task cap' in row['stop_reason']
    with pytest.raises(phases.PhaseError, match='task cap reached'):
        run(task, 'P2', cfg=config(task_cap_usd=5))


def test_usage_without_a_price_stops_the_phase(task, monkeypatch):
    monkeypatch.setenv('STUB_MODEL', 'unknown-model')
    monkeypatch.setenv('STUB_SLEEP', '60')
    row = run(task)
    assert row['status'] == 'stopped' and 'no price' in row['stop_reason'] and row['unpriced_records'] > 0


def test_provider_error_fails_although_pi_exits_zero(task, monkeypatch):
    monkeypatch.setenv('STUB_STOP', 'error')
    row = run(task)
    assert row['exit'] == 0 and row['status'] == 'failed'


def test_a_phase_label_runs_once(task):
    run(task)
    with pytest.raises(phases.PhaseError, match='already started'):
        run(task)


def test_resume_charges_only_the_new_usage(task):
    first = run(task)
    second = run(task, 'P2', resume_from='P1')
    assert second['status'] == 'done' and second['session_id'] == first['session_id']
    assert second['phase_usd'] == pytest.approx(first['phase_usd'])  # the earlier entries in that session are not charged again


def test_reused_session_is_flagged(task, monkeypatch):
    first = run(task)
    monkeypatch.setattr(phases.PiHarness, 'outcome', lambda self, stream: (first['session_id'], 'words', False))
    assert run(task, 'R1')['status'] == 'session_reused'


def test_review_execution_satisfies_the_delivery_verifier(task):
    row = run(task, 'R3', template='R1', cfg=config(phase_caps_usd={'R1': 50}))
    assert row['role'] == 'reviewer' and row['status'] == 'done'
    work = task / 'work'
    record = Path(row['execution'])
    assert record.parent == work / 'reviews/executions/R3'
    assert not runtime_managed(work)
    review = {'reviewer_id': 'R3', 'execution': {'path': str(record.relative_to(work)), 'sha256': sha256_file(record)}}
    assert verify_review_independence(work, review, {}, phases.PRODUCER_ID, set()) == 'declared-execution-record'


def test_claude_uses_the_reported_cost(task):
    row = run(task, cfg=config('claude'))
    assert row['status'] == 'done' and row['phase_usd'] == .5 and row['session_id']
    assert (task / 'phases/P1/final.md').read_text().strip() == 'claude words'


def test_codex_counts_cached_input_once(task):
    row = run(task, cfg=config('codex', model='model-a'))
    assert row['status'] == 'done' and row['phase_usd'] == pytest.approx((1000 + 100 + 100 * 10) / 1e6)
    assert (task / 'phases/P1/final.md').read_text().strip() == 'codex words'


def test_codex_cap_applies_during_the_turn(task, monkeypatch):
    monkeypatch.setenv('STUB_OUTPUT', '1000000')
    monkeypatch.setenv('STUB_SLEEP', '60')
    row = run(task, cfg=config('codex', model='model-a', phase_caps_usd={'P1': 1}))
    assert row['status'] == 'stopped' and 'phase cap' in row['stop_reason']


def test_long_prompt_tier():
    table = {**PRICES['model-a'], 'long_prompt': {'above_tokens': 10, 'input_multiplier': 2, 'output_multiplier': 1.5}}
    assert phases.token_cost(table, 100, 0, 0, 100) == pytest.approx((100 * 2 + 1000 * 1.5) / 1e6)
    assert phases.token_cost(table, 5, 0, 0, 100) == pytest.approx((5 + 1000) / 1e6)


@pytest.mark.parametrize('change,message', [({'schema': 'x'}, 'schema'), ({'harness': 'x'}, 'harness'),
                                            ({'prices_usd_per_mtok': {}}, 'prices'), ({'task_cap_usd': 0}, 'task_cap'),
                                            ({'harness': 'codex'}, 'explicit model')])
def test_config_is_validated(tmp_path, change, message):
    path = tmp_path / 'c.json'
    path.write_text(json.dumps(config(**change)))
    with pytest.raises(phases.PhaseError, match=message):
        phases.load_config(path)


TEMPLATES = sorted(p.stem for p in (ROOT / 'prompts/phases').glob('*.md') if p.stem not in {'common', 'protocol'})


@pytest.mark.parametrize('name', ['phases-solpi.json', 'phases-claude-code.json', 'phases-codex.json'])
def test_example_configs_load_and_cap_every_phase(name):
    value = phases.load_config(ROOT / 'examples/runtime' / name)
    assert set(value['phase_caps_usd']) == set(TEMPLATES)


@pytest.mark.parametrize('template', TEMPLATES)
def test_templates_render_and_name_real_files(task, template):
    values = phases.brief_values(task, config(phase_caps_usd={template: 5}), template, template, source=ROOT,
                                 quoted='claim one\nclaim two', settings={'DENIAL': 'R2', 'PREVIOUS_HANDOFF': 'P2-handoff.md'})
    text = phases.render_brief(template, values)
    assert '$5' in text and f'{ROOT}/prompts/phases/protocol.md' in text
    assert '.runtime' not in text  # a .runtime/reviews folder would make the verifier require runtime reviews
    if '{PRODUCER}' in (ROOT / 'prompts/phases' / f'{template}.md').read_text():
        assert '> claim one\n> claim two' in text
    for ref in re.findall(re.escape(str(ROOT)) + r'/([\w./-]+)', text):
        assert (ROOT / ref.rstrip('./')).exists(), ref


def test_review_template_requires_all_views_and_raw_hashes():
    text = (ROOT / 'prompts/phases/R3.md').read_text()
    assert 'Look at every contact sheet' in text and '`raw_files`' in text and '"sha256"' in text
    assert '`raw_files`' in (ROOT / 'prompts/phases/R3b.md').read_text()
    assert '`--status`' in (ROOT / 'prompts/phases/R2.md').read_text()


def test_missing_brief_value_is_refused(task):
    values = phases.brief_values(task, config(phase_caps_usd={'P2r': 5}), 'P2r', 'P2r', source=ROOT)
    with pytest.raises(phases.PhaseError, match='DENIAL'):
        phases.render_brief('P2r', values)


def test_phase_needs_a_cap(task):
    with pytest.raises(phases.PhaseError, match='no cap'):
        phases.brief_values(task, config(phase_caps_usd={}), 'P1', 'P1', source=ROOT)
    with pytest.raises(phases.PhaseError, match='no cap'):
        run(task, cfg=config(phase_caps_usd={}))


def freeze(task, decisions):
    folder = task / 'work/reviews/R2'
    folder.mkdir(parents=True)
    (folder / 'attempt_gate.py').write_text(
        "from pathlib import Path\nprint(len(list(Path('.').glob('ran-*'))) + 1)\n")
    (folder / 'capture_attempt.sh').write_text(
        'set -e\ntouch ran-$1\ncase $1 in\n' + ''.join(
            f"{i}) echo noise; echo '{{\"decision\": \"{d}\"}}';;\n" for i, d in enumerate(decisions, 1)) + 'esac\n')


def test_capture_runs_until_the_gate_stops_it(task):
    freeze(task, ['continue', 'continue', 'pause_for_batch_review', 'continue'])
    result = phases.capture(task, 'R2')
    assert result['attempts_run'] == 3 and result['decision'] == 'pause_for_batch_review'
    assert 'driver stop' in (task / 'work/capture/driver.log').read_text()


def test_capture_stops_when_the_gate_prints_no_decision(task):
    freeze(task, ['continue'])
    result = phases.capture(task, 'R2')
    assert result['attempts_run'] == 2 and result['decision'] == ''


def test_capture_needs_the_frozen_interface(task):
    with pytest.raises(phases.PhaseError, match='capture_attempt.sh'):
        phases.capture(task, 'R2')


def test_cli_runs_a_phase_and_reports_cost(task, tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config(environment=['Simulator: stub'])))
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    tool = [sys.executable, str(ROOT / 'tools/pipeline_phases.py')]
    shown = subprocess.run([*tool, 'run', '--task-dir', str(task), '--config', str(path), '--phase', 'P1', '--print-brief'],
                           capture_output=True, text=True, env=env)
    assert shown.returncode == 0 and '- Simulator: stub' in shown.stdout and not (task / 'phases/P1').exists()
    ran = subprocess.run([*tool, 'run', '--task-dir', str(task), '--config', str(path), '--phase', 'P1'],
                         capture_output=True, text=True, env=env)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout)['status'] == 'done'
    cost = subprocess.run([*tool, 'cost', '--task-dir', str(task)], capture_output=True, text=True, env=env)
    assert json.loads(cost.stdout)['phases'][0]['phase'] == 'P1'
