"""CPU checks of the CLI driver with stub executables; no agent CLI or model is called."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from datafarm.agent.runtime import register_identity, runtime_dir, supervise
from datafarm.common import sha256_file
from test_configurable_runtime import policy
from test_task_cost_budget import driver_module, workspace

RUNTIME = Path(__file__).resolve().parents[1] / 'examples/runtime'
DRIVER = RUNTIME / 'agent_cli_driver.py'

CLAUDE = '''#!/usr/bin/env python3
import json, sys, uuid
prompt = sys.stdin.read()
print(json.dumps({"type": "system", "subtype": "init", "session_id": str(uuid.uuid4())}))
print(json.dumps({"type": "result", "is_error": False, "prompt": prompt, "args": sys.argv[1:]}))
'''
PI = '''#!/usr/bin/env python3
import json, pathlib, sys
prompt = pathlib.Path(sys.argv[-1][1:]).read_text()
print(json.dumps({"type": "session", "id": "pi-session"}))
print("not json")
print(json.dumps({"type": "message_end", "message": {"role": "user"}}))
print(json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "STOP",
                                                     "prompt": prompt, "args": sys.argv[1:]}}))
'''


def stub(tmp_path, name, source):
    folder = tmp_path / 'bin'
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return folder


def run(tmp_path, folder, *args):
    session = tmp_path / 'sessions' / '1'
    session.mkdir(parents=True)
    env = {**os.environ, 'PATH': f'{folder}{os.pathsep}{os.environ["PATH"]}', 'DATAFARM_SESSION_DIR': str(session),
           'DATAFARM_AGENT_SESSION_KEY': 'author', 'DATAFARM_AGENT_ROLE': 'orchestrator'}
    result = subprocess.run([sys.executable, str(DRIVER), *args], input='the prompt', text=True,
                            capture_output=True, cwd=tmp_path, env=env, timeout=30)
    return session, result


def test_claude_identity_comes_from_the_stream_and_is_accepted(tmp_path):
    session, result = run(tmp_path, stub(tmp_path, 'claude', CLAUDE), 'claude', '--', '--model', 'm')
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[1]['prompt'] == 'the prompt' and events[1]['args'][:3] == ['--model', 'm', '-p']
    identity = json.loads((session / 'identity.json').read_text())
    assert identity['session_id'] == events[0]['session_id'] and identity['fresh'] is True
    assert identity['evidence'] == [{'path': 'sessions/1/agent-output.jsonl',
                                     'sha256': sha256_file(session / 'agent-output.jsonl')}]
    register_identity(tmp_path, session, 'author', 'orchestrator')


def test_pi_reads_the_prompt_file_and_skips_non_json_lines(tmp_path):
    session, result = run(tmp_path, stub(tmp_path, 'pi', PI), 'pi', '--', '--approve')
    assert result.returncode == 0, result.stderr
    final = json.loads(result.stdout.splitlines()[-1])['message']
    assert final['prompt'] == 'the prompt' and final['args'][0] == '--approve'
    assert json.loads((session / 'identity.json').read_text())['session_id'] == 'pi-session'


def test_pi_provider_error_fails_although_pi_exits_zero(tmp_path):
    source = PI.replace('"STOP"', '"error"')
    session, result = run(tmp_path, stub(tmp_path, 'pi', source), 'pi')
    assert result.returncode == 1
    assert (session / 'identity.json').is_file()


def test_later_success_clears_an_earlier_provider_error(tmp_path):
    source = PI.replace('"STOP"', '"error"') + (
        'print(json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}}))\n')
    assert run(tmp_path, stub(tmp_path, 'pi', source), 'pi')[1].returncode == 0


def test_claude_result_error_fails(tmp_path):
    source = CLAUDE.replace('"is_error": False', '"is_error": True')
    assert run(tmp_path, stub(tmp_path, 'claude', source), 'claude')[1].returncode == 1


def test_missing_session_id_writes_no_identity(tmp_path):
    source = '#!/bin/sh\ncat > /dev/null\necho "{}"\n'
    session, result = run(tmp_path, stub(tmp_path, 'claude', source), 'claude')
    assert result.returncode == 1 and 'no session ID' in result.stderr
    assert not (session / 'identity.json').exists()


def test_unknown_profile_is_refused(tmp_path):
    result = subprocess.run([sys.executable, str(DRIVER), 'other'], input='', text=True, capture_output=True)
    assert result.returncode != 0 and 'usage' in result.stderr


@pytest.mark.parametrize('name,profile', [('claude-code.json', 'claude'), ('pi-solpi.json', 'pi')])
def test_example_configs_declare_the_identity_protocol(name, profile):
    driver = json.loads((RUNTIME / name).read_text())['driver']
    assert driver['session_identity_protocol'] == 'datafarm.agent-identity/1' and driver['input'] == 'text'
    assert Path(driver['argv'][1]).name == DRIVER.name and driver['argv'][2] == profile


def test_supervise_registers_each_stub_session(tmp_path, monkeypatch):
    work = workspace(tmp_path)
    policy(work)
    module = driver_module(monkeypatch, work)
    fixture_context = module.write_context

    def context(folder, *args, **kwargs):
        result = fixture_context(folder, *args, **kwargs)
        (folder / 'context.md').write_text('the prompt')
        return result
    monkeypatch.setattr(module, 'write_context', context)
    monkeypatch.setenv('PATH', f'{stub(tmp_path, "claude", CLAUDE)}{os.pathsep}{os.environ["PATH"]}')
    config = work / 'runtime.json'
    config.write_text(json.dumps({
        'schema': 'datafarm.agent-runtime/1',
        'driver': {'argv': [sys.executable, str(DRIVER), 'claude'], 'input': 'text',
                   'session_identity_protocol': 'datafarm.agent-identity/1'},
        'max_sessions': 2, 'timeout': 20, 'poll_seconds': .02, 'heartbeat_seconds': .03, 'max_retries': 0,
        'backoff_seconds': .03, 'backoff_max_seconds': .1}))
    args = type('Args', (), {'work': work, 'config': config, 'resume': False})()
    assert supervise(args, module.drive)['status'] == 'session-budget-exhausted'
    known = json.loads((runtime_dir(work) / 'identities.json').read_text())
    assert len(known) == 2 and {row['session_key'] for row in known.values()} == {'author'}
