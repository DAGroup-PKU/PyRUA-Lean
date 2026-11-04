#!/usr/bin/env python3
"""Run a coding-agent CLI as a DataFarm argv driver.

The runtime pipes the compiled context to stdin and sets DATAFARM_SESSION_DIR. This driver runs
one fresh CLI session on that prompt, keeps the CLI's JSON stream as agent-output.jsonl and writes
identity.json (datafarm.agent-identity/1) with the session ID the CLI reported. A runtime config
selects a profile and may add CLI arguments:

    "argv": ["python3", "/abs/path/agent_cli_driver.py", "claude", "--", "--model", "<model>"]

To add another harness, add a profile: its command, and how its stream names the session and
reports a failed run.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def claude_command(session):
    return ['claude', '-p', '--output-format', 'stream-json', '--verbose'], True


def claude_session(event):
    return event.get('session_id')


def claude_failure(event):
    return event.get('is_error') is True if event.get('type') == 'result' else None


def pi_command(session):
    # Pi reads the prompt from an @file; context files are off because the prompt already holds them.
    return ['pi', '-p', '--mode', 'json', '--no-context-files', '--session-dir', str(session / 'pi-sessions'),
            f'@{session / "prompt.md"}'], False


def pi_session(event):
    return event.get('id') if event.get('type') == 'session' else None


def pi_failure(event):
    # Pi exits 0 after a provider error, so the outcome is read from the last assistant message.
    message = event.get('message') if event.get('type') == 'message_end' else None
    if isinstance(message, dict) and message.get('role') == 'assistant':
        return message.get('stopReason') == 'error'
    return None


PROFILES = {'claude': (claude_command, claude_session, claude_failure),
            'pi': (pi_command, pi_session, pi_failure)}


def write_identity(session, session_id, evidence):
    work = session.parents[1]
    (session / 'identity.json').write_text(json.dumps({
        'schema': 'datafarm.agent-identity/1', 'session_id': session_id,
        'session_key': os.environ['DATAFARM_AGENT_SESSION_KEY'], 'role': os.environ['DATAFARM_AGENT_ROLE'],
        'fresh': True, 'evidence': [{'path': str(evidence.relative_to(work)),
                                     'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}]}))


def main(argv):
    if not argv or argv[0] not in PROFILES:
        sys.exit(f'usage: agent_cli_driver.py {{{"|".join(PROFILES)}}} [-- extra CLI arguments]')
    build, read_session, read_failure = PROFILES[argv[0]]
    extra = argv[2:] if argv[1:2] == ['--'] else argv[1:]
    session = Path(os.environ['DATAFARM_SESSION_DIR']).resolve()
    prompt = sys.stdin.read()
    (session / 'prompt.md').write_text(prompt)
    command, prompt_on_stdin = build(session)
    evidence = session / 'agent-output.jsonl'
    session_id, failed = None, False
    # Children stay in this process group, so a runtime stop or timeout reaches them.
    with evidence.open('w') as log:
        process = subprocess.Popen(command[:1] + extra + command[1:], stdout=subprocess.PIPE, text=True,
                                   stdin=subprocess.PIPE if prompt_on_stdin else subprocess.DEVNULL)
        if prompt_on_stdin:
            process.stdin.write(prompt)
            process.stdin.close()
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                session_id = session_id or read_session(event)
                outcome = read_failure(event)
                failed = failed if outcome is None else outcome
    code = process.wait()
    if not session_id:
        print(f'{command[0]} reported no session ID; identity not written', file=sys.stderr)
        return code or 1
    write_identity(session, session_id, evidence)
    return code or (1 if failed else 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
