"""Phase-by-phase task creation: one fresh agent session per phase, with USD caps the host enforces.

The host renders a brief from ``prompts/phases/``, runs one harness session on it, and records the
outcome. Producers and independent reviewers never share a session, and capture attempts run with no
model at all. See docs/reference/phased-runs.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from datafarm.common import sha256_file
from datafarm.expert.session import ClaudeStreamParser, CodexJsonParser

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT / 'prompts/phases'
PRODUCER_ID = 'phase-producer'
TERMINATE_GRACE_S = 10


class PhaseError(ValueError):
    pass


def price_for(model, prices):
    """Prices match by substring because harnesses prefix model IDs with a provider route."""
    matches = [name for name in prices if name in (model or '')]
    return prices[max(matches, key=len)] if matches else None


def token_cost(table, input_tokens=0, cache_read=0, cache_write=0, output=0):
    prompt = input_tokens + cache_read + cache_write
    tier = table.get('long_prompt') or {}
    long_prompt = bool(tier) and prompt > tier['above_tokens']
    side = tier.get('input_multiplier', 1) if long_prompt else 1
    out = tier.get('output_multiplier', 1) if long_prompt else 1
    return ((input_tokens * table['input'] + cache_read * table['cache_read']
             + cache_write * table['cache_write_5m']) * side + output * table['output'] * out) / 1e6


def read_events(path):
    if not path.exists():
        return
    with path.open(errors='ignore') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


class Harness:
    """One headless agent CLI: its command, how it reports usage, and how a session ended."""

    def __init__(self, config):
        self.extra = list(config.get('extra_args', []))
        self.model = config.get('model')
        self.prices = config['prices_usd_per_mtok']
        self.env = dict(config.get('env', {}))

    def outcome(self, stream):
        raise NotImplementedError

    def compact(self, stream):
        pass


class PiHarness(Harness):
    """Pi, with SoL-Pi when `pi install` has added it. Usage comes from the session files, which also
    hold compaction and reducer calls that the JSON stream does not show."""

    name = 'pi'

    def compact(self, stream):
        # Each token delta repeats the partial message; the finished messages are kept.
        kept = stream.with_suffix('.tmp')
        with stream.open(errors='ignore') as lines, kept.open('w') as out:
            out.writelines(line for line in lines if not line.startswith('{"type":"message_update"'))
        kept.replace(stream)

    def session_root(self, phase_dir, resume):
        return (resume['dir'] if resume else phase_dir) / 'pi-sessions'

    def command(self, phase_dir, prompt, resume):
        argv = ['pi', '-p', '--mode', 'json', '--no-context-files', '--session-dir',
                str(self.session_root(phase_dir, resume)), *self.extra]
        argv += ['--model', self.model] if self.model else []
        argv += ['--session', resume['session_id']] if resume else []
        return argv + [f'@{prompt}'], None

    def usage(self, phase_dir, stream, session_id, resume):
        usd, unpriced = 0.0, 0
        seen = set()
        for path in sorted(self.session_root(phase_dir, resume).rglob('*.jsonl')):
            model = None
            for event in read_events(path):
                key = (event.get('id'), event.get('timestamp'))
                if key in seen:
                    continue
                seen.add(key)
                kind, message = event.get('type'), event.get('message') or {}
                data = event.get('data') if isinstance(event.get('data'), dict) else {}
                if kind == 'model_change':
                    model = event.get('modelId') or model
                if kind == 'message' and message.get('role') == 'assistant':
                    entry, entry_model = message.get('usage'), message.get('model')
                    model = entry_model or model
                elif kind == 'usage':
                    entry, entry_model = event.get('usage'), event.get('model')
                elif kind in ('compaction', 'branch_summary'):  # these run on the session model
                    entry, entry_model = event.get('usage'), event.get('model') or model
                elif data.get('kind') == 'provider_response':  # SoL-Pi reducer
                    entry, entry_model = data.get('usage'), data.get('model')
                else:
                    continue
                if not isinstance(entry, dict):
                    continue
                tokens = [int(entry.get(k) or 0) for k in ('input', 'cacheRead', 'cacheWrite', 'output')]
                if not any(tokens):
                    continue
                table = price_for(entry_model, self.prices)
                if table is None:
                    unpriced += 1
                else:
                    usd += token_cost(table, *tokens)
        return usd, unpriced

    def outcome(self, stream):
        session_id, text, failed = None, '', False
        for event in read_events(stream):
            if event.get('type') == 'session':
                session_id = session_id or event.get('id')
            message = event.get('message') if event.get('type') == 'message_end' else None
            if isinstance(message, dict) and message.get('role') == 'assistant':
                # Pi exits 0 after a provider error, so the last assistant message decides.
                failed = message.get('stopReason') == 'error'
                parts = [p.get('text', '') for p in message.get('content') or [] if isinstance(p, dict)]
                text = ''.join(parts).strip() or text
        return session_id, text, failed


class ClaudeHarness(Harness):
    name = 'claude'

    def command(self, phase_dir, prompt, resume):
        argv = ['claude', '-p', '--output-format', 'stream-json', '--verbose', *self.extra]
        argv += ['--model', self.model] if self.model else []
        argv += ['--resume', resume['session_id']] if resume else []
        return argv, prompt.read_text()

    def parsed(self, stream):
        parser = ClaudeStreamParser(self.prices)
        if stream.exists():
            with stream.open(errors='ignore') as lines:
                for line in lines:
                    parser.feed(line)
        return parser.finish()

    def usage(self, phase_dir, stream, session_id, resume):
        parsed = self.parsed(stream)
        if parsed.cost_usd is not None:
            return parsed.cost_usd, 0
        return 0.0, int(bool(parsed.input_tokens or parsed.output_tokens))

    def outcome(self, stream):
        parsed = self.parsed(stream)
        return parsed.session_id, parsed.result_text or '', parsed.is_error or not parsed.saw_result


class CodexHarness(Harness):
    """Codex reports usage in the stream only when a turn ends, and one phase is one turn. While the
    phase runs, usage comes from the rollout file of the thread."""

    name = 'codex'

    def command(self, phase_dir, prompt, resume):
        argv = ['codex', 'exec']
        argv += ['resume', resume['session_id']] if resume else []
        argv += ['--json', '--skip-git-repo-check', *self.extra]
        argv += ['-m', self.model] if self.model else []
        return argv + ['-'], prompt.read_text()

    def parsed(self, stream):
        parser = CodexJsonParser(self.prices, self.model)
        if stream.exists():
            with stream.open(errors='ignore') as lines:
                for line in lines:
                    parser.feed(line)
        return parser.finish()

    def rollout_usage(self, thread, table):
        home = Path(self.env.get('CODEX_HOME') or os.environ.get('CODEX_HOME') or Path.home() / '.codex')
        usd, seen = 0.0, set()
        for path in (home / 'sessions').rglob(f'rollout-*{thread}.jsonl'):
            for event in read_events(path):
                payload = event.get('payload') if isinstance(event.get('payload'), dict) else {}
                if event.get('type') != 'token_usage_record' or payload.get('response_id') in seen:
                    continue
                seen.add(payload.get('response_id'))
                used = payload.get('usage') or {}
                cached = int(used.get('cached_input_tokens') or 0)
                written = int(used.get('cache_write_input_tokens') or 0)
                usd += token_cost(table, max(0, int(used.get('input_tokens') or 0) - cached - written), cached, written,
                                  int(used.get('output_tokens') or 0))
        return usd

    def usage(self, phase_dir, stream, session_id, resume):
        parsed = self.parsed(stream)
        table = price_for(self.model or parsed.model, self.prices)
        if table is None:
            return 0.0, 1
        thread = session_id or parsed.session_id
        live = self.rollout_usage(thread, table) if thread else 0.0
        cached = parsed.cache_read_tokens or 0
        settled = token_cost(table, parsed.input_tokens or 0, cached, 0, parsed.output_tokens or 0)
        return max(live, settled), 0

    def outcome(self, stream):
        parsed = self.parsed(stream)
        return parsed.session_id, parsed.result_text or '', parsed.is_error or not parsed.saw_result


HARNESSES = {h.name: h for h in (PiHarness, ClaudeHarness, CodexHarness)}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'datafarm.phase-run/1':
        raise PhaseError('phase config schema must be datafarm.phase-run/1')
    if config.get('harness') not in HARNESSES:
        raise PhaseError(f'harness must be one of {sorted(HARNESSES)}')
    if not config.get('prices_usd_per_mtok'):
        raise PhaseError('prices_usd_per_mtok is required: caps cannot be enforced without prices')
    if config['harness'] == 'codex' and not config.get('model'):
        raise PhaseError('the codex harness needs an explicit model: its stream does not name one')
    if not float(config.get('task_cap_usd', 0)) > 0:
        raise PhaseError('task_cap_usd must be positive')
    return config


def ledger_path(task):
    return Path(task) / 'phase-usage.jsonl'


def ledger(task):
    return list(read_events(ledger_path(task)))


def task_total(task):
    return sum(row['phase_usd'] for row in ledger(task))


def cost_status(task):
    live = Path(task) / 'phases' / 'current-cost.json'
    current = json.loads(live.read_text()) if live.exists() else None
    return {'task_usd': round(current['task_usd'] if current else task_total(task), 4),
            'running_phase': current,
            'phases': [{k: row.get(k) for k in ('phase', 'status', 'phase_usd', 'exit', 'session_id')}
                       for row in ledger(task)]}


def role_of(template):
    return 'reviewer' if template.startswith('R') else 'producer'


def render_brief(template, values):
    text = (TEMPLATES / f'{template}.md').read_text()
    filled = dict(values, COMMON='')
    filled['COMMON'] = re.sub(r'\{([A-Z_]+)\}', lambda m: str(filled.get(m.group(1), m.group(0))),
                              (TEMPLATES / 'common.md').read_text()).rstrip('\n')
    text = re.sub(r'\{([A-Z_]+)\}', lambda m: str(filled.get(m.group(1), m.group(0))), text)
    left = sorted(set(re.findall(r'\{([A-Z_]+)\}', text)))
    if left:
        raise PhaseError(f'{template}: missing brief values {left}; pass them with --set KEY=VALUE')
    return text


def brief_values(task, config, phase, template, *, source, quoted=None, summary='', settings=None):
    task = Path(task).resolve()
    work = task / 'work'
    caps = config.get('phase_caps_usd', {})
    if template not in caps and phase not in caps:
        raise PhaseError(f'no cap for phase {phase} (template {template}) in phase_caps_usd')
    lock = json.loads((work / 'input.json').read_text())
    tools = f'{sys.executable} {source}/tools'
    values = {
        'NAME': config.get('task_name') or task.name, 'T': str(task), 'W': str(work), 'SOURCE': str(source),
        'PCAP': caps.get(phase, caps.get(template)), 'TASK_CAP': config['task_cap_usd'],
        'TARGET': lock.get('target_count'), 'PRODUCER_ID': PRODUCER_ID,
        'COST_COMMAND': f'{tools}/pipeline_phases.py cost --task-dir {task}',
        'VERIFY_COMMAND': f'{tools}/pipeline_agent.py verify --work {work}',
        'ENVIRONMENT': '\n'.join(f'- {line}' for line in config.get('environment', [])),
        'FORBIDDEN': ', '.join(f'`{p}`' for p in config.get('freshness_forbidden', [])) or 'none listed',
        'SUMMARY': summary or 'None.', 'PHASE': phase,
        'REQUEST': 'P2-source-request.json', 'HANDOFF': 'P2-handoff.md', 'BATCH_REVIEW': 'R2',
        'ADMISSION_REVIEW': 'R3', 'DELIVERY_PHASE': 'P4',
        'PRODUCER': '\n'.join(f'> {line}' for line in (quoted or '').strip().splitlines()),
    }
    values.update(settings or {})
    return values


def stop_group(process):
    for sig, wait in ((signal.SIGTERM, TERMINATE_GRACE_S), (signal.SIGKILL, TERMINATE_GRACE_S)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def write_execution(task, phase, transcript_source):
    """Host-written record that a review ran as its own session; verify_delivery checks it."""
    work = Path(task) / 'work'
    folder = work / 'reviews' / 'executions' / phase
    folder.mkdir(parents=True, exist_ok=True)
    transcript = folder / 'transcript.jsonl'
    shutil.copyfile(transcript_source, transcript)
    (folder / 'execution.json').write_text(json.dumps({
        'role': 'reviewer', 'actor_id': phase, 'producer_id': PRODUCER_ID,
        'transcript': {'path': str(transcript.relative_to(work)), 'sha256': sha256_file(transcript)}}, indent=2) + '\n')
    return folder / 'execution.json'


def run_phase(task, config, phase, template, brief, *, source, resume_from=None, poll_seconds=None):
    task = Path(task).resolve()
    if not (task / 'work' / 'input.json').is_file():
        raise PhaseError(f'{task}/work is not a prepared task; run pipeline_agent.py prepare --output {task}/work')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', phase):
        raise PhaseError('invalid phase label')
    phases = task / 'phases'
    phase_dir = phases / phase
    if phase_dir.exists():
        raise PhaseError(f'phase {phase} already started; use a new label with --resume-from to continue it')
    rows = ledger(task)
    resume = None
    if resume_from:
        previous = next((r for r in rows if r['phase'] == resume_from), None)
        if previous is None or not previous.get('session_id'):
            raise PhaseError(f'{resume_from} has no recorded session to resume')
        resume = {'dir': phases / resume_from, 'session_id': previous['session_id']}
    caps = config.get('phase_caps_usd', {})
    if phase not in caps and template not in caps:
        raise PhaseError(f'no cap for phase {phase} (template {template}) in phase_caps_usd')
    task_cap, phase_cap = float(config['task_cap_usd']), float(caps.get(phase, caps[template]))
    before = task_total(task)
    if before >= task_cap:
        raise PhaseError(f'task cap reached before {phase}: ${before:.2f} of ${task_cap:.2f}')
    harness = HARNESSES[config['harness']](config)
    phase_dir.mkdir(parents=True)
    agents = (Path(source) / 'AGENTS.md').read_text()
    prompt = phase_dir / 'prompt.md'
    (phase_dir / 'brief.md').write_text(brief)
    prompt.write_text(f'{agents}\n\n{brief}')
    stream = phase_dir / 'stream.jsonl'
    argv, stdin_text = harness.command(phase_dir, prompt, resume)
    (phase_dir / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
    baseline = harness.usage(phase_dir, stream, resume['session_id'], resume)[0] if resume else 0.0
    poll = poll_seconds if poll_seconds is not None else config.get('poll_seconds', 30)
    max_wall = config.get('max_phase_seconds')
    started = time.time()
    live = phases / 'current-cost.json'
    stop_reason = None
    spent = 0.0
    with stream.open('w') as out, (phase_dir / 'stderr.log').open('w') as err:
        process = subprocess.Popen(argv, cwd=task, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                   stdout=out, stderr=err, text=True, start_new_session=True,
                                   env={**os.environ, **harness.env})
        if stdin_text is not None:
            process.stdin.write(stdin_text)
            process.stdin.close()
        while True:
            try:
                process.wait(timeout=poll)
                break
            except subprocess.TimeoutExpired:
                pass
            session_id = harness.outcome(stream)[0]
            usd, unpriced = harness.usage(phase_dir, stream, session_id, resume)
            spent = max(0.0, usd - baseline)
            live.write_text(json.dumps({'phase': phase, 'phase_usd': round(spent, 4),
                                        'task_usd': round(before + spent, 4), 'at': time.time()}) + '\n')
            if unpriced:
                stop_reason = f'{unpriced} usage records have no price in prices_usd_per_mtok'
            elif spent >= phase_cap:
                stop_reason = f'phase cap ${phase_cap:g} reached'
            elif before + spent >= task_cap:
                stop_reason = f'task cap ${task_cap:g} reached'
            elif max_wall and time.time() - started > max_wall:
                stop_reason = f'max_phase_seconds {max_wall} reached'
            if stop_reason:
                stop_group(process)
                break
    code = process.wait()
    harness.compact(stream)
    session_id, text, failed = harness.outcome(stream)
    usd, unpriced = harness.usage(phase_dir, stream, session_id, resume)
    spent = max(spent, usd - baseline)
    live.unlink(missing_ok=True)
    (phase_dir / 'final.md').write_text(text + '\n')
    earlier = {r.get('session_id'): r['phase'] for r in rows if r.get('session_id')}
    reused = session_id in earlier and not (resume and session_id == resume['session_id'])
    status = ('session_reused' if reused else 'stopped' if stop_reason else 'failed' if code or failed else 'done')
    row = {'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'phase': phase, 'template': template,
           'role': role_of(template), 'harness': harness.name, 'model': harness.model, 'exit': code,
           'status': status, 'stop_reason': stop_reason, 'session_id': session_id, 'resumed_from': resume_from,
           'over_cap': spent > phase_cap or before + spent > task_cap,
           'unpriced_records': unpriced, 'task_usd_before': round(before, 4), 'phase_usd': round(spent, 4),
           'task_usd_after': round(before + spent, 4), 'wall_s': round(time.time() - started, 1)}
    with ledger_path(task).open('a') as rows_out:
        rows_out.write(json.dumps(row) + '\n')
    if role_of(template) == 'reviewer' and status == 'done':
        row['execution'] = str(write_execution(task, phase, stream))
    return row


def gate_decision(output):
    """The `decision` of the last top-level JSON object a gate printed; empty if there is none."""
    starts = [i for i, char in enumerate(output) if char == '{' and (i == 0 or output[i - 1] == '\n')]
    for start in reversed(starts):
        try:
            value = json.JSONDecoder().raw_decode(output[start:])[0]
        except ValueError:
            continue
        return value.get('decision', '') if isinstance(value, dict) else ''
    return ''


def capture(task, review):
    """Run the reviewer-frozen capture one attempt at a time, with no model, until the gate stops it."""
    work = Path(task).resolve() / 'work'
    folder = work / 'reviews' / review
    for name in ('capture_attempt.sh', 'attempt_gate.py'):
        if not (folder / name).is_file():
            raise PhaseError(f'{folder}/{name} is missing; the reviewer freezes the capture interface')
    (work / 'capture').mkdir(exist_ok=True)
    log_path = work / 'capture' / 'driver.log'

    def log(text):
        with log_path.open('a') as stream:
            stream.write(text.rstrip('\n') + '\n')

    decision = ''
    attempts = 0
    while True:
        result = subprocess.run([sys.executable, 'attempt_gate.py', '--next-index'], cwd=folder,
                                capture_output=True, text=True)
        index = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''
        if not index.isdigit():
            decision = 'bad_next_index'
            log(f'bad next index {index!r}: {result.stderr[-500:]}')
            break
        log(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} start index {index}')
        result = subprocess.run(['bash', 'capture_attempt.sh', index], cwd=folder, capture_output=True, text=True)
        log(result.stdout)
        log(result.stderr)
        attempts += 1
        decision = gate_decision(result.stdout)
        log(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} index {index} rc={result.returncode} decision={decision}')
        if decision != 'continue':
            break
    log('driver stop')
    return {'attempts_run': attempts, 'decision': decision, 'log': str(log_path)}
