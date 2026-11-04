"""Headless agent sessions: command construction, event parsing, wall caps, outages.

A round is a sequence of sessions of one conversation: the first starts the agent with
the brief, later ones resume it (``--resume <session id>``) with a short status prompt.
Each session is capped in wall-clock time (SIGTERM, then SIGKILL, to the whole process
group), its stdout event stream is appended to a transcript file as it arrives, and its
final ``result`` event supplies ``total_cost_usd``, ``usage``, ``num_turns`` and the
session id -- so cost comes from the provider's own accounting instead of being
reconstructed from transcripts by hand.

Backends: ``claude`` (``claude -p --output-format stream-json --verbose``), ``codex``
(``codex exec --json``), ``fake`` (a scripted stand-in used by tests and the dry run).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from datafarm.expert.util import ConfigError

API_ERROR_PATTERNS = ("API Error", "api_error", "overloaded", "Overloaded", " 529", "rate_limit",
                      "rate limit", "ECONNRESET", "ETIMEDOUT", "ECONNREFUSED", "EAI_AGAIN",
                      "ENOTFOUND", "fetch failed", "Connection error", "socket hang up",
                      "502 Bad Gateway", "503 Service", "504 Gateway", "Internal server error",
                      "Request timed out", "stream disconnected", "network error")
AUTH_ERROR_PATTERNS = ("Invalid API key", "authentication_error", "Please run /login", "401 ",
                       "OAuth token has expired", "Not logged in", "invalid x-api-key",
                       "Credit balance is too low", "permission_error")


# --------------------------------------------------------------------------- pricing


def estimate_cost(model: str | None, usage: Mapping[str, int | None], prices: Mapping[str, Any]
                  ) -> float | None:
    """Price-table estimate in USD (per-million-token prices), or ``None`` if unpriced."""

    if not model:
        return None
    table = prices.get(model)
    if table is None:
        for name, value in prices.items():
            if model.startswith(name):
                table = value
                break
    if table is None:
        return None

    def n(key: str) -> int:
        return int(usage.get(key) or 0)

    total = (n("input_tokens") * table["input"] + n("output_tokens") * table["output"]
             + n("cache_read_tokens") * table["cache_read"]
             + n("cache_write_5m_tokens") * table["cache_write_5m"]
             + n("cache_write_1h_tokens") * table.get("cache_write_1h", table["cache_write_5m"]))
    return round(total / 1_000_000.0, 6)


# --------------------------------------------------------------------------- parsing


@dataclasses.dataclass
class ParsedSession:
    session_id: str | None = None
    model: str | None = None
    saw_result: bool = False
    is_error: bool = False
    subtype: str | None = None
    result_text: str | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    turns: int | None = None
    tool_uses: int = 0
    events: int = 0
    bad_lines: int = 0
    error_texts: list[str] = dataclasses.field(default_factory=list)
    verified_transport_interruption: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["result_text"] = None if self.result_text is None else self.result_text[:2000]
        data["error_texts"] = [t[:500] for t in self.error_texts[-10:]]
        return data


class ClaudeStreamParser:
    """Parses ``claude -p --output-format stream-json --verbose`` output."""

    def __init__(self, prices: Mapping[str, Any] | None = None) -> None:
        self.prices = dict(prices or {})
        self.p = ParsedSession()
        self._usage_by_message: dict[str, dict[str, Any]] = {}
        self._tool_ids: set[str] = set()

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            self.p.bad_lines += 1
            return
        if not isinstance(event, dict):
            self.p.bad_lines += 1
            return
        self.p.events += 1
        kind = event.get("type")
        if event.get("session_id"):
            self.p.session_id = event["session_id"]
        if kind == "system":
            if event.get("subtype") == "init" and event.get("model"):
                self.p.model = event["model"]
            elif "error" in str(event.get("subtype", "")) or "retry" in str(event.get("subtype", "")):
                self.p.error_texts.append(json.dumps(event)[:500])
        elif kind == "assistant":
            message = event.get("message") or {}
            if message.get("model"):
                self.p.model = self.p.model or message["model"]
            if isinstance(message.get("usage"), dict):
                self._usage_by_message[str(message.get("id") or len(self._usage_by_message))] = message["usage"]
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    self._tool_ids.add(str(block.get("id") or len(self._tool_ids)))
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if any(p in text for p in API_ERROR_PATTERNS + AUTH_ERROR_PATTERNS) and len(text) < 2000:
                        self.p.error_texts.append(text)
            if event.get("error"):
                self.p.error_texts.append(str(event.get("error"))[:500])
        elif kind == "result":
            self.p.saw_result = True
            self.p.is_error = bool(event.get("is_error"))
            self.p.subtype = event.get("subtype")
            self.p.result_text = event.get("result") if isinstance(event.get("result"), str) else None
            if isinstance(event.get("num_turns"), int):
                self.p.turns = event["num_turns"]
            usage = event.get("usage") or {}
            if isinstance(usage, dict) and usage:
                self.p.input_tokens = usage.get("input_tokens")
                self.p.output_tokens = usage.get("output_tokens")
                self.p.cache_read_tokens = usage.get("cache_read_input_tokens")
                self.p.cache_write_tokens = usage.get("cache_creation_input_tokens")
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                self.p.cost_usd = float(cost)
                self.p.cost_source = "provider"
            if self.p.is_error and self.p.result_text:
                self.p.error_texts.append(self.p.result_text[:500])

    def finish(self) -> ParsedSession:
        self.p.tool_uses = len(self._tool_ids)
        if self.p.input_tokens is None and self._usage_by_message:
            totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                      "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}
            for usage in self._usage_by_message.values():
                totals["input_tokens"] += int(usage.get("input_tokens") or 0)
                totals["output_tokens"] += int(usage.get("output_tokens") or 0)
                totals["cache_read_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
                creation = usage.get("cache_creation") or {}
                one_hour = int(creation.get("ephemeral_1h_input_tokens") or 0) if isinstance(creation, dict) else 0
                totals["cache_write_1h_tokens"] += one_hour
                totals["cache_write_5m_tokens"] += int(usage.get("cache_creation_input_tokens") or 0) - one_hour
            self.p.input_tokens = totals["input_tokens"]
            self.p.output_tokens = totals["output_tokens"]
            self.p.cache_read_tokens = totals["cache_read_tokens"]
            self.p.cache_write_tokens = totals["cache_write_5m_tokens"] + totals["cache_write_1h_tokens"]
            if self.p.cost_usd is None:
                estimate = estimate_cost(self.p.model, totals, self.prices)
                if estimate is not None:
                    self.p.cost_usd = estimate
                    self.p.cost_source = "price_table"
        elif self.p.cost_usd is None and self.p.input_tokens is not None:
            estimate = estimate_cost(self.p.model, {
                "input_tokens": self.p.input_tokens, "output_tokens": self.p.output_tokens,
                "cache_read_tokens": self.p.cache_read_tokens,
                "cache_write_5m_tokens": self.p.cache_write_tokens}, self.prices)
            if estimate is not None:
                self.p.cost_usd = estimate
                self.p.cost_source = "price_table"
        return self.p


class CodexJsonParser:
    """Parses ``codex exec --json`` JSONL events (thread/turn/item)."""

    TOOL_ITEMS = ("command_execution", "file_change", "mcp_tool_call", "web_search")

    def __init__(self, prices: Mapping[str, Any] | None = None, model: str | None = None) -> None:
        self.prices = dict(prices or {})
        self.p = ParsedSession(model=model)
        self._tokens = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        self._saw_usage = False

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            self.p.bad_lines += 1
            return
        if not isinstance(event, dict):
            self.p.bad_lines += 1
            return
        self.p.events += 1
        kind = event.get("type")
        if kind == "thread.started" and event.get("thread_id"):
            self.p.session_id = event["thread_id"]
        elif kind == "turn.completed":
            self.p.saw_result = True
            self.p.turns = (self.p.turns or 0) + 1
            usage = event.get("usage") or {}
            for key in self._tokens:
                self._tokens[key] += int(usage.get(key) or 0)
            self._saw_usage = True
        elif kind == "turn.failed":
            self.p.saw_result = True
            self.p.is_error = True
            self.p.error_texts.append(str((event.get("error") or {}).get("message", event))[:500])
        elif kind == "error":
            self.p.is_error = True
            self.p.error_texts.append(str(event.get("message", event))[:500])
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") in self.TOOL_ITEMS:
                self.p.tool_uses += 1
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                self.p.result_text = item["text"]

    def finish(self) -> ParsedSession:
        if self._saw_usage:
            cached = self._tokens["cached_input_tokens"]
            self.p.input_tokens = max(0, self._tokens["input_tokens"] - cached)
            self.p.cache_read_tokens = cached
            self.p.output_tokens = self._tokens["output_tokens"]
            self.p.cache_write_tokens = 0
            estimate = estimate_cost(self.p.model, {
                "input_tokens": self.p.input_tokens, "output_tokens": self.p.output_tokens,
                "cache_read_tokens": cached}, self.prices)
            if estimate is not None:
                self.p.cost_usd = estimate
                self.p.cost_source = "price_table"
        return self.p


def classify_errors(parsed: ParsedSession, stderr_tail: str, exit_code: int | None
                    ) -> tuple[bool, bool, str | None]:
    """Classify outages from zero-tool errors or authoritative metered receipts.

    A metered transport interruption can follow useful tools; resume its same
    conversation without replaying tools, retaining all costs and trial records.
    """

    texts = "\n".join(parsed.error_texts + [stderr_tail[-4000:]])
    failed = parsed.is_error or not parsed.saw_result or (exit_code not in (0, None))
    auth = failed and any(p in texts for p in AUTH_ERROR_PATTERNS)
    api = failed and not auth and (parsed.verified_transport_interruption or
          (parsed.tool_uses == 0 and any(p in texts for p in API_ERROR_PATTERNS)))
    if auth:
        return False, True, "auth_error"
    if api:
        return True, False, "api_outage"
    if failed:
        if parsed.subtype and "budget" in parsed.subtype:
            return False, False, "max_budget"
        return False, False, parsed.subtype or ("no_result" if not parsed.saw_result else "error")
    return False, False, None


# --------------------------------------------------------------------------- backends


@dataclasses.dataclass
class SessionSpec:
    index: int
    prompt: str
    cwd: str
    resume_id: str | None = None
    new_session_id: str | None = None
    max_budget_usd: float | None = None


class AgentBackend:
    name = "agent"
    call_backend = "agent"
    uses_stdin_prompt = False

    def __init__(self, cfg: Mapping[str, Any], prices: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg)
        self.prices = dict(prices or {})
        self.model = str(self.cfg.get("model"))

    def command(self, spec: SessionSpec) -> list[str]:
        raise NotImplementedError

    def parser(self) -> Any:
        return ClaudeStreamParser(self.prices)

    def env(self) -> dict[str, str]:
        return {}

    @contextlib.contextmanager
    def session(self, spec: SessionSpec, directory: Path):
        yield

    def settle(self, parsed: ParsedSession) -> ParsedSession:
        return parsed


class ClaudeCliBackend(AgentBackend):
    name = "claude"
    call_backend = "claude-cli"

    def command(self, spec: SessionSpec) -> list[str]:
        cfg = self.cfg
        argv = [cfg.get("cli", "claude"), "-p", spec.prompt, "--output-format", "stream-json",
                "--verbose", "--model", self.model]
        if spec.resume_id:
            argv += ["--resume", spec.resume_id]
        elif spec.new_session_id:
            argv += ["--session-id", spec.new_session_id]
        tools = cfg.get("tools", ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "TodoWrite"])
        argv += ["--tools", ",".join(tools), "--allowedTools", ",".join(tools)]
        disallowed = cfg.get("disallowed_tools", ["WebFetch", "WebSearch"])
        if disallowed:
            argv += ["--disallowedTools", ",".join(disallowed)]
        if cfg.get("permission_mode"):
            argv += ["--permission-mode", cfg["permission_mode"]]
        argv += ["--strict-mcp-config", "--setting-sources", cfg.get("setting_sources", "user")]
        if cfg.get("settings_file"):
            argv += ["--settings", cfg["settings_file"]]
        if cfg.get("effort"):
            argv += ["--effort", cfg["effort"]]
        if spec.max_budget_usd is not None:
            argv += ["--max-budget-usd", f"{max(0.01, spec.max_budget_usd):.2f}"]
        return argv + list(cfg.get("extra_args", []))

    def env(self) -> dict[str, str]:
        return {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "IS_SANDBOX": "1"}


class CodexExecBackend(AgentBackend):
    """``codex exec --json``; the prompt goes through stdin (``-``), so it is neither in the
    process list nor appended to an empty ``<stdin>`` block. Codex's own sandbox is bypassed
    because the harness sandbox already confines it."""

    name = "codex"
    call_backend = "codex-exec"
    uses_stdin_prompt = True

    def command(self, spec: SessionSpec) -> list[str]:
        cfg = self.cfg
        argv = [cfg.get("cli", "codex"), "exec", "--json", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox", "-m", self.model]
        argv += list(cfg.get("extra_args", []))
        if spec.resume_id:
            return argv + ["resume", spec.resume_id, "-"]
        return argv + ["-C", spec.cwd, "-"]

    def parser(self) -> Any:
        return CodexJsonParser(self.prices, self.model)


class MeteredCodexBackend(CodexExecBackend):
    """Gateway and credentials stay in the trusted harness, outside author mounts.

    The author sees only a loopback endpoint. Costs come from upstream usage,
    never author stdout. Unknown usage blocks the round rather than freeing debt.
    """
    name = "codex-metered"
    call_backend = "codex-metered"

    @contextlib.contextmanager
    def session(self, spec: SessionSpec, directory: Path):
        from datafarm.agent.credentials import KeysFileCredentials
        from datafarm.agent.metered_provider import Meter, MeteredGateway, load_pricing
        if spec.max_budget_usd is None or spec.max_budget_usd <= 0:
            raise ConfigError('metered backend requires a finite positive session budget')
        try:  # agent.pricing: JSON object or file naming the metered model and its prices; no default
            pricing = load_pricing(self.cfg.get('pricing'))
        except (ValueError, OSError) as exc:
            raise ConfigError(f'codex-metered agent.pricing: {exc}') from None
        if pricing.model != self.model:
            raise ConfigError(f'codex-metered: agent.model {self.model!r} is not the priced model {pricing.model!r}')
        self.meter = Meter(directory / f'session-{spec.index:03d}.provider.jsonl', spec.max_budget_usd, pricing=pricing,
                           buffered=not bool(self.cfg.get("upstream_streaming", False)),
                           receipt_lookup=self.cfg.get("provider_receipt_lookup")=="openai_compatible",
                           read_timeout_s=self.cfg.get("upstream_read_timeout_s",900),
                           stream_max_s=self.cfg.get("upstream_stream_max_s"))
        creds = KeysFileCredentials(self.cfg['credentials_file'])()
        gateway = MeteredGateway(self.meter, creds.endpoint.rstrip('/')+'/v1/responses', creds.api_key)
        self.gateway = gateway
        try:
            yield
        finally:
            gateway.close()  # Finish charging requests even if the CLI was killed.
            self.receipt = {'schema':'datafarm.session-cost/1', 'complete':not self.meter.unknown,
                            'includes_descendants':True, 'cost_usd':float(self.meter.spent),
                            'cost_basis':'api-equivalent', 'provider_calls':self.meter.calls-self.meter.not_sent_calls,
                            'connection_attempts_not_sent':self.meter.not_sent_calls,
                            'denied_local':self.meter.denied, 'evidence':str(self.meter.path)}
            path = directory / f'session-{spec.index:03d}.cost.json'
            path.write_text(json.dumps(self.receipt, indent=2)+'\n')

    def command(self, spec: SessionSpec) -> list[str]:
        # No arbitrary config overrides that could reroute the metered provider.
        if self.cfg.get('extra_args'):
            raise ConfigError('metered backend does not accept arbitrary CLI overrides')
        argv = super().command(spec)
        options = ['model_provider="xh_metered"',
                   'model_providers.xh_metered.name="Trusted harness meter"',
                   'model_providers.xh_metered.base_url='+json.dumps(self.gateway.url),
                   'model_providers.xh_metered.wire_api="responses"',
                   'model_providers.xh_metered.requires_openai_auth=false',
                   'model_providers.xh_metered.request_max_retries=0',
                   'model_providers.xh_metered.stream_max_retries=0',
                   'model_providers.xh_metered.stream_idle_timeout_ms=960000',
                   'model_auto_compact_token_limit=64000',
                   'project_doc_max_bytes=0',
                   'features.responses_websockets=false',
                   'features.responses_websockets_v2=false', 'service_tier="default"']
        effort = self.cfg.get('reasoning_effort')
        if effort is not None:
            if effort not in ('low', 'medium', 'high'):
                raise ConfigError('metered reasoning_effort must be low, medium or high')
            options.append('model_reasoning_effort=' + json.dumps(effort))
        insertion = [v for item in options for v in ('-c', item)]
        argv[2:2] = insertion
        return argv

    def settle(self, parsed: ParsedSession) -> ParsedSession:
        parsed.cost_usd = self.receipt['cost_usd'] if self.receipt['complete'] else None
        parsed.cost_source = 'api-equivalent' if self.receipt['complete'] else None
        rows = [json.loads(l) for l in self.meter.path.read_text().splitlines()]
        last = rows[-1] if rows else {}
        parsed.verified_transport_interruption = bool(self.receipt['complete'] and
            (last.get('event') in ('not-sent', 'preflight-failed') or
             (last.get('event') == 'settled' and last.get('output_received') is False)))
        usage = [r['usage'] for r in rows if r.get('event')=='settled']
        parsed.cache_read_tokens = sum(u.get('input_tokens_details',{}).get('cached_tokens',0) for u in usage)
        parsed.cache_write_tokens = sum(u.get('input_tokens_details',{}).get('cache_write_tokens',0) for u in usage)
        parsed.input_tokens = sum(u['input_tokens'] for u in usage)-parsed.cache_read_tokens-parsed.cache_write_tokens
        parsed.output_tokens = sum(u['output_tokens'] for u in usage)
        return parsed


class FakeAgentBackend(AgentBackend):
    """Runs ``fakes/fake_agent.py`` (Claude stream-json output) with a JSON plan."""

    name = "fake"
    call_backend = "fake"
    uses_stdin_prompt = True

    def command(self, spec: SessionSpec) -> list[str]:
        cfg = self.cfg
        script = cfg.get("script") or str(Path(__file__).resolve().parent / "fakes" / "fake_agent.py")
        argv = [cfg.get("python", sys.executable), script, "--plan", cfg["plan"],
                "--session-index", str(spec.index), "--model", self.model]
        if spec.resume_id:
            argv += ["--resume", spec.resume_id]
        elif spec.new_session_id:
            argv += ["--session-id", spec.new_session_id]
        return argv


def make_agent_backend(agent_cfg: Mapping[str, Any], prices: Mapping[str, Any]) -> AgentBackend:
    kind = agent_cfg.get("backend")
    if kind == "claude":
        return ClaudeCliBackend(agent_cfg, prices)
    if kind == "codex-metered":
        return MeteredCodexBackend(agent_cfg, prices)
    if kind == "codex":
        codex = dict(agent_cfg.get("codex") or {})
        cfg = {**agent_cfg, **{k: v for k, v in codex.items() if k != "model"}}
        model = agent_cfg.get("model")
        if not model or str(model).startswith("claude"):
            model = codex.get("model")  # a Claude default is not a Codex model
        if not model:
            raise ConfigError("codex backend needs an explicit model: set agent.codex.model or pass --model")
        cfg["model"] = model
        return CodexExecBackend(cfg, prices)
    if kind == "fake":
        return FakeAgentBackend(agent_cfg, prices)
    raise ConfigError(f"unknown agent backend {kind!r}")


# --------------------------------------------------------------------------- runner


@dataclasses.dataclass
class RunInfo:
    exit_code: int | None
    wall_s: float
    stop_reason: str  # completed | wall_cap | <stop callback reason>
    killed: bool
    stderr_tail: str


class SessionRunner:
    """Run one agent process: stream stdout into the transcript and the parser, tick the
    broker while it runs, enforce the wall cap, terminate the process group on demand."""

    def __init__(self, *, tick: Callable[[], None] = lambda: None,
                 should_stop: Callable[[], str | None] = lambda: None,
                 tick_interval_s: float = 1.0, term_grace_s: float = 15.0) -> None:
        self.tick = tick
        self.should_stop = should_stop
        self.tick_interval_s = tick_interval_s
        self.term_grace_s = term_grace_s

    def run(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str, stdin_text: str | None,
            transcript: Path, stderr_path: Path, wall_s: float, parser: Any) -> RunInfo:
        start = time.monotonic()
        transcript.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript, "ab") as tx, open(stderr_path, "ab") as err:
            proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, cwd=cwd, env=dict(env), start_new_session=True,
                                    preexec_fn=_die_with_parent)
            lines: "queue.Queue[bytes | None]" = queue.Queue()

            def pump() -> None:
                assert proc.stdout is not None
                for raw in iter(proc.stdout.readline, b""):
                    tx.write(raw)
                    tx.flush()
                    lines.put(raw)
                lines.put(None)

            def feed_stdin() -> None:
                assert proc.stdin is not None
                try:
                    if stdin_text:
                        proc.stdin.write(stdin_text.encode("utf-8"))
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            threading.Thread(target=feed_stdin, daemon=True).start()
            stop_reason: str | None = None
            term_at: float | None = None
            killed = False
            eof = False
            try:
                while True:
                    try:
                        item = lines.get(timeout=self.tick_interval_s)
                        while True:
                            if item is None:
                                eof = True
                            else:
                                parser.feed(item.decode("utf-8", "replace"))
                            item = lines.get_nowait()
                    except queue.Empty:
                        pass
                    if eof:
                        try:
                            proc.wait(timeout=self.term_grace_s)
                        except subprocess.TimeoutExpired:
                            # stdout closed but the process lingers
                            stop_reason = stop_reason or "stdout_closed"
                            killed = True
                            _signal_group(proc, signal.SIGKILL)
                            proc.wait()
                        break
                    self.tick()
                    now = time.monotonic()
                    if stop_reason is None:
                        reason = self.should_stop()
                        if reason is None and now - start > wall_s:
                            reason = "wall_cap"
                        if reason is not None:
                            stop_reason = reason
                            term_at = now
                            _signal_group(proc, signal.SIGTERM)
                    elif term_at is not None and not killed and now - term_at > self.term_grace_s:
                        killed = True
                        _signal_group(proc, signal.SIGKILL)
            finally:
                _signal_group(proc, signal.SIGKILL)  # reap stragglers left in the group
                reader.join(timeout=5)
                if proc.poll() is None:
                    proc.wait()
        try:
            tail = stderr_path.read_bytes()[-8000:].decode("utf-8", "replace")
        except OSError:
            tail = ""
        return RunInfo(proc.returncode, time.monotonic() - start, stop_reason or "completed",
                       killed, tail)


def _die_with_parent() -> None:
    """Runs in the child before exec: if the harness dies, the agent gets SIGKILL
    (Linux ``PR_SET_PDEATHSIG``; a no-op elsewhere). bwrap adds its own --die-with-parent."""

    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except (OSError, AttributeError):
        pass


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass

