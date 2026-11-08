# Models and credentials

The paper's actor is **GPT-6 Astra** (reasoning effort high, at most 8,192 output tokens per
request) and its learner is **Claude Code with Claude Opus 5.5** (effort high). Keys are never
written into a config or a run directory: a config names the environment variable that holds a
key, and the coordinators read it at `run`. Learner sessions never see the actor's key.

## Actor

### OpenAI-compatible endpoint (`execution.runtime: openai`, the default)

Any Chat Completions endpoint with tool calling and image input:

```bash
# paths.env
ACTOR_MODEL=gpt-6-astra
ACTOR_BASE_URL=https://api.openai.com/v1      # or your provider / gateway
# your shell
export ACTOR_API_KEY=...
```

The templates set `"api_key_env": "ACTOR_API_KEY"`, `"extra_body": {"reasoning_effort": "high"}`
and the per-episode limits of the paper (cells, requests, nudges, wall clock). `check-services`
(learn) and the start of `eval/run.py` send one short request to confirm the endpoint answers.
Proxies follow your environment; a config's `environment` block can set or unset (`null`)
variables such as `HTTPS_PROXY` for its episodes.

### Codex CLI (`execution.runtime: codex`)

The episode runs `codex exec` with the episode's MCP tools only (`python`, `finish`): every
built-in Codex tool is switched off in [`tools/codex-home/config.toml`](../tools/codex-home/config.toml),
and `tools/codex-mcp-only` replaces the `workspace-write` sandbox `pyrualean.play` asks for by
`read-only`.

```bash
mkdir -p .cache/codex-home && cp tools/codex-home/config.toml .cache/codex-home/
CODEX_HOME=$PWD/.cache/codex-home tools/codex-mcp-only login        # your own login
```

then in the config's `execution`: `"runtime": "codex"`, `"codex_bin": "${CE_ROOT}/tools/codex-mcp-only"`,
`"codex_home": "${CE_ROOT}/.cache/codex-home"`, `"model": "gpt-6-astra"`, and either
`"base_url": null, "api_key_env": null` (the home's login) or an endpoint and key variable. Codex
adds its own base instructions and environment context to the prompt, so keep results of the two
runtimes apart.

## Learner (Claude Code)

`worker.command` is `${CE_ROOT}/tools/claude-clean`: the `claude` CLI (`CLAUDE_REAL_BIN`, default
the one on `PATH`) with its own config directory (`CE_CLAUDE_HOME`, default `.cache/claude-home`),
so that none of your settings, CLAUDE.md, memory, skills or plugins reach the learner. The
campaign also runs it with `--setting-sources project,local` and the config's `worker.settings`.
Three ways to give it access:

1. **Its own login**: `tools/claude-clean auth login` once; no key variable in the config.
2. **An Anthropic API key**: `export ANTHROPIC_API_KEY=...`, which the CLI reads; or name another
   variable in `worker.api_key_env`, whose key the campaign hands to the CLI as
   `worker.token_env` (default `ANTHROPIC_API_KEY`), dropping the CLI's other credential variable.
3. **An Anthropic-compatible gateway**: in `worker.settings.env` set `ANTHROPIC_BASE_URL` (and,
   if the gateway names the models differently, `ANTHROPIC_DEFAULT_OPUS_MODEL` with
   `"model": "opus"`); set `worker.api_key_env` to the variable that holds the gateway token and
   `"token_env": "ANTHROPIC_AUTH_TOKEN"`.

The model is `worker.settings.model` (templates: `${LEARNER_MODEL:-claude-opus-5-5}`), the effort
`worker.effort` (`high`). `check-services` runs the CLI exactly as a worker does and prints which
model answered.

The learner's tools are limited by allow and deny rules built per session
(`pyrualean.learn.worker.command`): it reads its session, the task's development runs and the
robot API source (and, w/ CE, the task's criteria), writes only its candidate library, its
verification rollouts and its notes, and may run only the replay and check commands. These rules
are not an operating-system sandbox; run campaigns under an account that holds nothing else.

## Distillation and translation sessions

`transfer/` runs one Claude Code session per library the same way (the CLI's own login or
`ANTHROPIC_API_KEY`); see [`transfer/README.md`](../transfer/README.md).
