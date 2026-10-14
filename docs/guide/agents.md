# Agent Configuration

Pick who runs the agent loop with one flag:

```bash
pyrualean play --runtime {codex,claude,openai} ...
```

Every runtime gets the same prompt, the same `python` (or `run_program`) and `finish` tools and the same budgets,
and the episode is judged the same way. They differ in which product drives the conversation and what it adds:
Codex keeps its own shell, sandboxed to the workspace; Claude Code its read-only `Read`, `Glob` and `Grep`; the
`openai` loop adds nothing.

| `--runtime` | What it is | When to pick it |
|---|---|---|
| `codex` (default) | The [Codex CLI](https://github.com/openai/codex) runs the loop; PyRUA-Lean serves it the tools. | GPT models, with a Codex login or an OpenAI Responses endpoint. |
| `claude` | Claude Code, through the Claude Agent SDK. | Claude models. |
| `openai` | A plain loop in PyRUA-Lean against any OpenAI-compatible chat-completions endpoint: a gateway, or a local SGLang or vLLM server. | Open models such as [PyRUA-Lean Coder](../models/coder.md); RoboDojo. |

`--model` names the model as the endpoint knows it. `--reasoning` sets the reasoning effort of Codex and Claude Code
(default `xhigh`); with `openai`, put it in `--openai-extra-body`.

## Codex

```bash
npm install -g @openai/codex@0.155.1
```

- Without further options the code arm uses whatever provider the Codex home is logged in to.
- `--base-url https://host/v1 --api-key-env MY_KEY` sends it to an endpoint instead (the key is read from the
  variable `MY_KEY`). Codex speaks the Responses API there; a server that serves only chat completions needs
  `--runtime openai`.
- For a run nothing else may leak into, give it a clean Codex home with `--codex-home`: a directory with only the
  credentials and a `config.toml` that sets `approval_policy = "never"` and `memories = false` under `[features]`.
  `--codex-bin` should point at the real Codex binary, not a wrapper.
- Long episodes: with `gpt-6-astra` Codex assumes a 272k-token window and compacts the conversation at a threshold
  derived from it. The paper's runs set `model_auto_compact_token_limit = 945000` in the Codex home's `config.toml`
  (or `CODEX_AUTO_COMPACT_TOKEN_LIMIT=945000`), so no episode was compacted; `result.json` records the setting
  (`host.codex_context`) and any compaction (`generation.compactions`).

!!! tip "GPT-6 Astra on Codex 0.155.1"

    Codex 0.155.1 runs `gpt-6-astra` in "code mode", which hides every tool behind one JavaScript tool. PyRUA-Lean
    passes Codex a model catalog that turns this off (direct function calls, as in the paper's runs), so nothing needs
    to be done for `play` and `generate`.

## Claude Code

```bash
pip install claude-agent-sdk          # into the robot stack's environment
pyrualean play --runtime claude --model <claude model> --claude-cli /path/to/claude ...
```

Claude Code reaches the model the way it is configured to (`ANTHROPIC_*` variables). Opus 5.5 needs Claude Code
2.1.280 or newer. `--reasoning none` turns thinking off.

## OpenAI-compatible endpoints

```bash
pyrualean play --runtime openai --base-url http://127.0.0.1:30000/v1 --api-key-env MY_KEY \
    --model <model name> ...
```

Without `--base-url` and `--api-key-env`, the loop reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

| Option | Meaning (default) |
|---|---|
| `--openai-max-tokens` | the longest reply (`4096`) |
| `--openai-extra-body` | JSON merged into every request, e.g. `{"reasoning_effort": "high"}` |
| `--openai-nudges` | replies without a tool call tolerated before the episode ends (`2`) |
| `--openai-tool-choice` | `auto` or `required` (`auto`) |
| `--openai-temperature`, `--openai-timeout-s` | sampling temperature, request timeout (`900` s) |
| `--image-max-side` | downscale images to this longest side (`0`: keep) |

A rate-limited endpoint (HTTP 429) is waited out, not counted as a failure.
