# Command-Line Reference

## Commands

| Command | What it does |
|---|---|
| `pyrualean api` | Print the API reference the agent reads (LIBERO-PRO's robot). `--primitives no-vla` leaves the VLA out. |
| `pyrualean prompt` | Print the prompt of a single shot (`generate`); `--card` renders it for a task card. A `play` episode saves the prompt its model read as `prompt.txt` in the run directory. |
| `pyrualean play` | Run one episode with an agent ([Running Episodes](running.md)). |
| `pyrualean card` | Boot only the simulator and save a task card. |
| `pyrualean generate` | Ask the model for a whole program from a task card (single shot). |
| `pyrualean run` | Run a program file for one episode. |
| `pyrualean summarize` | Tabulate run directories ([Output Files](outputs.md)). |

## The task instance

`play`, `card` and `run` take the same options:

| Option | Meaning (default) |
|---|---|
| `--robot` | `libero`, `robotwin` or `robocasa` (`libero`). `play` also takes `robodojo_xpl_rgb`, which starts RoboDojo's own client itself (it needs `ROBODOJO_SOURCE_ROOT` and one environment with both sides); RoboDojo's official evaluation runs through XPolicyLab ([RoboDojo](../robots/robodojo.md)) |
| `--suite` | LIBERO suite, RoboTwin task config or RoboCasa split |
| `--task` | LIBERO task index, or the robot's task name |
| `--seed` | the scene seed (`0`; always pass it, since the robots' own defaults are not used) |
| `--out` | the run directory |
| `--cuda-device` | the GPU the episode's servers use |
| `--max-episode-steps` | the simulator step limit (`10000`) |
| `--rpent-root`, `--python` | the RPent checkout and the interpreter it is installed in |

## `play`

| Option | Meaning (default) |
|---|---|
| `--arm` | `cells` or `program` |
| `--runtime` | `codex`, `claude` or `openai` (`codex`; [Agent Configuration](agents.md)) |
| `--model`, `--reasoning` | the model (`gpt-6-astra`) and its reasoning effort (`xhigh`) |
| `--base-url`, `--api-key-env` | an OpenAI-compatible endpoint, and the variable holding its key |
| `--max-turns` | the decision budget, in code cells |
| `--budget-s` | the wall-clock budget, in seconds (`1200`) |
| `--images` | `on-motion`, `on-demand` or `none` (`on-motion`) |
| `--feedback` | `rich` or `pure` (`rich`) |
| `--primitives` | `full` or `no-vla` (`full`) |
| `--max-programs` | the most programs on the `program` arm |
| `--guides`, `--guides-dir` | operating guides in the agent's workspace |
| `--vla-endpoint`, `--sam3-endpoint` | running model servers to attach to |
| `--codex-home`, `--codex-bin`, `--claude-cli` | the runtimes' homes and executables |

## Environment variables

| Variable | Meaning |
|---|---|
| `RPENT_ROOT` | the RPent checkout, when RPent is not installed as a package |
| `PYRUALEAN_RUNTIME` | default of `--runtime` |
| `PYRUALEAN_PRIMITIVES` | default of `--primitives` |
| `PYRUALEAN_CODEX_BIN`, `PYRUALEAN_CODEX_HOME` | defaults of `--codex-bin`, `--codex-home` |
| `PYRUALEAN_CLAUDE_CLI` | default of `--claude-cli` |
| `PYRUALEAN_OPENAI_EXTRA_BODY` | default of `--openai-extra-body` |
| `PYRUALEAN_CODEX_RETRIES` | retries of a request from Codex (`play`, `generate`) to a `--base-url` endpoint (`10`) |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | the endpoint and key of `--runtime openai` when `--base-url` / `--api-key-env` are not given |
| `PYRUALEAN_OPENAI_RETRIES`, `PYRUALEAN_OPENAI_TIMEOUT_S` | retries (`10`) and request timeout in seconds (`900`) of `--runtime openai` |
| `CODEX_AUTO_COMPACT_TOKEN_LIMIT` | Codex's compaction threshold for the episode ([Agent Configuration](agents.md#codex)) |

`pyrualean play` passes `--runtime` and `--primitives` on only when they differ from `codex` and `full`, so a
`PYRUALEAN_RUNTIME` or `PYRUALEAN_PRIMITIVES` in the environment also wins over an explicit `--runtime codex` or
`--primitives full`: unset it to get those.

Each robot page lists the variables of its stack: checkpoints, assets and rendering.

## Development

```bash
pip install -e '.[dev]'
pytest          # offline tests against scripted backends
ruff check .
```
