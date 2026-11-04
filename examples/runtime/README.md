# Runtime examples

- `native-codex.json`: native driver for an existing Codex CLI login; the model follows the deployment settings.
- `claude-code.json`: Claude Code through `agent_cli_driver.py`; it uses your Claude Code login and `bypassPermissions`.
- `pi-solpi.json`: Pi through `agent_cli_driver.py`; it uses your Pi settings and loads SoL-Pi once `pi install` has added it.
- `sol-pi.template.json`: SoL-Pi settings with all four mechanisms on; copy it to `~/.pi/agent/sol-pi.json` and fill in the reducer model.
- `agent_cli_driver.py`: the argv driver behind those two configs.
- `phases-solpi.json`, `phases-claude-code.json`, `phases-codex.json`: harness, prices and USD caps for [phased runs](../../docs/reference/phased-runs.md). Check the prices against your provider and fill in `environment`.
- `budget-disabled.json`: cost policy whose target and limit are not acted on; usage is still recorded and unknown costs stay unknown.
- `budget-enforced.json`: strict cost policy; it needs a metered driver and cannot be used with the native CLI.
- `metered_codex_driver.py`: an example metered driver; see [cost control](../../docs/reference/cost-control.md).

## How the harnesses differ

| | Codex (native) | SoL-Pi, Claude Code (`agent_cli_driver.py`) |
|---|---|---|
| A later session | resumes the same thread | starts fresh and continues from the task's files |
| Author and reviewer identity | taken from the Codex thread | taken from the session ID in the CLI's stream |
| Output | `stdout.log` | also `agent-output.jsonl` in the session folder |

Set `/path/to/PyRUA-Lean` in the config to the absolute path of your checkout, because the driver
runs from the task directory. Anything after `--` in `argv` goes to the CLI, for example `"--model", "<model>"`.

To confirm that SoL-Pi loaded, look in a session's `pi-sessions/*.jsonl` for a `sol-pi-online-context-state-v1` entry
(it needs `onlineContextCompact`). Pi runs without any error when SoL-Pi is missing, so this is easy to overlook.

`agent_cli_driver.py` does not report cost, so the ledger records these sessions as unknown and an enforced
budget refuses them. Token usage stays in `agent-output.jsonl`.

To add another harness, add a profile to `agent_cli_driver.py`: the command, and how its JSON stream gives the
session ID and reports a failed run. Copy `claude-code.json` and change the profile name. The driver contract is in
[agent runtime](../../docs/reference/agent-runtime.md).

The optional [expert harness](../../docs/reference/expert-harness.md) starts its authoring agent itself and has its
own `agent.backend` setting, which supports Claude Code and Codex but not Pi.

The default cost policy is `configs/task-cost-policy.json`. These files configure the runtime; they do not show that a
simulator, a provider or a result works. See [quickstart](../../docs/quickstart.md) and [agent runtime](../../docs/reference/agent-runtime.md).
