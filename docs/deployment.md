# Deployment bindings

The code never hardcodes where simulators, models, data or credentials live. Every external
location is a **runtime binding**. A binding defaults to a path under the workspace (the checkout, or `DATAFARM_WORKSPACE`) and can
be overridden in `bindings.json` in the checkout (or the file named by `DATAFARM_BINDINGS`), or per key with
the environment variable `DATAFARM_<KEY>`. Values are paths, never secrets. Tools that need a
binding fail closed with a message naming it. They do not fall back to a guess.

Implementation: `src/datafarm/runtime.py::defaults`, `::load_bindings`, `::expand`
(configs use `${KEY}` placeholders). The H3 control hybrid uses its own fail-closed resolver,
`src/datafarm/render/h3/h3_paths.py`, which reads only explicit environment variables or
`bindings.json` entries: apart from `H3_PYTHON` its keys have no workspace default.

```bash
pip install -e '.[dev,workflow]'                    # from the source checkout (distribution pyrua-datafarm)
$EDITOR bindings.json                               # point bindings at your installations
datafarm doctor [--workspace DIR] --profile core    # no API or GPU calls
```

`doctor` profiles: `core`, `seedance`, `training`, `h3-control`, `h3-ood`, `usdcraft`.

A minimal `bindings.json` holds path strings for the keys you use:

```json
{
  "TRAINING_PYTHON": "/opt/envs/training/bin/python",
  "OPENPI_CACHE": "/data/openpi-cache",
  "DATASETS": "/data/lerobot",
  "GPU_SCHEDULER": "/opt/bin/gpu-scheduler",
  "H3_MODELS": "/data/h3-models"
}
```

## Bindings

| Key | Default (under the workspace `<ws>`) | Used for |
|---|---|---|
| `WORKSPACE` | `<ws>` | root of all defaults |
| `TRAINER` | `<ws>/vendor/trainer` | vendored openpi trainer |
| `GPU_SCHEDULER` | `<ws>/external/gpu-scheduler` | executable the deployment supplies; see [GPU scheduling](#gpu-scheduling) |
| `ROBOTWIN`, `ROBOTWIN_PYTHON` | `<ws>/external/robotwin`, `<ws>/environments/robotwin/bin/python` | SAPIEN tasks via RoboTwin 2.0 |
| `ISAAC_PYTHON`, `ISAAC_KIT`, `ISAAC_ENV` | `<ws>/environments/isaac/...` | Isaac Sim |
| `ANALYSIS_PYTHON`, `ANALYSIS_ENV` | current interpreter | CPU analysis tools |
| `TRAINING_PYTHON`, `OPENPI_CACHE`, `DATASETS` | `<ws>/environments/training/bin/python`, `<ws>/external/openpi-cache`, `<ws>/data/lerobot` | pi0.5 training |
| `MODELS`, `EXTERNAL` | `<ws>/external/models`, `<ws>/external` | model weights and other external checkouts |
| `H3_PYTHON` | `<ws>/environments/h3/bin/python` | H3 runtime (obtained separately) |
| `H3_MODELS`, `H3_VIDEOX_FUN`, `H3_TURBO_LORA` | none for the H3 harness (set explicitly) | H3 control hybrid; `H3_VIDEOX_FUN` is VideoX-Fun with `vendor/videox_fun/videox-fun-datafarm.patch` applied |
| `H3_EVAL_DINO`, `H3_SAM2` | none for the H3 harness (set explicitly) | frozen render evaluation; OOD masks |
| `H3_HED_ANNOTATOR_REPO` | none for the H3 harness (set explicitly) | ControlNet v1.1 HED annotator (required by the soft-HED tool) |
| `EPISODES` | `<ws>/data/episodes` | real input episodes |
| `TRAINING_RUN` | `<ws>/runs/training` | training outputs |
| `ARK_CREDENTIALS`, `AGENT_CREDENTIALS` | `<ws>/secrets/*.env` | credential **files**, mode 0600; read by explicit keys-file loaders, never placed in argv |

## Settings without defaults

These have no default on purpose. A missing value stops the tool before any call is made.

| Setting | Where | Meaning |
|---|---|---|
| `DATAFARM_SEEDANCE_UPLOAD_HOST` env, or `upload_host` in a Seedance batch config | `src/datafarm/render/seedance/client.py::upload_host` | HTTPS base URL of a file host that serves the carrier video to Seedance. **Opting in publishes real-scene media at a URL until it expires.** |
| `agent.model` and, for `codex-metered`, `agent.pricing` | expert-harness adapter; `src/datafarm/expert/session.py` | The coding model, and its price table: `{"model", "source", "usd_per_million_tokens": {"input", "cached_input", "cache_write", "output"}, "long_context"?}`. |
| `--pricing` / `DATAFARM_METERED_PRICING` | `examples/runtime/metered_codex_driver.py` | Same price-table format, for the example metered driver. |
| `H3_EVAL_DINO` / `DATAFARM_H3_EVAL_DINO` | `evaluations/render-h3/h3eval.py` | DINOv2 directory for `realpool`/`score`. |

## Coding-agent drivers

`tools/pipeline_agent.py supervise --config <runtime.json>` drives the agent.
- **Native Codex CLI** (`{"driver": {"type": "codex"}}`): the deployment must provide
  `codex exec --json` and `codex exec resume` with an existing login. The runtime never reads
  or copies credentials.
- **Any other agent**: implement the argv driver protocol `datafarm.agent-identity/1`, described
  in [agent runtime](reference/agent-runtime.md).

Budget policies are in `examples/runtime/` and described in
[cost control](reference/cost-control.md).

## GPU scheduling

The release ships no GPU scheduler. The deployment supplies an executable with `submit` and `leases` subcommands, and tools receive its path through the `GPU_SCHEDULER` binding (default `<ws>/external/gpu-scheduler`). A thin wrapper around Slurm, Kubernetes or an SSH lease script is enough. A single-node setup can avoid it: the expert harness also has a `local` rollout backend.

Three things call the scheduler: `tools/launch_pi05_training.py` (training jobs), the expert harness `command` rollout backend through the adapter's `rollout.submit.argv` (`src/datafarm/expert/rollout.py`), and the H3 workers (`src/datafarm/render/h3/launch_control_worker.sh`, `h3_requeue_orphans.py`).

### `submit`

```text
gpu-scheduler submit [--host HOST]... --gpus N [--min-free-mb MB] [--max-util PCT] \
                     --name NAME [--cwd DIR] [--] COMMAND [ARG...]
```

| Option | Requirement |
|---|---|
| `--gpus N` | Allocate N GPUs, make them visible to the job and start the command in the background. `submit` returns once the job has started; it does not wait for the job to finish. |
| `--min-free-mb MB`, `--max-util PCT` | Use only GPUs with at least this much free memory and at most this utilization. |
| `--name NAME` | Job name, reported back by `leases`. |
| `--cwd DIR` | Working directory of the command. |
| `--host HOST` | Repeatable. Restrict placement to these hosts. The H3 workers always pass it, from `<run>/queue/ALLOWED_HOSTS`. |

- The command starts after the options, optionally behind `--`. `tools/launch_pi05_training.py` omits the separator; the harness adapter template and the H3 workers use it.
- Exit 0: the job was started. Exit 2: no matching GPU is free and nothing was started; `tools/launch_pi05_training.py` then tries its next GPU count. Any other nonzero exit is a failure.
- On success, print `key=value` lines. The expert harness records `id`, `host`, `gpus`, `pid` and `log` in the attempt receipt.
- `submit` must not rely on the caller's environment: the H3 launcher passes every variable it needs inside the command.
- A slow `submit` can be killed by its caller (120 s by default in the harness). A job started by a killed `submit` must still appear in `leases`.

### `leases`

`leases` prints one lease per line in whitespace-separated columns, and is read by `h3_requeue_orphans.py`: column 1 is the lease ID, of the form `<name>-YYYYMMDD-HHMMSS-<hex>`; column 3 is the host; column 5 is the process ID of the job.

A failed query must exit nonzero, not print an empty table: an unknown lease table is treated as unknown, whereas an empty one makes a caller act on a running job. `h3_requeue_orphans.py` returns the tickets of workers that are not listed to the queue, so the listing has to be complete.

