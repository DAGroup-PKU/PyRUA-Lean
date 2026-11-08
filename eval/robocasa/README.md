# eval/robocasa: RoboCasa365 evaluation of `robocasa_rgb`, zero-shot and with skill libraries

The RoboCasa365 counterpart of `eval/`. It runs the cell agent of the bundled `pyrualean` package
on RoboCasa365 tasks with the `robocasa_rgb` robot (the colour-camera primitives, SAM3 for
perception) and the `no-vla` primitive set, either with no library (zero-shot, on the same
(task, seed) cells as the PyRUA-Lean paper's RoboCasa365 runs, optionally paired cell by cell with
reference runs) or with several skill libraries ("arms") on the same tasks and seeds, paired with the episodes of
a library-free root.

| File | Role |
|---|---|
| `run.py` | `init` / `run` / `report` of an evaluation root |
| `robocasa_episode.py` | one episode in its own process; the kitchen runs in RoboCasa's own Python |
| `run_env.sh` | starts `run.py ROOT run` detached, log in `ROOT/run.log` |
| [`configs/eval/robocasa/`](../../configs/eval/robocasa) | config templates of the paper's RoboCasa365 experiments |
| `smoke_cells/` | a hand-written cell for the replay smoke (no model) |
| `patches/` | the patch robocasa needs for reproducible scenes (docs/SETUP.md, section 4) |

## Relation to `eval/robodojo/run.py`

Scheduling, ledger and budgets are the `Evaluation` class of `eval/robodojo/run.py`, imported and not
changed: at most `concurrency` episodes at once, GPU admission (`gpus`, `per_gpu`, `need_gib`,
`stagger_seconds`), infrastructure retries (`budgets.max_infra_retries`), and the pause after
`pause_after_provider_failures` episodes in a row ended by the model provider (`PAUSED.json`;
`run` again resumes). What is specific to RoboCasa365:

- A task is a RoboCasa task name (`OpenDrawer`) and its layout `n` is the scene seed
  `seeds_of(config, task)[n]`: `seed_overrides[task]` replaces `seeds` for one task (the
  PyRUA-Lean paper ran `CoffeeSetupMug` on seed 6 instead of 4), and `only_seeds[task]`
  (optional) restricts a task to some of its seeds. Every task runs every seed; no layout replaces
  another.
- The outcome is `environment_success`, the task's `_check_success` checked after every native
  action. There is no verdict record and no stages, so a failed episode is a judged (`policy`)
  episode. An episode the kitchen ended (solved, or its native-action budget spent) is judged
  whatever the planner did after it, so a provider failure then does not run it again. The model
  sees no success signal (no `done`, `task_progress` or criteria source); the kitchen ends the
  episode itself.
- The native-action budget is RoboCasa's official horizon of the task (`max_episode_steps` is
  nominal).
- Without `arms` there is no library (a library, even an empty one, adds a note to the prompt).
  With `arms` (`{"name": {"library": DIR}}`, names are identifiers other than `v0`) every task runs
  once per arm as `<arm>/<task>`, the arms of a task side by side in the order.
- GPUs: the physical GPU index is passed to the kitchen (EGL rendering) and to SAM3;
  `CUDA_VISIBLE_DEVICES` is removed from the episode environment (robosuite refuses an EGL device
  outside the mask). One root uses the GPUs of one machine; an evaluation spread over machines is
  one root per machine, each with its share of the tasks and the same settings, merged by
  `report --with`.
- The report gives the judged episodes and successes, and pairs every judged episode with the
  reference run of the same (task, seed) when the config has an optional `reference`
  (`{"label": ..., "dirs": [...]}`, directories of `<prefix>-cells-target-<Task>-s<seed>/result.json`,
  e.g. earlier runs of your own; the templates have none): the pairs where only one side succeeded
  and the two-sided exact sign test on them. With arms it adds each arm's episodes and successes
  (`arms`) and pairs each arm with `v0`, the episodes of the library-free roots given with
  `--with` (`paired_with_v0`). The merged `ours` / `paired` counts mix all arms; compare arms with
  `arms` and `paired_with_v0`.

## The identity rule

`init` freezes the config into `ROOT/config.json` with an identity hash of everything that must
not change during an evaluation: the policy code (`src/pyrualean` without `learn/` and
`evolve/`), the Python sources of `robocasa` and `robosuite` in the kitchen's interpreter
(`paths.sim_packages`, not the assets), the SAM3 checkpoint hash and the `execution` settings. It
also records the commit of `paths.lean` and whether its `src/` is clean (informational only). Each
arm's library is copied into `ROOT/libraries/<arm>/<task>/` and its hash recorded. `run` and every
episode refuse to start when the identity differs, and an episode that read another library than
`init` copied (or any library, in a root without arms) stops the run.

Libraries and `replay_dir` are not part of the identity: a root with arms has the same identity as
the library-free root of the same settings, which is what lets `report --with` pair them, and a
replay root has the same identity as a model root with the same settings. `report --with`
refuses roots of another identity and roots that share a task.

## Setup

The kitchen's interpreter is described in [`docs/SETUP.md`](../../docs/SETUP.md) (section 4): MuJoCo 3.3.1
exactly, and robocasa with [`patches/robocasa-counter-geom-order.patch`](patches/robocasa-counter-geom-order.patch),
without which the same seed does not give the same kitchen. The patch changes robocasa's sources, so it is part
of the identity below.

Fill in `paths.env` (copied from `paths.env.example`): `ROBOCASA_SIM_PYTHON`,
`ROBOCASA_SIM_PACKAGES`, `ROBOCASA_ASSETS_PATH` (with `objects/` and `fixtures/`),
`SAM3_CHECKPOINT_PATH`, `ACTOR_MODEL`, `ACTOR_BASE_URL`. The templates also name:

| Variable | Meaning |
|---|---|
| `CE_PYTHON` | the Python that runs `run.py` and the episodes: the repository's dependencies and SAM3 installed |
| `ROBOCASA_LIB_XBENCH_WO_CE`, `ROBOCASA_LIB_XBENCH_W_CE` | `heldout10-*.json`: the libraries translated to `robocasa_rgb` from your RoboDojo runs without / with criteria exposure |
| `ROBOCASA_LIB_RC10_WO_CE`, `ROBOCASA_LIB_RC10_W_CE` | `heldout10-*.json`: the libraries distilled from your runs on the 10 RoboCasa365 source tasks without / with criteria exposure |
| `SAM3_SHA256` | optional: the checkpoint's sha256 if it is not the paper's |
| `ACTOR_API_KEY` | the key of the actor endpoint, exported in your shell (never written to a config) |

A library is a directory with `lessons/*.md` and `skills/*.py`. `CE_ROOT` (set by
`tools/env.sh`) is `paths.lean`: the episodes run `src/pyrualean` of this repository, and their
caches are under `.cache/`.

```bash
source tools/env.sh                       # CE_ROOT, paths.env, caches
export CE_PYTHON=$(command -v python)     # the Python with the repository's dependencies
python tools/render_config.py configs/eval/robocasa/atomic-zero-shot.json runs/robocasa/atomic.json
```

A placeholder that is unset and has no default stops the rendering.

## Run

```bash
export ACTOR_API_KEY=...
"$CE_PYTHON" eval/robocasa/run.py ROOT init --config runs/robocasa/atomic.json
bash eval/robocasa/run_env.sh ROOT        # detached; logs ROOT/run.log and ROOT/eval.log
"$CE_PYTHON" eval/robocasa/run.py ROOT report [--with ROOT2 ...]   # writes ROOT/report.json
```

`init` checks the config (paths, assets, budgets, arms) and the SAM3 checkpoint's sha256. `run`
first checks that the endpoint answers (`base_url` with the key in `api_key_env`; for
`execution.runtime` `codex` without `base_url`, `codex login status` in `execution.codex_home`).
Running `run_env.sh ROOT` again resumes: judged episodes are not run again. `run --tasks
OpenDrawer ...` (with arms: `wo_ce/OpenCabinet ...`) restricts a run to some tasks. Each episode
writes `ROOT/runs/<task>/l<n>-a<attempt>/` (`request.json`, `episode.log`, `episode/result.json`
and the episode's `workspace/`).

Runtimes (`execution.runtime`): `openai` (any OpenAI-compatible Chat Completions endpoint:
`base_url` and `api_key_env`) and `codex` (the Codex CLI: `codex_bin` and `codex_home`, which
must hold a login when there is no `base_url`). A config with `replay_dir` runs the saved cells
of that directory instead of the model.

## Config templates

All of them use the paper's per-episode settings: `robocasa_rgb` / `no-vla`, split `target`, at
most 7200 s of policy time per episode, 8192 output tokens per request, 20 nudges, images on
demand, pure feedback, reasoning effort high; atomic tasks get 40 cells and 80 model requests,
composite tasks 100 cells and 200 model requests. The actor is `${ACTOR_MODEL:-gpt-6-astra}` at
`${ACTOR_BASE_URL}` with the key in `ACTOR_API_KEY`. They run 4 episodes at once on GPU 0 (an
episode needs about 10-11 GB of GPU memory; `need_gib` is 20); change `concurrency`, `gpus` and
`per_gpu` to your machine. `budgets.max_hours` counts from the root's first `run`.

| Template | Episodes |
|---|---|
| `atomic-zero-shot.json` | the 18 atomic tasks of Target50 x seeds 1-5 (`CoffeeSetupMug`: 1, 2, 3, 5, 6), no library, paired with the paper's no-vla runs (59/90): 90 |
| `composite-zero-shot.json` | the 32 composite tasks of Target50 x seeds 1-5, no library, paired with the paper's no-vla runs (32/160): 160 |
| `heldout10-atomic.json` | the 3 atomic held-out target tasks x seeds 10-19 x arms `wo_ce`, `w_ce`, `rc10_wo_ce`, `rc10_w_ce`: 120 |
| `heldout10-atomic-v0.json` | the same tasks and seeds, no library (`v0`): 30 |
| `heldout10-composite.json` | the 7 composite held-out target tasks x seeds 10-19 (`SteamInMicrowave`: 20 instead of 18) x the same four arms: 280 |
| `heldout10-composite-v0.json` | the same tasks and seeds, no library (`v0`): 70 |
| `smoke-replay.json` | `OpenDrawer` seed 1 with the atomic settings, no library, the cell of `smoke_cells/` instead of the model: 1 |

The empty library (`v0`) of the held-out evaluation is a root without arms, with the same settings
and so the same identity as the root of the arms; drop the arms you have no library for. Pair them
with:

```bash
"$CE_PYTHON" eval/robocasa/run.py ROOT_ARMS report --with ROOT_V0
```

The replay smoke checks the chain (kitchen boot, primitives, SAM3 `locate`, the identity) at no
model cost; its identity equals that of `atomic-zero-shot.json` rendered in the same environment.

## Tests

The tests use a fake code tree, fake kitchen packages, fake reference runs, fake episodes and fake
GPUs: no simulator, GPU or model. `eval/robocasa/tests` and `eval/robodojo/tests` both `import run`, so
they run as two pytest invocations:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider eval/robocasa/tests
PYTHONPATH=src python -m pytest -q -p no:cacheprovider eval/robodojo/tests
```
