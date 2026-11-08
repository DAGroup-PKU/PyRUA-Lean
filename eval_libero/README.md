# eval_libero: paired evaluation of skill libraries on LIBERO-PRO

The LIBERO-PRO counterpart of `eval/`. It runs several skill libraries ("arms"), for example the
empty library and a library translated from RoboDojo, on the same LIBERO-PRO tasks and seeds,
and pairs their outcomes. The robot is `libero_rgb` (the colour-camera primitives of the bundled
`pyrualean` package) with the `no-vla` primitive set: no Pi0.5 server is started, only RPent's
LIBERO environment server and SAM3.

| File | Role |
|---|---|
| `run.py` | `init` / `run` / `report` of an evaluation root |
| `libero_episode.py` | one episode in its own process, in the RPent LIBERO Python |
| `run_env.sh` | starts `run.py ROOT run` detached, log in `ROOT/run.log` |
| `configs/` | config templates of the paper's LIBERO-PRO experiments |
| `libraries/empty/` | the empty library (`lessons/none.md`, an empty file) |

## Relation to `eval/run.py`

Scheduling, ledger and budgets are the `Evaluation` class of `eval/run.py`, imported and not
changed: at most `concurrency` episodes at once, GPU admission (`gpus`, `per_gpu`, `need_gib`,
`stagger_seconds`), infrastructure retries (`budgets.max_infra_retries`), and the pause after
`pause_after_provider_failures` episodes in a row ended by the model provider (`PAUSED.json`;
`run` again resumes). What is specific to LIBERO:

- A task is `<arm>/<suite>-t<index>` and its layout `n` is the LIBERO seed `seeds[n]`. Every task
  runs every seed; LIBERO has no unstable scenes, so no layout replaces another.
- The outcome is `environment_success`, LIBERO's task predicate checked after every simulator
  step. There is no verdict record and no stages, so a failed episode is a judged (`policy`)
  episode, not infrastructure.
- The order interleaves the arms task by task (`v0/t5`, `wo_ce/t5`, `w_ce/t5`, `v0/t6`, ...), so
  the arms of a task run side by side.
- GPUs: the physical GPU index is passed to RPent's servers, which pin it themselves;
  `CUDA_VISIBLE_DEVICES` is removed from the episode environment (a mask would renumber it).
- The report gives, per arm, the judged episodes, successes and missing episodes, and pairs each
  arm with the first one by (task, seed): the pairs where only one arm succeeded and the two-sided
  exact sign test on them.

## The identity rule

`init` freezes the config into `ROOT/config.json` with an identity hash of everything that must
not change during an evaluation: the policy code (`src/pyrualean` without `learn/` and
`evolve/`), RPent's `robots/libero` and `rpent/`, the suite's BDDL and init files, the SAM3
checkpoint hash and the `execution` settings. Each library is copied into
`ROOT/libraries/<task>/` and its hash recorded. `run` and every episode refuse to start when the
identity differs, and an episode that read another library than `init` copied stops the run.
`replay_dir` is not part of the identity, so a replay root has the same identity as a model root
with the same settings.

## Setup

Fill in `paths.env` (copied from `paths.env.example`): `RPENT_ROOT`, `LIBERO_PYTHON`,
`LIBERO_PRO_ASSET_PATH`, `SAM3_CHECKPOINT_PATH`, `ACTOR_MODEL`, `ACTOR_BASE_URL`. The templates
also name:

| Variable | Meaning |
|---|---|
| `LIBERO_PRO_ROOT` | the `liberopro/liberopro` directory of the LIBERO-PRO package (it holds `bddl_files/` and `init_files/`) |
| `LIBERO_LIB_WO_CE`, `LIBERO_LIB_W_CE` | the libraries translated to `libero_rgb` from your RoboDojo runs without / with criteria exposure (directories with `lessons/*.md` and `skills/*.py`) |
| `LIBERO_REPLAY_DIR` | `smoke-replay.json` only: a directory of `cell_NNN.py` files |
| `SAM3_SHA256` | optional: the checkpoint's sha256 if it is not the paper's |
| `ACTOR_API_KEY` | the key of the actor endpoint, exported in your shell (never written to a config) |

```bash
source tools/env.sh                       # CE_ROOT, paths.env, LIBERO_CONFIG_PATH, caches
export LIBERO_PRO_ROOT=$("$LIBERO_PYTHON" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("liberopro").origin).parent / "liberopro")')
mkdir -p "$LIBERO_CONFIG_PATH"            # LIBERO-PRO writes its default config.yaml there
export LIBERO_LIB_WO_CE=/path/to/library-wo-ce LIBERO_LIB_W_CE=/path/to/library-w-ce
python tools/render_config.py eval_libero/configs/xbench-10swap-t569.json runs/libero/xbench-10swap.json
```

A placeholder that is unset and has no default stops the rendering.

## Run

```bash
export ACTOR_API_KEY=...
"$LIBERO_PYTHON" eval_libero/run.py ROOT init --config runs/libero/xbench-10swap.json
bash eval_libero/run_env.sh ROOT          # detached; logs ROOT/run.log and ROOT/eval.log
"$LIBERO_PYTHON" eval_libero/run.py ROOT report   # writes ROOT/report.json
```

`run` first checks that the endpoint answers (`base_url` with the key in `api_key_env`; for
`execution.runtime` `codex` without `base_url`, `codex login status` in `execution.codex_home`).
Running `run_env.sh ROOT` again resumes: judged episodes are not run again. `run --tasks
v0/libero_10_swap-t5 ...` restricts a run to some tasks. Each episode writes
`ROOT/runs/<task>/l<n>-a<attempt>/` (`request.json`, `episode.log`, `episode/result.json` and the
episode's `workspace/`).

Runtimes (`execution.runtime`): `openai` (any OpenAI-compatible Chat Completions endpoint:
`base_url` and `api_key_env`) and `codex` (the Codex CLI: `codex_bin` and `codex_home`, which
must hold a login when there is no `base_url`). A config with `replay_dir` runs the saved cells
of that directory instead of the model.

## Config templates

All of them use the paper's per-episode settings: `libero_rgb` / `no-vla`, at most 40 cells, 80
model requests and 7200 s of policy time per episode, 8192 output tokens per request, 20 nudges,
images on demand, pure feedback, reasoning effort high, 10000 simulator steps; the actor is
`${ACTOR_MODEL:-gpt-6-astra}` at `${ACTOR_BASE_URL}` with the key in `ACTOR_API_KEY`. They run 4
episodes at once on GPU 0 (an episode needs about 25 GB of GPU memory); change `concurrency`,
`gpus` and `per_gpu` to your machine.

| Template | Episodes |
|---|---|
| `xbench-10swap-t569.json` | `libero_10_swap` tasks 5, 6, 9 x seeds 0-9 x arms `v0` (empty library), `wo_ce`, `w_ce`: 90 |
| `xbench-goalswap-t159.json` | `libero_goal_swap` tasks 1, 5, 9 x seeds 0-9 x the same three arms: 90 |
| `smoke-replay.json` | `libero_10_swap` task 5 seed 0, `v0`, the cells of `LIBERO_REPLAY_DIR` instead of the model: 1 |

A config holds one suite, so the paper's six LIBERO-PRO tasks are the two `xbench-*` roots
together. For the replay smoke, `LIBERO_REPLAY_DIR` can be the `workspace/` of an episode of the
same task and seed (`ROOT/runs/v0/libero_10_swap-t5/l0-a0/episode/workspace`) or cells written by
hand; the smoke makes no model request. A replay stops at the first cell that raises, where the model
would have read the error and gone on: with the cells of a model episode it may end early ("replayed 4 of
12 cells") and fail although that episode succeeded.

## Tests

The tests use a fake RPent/LIBERO tree, fake episodes and fake GPUs: no simulator, GPU or model.
`eval_libero/tests` and `eval/tests` both `import run`, so they run as two pytest invocations:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider eval_libero/tests
PYTHONPATH=src python -m pytest -q -p no:cacheprovider eval/tests
```
