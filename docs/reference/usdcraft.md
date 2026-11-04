# Object assets with USDCraft

DataFarm needs simulation assets for objects seen in a real episode. [USDCraft](https://github.com/NVlabs/USDCraft) ([paper](https://arxiv.org/abs/2610.11322)) lets a coding agent write an object as a Python program, then exports it as a jointed URDF and USD model. It is optional and lives in its own repository; the adapter calls its command line and imports none of its code.

## Install

USDCraft needs Linux or macOS, [`uv`](https://docs.astral.sh/uv/), Python 3.12 (`uv` provides it) and Node.js ^20.19 or >=22.12. Keep it in its own environment, apart from the simulator. The adapter was tested with commit `46fec18`:

```bash
git clone https://github.com/NVlabs/USDCraft.git && cd USDCraft
git checkout 46fec18
uv sync --frozen
uv run --frozen usdcraft setup
```

The adapter uses USDCraft's Codex workflow, so an authenticated `codex` must be on `PATH`, or named by `USDCRAFT_CODEX_BIN`. It does not use USDCraft's API-key workflow. Set the `USDCRAFT` binding to the checkout, then check the installation:

```bash
datafarm doctor --profile usdcraft
```

`doctor` checks `uv`, `codex` and that `usdcraft/cli.py` is the tested version. `plan.json` records the checkout's commit in `upstream`; another commit is allowed, and `tested` shows it was not the one tested here.

## Describe the actual instance

Copy `examples/usdcraft.json` into your deployment configuration. Its `checkout` uses the `USDCRAFT` binding, and an absolute path also works. Observation paths should be absolute, so a supervised process sees the same files. Save an observation JSON:

```json
{
  "object_id": "observed-cabinet",
  "image": "/data/episode/evidence/cabinet-front.png",
  "description": "The observed cabinet has one left-hinged door and a recessed handle. Preserve its visible proportions and the handle contact space. Width 0.45 m is inferred from the robot reference, not measured. Interior shelves are unobserved; do not add task-critical hidden structure.",
  "context_files": ["/data/episode/evidence/cabinet-open.png", "/data/episode/evidence/object-observations.json"]
}
```

The context document should identify the source video, view, frame or time, crop coordinates, scale source, observed articulation and unresolved properties. Only `image` is passed to the single-image CLI; additional files are provenance for later comparison. With a corresponding mesh add `mesh`, and optionally `up_axis`, `front_axis` and `meters_per_unit`. These select Reconstruction; without a mesh the route is Generation. Keep external mesh textures and buffers together. Image generation is an inferred asset, not evidence that real hidden geometry or mass has been recovered.

```bash
python tools/build_usd_asset.py plan --config /data/deployment/usdcraft.json \
  --observation /data/task/objects/cabinet/observation.json \
  --output /data/task/objects/cabinet/build-001
python tools/build_usd_asset.py run --work /data/task/objects/cabinet/build-001 --task-work /data/task
```

`plan` does not call a model. `run` executes the recorded argv without a shell, using the `generation-codex` or `reconstruction-codex` interface with `--direct --keep-workspace --json`, and retains failed output. For a long build, launch the run command through the run ledger and monitor its worker and output; do not start it repeatedly from a polling agent. An optional timeout terminates the subprocess group; none is imposed by default.

`--task-work` binds the build to a task: it reserves a persistent external-usage ID in the task's cost ledger, records unknown descendant cost (never zero), and attaches a `status_file`-backed `external_jobs` entry. The supervisor wakes after transport finishes. Without it, a standalone run makes no claim of task budget coverage.

## Guarantees

1. A build routes to Reconstruction when the observation names a `mesh`, otherwise to Generation, and hashes every input. A changed input refuses to run.
2. Once `execution.json` exists, a build is never re-submitted: a failed, interrupted or unknown execution is returned as is. A concurrent caller fails with `BuildLocked`.
3. In the task-bound form, the build lock is held across job registration, budget reservation, execution and status writes. An enforced task budget is refused before launch.
4. An export is bound only if the tool reported `<mode>_ready` from `codex-cli`, the model file lies inside the canonical record (a relative `model_path` resolves against the checkout), the URDF has links, every mesh is a relative path inside the materialization, and the export closure has no external symlinks. The result status is `exported-unqualified`, never task complete. `build_usd_asset.py run` exits 0 only with that status.

Code: `src/datafarm/simulation/usdcraft.py`, `tools/build_usd_asset.py`.

## Data model

- **Deployment config** (`examples/usdcraft.json`): `checkout`, `model`, `reasoning`, `launcher` (default `["uv", "run", "--frozen", "usdcraft"]`), `timeout_seconds` (default `null`). `model` and `reasoning` have no default.
- **Observation:** `object_id` (lowercase slug), `image`, `description`, optional `mesh`, `up_axis` and `front_axis` (`±X/±Y/±Z`), `meters_per_unit` (>0), `context_files`.
- **Build directory:** `plan.json` (`datafarm.usdcraft-plan/1`: mode, argv, input hashes, `qualification: unverified`, cost `unknown`), `execution.json` (status `starting` → `running` → `returned`/`failed`/`interrupted` → `exported-unqualified` or `invalid-export`), `stdout.json`, `stderr.log`, `asset.json` (`datafarm.object-asset/1`: record ID, URDF, artifact hashes, joints, `required_next_step`), `.lock`. Task-bound runs add `job-status.json` and an `external_jobs` entry (`kind: observed-object`) in the task's `state.json`.

## Using the asset

Load the URDF in the target simulator and map names, frames, units, the fixed-base choice and relevant joint ranges into the generated task scene. Do not assume USD physics settings transfer between simulators. Validate actual joint motion, collision cavities, grasp and contact surfaces, support and settling. Compare new simulator images with real views and retain evidence and import settings.

`exported-unqualified` only means the asset is ready for inspection. The task author collects simulator qualification probes, then queues an independent review through `state.review_requests` with source hashes. Independent grounding review remains required. Nothing is adopted or physically certified automatically.

## Failure behaviour

Changed inputs require a new versioned plan. A non-zero exit is kept as `failed` with stdout and stderr. An invalid export is kept as `invalid-export` with the reason. An interrupted or stale `starting` or `running` record is returned unchanged and needs an operator to reconcile. A job ID in `external_jobs` without an execution record is refused ("reconcile before resubmission").

## Limitations

- At the tested commit, Generation has run end to end through the adapter with Codex, and SAPIEN loaded the exported URDF with its joints. Reconstruction has been checked only against the command-line options, not run, and Isaac Sim import has not been tested.
- Exported assets are never simulator-qualified by the adapter, and generation infers geometry.
- Only `image` reaches the single-image CLI; `context_files` are hashed for provenance only.
- Costs stay `unknown` until upstream call records are reconciled with the task's budget policy, and enforced budgets cannot be met. The adapter neither enforces a dollar cap nor treats native login as zero-cost usage.
- Loading `model.py` executes Python, so the adapter collects metadata without importing it. Package or remote mesh URIs need an explicitly implemented resolver.
- `build_lock` is an advisory `flock` on `<build>/.lock`, which some network filesystems do not honour across hosts.
