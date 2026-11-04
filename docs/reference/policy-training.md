# Policy training (pi0.5)

Training is downstream of admission. `tools/launch_pi05_training.py` (`src/datafarm/training/launch.py`) turns a JSON
training spec, started from `configs/training/template.json`, into a launch manifest and a scheduler command
for the vendored openpi trainer (`vendor/trainer/`). First it validates the dataset, the action representation, the state
conditioning mode, the distribution claim and the admission binding. It writes nothing
launchable unless every check passes. Two action families exist:

- **EEF**: `absolute_eef` (what `configs/training/template.json` starts from and what
  `tools/build_pi05_dataset.py --task <instruction> --expected-episodes <N>` builds), and `delta_eef_current_rpy_v1` (`delta_eef.py`). The
  builder keeps every source frame; `--excise-recovery` drops deliberate misaligned-approach recovery detours, whose
  revisited joint state gives a state-free policy two futures for one observation.
- **Joint** (`joint_actions.py`): `delta_joint_current_v1` and `absolute_joint_v1`. With `absolute_joint_v1` the
  model predicts absolute joint targets and ignores the request state. These modes exist in the launcher, but
  they need a dataset with a joint `state_action_contract` and the joint source receipts below, and no shipped
  tool builds such a dataset; you must supply your own.

## Input contract

For the dual-arm PiPER policy family, one training sample contains:

- three independent `224 x 224` RGB observations: head, left wrist, and right
  wrist;
- an explicitly declared state-conditioning mode, `none`;
- a 14-dimensional action in the declared semantic order and the admitted temporal
  convention;
- one task prompt and an action horizon of 50.

Adding joint proprioception to a dataset whose state is EEF-only requires a new, versioned state schema and
a train/inference parity check; it must not be silently appended.

Padding state and action to the model width of 32 is an implementation detail.
It does not change the semantic dimension. Composite review videos such as a
T-layout carrier are transport artifacts; they must be decoded into
the three camera tensors before training.

PI0.5 can consume normalized state by discretizing it into 256 bins and placing the result in the language
prefix. This launcher does not use that path: every launch declares `state_conditioning: none`, and the model
flag and CLI flag must agree. Any other value is refused.

## Why state conditioning is explicit

In the PI0.5 tokenizer, `discrete_state_input=False` passes `state=None` to the tokenizer, and PI0.5 has no continuous suffix state token. A dataset can therefore declare a 14-D `observation.state` that the loader reads, normalizes and pads to `(batch, 32)`, while the model is conditioned only on images and prompt. The model configuration defaults `discrete_state_input` to true when no override is supplied, so the false value is a project-specific override and not the upstream default. Checkpoints trained this way are not proprioceptive.

## Guarantees

1. **A launch needs a verified admission certificate.**
   `train-command` verifies `--admission-certificate`: the stored document, a passing
   admission-root receipt whose verifier differs from the producer and no rejecting root receipt, a fresh
   re-derivation of the admitted set from the stored evidence, and the binding to this dataset's
   provenance, allowlist, members, repo, waivers and files. Every other launch raises `AdmissionRequired`.
2. **The state conditioning mode is explicit and consistent.** `state_conditioning` must be
   `none`, and the command passes `--model.no-discrete-state-input`.
3. **The model interface is fixed.** The command always passes `--model.action-dim 32`,
   `--model.action-horizon 50` and `--model.pi05`. The audit requires these values, 14-D
   native actions, and exactly three views `cam_high`, `cam_left_wrist` and
   `cam_right_wrist`. The dataset must expose a 14-D `observation.state` and `action` and
   all three image features.
4. **The dataset is bound to an accepted distribution and is fully generated.** The spec
   must use `datafarm.pi05-finetune/1.1.0` and declare `target_task` and a
   `required_distribution_claim`:
   - `broad_deployment` requires `training_scope: production_policy`;
   - `local_robustness` is exploratory only, with an authorization ID.

   `distribution_acceptance` must match its sha256, and the dataset provenance must name
   that receipt. The provenance must also:
   - be `training_eligible`;
   - pin a semantic allowlist and its digest;
   - mark every episode `image_source: generated`;
   - match the spec's `allowed_semantic_waivers`, `repo_id` and `dataset_fps` (default 30).
5. **The action representation is declared and matches the trainer config.** `action_representation` is
   required and must agree with the `base_config` suffix (`_delta_eef_current_rpy_v1`,
   `_delta_joint_current_v1`, `_absolute_joint_v1`, or none). The command runs `src/datafarm/training/run_trainer.py` with
   `--action-adapter delta-eef`, `joint` or `none`. The joint adapter registers both joint configs for every
   Piper base config, without editing the trainer checkout.
6. **Joint actions have exact, declared semantics.** The dataset stores absolute joint
   targets (`absolute_next_joint_current_gripper`, 14 fields `left_joint_1..6,
   left_gripper, right_joint_1..6, right_gripper`, radians for the 12 arm joints, one
   gripper unit `m` [0, 0.105] or `normalized_0_1`).
   - `delta_joint_current_v1` subtracts the current measured joint state from the 12 arm
     joints, using one anchor per chunk.
   - `absolute_joint_v1` passes targets through unchanged (shapes are still validated).
   - Grippers stay absolute in both modes.
   - `model_representation` must match the mode.
7. **Joint labels come from simulator readback or an audited drive target.** The dataset
   contract must declare `action_joint_t` as either:
   - the next retained frame's joint readback, or
   - the joint *drive target* (for exports without readback), which must carry a
     `joint_tracking_audit` object.

   Grippers must be the current frame's expert command, and field order and units must
   equal the launch contract.
8. **Joint launches bind exact source evidence.** The spec must bind, by sha256:
   - the dataset provenance;
   - an action-conversion certificate file (and its store);
   - a terminal joint readback manifest covering every episode with exact final joints and
     unchanged source states;
   - an independent joint machine audit that uniquely matches the dataset's episode and
     frame counts and passes the required exact checks.

   Gripper identity is proven against an EEF baseline by default. The independent command
   re-derivation replaces it only when the spec declares
   `eef_baseline: {status: not_applicable, reason}`, the provenance names no EEF source,
   and the audit records that check as not applicable.
9. **The trainer source and tokenizer are pinned.** `trainer_source_sha256` and the PaliGemma tokenizer hash
    must match, and required trainer files must exist. The
    global `batch_size` must divide every GPU candidate count. Runs use `--no-overwrite
    --no-resume`.

Code: `tools/launch_pi05_training.py`, `src/datafarm/training/`.

## Data model

- **Training spec** (`datafarm.pi05-finetune/1.1.0`). The fields are listed under
  *Configuration*.
- **Dataset** (LeRobot): `meta/info.json` (`fps`, `total_episodes`, `total_frames`,
  `features`) and `meta/lerobot-provenance.json`. The provenance holds `repo_id`,
  `training_eligible`, `semantic_allowlist(_sha256)`, `episodes_detail[].image_source`,
  `allowed_semantic_waivers` and `distribution_acceptance_sha256`. Joint datasets add
  `state_action_contract` and `joint_conversion`.
- **Launch manifest** `<run_manifest_dir>/launch-manifest.json`
  (`datafarm.pi05-train-launch/1.2.0`): the command, the contract audit (`checks`,
  `statuses`, `admission_mode`), the admission record, and the dataset provenance digest. A
  superseded manifest is kept.

## How it works

1. Build the LeRobot dataset from admitted episodes and issue a certificate
   ([delivery and admission](delivery-and-admission.md)).
2. `preflight` validates the spec and dataset, and with `--admission-certificate` also
   verifies the certificate.
3. `norm-stats` and `loader-smoke` materialize normalization and test the data loader.
4. `train-command` builds the trainer argv, resolves admission, audits the contract and
   writes the launch manifest. With `--execute`, it submits through `submit`, which tries
   each GPU-count candidate in order (exit 2 when none is free).

## Entry points

```bash
python tools/launch_pi05_training.py preflight --spec <spec.json> --admission-certificate <digest> --admission-store <store>
python tools/launch_pi05_training.py norm-stats --spec <spec.json>
python tools/launch_pi05_training.py train-command --spec <spec.json> --admission-certificate <digest> --admission-store <store> --execute
```

## Configuration

| Key | Default | Notes |
|---|---|---|
| `state_conditioning` | required | `none` |
| `action_representation` | required | `absolute_eef` (template), `delta_eef_current_rpy_v1`, `absolute_joint_v1`, `delta_joint_current_v1` |
| `batch_size`, `num_train_steps` | required | Not enforced by code. |
| `fsdp_devices`, `gpu_candidates` | `[fsdp_devices]` | Each candidate must divide `batch_size`. |
| fixed in the command | — | action dim 32, horizon 50, max tokens 200; warmup 1000, peak LR 2.5e-5 decaying to 2.5e-6 over 30 000 steps; AdamW (0.9, 0.95, 1e-8, wd 1e-10); clip 1.0 |
| `dataset_fps`, `minimum_episodes` | 30, 1 | Dataset checks. |
| `training_action_contract`, `joint_source_evidence`, `dataset_provenance_sha256`, `admission_certificate {path, sha256, store}` | — | Required for joint modes. |

Paths for the trainer, checkpoints and tokenizer are spec fields. Deployment-wide bindings
are described in [deployment](../deployment.md).

## Failure behaviour

- Spec, dataset or evidence drift raises `ValueError` or `FileNotFoundError` before any
  manifest is written. A missing certificate raises `AdmissionRequired` with the exact
  remedy.
- A failed contract audit raises with the failing check names.
- `submit` exits 2 when no candidate allocation is free, so a queue can retry. Other
  scheduler errors pass through.

## Filling the template

`configs/training/template.json` is not runnable as shipped. Before `preflight` passes you need:

- the `<task>`, `<org>/<dataset>` and `<authorization-id>` placeholders, and bindings for `TRAINER`, `TRAINING_PYTHON`,
  `OPENPI_CACHE`, `DATASETS` and `TRAINING_RUN`;
- `terminal_home_contract.canonical_home_state_14d`, the 14-number home state of your robot;
- the base checkpoint directory and the tokenizer under `OPENPI_CACHE`, whose hash must equal `paligemma_tokenizer_sha256`;
- `distribution_acceptance.path` and `.sha256`, bound to a receipt (see below);
- a dataset with `meta/info.json`, parquet episodes that end at the home state, and `meta/lerobot-provenance.json` holding
  `repo_id`, `training_eligible`, `semantic_allowlist`, `semantic_allowlist_sha256`, `episodes_detail`,
  `allowed_semantic_waivers`, `distribution_acceptance_sha256`, `training_scope` and `production_policy_eligible`. The dataset
  builder writes these.

`train-command` additionally needs `--admission-certificate`.

## Distribution-acceptance receipt

`distribution_acceptance` points at a `datafarm-distribution-acceptance/1.0.0` JSON file that records the setup coverage
actually captured (`src/datafarm/training/distribution.py::validate_distribution_acceptance_contract`). Task code or the
agent writes it; no shipped tool does.

| Field | Meaning |
|---|---|
| `schema_version` | `datafarm-distribution-acceptance/1.0.0` |
| `status` | `frozen` or `accepted` |
| `target_task` | must equal the spec's `target_task` |
| `claim` | `fixed_setup`, `local_robustness` or `broad_deployment`; must equal `required_distribution_claim` |
| `variable_objects` | non-empty list of object names that vary across episodes |
| `frozen_denominator`, `accepted_count` | positive integers; they must be equal |
| `frozen_plan_sha256`, `cohort_manifest_sha256`, `coverage_authority_sha256` | SHA-256 bindings of the plan, the cohort and the coverage evidence |
| `unique_joint_layouts` | must equal the denominator |
| `unique_relational_layouts` | positive; `broad_deployment` needs more than one |
| `required_world_span_xy_m` | per object, the required `[x, y]` span in metres |
| `observed_per_object` | per object, `unique_positions` (more than one) and `span_xy_m`, at least the required span |
| `required_image_span_px`, `observed_image_span_px` | per camera and object, required and observed `[x, y]` spans in pixels; the camera sets must match |

For `broad_deployment` every required span must be strictly positive. The dataset provenance must name the receipt's hash
in `distribution_acceptance_sha256`.

## Limitations

- **Only the horizon is pinned.** Batch size and steps are ordinary spec values. The learning-rate decay length (30 000 steps) is fixed
  independently of `num_train_steps`.
- **The certificate does not depend on delivery verification.** Admission profiles read
  allowlists and audits, not `delivery.json`.
- **No shipped builder writes joint datasets.** `tools/build_pi05_dataset.py` writes the XYZ-RPY contract, so
  `absolute_joint_v1` and `delta_joint_current_v1` need a dataset and receipts you build yourself.
- **The drive-target tracking audit is only checked for shape.** `joint_tracking_audit` must
  be an object; its contents are not checked here. Task code produces it.
- **Generated images only.** The launcher rejects datasets with any non-generated image source.

