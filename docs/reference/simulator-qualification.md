# Simulator qualification

Simulator qualification checks, before an expert is written, that the simulator can carry out
the skills a task needs. Scripted skill probes run over candidate simulator variants (dotted-key
patches of the material and scene files plus optional runtime flags such as a gripper contact
model); each probe is judged against declared criteria; the variant with the fewest changes that
passes every required probe is selected and issued as a certificate binding the exact variant
files and all probe evidence.

This release ships the decision logic and the run-binding check. A task's probe runner (launching
probes on the GPU scheduler and extracting metrics from simulator state) is an output the agent writes
for that task. The shipped probe kinds, `hold` and `fold`, are written for cloth.

Simulator qualification and render style are separate responsibilities. Passing the simulator checks
does not mean OOD rendering is allowed, nor that a generated video is photorealistic.

## Guarantees

1. A variant patch can only replace values at existing dotted paths (a `"*k"` string multiplies a
   numeric value); a missing path or a non-numeric multiplier is an error and the base document is
   not mutated.
2. A `hold` probe passes only if the hang was reached, the cloth rose at least `min_lift_m`, every
   gripped patch slipped at most `max_slip_m`, and the p95 strain is at most `max_strain_p95`
   (and p99.9 at most `max_strain_p999` when set); a missing metric fails; flatness after lay-down
   is advisory.
3. A `fold` probe passes only if `m1_at_release` is `True`; when a spec sets `require_settled` or
   `max_creep_m`, the settled verdict and the rest drift (over at least `min_rest_window_s`) are
   required too and fail when unmeasured; bottom-layer capture and robot-gate failures are advisory.
4. Selection requires at least one required probe and picks, among variants passing every required
   probe, the smallest `change_magnitude` (number of material keys + scene keys + runtime flags);
   ties go to the higher score, then the name; if none passes, nothing is selected.
5. A certificate is issued only for a selected variant with file digests, and records the selected
   patches, flags, raw and canonical file digests, criteria, the full ranking and every probe's
   pass flag, failures, advisories and metrics for every variant.
6. A certificate check recomputes the selection from the probe outcomes, compares each claimed
   per-probe pass flag and requires bound files; a forged selection or pass flag is reported.
7. A run binds to the certificate only if every certified file is reported with a matching digest
   (canonical JSON digest for `.json` names, byte digest otherwise) and no uncertified file is
   reported.

Code: `src/datafarm/simulation/sim_qualification.py`, `tools/check_sim_binding.py`.

## Data model

- **Spec** (JSON): `task_family`; `base_material`, `base_scene` (paths opened relative to the
  current directory); `variants[]` with `name`, `note`, `material` and `scene` (dotted-key patches),
  `runtime_flags` (a flag `file:<path>` binds that file), `launcher` (relative to the spec);
  `probes[]` with `name`, `kind` (`hold` | `fold`), `expert` (relative to the spec), `seed` (or null);
  `required_probes`; `criteria` (`{"hold": {...}, "fold": {...}}`, keyword arguments of the probe
  evaluators).
- **Certificate** (`datafarm.sim-qualification.certificate/1.0.0`): `task_family`, `selected_variant`
  (`name`, `note`, `material_patch`, `scene_patch`, `runtime_flags`, `files_sha256`,
  `files_canonical_sha256`), `required_probes`, `criteria`, `ranking` (rows
  `[name, passes_required, change_magnitude, score]`), `evidence` (`{variant: {probe: {passed,
  failures, advisories, metrics}}}`), `reason`.
- **ProbeOutcome** / **VariantResult** / **QualificationDecision**: frozen dataclasses in
  `sim_qualification.py`; `VariantResult.score` = 2 per passed probe − 0.25 per advisory
  (tie-break only).

## How it works

1. A task's probe runner materialises `material.json` and `scene.json` for every variant with
   `apply_patch`, and runs one job per (variant, probe) on the GPU scheduler.
2. It extracts metrics from each finished run. Unfinished runs or failed extraction yield empty
   metrics, which fail the probe.
3. Each outcome is judged by `evaluate_hold_probe` or `evaluate_fold_probe` with the spec's criteria,
   or by evaluators written for the task, and `select_variant` ranks the variants.
4. If a variant is selected, its files are hashed, `build_certificate` produces the certificate and
   `verify_certificate` re-checks it against the results.
5. Later, `tools/check_sim_binding.py` compares the files a run configuration uses with the
   certificate (`verify_run_binding`) and with each run's `environment.txt`.

## Entry points

```bash
# check that runs used exactly the certified files (exit 0 only if everything binds)
python tools/check_sim_binding.py /data/task/simqual/certificate.json \
  --file material.json=/data/task/sim/material.json --file scene.json=/data/task/sim/scene.json \
  --runs /data/task/runs/ep-0001 /data/task/runs/ep-0002
```

## Configuration

- Criteria defaults — hold: `max_slip_m 0.03`, `min_lift_m 0.10`, `max_strain_p95 1.15`,
  `max_strain_p999 None`, `min_flat_area 0.85` and `max_flat_height_m 0.02` (advisory);
  fold: `min_bottom_layer_capture 0.5` (advisory), `require_settled False`, `max_creep_m None`,
  `min_rest_window_s 8.0`.
- `check_sim_binding.py`: `--file NAME=PATH` (repeatable), `--runs DIR...`; exit 64 on usage errors.

## Failure behaviour

- No passing variant: the decision records the ranking and reason, and no certificate can be built.
- A failed certificate re-check is reported by `verify_certificate` as a list of failures.
- `check_sim_binding.py` exits 1 on any `mismatch`, `not_reported`, `not_certified`,
  `canonical_digest_missing`, `files_unbound`, `changed_since_run` or `no_recorded_digest` problem.

## Limitations

- **Probe families are cloth-specific.** Only `hold` and `fold` exist. There is no rigid-body or
  articulated-object probe family in shared code; a new task needs its own probe evaluators and
  metric extractors.
- **`verify_certificate` is not an independent review.** It compares selection and pass
  flags on the results it is given. It does not re-extract metrics from raw run files or re-hash the
  variant files.
- **Admission does not require a certificate.** Neither `task/contract.py::verify_delivery` nor
  `white_delivery.py::verify_white_delivery` checks a qualification certificate or a run binding;
  producing all episodes on the certified simulator holds only if the producer
  runs `check_sim_binding.py` and a reviewer checks it.
- `check_sim_binding.py` digests the files the operator names on the command line; each run
  directory needs to have recorded at least one of them in `environment.txt`, not all.
- Runtime flags without the `file:` prefix are bound only by name. A `file:` flag whose name ends in
  `.json` receives a raw digest but no canonical digest, so `verify_run_binding` would report
  `canonical_digest_missing` for it.
- `change_magnitude` counts changed keys, not the size of a change. Passing probes shows the simulator
  supports the actions; it is not a measurement of physical parameters.
