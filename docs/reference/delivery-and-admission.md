# Delivery checks and admission

Three separate mechanisms stand between a rendered episode and policy training:

- **Render verifier** (`src/datafarm/verify/`): one program shared by all tasks, configured by a spec the agent writes for each task. See [render-verifier](render-verifier.md).
- **Delivery verification** (`src/datafarm/task/contract.py::verify_delivery`, and `white_delivery.py` for white-only data): a task delivery is complete, unmodified, and bound to its contract, attempts, grounding record and independent review.
- **Admission certificate** (`src/datafarm/core/admission.py`): consumed by the training launcher. An independent root re-derives the admitted cohort from stored evidence.

The three are not chained in code; see [Limitations](#limitations). The schema of `delivery.json` and the review file is in [`prompts/instructions/task-artifact-contract.md`](../../prompts/instructions/task-artifact-contract.md).

## Guarantees

1. **A rendered delivery binds its inputs, count and attempt denominator.** `input.json` re-hashes every locked input and its digest; `stop_after` is `render`; `delivery.json` uses `datafarm.agent-delivery/1` and matches the input digest; `target_count` equals the locked count, and that many unique episode IDs are delivered; the attempts file lists unique IDs with outcome `admitted`, `rejected` or `infrastructure_failure`, and the admitted IDs equal the delivered IDs; the dataset manifest (`datafarm.dataset-files/1`) has exactly these IDs and hash-bound files.
2. **Each episode is consistent with its render contract and actual submission.** The five artifacts (`action`, `source`, `video`, `prompt`, `references`) are hash-bound; the contract still loads; the submission receipt matches the contract file, intent and prompt hashes and the real-image prefix; the delivered prompt hash and references equal the contract's; every reference's provenance source is one of the locked input files.
3. **Every episode has a fully passing, evidence-backed review.** The review (`datafarm.task-review/2`) names the producer. Its subjects equal the current artifact hashes. Verdicts `task`, `motion`, `reference`, `style`, `alignment` and `joint_margin` must all be `pass`, and its evidence files must be hash-bound.
4. **The visual review compares decoded real references with the generated video.** `visual_review` declares `inspection_method: decoded-real-reference-and-generated-video`, binds the video hash and every real-reference hash, and covers all six `APPEARANCE_COVERAGE` items with generated frame numbers, compared reference hashes, views from the contract, and decodable comparison images that also appear in the review evidence. Whole-scene items need `full-view` evidence, and `declared_view_photography` covers every view. Only `task_objects` may be `not_applicable`, with a rationale, and only when the contract does not require it.
5. **Review independence is verified under the agent runtime and only declared outside it.** `reviewer_id` must differ from the producer. In a runtime-managed workspace (`.runtime/` holds `reviews`, `author.json` or `control.json`), the review cites `runtime_review`. That review must have a safe ID and a `completion.json`, pass `validate_completion` (a `task-verifier` role, matching context, transcript, report, execution and identity hashes, no reuse of the author's native session), have verdict `pass`, have a frozen request covering every delivered artifact hash, and have a report whose evidence binds this episode's review file. A hand-written execution record is refused there. Outside the runtime, it is hash-checked and reported as `declared-execution-record`. `review_independence` is `runtime-verified` only if every episode was.
6. **Every delivery binds a valid grounding ledger.** See [grounding](grounding.md). The result returns counts per label as `physical_quantities`.
7. **A white-only delivery is never mistaken for rendered or training data.** See [White-only delivery](#white-only-delivery).
8. **An admission certificate verifies only with an independent root receipt and fresh evidence.** `verify_certificate` loads the certificate by digest from the content-addressed store. It requires an `admission` receipt with verdict `pass`, authority `admission-root`, a trusted root ID different from the certificate producer, and all checks true. No trusted root may reject it. The root's checks are re-run against the stored evidence. When an `AdmissionExpectation` is given, the provenance digest, allowlist digest, `repo_id`, waiver set and member IDs must match the launch.

Code: `src/datafarm/task/contract.py`, `src/datafarm/task/white_delivery.py`, `src/datafarm/core/admission.py`, `src/datafarm/training/cohort_auditors.py`.

## Data model

- **Task workspace:** `input.json` (`files [{path, sha256}]`, `sha256`, `target_count`, `stop_after`) and `delivery.json` (`schema`, `input_sha256`, `producer_id`, `target_count`, `grounding`, `attempts`, `dataset`, and `episodes[]` with `id`, `artifacts`, `render_contract`, `submission`, `review`). Every file reference is `{path, sha256}`.
- **Delivery result** (`datafarm.delivery-verification/1`): `valid`, `episodes`, `attempts`, `delivery_sha256`, `review_independence`, `physical_quantities`, `scope`.
- **Admission certificate:** an `admission-certificate` IR document in an `ArtifactStore` (profile `datafarm.training-admission/1.0.0`, producer `datafarm-training-certificate-compiler-v1`). It binds the admitted members and media digests, the dense allowlist and audit evidence, the dataset provenance, a digest manifest of the dataset files and the waiver policy, and is signed by a gate receipt from `datafarm-training-admission-root-v1`.

## How it works

1. The agent writes a verifier spec, verifies one accepted render, and calibrates the spec on mutations of it.
2. `verify_render.py` runs the calibrated spec on each episode and writes the receipt and an HTML report.
3. An independent reviewer (a runtime review under `pipeline_agent.py drive`) compares the decoded video with the real references and records verdicts and evidence.
4. `pipeline_agent.py verify` runs `verify_delivery` over the finished workspace.
5. For training, the dataset is built and `issue_admission_certificate.py` stores the evidence, compiles the certificate and has the root sign or reject it. The launcher calls `verify_certificate` (see [policy-training](policy-training.md)).

## White-only delivery

When the user explicitly asks to end at the white-mesh stage, use `pipeline_agent.py prepare --stop-after white --target-count N ...`. The input lock records this boundary; the default is `render`. An existing rendering task cannot skip rendering by changing the delivery type. If an existing input lock has no white-mesh boundary, save the user's request and bind the original input and existing outputs in a new delivery workspace; never rewrite the lock or success state of the earlier run.

The white mesh is the complete multi-view video output by the simulator, plus the actual states and actions from the same execution. It does not claim real photographic appearance and is not a training admission certificate. Provider transport limits belong to the rendering profile; the white mesh keeps the complete trajectory, and neither the actions nor the clock are altered to fit a provider length.

`delivery.json` uses `datafarm.white-delivery/1` and binds `input_sha256`, `producer_id`, the fixed `target_count`, `episodes`, all `attempts`, `dataset` and `reading_guide`.

- `attempts` keep admitted, rejected and infrastructure failures, and the qualified set equals the delivered set. The dataset order equals the episode order.
- `dataset` uses `datafarm.dataset-files/1`, listing `episode_ids` and the path and sha256 of all data files. Data may be at external paths, while manifests and review records are in the delivery workspace.
- The reading guide explains the difference between actual states and commands, the dimensions, order, units, absolute or delta semantics, the clock, and a runnable reading example.
- Each episode's artifacts are exactly three file references: `trajectory`, `video` and `requirements`. The trajectory manifest (`datafarm.white-trajectory/1`) binds the state and action files and their `state_schema`, `action_schema` and `clock`. The video manifest (`datafarm.white-video-set/1`) has views covering the episode's `required_views`, and every medium of each stream is listed in `dataset`.
- Each episode binds an independently executed review (`datafarm.white-review/1`): `producer_id`, `reviewer_id`, `subjects`, the `task`, `motion`, `alignment`, `joint_margin` and `visual` verdicts, `execution`, `evidence`, and decodable `visual_evidence` images for all declared views. `execution` binds the reviewer's `actor_id`, `producer_id`, `role=reviewer` and the transcript. Different names do not replace independent execution. Frame-by-frame decoding, state and action comparison, the physical judge and the joint check are executed by the independent review, with evidence saved. Photographic style is not part of this review.

`pipeline_agent.py verify` returns `datafarm.white-delivery-verification/1` with `rendering_certified` and `training_certified` always false. It verifies file binding and the existence of review evidence, not physical or visual correctness.

## Issuing a certificate

```bash
python tools/issue_admission_certificate.py \
  --store runs/<run>/admission-store \
  --dataset-root <lerobot_home>/<repo_id> \
  --output runs/<run>/training-admission-certificate.json
```

- Exit 0 means admitted. The tool prints `certificate_sha256`, `receipt_sha256`, the evidence sources and their digests.
- Exit 2 means rejected. `failed_checks` and `evidence_notes` say why, and the rejected certificate and receipt stay in the store.
- Exit 1 means an evidence or usage error.

Evidence defaults are the allowlist named in the provenance, the allowlist's sibling `semantic-audit.json` and `pi05-candidate-manifest.json`, and the distribution receipt named in the provenance. Override them with `--semantic-allowlist`, `--dense-audit`, `--candidate-manifest` or `--distribution-acceptance`. `--output` writes the canonical certificate bytes, whose SHA-256 is the certificate digest.

The dataset directory must not change after certification, because every launch re-hashes its files.

```bash
python tools/launch_pi05_training.py train-command --spec <spec> \
  --admission-certificate runs/<run>/training-admission-certificate.json \
  --admission-store runs/<run>/admission-store
```

`--admission-certificate` accepts the 64-hex digest, which needs `--admission-store`, or the certificate file. If the file lives inside the store under `sha256/xx/<digest>.json`, the store is inferred.

The launch manifest (`datafarm.pi05-train-launch/1.2.0`) records `admission.mode = certificate-verified`, the verification summary, whether the dataset files were verified, and the contract audit with per-check statuses. A manifest it replaces is kept as `launch-manifest.<sha12>.superseded.json`.

The certificate admits exactly one built dataset. Its waiver policy is `zero-waiver`, or `explicit-waivers` naming each approved waiver (sample, check, authorizer, reason). A waiver hidden by the provenance (`allowed_semantic_waivers: []` while the evidence waives a gate) is rejected. A launch without a verified certificate is refused.

## Handing off to training

- A certificate binds the training input to the independent checks it passed, so a pass is not reused after a video or configuration is swapped. A textual allowlist or a hand-written status file cannot replace it. The producer cannot act as the independent verifier.
- The certificate binds members, media digests, the allowlist, audits and dataset files. The mode intent, actual reference payload and real-style review are checked by the delivery check, not bound into the certificate. A certificate lacking style evidence cannot be upgraded automatically: original metrics are not modified and authorization is not backfilled.
- Data building checks the certificate and input hashes, then verifies action-image timing, terminal state, trim and splicing, and the state schema. Rejected or missing items stay in the registered denominator, and a replacement trajectory has its own ID and a mapping to its parent candidate. Reports state both total production attempts and the final number admitted.
- A run explicitly declares `state_conditioning: none`, with model parameters consistent with the data. The reference frame, units, time step and rollout interface of delta EEF must match. Training success only means the steps finished; report the evaluation protocol, the real and simulated boundary and deployment limits separately.
- Training is bound by the ledger, GPU authorization and persistence checks. Close out only after the checkpoint, metrics, commands, code version and certificate locations can be verified.

## Failure behaviour

- The verifier returns `abstain` rather than `pass` when evidence is missing. The CLI exits 2 for both `fail` and `abstain`.
- `verify_delivery` raises `ContractError` (a `ValueError`) on the first violated rule. A changed byte in any bound file fails with "artifact changed". `pipeline_agent.py drive` does not accept a completed task without a verified delivery.
- `verify_certificate` raises `AdmissionError` with every failure listed.

## Limitations

- **Delivery does not require a verifier or calibration receipt.** The render verifier gates the render stage by instruction: the [render-batch skill](../../.agents/skills/render-batch/SKILL.md) requires a calibrated spec per task, and on the H3 route acceptance also needs `evaluations/render-h3`.
- **The delivery check does not gate the certificate.** Admission reads allowlists and audits, not `delivery.json`.
- **The admission root runs in-process.** Its independence is a component identity (a verifier ID different from the producer, with re-derivation from stored bytes), not a separate execution.
- **Declared independence is not verified.** Outside the agent runtime it is a hash-checked declaration.
- **Semantics remain with the reviewer.** The delivery check binds content and membership. Physical and visual correctness come from the review it requires, and it cannot prove that a host isolated identities.
