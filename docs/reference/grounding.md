# Grounding from the calibration episode

Grounding turns the real calibration episode into the record G that the simulator and the
renderer consume: the physical quantities the simulator uses, the camera/scale fit, the
real-world appearance references for rendering and, optionally, newly built object assets (see [usdcraft](usdcraft.md)).
The shipped code does not perform the fit itself. It fixes the *form* of G (a typed
physical-quantity ledger, real-reference provenance, a source-bound object-asset record)
and refuses deliveries whose G is incomplete or whose bound files changed. The fitting work
is done by the agent following `.agents/skills/sim-grounding/SKILL.md`.

## Guarantees

1. Every ledger entry has a unique non-empty `name`, a `label` in
   `measured | fitted | assumed | unknown`, and a string `unit` (`""` if dimensionless).
2. A `measured` or `fitted` quantity needs at least one evidence reference.
3. Only an `unknown` quantity may have a `null` value; any other value must be a finite number,
   a list or a string.
4. An `assumed` or `unknown` quantity must declare a `randomization` or a non-blank
   `not_randomized_reason`.
5. A declared `randomization` (for any label) is either `choices` with at least two values or
   finite `low < high`; a numeric nominal value must lie inside `[low, high]`.
6. A grounding record file must carry `schema: "datafarm.grounding/1"`; an evidence binder, when
   given, is applied to every evidence reference.
7. A Grounding IR document must contain `payload.system_identification.physical_quantities`,
   validated by the same rules, with each evidence entry an ArtifactRef of exactly
   `{kind, sha256, media_type}`.
8. Both delivery verifiers require `delivery.grounding` to bind a non-empty file inside the
   task workspace whose SHA-256 matches, and re-hash every evidence file it cites.
9. A render intent needs at least one real-world reference; each is a decodable PNG/JPEG/WEBP
   bound by hash, with provenance `real-episode-frame` (non-negative integer `frame`) or
   `real-photo`, a bound `source` file and a description; every declared appearance view needs a
   `full-view` reference covering `full_scene`, and required appearance areas must be covered.
10. A rendered delivery is refused unless each real reference's `provenance.source.sha256` is one
    of the locked input files (`input.json`).
11. Camera-pose and grasp randomization plans are deterministic per seed, content-addressed and
    episode-constant; per-frame camera jitter is rejected.

Code: `src/datafarm/core/grounding.py`, `src/datafarm/core/schema.py`, `src/datafarm/task/contract.py`, `src/datafarm/task/white_delivery.py`, `src/datafarm/render/policy.py`, `src/datafarm/simulation/randomization.py`.

## Data model

- **Grounding record** (`grounding.json`, bound from `delivery.json` as `grounding: {path, sha256}`):
  `{schema: "datafarm.grounding/1", physical_quantities: [entry...]}`. Entry fields: `name` (str),
  `label`, `unit` (str), `value` (number | list | str | null), `evidence` (list of `{path, sha256}`,
  workspace-relative), `randomization` (`{low, high}` or `{choices: [...]}`), `not_randomized_reason`
  (str). `validate_quantities` returns the count per label; both verifiers report it as
  `physical_quantities` in their verification result.
- **Grounding IR** (`IRKind.GROUNDING`, `src/datafarm/core/schema.py::REQUIRED_PAYLOAD_KEYS`):
  payload sections `source`, `transfer_scope`, `camera_clock`, `reference_roles`,
  `system_identification`, `rights`; ledger evidence entries are `{kind, sha256, media_type}`.
- **Real reference** (`intent.real_references[]` in a render intent, `datafarm.render-intent/1`):
  `path`, `sha256`, `provenance {kind, source {path, sha256}, frame?, description}`,
  `scope` (`full-view` | `local-detail`), `views`, `view_regions {view: [x, y, w, h]}`,
  `coverage` (subset of `REFERENCE_AREAS` in `src/datafarm/render/policy.py` plus `full_scene`).

## How it works

1. The agent reads the full episode and follows the sim-grounding skill: metric scale from robot
   or known-object dimensions, camera intrinsics/extrinsics, table and object layout, cross-checked
   by projection at several times and views; small simulator image checks per camera before the
   first full white capture.
2. It writes `grounding.json` listing every physical quantity the simulator uses, with label,
   evidence for measured/fitted values, and randomization or a stated reason for uncertain ones.
3. It extracts real frames (or uses user-supplied real photos placed in the locked inputs) as
   appearance references and records their provenance in the render intent.
4. If an object asset is missing, it writes an observation and runs the USDCraft bridge
   (`plan`, then `run`); the exported asset stays unqualified until simulator import, comparison
   and contact probes are reviewed independently.
5. Delivery verification (`verify_delivery` / `verify_white_delivery`) re-validates the ledger and
   re-hashes its evidence; the rendered path also re-validates reference provenance against the
   input lock.

## Entry points

```bash
# verify a finished task workspace (includes the grounding ledger)
python tools/pipeline_agent.py verify --work /data/task
```

## Failure behaviour

- An invalid ledger raises `GroundingError`; verifiers re-raise it as `ContractError`
  ("grounding record invalid: ..."); the IR validator raises `SchemaError`. Nothing is repaired.
- A changed evidence or grounding file fails with "artifact changed"; the delivery must be rebuilt.
- Reference provenance outside the input lock fails with an instruction to include the real
  photos at `prepare`; the lock cannot be extended after the fact.

## Limitations

- **The ledger is declarative.** No shared code checks that a declared randomization is applied
  in the simulator, or that a label is truthful; the evidence binding proves bytes, not that the
  file supports the value. Correctness is left to independent review.
- **Randomization implemented in shared code** is the camera-pose and grasp plans of
  `randomization.py` (default camera names match a dual-arm rig). There is **no shared mass,
  friction or material randomizer**; randomizing uncertain quantities is a declaration requirement in the ledger.
- `choices` are not checked against the nominal value; list or string values are never range-checked.
- **Camera fitting and scale cross-checks are agent-performed per the skill.** No shared calibration
  entry point ships, and no verifier requires camera-fit evidence. The delivery verifiers do not
  directly check camera calibration or joint-limit occupancy.
- **Reference frames**: no shared tool emits `real-episode-frame` references; `validate_intent`
  checks structure, image decoding, region bounds and source binding, but does not decode the
  source video to confirm the frame index. The rendered delivery checks provenance against the
  input lock; the white delivery has no real-reference check.
