---
name: build-observed-object
description: Build a new observed rigid or articulated object with the optional external tool USDCraft, export source-bound URDF/USD assets, and qualify them for the current simulator. Use during scene grounding when a suitable object asset is missing; deformable objects need a different backend.
---

# Build an observed object

Use this during `sim-grounding` when missing object geometry or articulation prevents task
execution. Reuse a suitable source-verified asset when available. USDCraft is an optional external
tool for asset authoring, not the robot controller or a replacement for task verification. Load its SDK
references only when debugging the generated model. Upstream agent instructions describe that
framework's authoring context; they do not override the current task's authority or acceptance.

## Observe and choose the input route

Open the real full-scene and object detail images, including different observed joint poses.
Save object identity, frame/view/time provenance, task-relevant affordances, visible parts,
joint type/axis/range, scale anchors and uncertainties in `task/objects/<id>/observation.json`.
Separate measured dimensions from inferred parameters. Do not invent invisible articulation
or declare category priors to be measurements. Include container cavities, handles and contact
surfaces that matter to the task; a visually similar solid box is not a usable open container.

- Image + description only: **Generation** (`generation-codex`). It infers geometry; it does
  not establish a measured reconstruction or exact physical parameters.
- Corresponding source mesh + photo: **Reconstruction** (`reconstruction-codex`). Declare
  units and axes when supported by evidence, retain mesh dependencies, and do not substitute
  an unrelated generated mesh as measured ground truth.
- Soft or deforming objects: route to the simulator's deformable model, not a forced joint tree.

The upstream CLI consumes one image. Choose a clear real frame of the same instance; summarize
other views in the description and retain their evidence paths. Context files are hashed by the
bridge but are **not automatically passed as images to USDCraft**. Confirm multiview fit later.

## Execute and retain the result

Read [USDCraft integration](../../../docs/reference/usdcraft.md) for the config and observation
schema, setup and exact commands. Model/reasoning, checkout, launcher and optional timeout are
configuration; do not bake this deployment's paths or credentials into generated tools.

1. `build_usd_asset.py plan` validates paths, binds input hashes and prepares argv without
   calling a model. An authorized build uses the upstream native Codex `--direct` route;
   existing task authorization covers routine asset generation, not new external service use.
2. Run the plan through the pipeline run ledger/supervisor, using `build_usd_asset.py run --work BUILD --task-work TASK`. Native USDCraft is refused for enforced budgets; advisory/disabled retain unknown descendant cost in the same task ledger.
   Store the run/worker ID and progress path in the task's external jobs. Wait in program code,
   not repeated model polling. The task-bound bridge creates an external_jobs ID/status_file; the existing supervisor refreshes it. When only waiting, set model_wait_for to that exact ID. No photorealistic video service is used; local asset inspection
   renders are part of geometry authoring, even when downstream video rendering is disabled.
3. Retain stdout/stderr, upstream run/record IDs and failed workspaces. Never resubmit an
   existing `running`, interrupted or failed build automatically: inspect the existing upstream
   workspace and worker first. A changed hypothesis gets a new versioned plan, preserving cost.
4. `asset.json` binds the canonical model and export dependencies, records joint names and
   leaves qualification **unverified**. Keep exports in their existing dependency-complete
   location. No generated source is copied into the pipeline's universal skills or registry.
   Account for all upstream authors/observation calls; absent usage is unknown, never zero.

## Qualify in the target simulator

Use exported URDF for a compatible URDF loader (such as the SAPIEN loader); use a USD
variant only with the appropriate simulator/backend. PhysX/Newton fields are not automatically
portable. Record import options, version, mass/inertia/friction, drive defaults, fixed/floating
base, link and joint maps, axes, units and collision simplifications. Upstream compilation is not
proof of a successful simulator import.

Inspect real-versus-sim views at the observed poses. Sweep relevant joint ranges to check limits,
axes, collisions, gaps and handles; test free settling, support and representative manipulation
contacts. Save actual state/contacts and images. For articulation, check the actuator/passive
behavior needed by the task rather than only moving a viewer slider. Fit uncertainty is allowed
when explicit and tested. Revise the asset on a demonstrated mismatch; preserve failed versions.

Only attach the asset to the task scene after the normal independent grounding/task review
accepts these source-bound probes. USDCraft's own mechanical checks are upstream evidence, not a
DataFarm success certificate. Keep simulation qualification separate from asset export and final
white-episode admission. Recollect matched states/images if geometry or articulation changes.
