---
name: sim-grounding
description: Reconstruct and validate the task-relevant robot, cameras, scene and physical model from real evidence, iterating without requiring a prebuilt task scene.
---

# Automatic scene and physics adaptation

First read the complete input, and list the parameters that have evidence and the items that cannot be identified. Prefer reusing verifiable robot URDF/mesh,
joint order/limits, gripper dimensions and existing calibration; these are device assets, not a requirement for the user to write a task adapter.
A robot that is not among the deployment's assets arrives with the input as a URDF package: run `tools/check_robot_package.py` (links, joints,
root, joints without limits, unresolved meshes, calibration files), then verify the joint mapping and calibration against the real views with
overlays at several instants. A package or vendor note is evidence to check, not an accepted device fact; disclose every assumption.

- Establish a metric baseline and the world/base/camera coordinate relations from the real robot / known object scales. Fit the tabletop, object layout,
  end-effector pose and the camera intrinsics/extrinsics; validate by projecting at multiple times and from multiple views, and do not let a single-image alignment mask a scale error.
  A wrist camera follows its corresponding end-effector; distinguish RGB, depth, EEF and visual estimates. A candidate calibration is not an accepted device fact.
- Before the first complete white-model collection, use a few real simulator images to check each camera: the initial state, the holding/manipulation pose,
  and the place/retreat pose. The author and the independent reviewer must actually open the corresponding real reference and the simulated image, and confirm the camera identity,
  orientation, task region and self-occlusion; the existence of a matrix, a decodable image or a passing physics probe cannot replace this check.
  When a single real camera is genuinely occluded, keep that fact and do not insist that every camera see all objects throughout; but whole-run self-occlusion caused by a wrong mounting
  must not be passed off as real camera behavior. Record the config/source hashes and the image evidence before entering complete collection.
  After changing a camera, redo the small-sample check and recollect matching data; never write new extrinsics onto old images.
- Build geometry and contact sufficient to express the task. For rigid bodies check stable placement, collision, mass/inertia, friction and gripper contact;
  for soft bodies check the material, topology / layer identity, tethers or attachments, solver stability and the free evolution after the action is released.
  A simplified model is usable only after task-relevant validation; do not pursue pixel-level appearance agreement with the coarse white model.
- Design cheap, discriminating probes: static hold, grasp/lift/release, friction slip, tiny joint/end-effector motions, camera
  reprojection, and independent randomized initial states. Calibrate against real evidence; do not use rendering to "paint in" a physical success that does not exist.
- Attribute failures to coordinates/scale, collision/friction, control timing, the observation interface or the soft-body model; change one explainable factor at a time,
  and compare the same events in real and simulation. Keep the raw traces and failure records. Parameters and physical assumptions carry versions and sources.
- Randomized initialization must really reset the physical state and wait for it to settle; changing only the image or text is not enough. Check legal contact/penetration, reachability,
  object count/identity and the task's supported range. Every new round uses independent seeds; reusing a fixed successful animation as data is forbidden.
  For a `broad_table` placement, record `reachable_tabletop_xy` (`x`, `y` ranges on the table plane that both arms reach within their joint
  limits minus a margin) in grounding.json with the probe that measured it.
- For a complete episode, check the limit margins of actions against the actual readback, next-state, and the home/wait/recovery/crop connections; replan
  and recapture images when necessary. Candidate margins are not hardware limits and do not carry over to a new robot.

When a needed new rigid-body/articulated object asset is found to be missing and not reusable, read `.agents/skills/build-observed-object/SKILL.md`
and, per the deployment config, generate and validate it with USDCraft (`docs/reference/usdcraft.md`); image inference is not a measured reconstruction, and an export is not simulator admission.

grounding.json uses the schema `datafarm.grounding/1`, and physical_quantities lists, item by item, the physical quantities the simulator uses:
name, unit, value, label (measured/fitted/assumed/unknown). measured/fitted must bind an evidence
file (path/sha256); assumed/unknown must give a randomization (low/high containing the nominal value, or at least two
choices), or state not_randomized_reason. Delivery validation checks each item (src/datafarm/core/grounding.py).

Output grounding.json/md, the scene/camera/physics config, the probe commands, the raw observations and a qualification report.
If the scale/calibration cannot be determined from the existing evidence, state clearly which measurement could disambiguate it; do not fake a calibration, and do not vaguely ask
to "supply more context". Read docs/reference/simulator-qualification.md and configs/joint-margin-requirements.json.
