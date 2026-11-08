# Operating knowledge for RoboTwin: how the tools behave (no task recipes)

## Frames, units, robot
- Two arms, `"left"` and `"right"`, on one base at the near edge of a table.
  World frame: x across the table (the LEFT arm starts at negative x and is
  on the LEFT side of the head image), y away from the robot towards the far
  edge, z up; metres and radians; quaternions are `[w, x, y, z]`.
- Both arms start above the table with the grippers open (1.0) and pointing
  along +y (the approach axis).  Read the table height from the head world
  map (`np.nanmedian(robo.world_map("head")[..., 2])`); it is randomised per
  scene, never hard-code it.
- EEF vs TCP: `move_to` targets the EEF (wrist link).  The TCP - the gripper
  centre, where an object is held - is 0.12 m further along the approach
  axis (`state().left.approach`).  To put the gripper centre on a point p,
  command the EEF at `p - 0.12 * approach`; sending an object's surface point
  as an EEF target parks the gripper 12 cm past it.
- The task instruction (`robo.task`) is authoritative: obey it verbatim,
  including which arm it names.  When it names none, the arm on the object's
  side (left for x < 0) is the natural choice; the other arm stays out of the
  way.

## Gripper
- Gripper values are floats in [0, 1]: `robo.OPEN` (1.0) open, `robo.CLOSE`
  (0.0) closed.  `set_gripper(arm, val, steps=10)` interpolates linearly over
  `steps` native actions; `release(arm)` opens to 1.0.
- `move_to(..., gripper=None)` keeps the current command, so a held object
  stays held by default; pass a value only to change it during the motion.
- `state().left.gripper` reports the commanded value, not contact.  Closure
  alone proves nothing: a grasp is verified when the object leaves its
  resting place and moves with the TCP (compare wrist images / world samples
  before and after a small lift).

## Motion primitives
- `move_to(arm, xyz, quat=None, gripper=None, substeps=25)`: a motion planner
  computes a collision-free joint path to the EEF pose and executes it as
  `substeps` native actions.  `Move.planned=False` means nothing moved (the
  target is unreachable, in collision with the table, the other arm or an
  object): do not repeat the same target; retreat or change one variable
  (approach height, waypoint, orientation).
- A successful plan does not prove arrival: compare `final_eef_pos` with the
  target (`reached`, `final_dist_m`) and check the images for the intended
  scene change.
- Guarded low approaches: near the table, a container rim, a button, a
  hinge, a stacked object or the other arm never queue several unobserved low
  waypoints.  Keep x/y, orientation and gripper, change only z by 5-10 mm per
  call with at most 8 substeps, and check after every increment.  Do the
  increments in a loop inside one cell: after each `move_to` test the result
  and the state in code (the plan failed, z made no progress, x/y drifted,
  `robo.done`) and stop the loop on any of them; look at an image only when
  the hold becomes uncertain or a contact cannot be interpreted.
- `rotate_wrist(arm, delta_yaw_deg)` keeps the EEF position but sweeps the
  TCP and a held object on a 0.12 m arc.  Rotate only after a verified hold,
  with clearance around the gripper, at a safe transport height, in small
  increments; then re-check the hold and the object's actual orientation
  (EEF yaw change is not object yaw change).
- Every native action counts against `step_lim`; the limit is a safety
  ceiling, not a target.  `State.actions_left` and the truncated flag tell you
  where you stand.

## Perception
- Three cameras, 320x240 each.  `head` is the semantic authority: identity,
  distractors, destinations, the relation the language asks for, global
  progress.  The wrist views refine the geometry of the SAME candidate the
  head view selected; do not let a wrist view silently switch to a
  look-alike.
- World maps are `[row, col] -> [x, y, z]` metres and may contain NaN.
  `sample_world_xyz(view, pixels)` gives the median of a small window;
  `query_world_map(view, bbox)` gives extent and median of a box;
  `world_map(view)` is the whole array.  Pair pixels with the world map of
  the same step and view.
- A visible surface point is not an object centre: sample several interior
  pixels and take robust statistics; pixels on edges, shadows or the gap to
  the table return background points.  Flat objects at table height need the
  RGB image, not depth, to be told apart.
- Re-localise after occlusion, contact or substantial arm / object motion;
  call `render()` first when time has passed without a motion call, since
  images belong to recorded steps.

## Observable gates (what "done with this phase" means)
- Grasp: the target leaves its source and moves with the TCP.
- Transport: the hold stays stable through a clearance waypoint and the
  lateral move.
- Placement on a support: the object rests on the correct support before
  release, then stays put and separated while the arm withdraws.
- Container: the object body crosses the opening and remains inside after
  release; rim or nearby placement is incomplete.  A pad, plate, scale,
  skillet or stand is not a container.
- Short contact (button, switch, bell): the intended control visibly changes
  after one guarded contact; distinguish the control from printed markings.
- Articulation (lid, door, hinge, knob): contact is retained while the part
  moves in the requested direction; verify the state change before letting
  go.  A momentary press is not an articulation.
- Handover: the receiving arm's hold is verified before the giving arm
  releases.  A task name alone does not prove a handover is required; the
  instruction does.
- Ranking / stacking: follow the order the language gives; each correct
  relation is protected from later paths and actions.  Ranking does not
  imply stacking.
- Orientation / hold: the requested orientation is visible while control is
  retained; do not release when the instruction asks to hold, lift or shake.
- Only `eval_success` (`terminated`, `robo.done`) proves task success; stop
  acting as soon as it fires.  Primitive success is not task success.

## Recovery and budget
- After a failed phase name the first unmet gate and the blocker: wrong
  identity or destination, missed grasp, lost hold, planning / collision,
  insufficient contact, premature release, unstable placement, incomplete
  relation.  Change one meaningful variable and verify; do not repeat the
  same primitive target or the same hand-written recovery twice.
- Near success, repair only the remaining blocker; do not restart the task
  or disturb objects that are already correct.
- Work in phases with in-code checks (`Move.planned` / `reached`, gripper
  value, world samples before and after, `robo.done`), and look at an image
  only when a decision depends on it.
- There is no reset.  Recover in place: re-observe, re-localise, re-grasp.
