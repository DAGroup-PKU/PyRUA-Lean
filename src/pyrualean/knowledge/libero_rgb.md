# Operating knowledge for LIBERO: how the tools behave (no task recipes)

## Frames, units, robot
- One Franka arm with a two-finger gripper stands at the edge of a table and
  reaches across it.  World frame: x and y across the table, z up; metres
  and degrees.  "left" / "right" in the task text are as seen in the
  agentview image, not the robot's own: "left" is -y (the LEFT of the
  agentview image, the robot's own right) and "right" is +y.
- The table's height differs from scene to scene (the task card states the
  one of this scene): about 0.0 in object scenes, 0.43 in living-room scenes,
  0.90 in kitchen scenes.  Read object heights from the scene (`locate`),
  never assume them.
- `state().object_names` (e.g. `akita_black_bowl_1`) carry no pose and no
  visual difference between `_1` / `_2`; identify targets by what they look
  like and where they are.  The task instruction (`robo.task`) is
  authoritative: read it before planning and obey it verbatim.

## The grasp point and the gripper's orientation
- `move_to` drives the GRASP POINT, between the finger pads;
  `state().tcp_pos` reports it.  The fingertips end about 1 cm beyond it, so
  pointing straight down the grasp point cannot go lower than about 1 cm
  above the surface under it.
- The orientation is three angles (`state().pitch`, `.yaw`, `.roll`):
  `pitch` 0 points level and 90 straight down; the gripper starts pointing
  down (pitch close to 90).
- Pointing straight down (`pitch=90`), `yaw` alone sets how the fingers lie:
  they close along the line at `yaw` degrees from +x (counter-clockwise seen
  from above), and `roll` reads 0.  An EMPTY open gripper's fingers are
  interchangeable, so `yaw` and `yaw + 180` are the same grasp and such a
  move may arrive at either (`note` says so).  Holding something, a move
  always arrives at exactly the angles asked: that is how a held object is
  turned.  Tilted, `yaw` is the heading the gripper points towards (0: +x),
  and `roll` 0 keeps the fingers closing along that heading, one above the
  other; `roll` 90 turns them to close across it, side by side, as at the
  start (which reads roll -90).
- An angle you leave out of `move_to` keeps its current value: a plain
  position move does not turn the gripper.  Near straight down (within a
  degree or two) yaw and roll are not separately defined: give `pitch=90`
  and `yaw` together for a top grasp.

## Motion
- `move_to(xyz, gripper, pitch, yaw, roll)` servos position and orientation
  together with LIBERO's own controller, at most 2 cm and about 5 degrees per
  simulator step, until the pose is reached or the arm stops making progress.
  The path is close to a straight line and is not planned around anything.
  `reached` means within 0.01 m and 5 degrees.
- `blocked` set means the arm stopped short: something is in the way, a held
  object is pushing against something, or the pose is beyond the arm's
  reach (low and far, or tilted far from straight down).  Look before trying
  again; do not repeat the same target.
- Reach: pointing straight down the gripper reaches the table everywhere in
  front of the robot.  Level or steeply tilted it does not get low: over the
  low table of object scenes a level gripper stopped (`blocked`) about 11 cm
  above the table, and carrying one level across the table stopped short as
  well.  Grasp and carry pointing down, or tilted by at most a few tens of
  degrees.
- Turning: the arm's joints have limits, and a big turn of the gripper can
  wind one against its limit.  Turn by at most about 90 degrees in one move,
  and turn a held object high above the table.  `joint_limit` set means a
  motion ended with a joint of the arm at its limit (it says which, and
  what to do): the arm will not unwind by itself; turn back the way it came,
  or reach the pose by a smaller turn.
- Carrying past things: lift first (straight up), travel above everything in
  the way, come down over the destination.  A container an arm bumps moves.
- Every simulator step (also each VLA action) counts against the episode's
  budget (the task card states it).  When it is spent the state reports `truncated` and no motion
  is possible.

## Gripper
- Two conventions.  Commands are signed: `robo.CLOSE` (1) closes / holds,
  `robo.OPEN` (-1) opens, `robo.KEEP` (0) keeps commanding what the gripper
  was last told (a held object stays squeezed).  Measurements are openings:
  `state().gripper` and `Gripper.opening` read 1.0 when open (8 cm between
  the fingers) and 0.0 when closed on nothing.
- `set_gripper(command)` runs a full stroke (repeats the command until the
  fingers stop, about 12 simulator steps) without moving the arm.  After a
  close, `holding` says the fingers stopped on something and `width_m` how
  far apart they stand.  `set_gripper(robo.OPEN, opening=0.5)` stops opening
  at that reading, to reach between close neighbours.
- `holding` is not proof that the INTENDED object is held: confirm that it
  moved with the arm after a lift (wrist camera, or `locate` it at its old
  place and see it gone).
- Carry with `gripper=robo.CLOSE` (or `robo.KEEP` after a close) in every
  `move_to` that transports an object; `robo.OPEN` in a motion drops it.
- A round bottle with a thin neck is taken from above too: the open fingers
  (8 cm) pass either side of the neck and close on the body; a descent that
  stops on the shoulders (`blocked`) still closes on the body.
- A top grasp with these tools: `locate` the object; `set_gripper(OPEN)`;
  `move_to` above it pointing down (`pitch=90`, 8-10 cm above `top_z`) and
  `locate` it again from there (see Perception); turn to `yaw=grasp_yaw` and
  descend until the pads cover the top part of the object; `set_gripper(CLOSE)`;
  lift straight up; confirm.
- To put something down, lower it until its bottom is just above the surface
  (support top + object height, plus a few mm), open, and lift straight up
  before moving sideways.  Over a container, an object may be released from a
  few centimetres above its floor.

## The VLA skills
- `pi0_pick` runs the frozen Pi0.5 policy for the grasp only; you do every
  `move_to` and the release yourself.  Typical call:
  `pi0_pick("pick up the <object>", max_chunks=20, lift_thresh=0.05,
  gripper_closed_thresh=0.06)`; use `lift_thresh=0.08` for slippery tall
  bottles.  Pre-position the gripper 15-20 cm above the target first.
- Judge the grasp yourself: `Pick.success` (lift + closure heuristic) is a
  hint, not proof.  After the pick read `state().gripper` and, if in doubt,
  look at the wrist image: the object should now be raised into the gripper
  and its original spot empty.  Then `set_gripper(robo.CLOSE)` and carry
  with `move_to(..., gripper=robo.CLOSE)`.
- `pi0_doubled` is the contact skill for drawers, doors, knobs and pushes
  (its `success` only mirrors the task predicate); call it repeatedly,
  alternating with short scripted aligns.

## Perception
- Two colour cameras, 1024 x 1024, and no depth image.  `agentview` (fixed,
  over the table) is the authority for identity and coarse geometry: which
  object, which destination, the overall layout.  `wrist` (on the gripper)
  shows what the gripper is about to grasp or is holding, and a measurement
  needs both cameras to be three-dimensional.
- `locate(text, camera)` segments the object and measures it in 3D from the
  cameras that see it: `center_xyz` (footprint centre, half its height),
  `top_z`, `height_m`, the footprint `length_m` x `width_m` (the smallest
  rectangle around it), its long axis `long_axis_deg` (from +x) and
  `grasp_yaw`, the `yaw` that closes pointing-down fingers across its narrow
  side.  A handle or a spout belongs to the footprint and pulls the middle
  away from the body's.  A hollow object (bowl, basket) gives its rim as
  `top_z`.  Nothing moves.
- One camera cannot tell how tall a thing is: a low deep object and a tall
  shallow one look the same from it.  `views` says how many cameras agreed on
  the measurement.  With `views == 2` and no `note` the measurement is good,
  its height included.  A `note` that the cameras agree only roughly means
  the second camera saw part of it or saw it badly: the position is usable,
  the height may be centimetres off.  With `views == 1` the height is a
  guess, the tallest upright shape that fits the one image: close for a box
  or a can, wrong for a bowl, a cup or anything wider above than below;
  `center_xyz` is then off, along the line away from the camera, by about
  half of what the height is wrong by, and `note` says that it is a guess.
- The view that settles a measurement is from above, the one a top grasp
  passes through anyway: the gripper pointing straight down with its grasp
  point 8-10 cm above the object (`move_to([x, y, top_z + 0.09], pitch=90)`).
  So: `locate` from agentview for a first position, move above it, `locate`
  again, and grasp or place by the second reading.  From straight above, a
  text prompt often misses an object whose top is all the wrist sees (a lid,
  a cap): `locate(..., camera="wrist")` then measures from agentview's mask
  with the wrist image as the second view (`camera` reads `"agentview"` and
  `note` says so); the gripper above the object still settles its height.  Do the same over the
  place where something is to be put down just before lowering onto it.
- An object measured once with `views == 2` keeps its height: a later
  `locate` with the SAME prompt uses it when only one camera sees the object
  then (`note` says so).
- `locate` takes the object to stand on the table.  Something lying on top of
  another object, inside a container or in the gripper is not measured
  correctly (it comes out further away and larger): work out where it is from
  where you put it and from the images.
- `segment` + `back_project` give one mask and one point.
  `back_project(row, col, camera)` segments whatever is under that pixel,
  measures it as `locate` does and returns where the pixel's ray meets it
  (`surface == "object"`); a pixel on the table top itself returns that point
  of the table, exactly (`surface == "table"`), and so does one on something
  too flat to measure.
- All of them read the CURRENT scene: localise, act, then localise again.
  Positions from before a contact are stale.
- A mask is the best match for the text, not proof of identity.  Describe one
  object by colour, shape and position ("the black bowl on the stove");
  brand or internal names ("akita", "alphabet soup") score badly.  Check
  `score` and where `centroid` falls in the image before acting on it.
- `robo.image(camera)` returns the recorded picture as an array for your own
  analysis.

## Judging progress
- The benchmark checks its task predicate after every simulator step:
  `robo.done` turns True (and motion results carry `terminated=True`) the
  moment the task is solved, and every further motion is refused.  When you
  believe you are done but `robo.done` is False, something the benchmark
  checks is not satisfied yet: find it and fix it while budget remains.
- Grasp: the target leaves its place and moves with the arm.
- Placement: the object rests where the instruction wants it after the
  gripper has opened; an object on a rim or beside its destination is not
  placed.

## Safety and recovery
- Inspect the scene after motion.  Stop the current plan if an object is
  unstable or the arm is obstructed, and re-localise before recovering.
- After a failed phase name what went wrong (wrong object, missed grasp, lost
  hold, unreachable target, premature release) and change one meaningful
  variable; do not repeat the same target or the same recovery twice.
- There is exactly one episode and no reset.  Recover in place: re-observe,
  re-localise, re-grasp.
