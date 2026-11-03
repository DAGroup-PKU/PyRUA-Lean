# Operating knowledge for the G2 in GenieSim: how the tools behave (no task recipes)

## Frames, units, robot
- A humanoid G2 with two seven-joint arms, `"left"` and `"right"`, each with
  a two-finger gripper, on a torso that stands on a knee-hip-waist chain and
  (in some scenes) a wheeled base.  Every position is in `robo.FRAME`
  (`base_link`, fixed to the robot's base): +x forward, +y to the robot's
  left, +z up; metres and degrees.
- The upper body may start turned on its waist: `state().torso_heading_deg`
  (also on the task card) is where the chest faces, and the work surface is in
  front of the chest.  A heading of 90 means the table is towards +y.
- Hands start in different places in different scenes: read `state()`, do not
  assume.  Object positions come from `locate`, never from assumption.
- The task instruction (`robo.task`) is generated for this layout and is
  authoritative: read it before planning and obey it verbatim.

## The grasp point and the orientation
- `move_to` drives an arm's GRASP POINT: the gripper's centre between the
  fingers, 0.143 m out from the wrist flange along the pointing direction
  (`state().right.tool_centre`) — the same point the benchmark measures the
  gripper from.  The fingertips reach about 2 cm beyond it, so pointing
  straight down the grasp point cannot go lower than about 2 cm above the
  surface under it.
- The orientation is three angles (`state().right.pitch`, `.yaw`, `.roll`):
  `pitch` 0 points level, 90 straight down; `yaw` is the heading of the
  pointing direction, 0 along +x, +90 towards +y (the robot's left); `roll`
  turns the fingers about the pointing direction, and at 0 they close
  horizontally.  Pointing straight down, the fingers close along the line at
  `yaw + 90` degrees from +x: `yaw = locate(...).grasp_yaw` closes them across
  an object's narrow side.
- An angle left out of `move_to` keeps its current value: a plain position
  move does not turn the gripper.  Holding something, a turn of the wrist
  turns the object with it (pouring, standing something up, twisting).

## Motion
- `move_to(xyz, arm, gripper, pitch, yaw, roll, path)` solves the arm's
  inverse kinematics for the whole pose (the same solver the benchmark uses)
  and commands joint angles along a planned path.  `path="line"` moves the
  grasp point along a straight line, about 2.5 cm or 8 degrees per native
  action, or does nothing: use it for every approach, descent and lift near
  objects.  `path="joint"` swings the joints straight to the solution (up to
  0.15 rad per joint per action; the grasp point does not follow a line): only
  away from objects.  `"auto"` (default) takes the line where it can.
- Measured on this robot: targets arrive within 1-4 mm and 1 degree; a 10 cm
  line costs about 4 actions, a turn of the fingers by 60 degrees about 13.
  `reached` means within 1 cm and 5 degrees.
- Nothing moves when the pose is out of reach: `steps_used` is 0,
  `stop_reason` is `"out_of_reach"`, and `alternatives` names nearby requests
  that ARE reachable, each as the exact arguments to pass (another `pitch`,
  another height, another `yaw`, or `move_waist` first).  Take one of those or
  the other arm; do not repeat the pose.  A pose missed by at most 2 cm and 8
  degrees is taken anyway and `note` says how far off.  A line that cannot be
  followed is refused the same way (`path="auto"` then swings).
- Reach depends on height, pitch and yaw (the right arm with the starting
  waist; the left arm is the mirror image, y negated):
  - pointing straight down (`pitch=90`), with a suitable `yaw`: forward x from
    0.35-0.45 out to 0.70-0.75 for y between -0.45 and 0.0 at z 0.80-1.00,
    a little less at z 1.10; towards the other side (y +0.15) only x
    0.35-0.55.  Not every `yaw` reaches everywhere: at y +0.15 `yaw=0` reaches
    nowhere.  A refused pose names the `yaw` that works (see below).
  - `pitch=45`, heading forward: x up to 0.85-0.95, from about 0.35-0.80
    inward depending on height; not past y = 0 towards the other side.
  - level (`pitch=0`), heading forward: x up to 0.85-1.00, from about 0.35.
  So a top grasp suits low objects close in; for anything further out, or tall
  (a bottle, a cup), approach tilted or level, `yaw` towards the object, at
  about half its height.  The waist moves all of this with the chest.
- An EMPTY open gripper's two fingers are interchangeable, so `yaw` and
  `yaw + 180` are the same grasp: `move_to` takes whichever the arm reaches
  (`note` says so).  Holding something, it takes exactly the angles asked.
- Contact: `stop_reason == "blocked"` means the arm was held back for several
  actions (each took the benchmark's whole settling time): something is in the
  way, or it pushes against what it holds — at the end of a push, the end of
  the travel.  Look before trying again.  `resisted_actions` counts them.
- The other arm holds its joints, and its gripper its last command, during a
  motion.  Move one arm at a time and keep the idle one clear.
- `move_both(left=[x,y,z], right=[x,y,z])` moves both grasp points together
  along straight lines, each keeping its orientation, arriving together — for
  something carried in two hands.
- `home(arm)` returns one arm or both to the joint angles they started with,
  by a joint swing: release what you hold and get clear of tall things first.
- A press (button, switch) or a push is a motion: `move_to` a point 1-2 cm
  PAST the surface with `path="line"`; it stops `blocked` where the part stops.
  A hinged part (door, lid) moves on an arc about its hinge: several short
  `path="line"` moves a few centimetres apart along that circle.
- Every native action counts against the stage's budget (the card lists
  each stage's when they are known).  `steps_used` reports what a call spent.

## The waist and the base
- `move_waist(forward=, up=, turn_deg=, lean_deg=)` moves the chest, and both
  arms with it (they keep their joint angles; a held object comes along).
  `forward`/`up` are along base_link +x/+z; `reset=True` returns to the start.
  It costs 10-30 actions and changes every hand position: re-read `state()`
  and re-`locate` afterwards.  Use it when `move_to`'s `alternatives` say so,
  or to reach past the arm's range; otherwise keep it still.
- In scenes with a free wheeled base, `move_base(forward, left, turn_deg)`
  drives it, open loop: afterwards every measured object position is stale.

## Gripper
- Two conventions.  Commands: `robo.CLOSE`, `robo.OPEN`, and `robo.KEEP`
  (None: keep commanding what was last commanded — a held object stays
  squeezed).  Measurements: `state().right.opening` and `Gripper.opening` read
  1.0 open and 0.0 closed on nothing.
- `set_gripper(arm, command)` repeats the command until the fingers stop
  (an empty close takes about 16 actions) without moving the arm.  `holding`
  says the fingers stopped apart: something is between them — not proof that
  it is the INTENDED object.
- After a close that holds something, `set_gripper` waits (up to 30 actions)
  for the benchmark's next check, because a pick-up is measured from where the
  object was when its stage started.  `wait_for_check=False` skips the wait
  when the next stage is not judged on a lift.
- Carry with `gripper=robo.CLOSE` in every `move_to` that transports
  something; `robo.OPEN` in a motion drops it.
- Close across the narrow side: for a long object `locate` gives `grasp_yaw`
  (pointing down) and `grasp_fingers` (the direction across it) and
  `width_m`.  Fingers closing along a long object slide off it or catch what
  lies next to it.
- Confirm a grasp: lift a few centimetres, then `locate` the object again (by
  the same words, `near=` its old pixel): if it did not rise with the gripper,
  something else is held — open, and approach again.

## Perception
- Three cameras, each with a colour and a depth image: `head` sees the
  scene; `hand_left` / `hand_right` sit on the grippers and see what they are
  about to take.
- `locate("words", camera)` finds an object by a description (SAM3) and
  measures it with the depth image: `centre_xyz`, `top_z`, `size_xyz`,
  `width_m` / `length_m`, `tilt_deg`, `long_axis_deg`, `grasp_yaw`,
  `pixel_uv` (the pixel measured), `score` (the match's confidence), and
  `candidates`: every instance the words matched, best first.  The match is by
  WHAT a thing is, not where: words like "left" or "in the box" do not make it
  choose another of several alike objects.  When the one you mean is not the
  best match, pass `near=(row, col)` — a pixel on or next to it — to measure
  that instance.  Describe one object by colour, shape, size, material.
- `locate(row, col)` measures the object at a pixel: by its shape above the
  surface it stands on, else SAM3's mask at that pixel, else the pixel's
  colour region (`method` says which).  Read the head image yourself
  (`robo.image("head")`, RGB uint8; row 0 at the top) to pick the pixel.
- A low `score` (below about 0.4), a centre far from where the object was, or
  a size unlike the object means a wrong or partial match: look at the image,
  re-describe, or move the arm out of the view and look again.  An arm in
  front of an object hides part of it; a held object is hard to measure.
- The view that settles a measurement is close: for a precise grasp or
  placement, bring the gripper 8-12 cm above or in front of the object and
  `locate` it again from that arm's wrist camera, then act on that reading.
- All of them read the CURRENT observation: measure, act, then measure again.
  Positions from before a contact are stale.
- `back_project(row, col, camera)` returns the point seen at one pixel.

## How the benchmark judges an episode
- A task is a fixed sequence of stages, each judged by a check of the
  simulator's own state (the true object poses, not your measurements), run
  once every 30 native actions.  A stage starts when the one before it passes,
  and a passed check stays passed.
- ONLY WHAT A CHECK SEES COUNTS.  A state reached and left again between two
  checks is never seen: measured, a pot lifted 19 cm and set down again 15
  actions later scored 0 for "lift it 10 cm".  After reaching what a stage
  asks — lifted, inside, on top, upright, open — call `robo.wait_for_check()`
  (`checks=2` for the stages that need two checks in a row) before the next
  motion changes it.  It costs at most 30 actions per check.
- Each stage has its own action budget (on the card when known).  When a stage
  runs out of it THE EPISODE ENDS: the stages after it never start and score
  0.  The task's score is the average over its stages, so the early stages
  count as much as the last: get each one done, then go on.
- What the usual stages measure:
  - reaching an object: the gripper's centre within about 10 cm of it along
    every axis; nothing needs to be grasped.
  - picking up: the object rises more than 2 cm above where it was when the
    stage started and stays within 20 cm of the gripper.  A lift before the
    stage starts does not count (see Gripper).  Lift 5 cm or more and hold.
  - putting inside: the object's reference point inside the container's box
    (often its central 80 %) for two consecutive checks: after releasing, keep
    still for about 60 actions or until the next stage has begun.
  - putting on top / stacking: the placed object's bottom within 2 cm of the
    support's top and overlapping it; a stack's centres within 3-5 cm of the
    bottom one's, for two checks.
  - upright: the object's up axis within the stated angle (often 15 degrees)
    of vertical.  Release gently and low: a part that tips afterwards fails.
  - beside / left of / in front of: centres more than 5 cm apart along the
    robot's left-right or forward axis (or aligned within 5 cm), two checks.
  - into a region: the object's centre inside a fixed box.
  - opening / closing / pushing an articulated part: its joint ends inside a
    range, usually the end of its travel: push it all the way.
  - lifting by an amount: the object's height above where it started.
  - some stages are judged by a vision model from the cameras, with the robot
    hidden: make the scene look the way the instruction says.
- An object that drops below about 0.3 m (off the table) can END THE EPISODE
  at once.  Move carefully near edges, never drop things from height, keep
  what you carry low.
- Nothing else is judged: the arms need not go home, other objects may move,
  as long as nothing falls.  Finish when the last stage's condition holds in
  the scene; the robot then holds still until the benchmark ends the episode.
- Paraphrased instructions, distractors and look-alikes are part of the test:
  find the object the words single out (colour, shape, size, number,
  category, position, a logical or common-sense description), not the nearest
  similar one.  Count objects in the image when the words count them.

## Safety and recovery
- Inspect the scene after each phase (an image, or `locate`).  Stop the plan
  if an object is unstable or an arm is obstructed, and re-measure before
  recovering.
- After a failed phase name what went wrong (wrong object, missed grasp, lost
  hold, unreachable target, blocked path, premature release) and change one
  meaningful thing; do not repeat the same target or recovery twice.
- There is one episode and no reset: recover in place — re-observe,
  re-locate, re-grasp.  Thinking costs no actions: the simulator waits while
  you write code; measure before you move.
