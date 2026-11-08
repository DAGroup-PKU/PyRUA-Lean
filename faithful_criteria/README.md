# criteria_v2: instruction-faithful RoboDojo criteria

This directory holds the instruction-faithful criteria C^faith of the paper *Learning the Task or
Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement*: rewrites of the
RoboDojo judging code (`run_reward`, and `get_score` kept consistent with it) for the RoboDojo tasks
whose original criteria C^orig (RoboDojo commit cb2a37d) ask for more than the task instruction, or
disagree with it. The scope is every base task whose criteria were not found to match its
instruction, plus `solve_equation`, whose criteria match the instruction but contain a clear bug.
The paper's appendix ("Instruction-faithful Criteria") lists every change in a table; this file
gives the full detail.

## Principles

1. Only the criteria change (`run_reward`, with `get_score` following it). The instruction is never
   changed.
2. The return of both arms to home (`all_robot_back_to_origin`) in the original criteria is kept.
3. Hidden rules (requirements the instruction does not state and common sense does not imply) are
   removed or relaxed. Defects of the judging code are fixed. Where the criteria are looser than the
   instruction, they are tightened if the simulator state can detect the difference reliably;
   otherwise they are kept and the reason is given. Hard thresholds are relaxed to what still
   clearly satisfies the instruction, with the old and new values listed.
4. Asset annotation errors are corrected in memory only (`override_functional_frames`, patched again
   in every episode). The shared assets (`Assets/`) are not modified.
5. The original RoboDojo repository is not changed at all: a build copies the parts it overlays.

## Layout

| Path | Content |
|---|---|
| `reward_manager/`, `tasks/` | The modified files; a build puts them at their paths in the RoboDojo repository (`env/reward_manager/*.py`, `task/RoboDojo/tasks/*.py`) |
| `build_source.sh <tag>` | Builds `faithful_criteria/build/<tag>/` (see below) |
| `build/<tag>/` | An immutable RoboDojo source root judged by C^faith; created by `build_source.sh`, not shipped |
| `tests/test_offline.py` | Checks without Isaac Sim: the new primitives on the real asset annotations, reproductions of the old defects, and that every overlaid task registers its rules |
| `dual_table.py` | The 2x2 table of both criteria on the same episodes, from dual-judged evaluation roots |

`build_source.sh <tag>` needs `ROBODOJO_SOURCE_ROOT` (the RoboDojo checkout at cb2a37d; set by
`tools/env.sh` from `paths.env`). It copies RoboDojo's `env/` and `task/`, lays `reward_manager/` and `tasks/` over
them, and links the other parts the evaluation client reads (`src`, `utils`, `scripts`, `env_cfg`,
`XPolicyLab`, `Assets`, `third_party`) to the checkout. It puts the cb2a37d file of every overlaid
task under `task/RoboDojo/tasks/criteria_v1/` for the shadow judge (below), writes `MANIFEST.txt`
(the sha256 of every overlay file in the build) and `overlay.diff` (the diff against cb2a37d), and
makes the copied parts read-only. A tag that exists is never rebuilt, since runs pin the sha256 of
the judging code they were judged by; use a new tag instead (`OUT=<dir>` builds to a scratch
directory, for tests). The paper's build is tag `v2-b9`.

## Commands

From the repository root:

```bash
source tools/env.sh
bash faithful_criteria/build_source.sh v2-b9          # -> faithful_criteria/build/v2-b9/

# offline checks (no Isaac Sim needed)
ROOT=faithful_criteria/build/v2-b9 ROBODOJO_SOURCE_ROOT=$ROBODOJO_SOURCE_ROOT \
    $ROBODOJO_SIM_PYTHON -m pytest -q -p no:cacheprovider faithful_criteria/tests/test_offline.py

# the 2x2 of C^orig and C^faith over dual-judged evaluation roots
python faithful_criteria/dual_table.py ROOT [ROOT ...] [--json OUT]
```

To judge with C^faith, use the build as the `source_root` of an evaluation (`eval/`, e.g. through
`ROBODOJO_FAITHFUL_ROOT` in `paths.env`) or of a learn campaign whose learner should read C^faith.
With `ROBODOJO_SOURCE_ROOT` (the cb2a37d checkout) as `source_root` instead, episodes are judged by
C^orig alone.

### Dual judging

Every overlaid task calls `RewardManager.attach_v1_shadow(self)` at the end of its `run_reward`: the
rules of the cb2a37d file of the same task (`task/RoboDojo/tasks/criteria_v1/<task>.py`) are
registered on a RewardManager of their own, which is evaluated after the main judge at every step.
The shadow only observes: its failures are kept in its own flags, it does not change `env.success`
and does not end the episode; it reads the asset metadata as cb2a37d has it, before any in-memory
correction; when it registers its rules it does not reload the scripted trajectory of the
supporting arm (`make_kong` draws the opponent's discarded tile there at random), and it saves and
restores the random-number state. Both criteria thus judge the same episode, which avoids the
run-to-run variation of a program on the same layout (perception and physics are stochastic).

- Each criterion's verdict is its first success or its first irreversible failure, locked from then
  on. An irreversible failure under C^faith is only recorded
  (`[criteria_v2] v2 judged failed at step N (the episode goes on)`); it does not end the episode.
- The episode ends early only when both criteria have reached a verdict (`ended=both decided`), so
  neither verdict is truncated. Otherwise it runs until the program finishes or the step limit is
  reached (`ended=over`).
- The ledger's `success` (the result the evaluation reports) is the C^faith verdict; if the program
  finishes after C^faith has succeeded, it is restored to success.
- The client log gets one line per criterion:
  `[criteria_v2] v1 shadow final: env 0 success=... first_success_step=... failed=... steps=...`
  (C^orig) and
  `[criteria_v2] dual final: env 0 v2 success=... first_success_step=... failed=... steps=... ended=...`
  (C^faith). After a C^faith failure its stages may still complete, so the `stages` field is not the
  verdict.

## New judge primitives

In `reward_manager/`, appended at the end of the classes; the existing code is unchanged.

| Primitive | Purpose |
|---|---|
| `override_functional_frames(env_idx, label, fixes)` | Replaces functional-point frames of one instance in this episode's metadata, by `(model_name, model_id)` (the record is replaced by a patched deep copy; asset files and other instances are untouched); logs `[criteria_v2] ...` in the client log when it applies |
| `add_functional_frames(env_idx, label, additions)` | Appends functional-point frames to one instance in this episode's metadata, in the same way |
| `is_A_point_close_and_aligned_to_any_B_point` | One functional point of A and the **same** functional point of B satisfy both the distance and the per-axis orientation conditions (the original `is_axis_aligned` compares frame 0 of both only) |
| `is_A_in_B_aligned_ellipse` | A's root point lies inside the ellipse inscribed in the xy face of B's `aligned_bbox`, at a height between B's bottom and top (in B's own frame) |
| `is_one_tower` | A set of parts forms one connected stack: a part rests on another when its bottom touches the other's top (height difference <1.5 cm) and their xy projections overlap, and on the table when it touches the table; every part of the largest connected set is supported, and the number of parts and the number of levels of the vertical support chain reach the required values |
| `count_joint_position_ratio_change_from_above_to_below` | Counts the falls of a joint ratio from above to below the thresholds (registered alone in a `query`, so it advances at every step); logs the ratios at the first step and every counted event |
| `is_joint_position_ratio_change_count_in_range` | Reads that count only: it lies in [`min_count`, `max_count`] |
| `is_all_A_in_B_local_box` | Every A root lies in a box given in B's root frame (the support-circle and functional-bbox checks compare world xy only, so they cannot tell an object in a lifted or tilted B from one on the table under it) |
| `has_A_been_held_by_every_arm` | Every target arm has held A off the table at some step (a handover); see `insert_key` |
| `RewardManager.attach_v1_shadow(task)` | Judges the episode with the cb2a37d rules of the task alongside, without acting on it (above) |

## Per-task changes

Each entry gives the task instruction. A `_random` variant gets the same change unless stated.

**hang_mugs / hang_mugs_random** (Hang all the mugs on the mug rack.)
- Defect 1: the orientation check compares with hook 0 of the rack only, so of the 6 hooks, hooks 2
  and 3 (whose x axis runs along the rack's ±y, the others' along ±x) never pass; the distance check,
  in contrast, iterates over all hooks, and the two are not bound to the same hook. Now
  `is_A_point_close_and_aligned_to_any_B_point`: on one and the same hook, distance <4.5 cm, z axis
  <60° and x axis ±60° (thresholds unchanged).
- Defect 2: frame 0 of the handle of mug/00014 is the identity quaternion (z axis up), while its own
  frame 1 and the handles of all other mugs have z along the mug's +x. Frame 0 becomes
  `[0.06, 0, 0.01, 0.5, 0.5, 0.5, 0.5]` (position unchanged, orientation as for mug 13 and others).
- Unchanged: each mug lifted >4.5 cm, handle above z = 0.9, both arms back home.
- Offline tests: 17 mugs × 2 racks × 6 hooks × 2 handle frames are all judged hung; the old criteria
  reproduce "hooks 2 and 3 never count" and "mug 14 never counts"; a handle lying across the hook
  (rotated 90°) or 5 cm lower still fails.

**solve_equation** (criteria match the instruction, but have a bug)
- The position, x-orientation and y-orientation checks each take an OR over the candidate pieces,
  without binding them to the same piece; now an AND per candidate piece, then an OR over pieces.
  Identical to the upstream fix in branch `origin/fix/task-check-and-instruction` (29910f3).
  Thresholds unchanged (1.8 cm, 45°).

**make_toast / make_toast_random** (Pick up two slices of bread, place them into the toaster, and press the lever down.)
- Defect: the checks of both slots accept "any slice of bread", so one slice can satisfy both; now
  two **different** slices, one in slot1 and one in slot2.
- Defect (random): the four corner points of slot1 and slot2 of toaster/00004 are mixed up, so the
  polygon of slot1 covers all of slot2. The two rectangles are rebuilt from the slot carriers
  measured in the USD, link_1 (y ∈ [-0.0267, -0.0073]) and link_2 (y ∈ [0.0073, 0.0267]), each with a
  3 mm margin; x range, height and orientation are kept from the metadata. Point 0 of slot2 of
  toaster/00002, (0.03, -0.026, 0.079), lies on the slot1 side and 13 cm high; it becomes the corner
  the other three points are missing, (-0.079, 0.006, -0.07).
- Hidden final-state constraints removed: "exactly 2 slices left on the bread rack" and "all 4
  slices upright (30°)".
- Tightened: the order of the instruction (bread first, then the lever). Stage 1: both slices in the
  slots and the lever not pressed (<85%); stage 2: both slices in the slots, the lever pressed >85%,
  both arms back home.
- Unchanged: the height window of the bread root relative to the slot's functional point (base
  (0.03, 0.065), random (-0.05, 0.068)), lever 85%.

**pour_balls_into_vase** (Pour all the balls from the cup into the vase.)
- Defect: "in the vase" uses the xy convex hull of the `oriented_bbox` and only requires being above
  the vase bottom. The oriented_bbox of vase/00002 is a tilted 27×24×28 cm box, so balls lying on the
  table outside the vase count as inside; for the other vases the box corners count as inside. Now
  `is_A_in_B_aligned_ellipse` (the elliptic cylinder inscribed in the vase body, at a height between
  bottom and rim).
- Hidden final-state constraint removed: "the cup is set upright after pouring (<15°)".
- Unchanged: both arms back home.

**pour_by_language** (…from the first … bottle into the first … bowl, … second …, … third …. Then reset the robot arm.)
- Defect: the order check is attached to the rising edge of "either arm back home" (`[[L, R]]` is an
  OR in nested rules) and almost never fires; now it is checked at every step (bowl 2 or 3 poured
  while bowl 1 is not, or bowl 3 poured while bowl 2 is not → permanent failure).
- Defect: the instruction asks for a reset, which the original criteria do not check; success now
  requires all three bowls poured and both arms back home at the same step.
- Timing trap: each bowl was judged only at the one step its bottle returned upright; now a
  level-triggered check at every step, and the bottle need not be set upright.
- Hard thresholds: liquid in the bowl ≥95% and left in the bottle ≤10% → bowl ≥90%, bottle ≤15%
  (other parameters unchanged).
- The trigger of `get_score` likewise becomes both arms back home (as upstream).

**build_tower** (Build a tower using the wooden blocks and wooden boards.)
- Hidden rules: the instruction gives no structure, but the original criteria accept a single
  6-level configuration (which board on which two pillars, pillars grouped by label {1,5}/{2,6}, all
  8 parts tilted <5°, top two levels centred within 2.3 cm, 20° between the boards). In addition, the
  original `is_A_up_B` compares z only, not xy.
- A "tower" in v2: the 8 parts (all blocks and boards) form one connected stack, each part on the
  table or on another part of the structure, longest vertical support chain ≥4 levels
  (`MIN_LEVELS`); all 8 parts at rest (displacement <2 mm per step); both arms back home. The table
  height is the lowest bottom of the parts at the start of the episode.
- `get_score`: ≥3 parts in 2 levels → 10, ≥5 parts in 3 levels → 30, complete → 100.
- Offline tests: the original configuration built by a scripted solution program and the 8 parts
  stacked into one pile both succeed; the roof left on the table, all parts laid flat, the structure
  floating, the top two parts off their support, and only 3 levels all fail.

**press_by_number** (Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm.)
- Defect (physics): once released, a button cap rests at about 75% of its travel in the evaluation
  simulator (a scripted program could only get a press counted by gripping the cap and pulling it
  above its top), while the original count requires a fall from >0.95 to <0.5 and "released"
  requires >0.9, so a normal press and release is never counted. Now a fall from >0.65 to <0.5
  counts once, and >0.65 is released (between 0.5 and 0.65 the state does not change, so a
  half-press is not counted twice). The new counting primitive logs the ratio of every button at the
  first step of the episode (`[criteria_v2] ... ratio at the first step`) and one line per counted
  press, to check 0.65 in the simulator.
- Contradiction between instruction and implementation: the original criteria require left red ×n0
  → blue → middle red ×n1 → blue, with the blue button pressed exactly twice. Now each red button is
  pressed exactly the number on the card behind it (num0 → left red, num1 → middle red; cards and
  buttons correspond by position only), the red buttons in any order and possibly alternating; once
  both red counts are reached, the blue button is pressed once; finally the blue button is released
  and both arms are back home. An extra press of any button fails immediately (wrong count or
  repeated confirmation); pressing blue before the red counts are reached fails immediately ("then …
  to confirm").
- New primitives `count_joint_position_ratio_change_from_above_to_below` (alone in a `query`, advancing
  the count at every step) and `is_joint_position_ratio_change_count_in_range` (reads the count only).

**swap_blocks** (Swap the two blocks using the empty mat, pressing the button after each move.)
- Button thresholds as in `press_by_number` (0.95/0.9 → 0.65).
- Kept: exactly 3 presses, an extra press fails immediately ("after each move" fixes the number);
  the intermediate placement stages after each move; "on the mat" 3 cm.
- Removed: the 3 stages requiring the block to be lifted 3 cm before each move (pushing it over is a
  move too).
- Tightened: the last stage (button released and arms back home) also requires both blocks to be
  still in the swapped positions (before, taking a block off its mat afterwards still passed).

**make_kong** (Wait for the opponent to discard a tile, then declare a kong with the matching tiles.)
- Declaring the kong = the three hand tiles matching the opponent's discarded tile are pushed down,
  face up (30°, unchanged).
- Removed: drawing the replacement tile mahjong9_0 and standing it at (0.319, -0.15) within 1.5 cm
  and 7° with the grippers open, and the stage-1 requirement that the replacement tile is untouched.
- The other hand tiles must stay standing, 7° → 20° (this is kept, since otherwise pushing all tiles
  down would pass); the pair mahjong4_0/4_1 must stay standing too (upstream fix 21b8d23).
- Kept: "wait for the opponent": if the player's arm leaves its origin while the opponent's arm is
  away from its origin, the episode fails permanently (the instruction says wait); layouts where the
  opponent fails to push its tile are marked unstable (the task's premise does not hold).

**play_tic_tac_toe** (Play tic-tac-toe as the first player and fill the board with the opponent.)
- Inconsistency fixed: an opponent's move is recognised at 1.8 cm, but success requires every piece
  within 1.5 cm of its cell centre, so an opponent move the environment accepts can block success;
  both now use 1.8 cm (`run_reward` and `get_score`).
- Kept: the turn mechanism (place one piece; the opponent moves only once the arm is back within
  0.3 m / 30° of its origin) and "moving the arm during the opponent's turn fails": this is the
  environment's protocol for taking turns and common sense, not a rule added by the criteria. Layouts
  where the opponent fails are excluded as unstable.

**pour_liquid_into_cup / pour_liquid_into_cup_random** (Pour the liquid from the bottle into the cup.)
- Timing trap: judged only once, on the rising edge of the bottle returning upright (30°) → a
  level-triggered check at every step; the bottle need not be set upright.
- Hard threshold: liquid in the cup 97% → 90%; left in the bottle ≤15% unchanged (as in the v2
  `pour_by_language`). The original criteria do not require a reset, and this is kept.

**fasten_screws** (Insert and tighten each screw into the nut of the same color.; the bolts are static geometry, the nuts are the movable parts)
- "Tighten" tightened: the nut's sinking depth 0.009 → 0.035 m (fully seated is about 0.038 m; USD:
  threaded section 15–52.5 mm, pitch 3 mm); the original 9 mm corresponds to a quarter of the way.
- Coaxiality relaxed 1 mm → 4 mm (radial clearance of engaged threads ≤0.7 mm; resting on the outer
  rim only gives at most 3.8 mm; a nut not on its bolt is off by at least 15 mm).
- Rotation is not checked: being fully seated already implies it was turned (if the SDF threads take
  effect); otherwise rotation has no physical meaning in the simulator. Upright 10°, grippers open
  and arms back home are kept.
- Note: whether the simulator lets a nut sink 35 mm has not been verified (see the paper's appendix).

**insert_key** (Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.)
- Hidden rule: the turn is accepted only clockwise by 30–90° → turned by more than 30° in either
  direction (the rotation cavity in the USD is symmetric about the slot's xz plane).
- "Hand it over" tightened: the new primitive `has_A_been_held_by_every_arm` is the first stage. Both
  hands must have held the key while it was >2 cm off the table. A hand holds the key when its grasp
  point (end-effector link6 + 0.145 m along the pointing axis, the same as PyRUA-Lean's `tcp_pos`)
  lies within the key's bbox enlarged by 1 cm, with a gripper reading of 0.01–0.6 (closed on
  something, neither closed on nothing nor open). In a smoke run, a scripted program's two-hand
  grasps of the key both read 0.36, which left too little margin to an upper bound of 0.4; the bound
  is 0.6. The first time each hand is recognised as holding the key, and a reading out of range while
  on the key, are logged.
- Depth, xy 7 mm and upright 7° unchanged. `get_score`: 15 (lifted) / 40 (handed over) / 100.

**fill_pen_holder** (Hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.)
- Hidden rule: pen tip down (4 checks) removed. The xy criterion changes from the pen's check end to
  the pen's centre within 3.5 cm of the holder axis (the check end is about 4.6 cm off the axis when a
  pen leans tip up, so the old criterion implied tip down); depth 3.5 cm unchanged.
- Tightened: the first stage requires the holder to have been lifted >2 cm; the final state requires
  the holder back on the table (within 1.5 cm of its start height).
- "Holding the holder while placing the pens" is not checked: the gripper readings of the possible
  grasps cannot be told apart reliably, and without a simulator calibration this would misjudge.

**fill_egg_holder** (Place the four eggs from the basket into the egg holder, then close the lid.)
- The height window and the lid >0.9 were rechecked and are kept (the inscribed radius of an egg cell,
  2.36 cm, is larger than the egg radius, 2.06–2.11 cm, so an egg reaches the cell bottom).
- Defect: the oriented_bbox used by `is_A_in_B` extends in xy behind the box body (y up to 0.115,
  while the body reaches only 0.06), so an egg lying on the table behind the holder passes. Each egg
  must also be within 0.055 m in xy of the centre, in the egg holder's frame.

**store_tools_in_toolbox** (Place each tool into its matching position in the toolbox, then reset the robot arm.)
- Orientation 10° → 15° (the annotated slot frames themselves differ from the tools' resting poses
  by 0.6–5.4°, leaving the pliers only about 4.6° of margin; a wrongly turned tool still does not fit
  the recess). Slot distance 5 cm and the top not above the box rim unchanged; `get_score` checks the
  orientation the same way (pliers either side up).

**pack_objects_into_box / pack_objects_into_box_random** (Place all the objects into the box with their front sides facing left.)
- The asset definition of "front" was checked one by one (car front, brush head, hammer head, shoe
  toe); "front within 60° of world −x" is kept.
- Removed: object long axis within 45° of the box's long side, and box heading within 40° (in
  `run_reward` and `get_score`). Box upright 15° and root xy inside the box bottom are kept.

**store_laptop_and_headphones / store_laptop_and_headphones_random** (Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.)
- Headphones: the whole quaternion within 30° → headband point within 5 cm of the holder point (kept)
  plus the headphones' z axis up within 30°, heading free.
- Laptop: lid closed >0.85 kept; vertical changes from "hinge up within 10°" to any edge down (±y or
  ±x up within 10°); "in the stand" changes to the midpoint of any edge of the base inside the stand's
  bbox (the new primitive `add_functional_frames` adds the midpoints of the other three edges, with
  coordinates from the USD of laptops 0–3).

**organize_table** (Place the alarm clock on the drawer, put the figurine on the stand, place the mouse on the mouse pad, and push the keyboard into the frame.)
- Figurine "on the stand": the centre of the square plate within the contact radius of the
  figurine's round base, {5: 0.045, 6: 0.025, 7: 0.045, 8: 0.02, 11: 0.032} → {5: 0.0715, 6: 0.0427,
  7: 0.0678, 8: 0.0241, 11: 0.0483} (the old values amount to requiring the 5 cm plate to be fully
  covered).
- Mouse: the signed orientation (local y within 45° of world +x) is removed; instead it must lie
  upright (15°). Keyboard: orientation unsigned (turned 180° also counts); covering the frame centre
  kept.
- Tightened: besides its height, the alarm clock must be over the drawer (`is_A_in_B(alarm, drawer)`).
- `get_score` drops the at-rest checks, which shared pre_state with the reward and had no effect.

**sweep_blocks / sweep_blocks_random** (Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.)
- "Pick up the broom" tightened: the first stage requires the broom to have been lifted 5 cm above
  its start.
- The handover to the right hand is not checked: there is no reliable criterion for which hand holds
  the broom (it can be grasped anywhere, and the gripper opening is not calibrated). "Dustpan to the
  left of the broom" is kept as a cheap proxy for the right hand holding the broom, with the gap
  ≥0.1 m → >0.
- Final state: dustpan tilt 5° → 20°; "the dustpan must not be lifted 1 cm" removed; the new
  primitive `is_all_A_in_B_local_box` decides, in the dustpan's own frame, whether each block is in
  the pan (a block in the pan has its centre at local z ≥ −0.0114, one under the dustpan at ≤ −0.034;
  the bound is −0.02; USD broom_shovel/00000).

## License

The files under `reward_manager/` and `tasks/` are modified copies of RoboDojo files (MIT License, Copyright (c) 2026
RoboDojo Contributors). `overlay.diff` in a build lists every change against cb2a37d.
