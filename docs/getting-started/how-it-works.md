# How It Works

A tool-calling agent spends a full LLM call on every small step of a task and re-sends everything before it each
time. A PyRUA-Lean agent writes code against a robot object, `robo`, instead: one call can find an object, move
above it, grasp, check the gripper and try again.

## What the agent gets

One tool, `python(code)`, over a persistent namespace that holds `robo` (and `np`, `math`, `time`). Variables,
functions and files the agent made stay there from one cell to the next. A second tool, `finish`, ends the episode.
The code runs in a sandbox: a fixed set of imports, file access confined to its workspace, and no handle to the
simulator behind `robo`.

This is one cell exactly as GPT-6 Astra wrote it in a LIBERO-PRO episode, placing a held bowl on a plate; the three
comment lines are ours, and `plate` was located by an earlier cell:

```python
# compose: compute a placement target from the scene geometry
held_at_plate = robo.segment(point=(314, 795))
print('bowl above plate', held_at_plate)
assert held_at_plate.found
correction = np.array(plate.world_xyz[:2]) - np.array(held_at_plate.world_xyz[:2])
place_xy = np.array(robo.state().eef_pos[:2]) + correction
surface_map = robo.world_map()
bowl_patch = surface_map[290:365, 740:850]
bowl_heights = bowl_patch[:,:,2]
bowl_heights = bowl_heights[(bowl_heights>1.04) & (bowl_heights<1.17)]
bottom_z = float(np.quantile(bowl_heights, 0.02))
placement_z = plate.world_xyz[2] + robo.state().eef_pos[2] - bottom_z + 0.007
print('correction', correction, 'bottom', bottom_z, 'placement z', placement_z)
assert np.linalg.norm(correction)<0.07 and 0.95<placement_z<1.05
# compose: lower, check, and release within the cell
lower = robo.move_to([*place_xy, placement_z], gripper=+1, step_clip=0.012, tol=0.005, max_steps=100)
print('lower', lower)
if lower.reached and not robo.done:
    print('release', robo.release())
print('done', robo.done)
# select: ask for an image only if the task is unfinished
if not robo.done:
    robo.show('agentview')
```

The cell printed five lines and asked for no image: the bowl was on the plate and the task was done. With tool
calling, the same step took four LLM calls, and every move returned three camera images.

## What `robo` returns

- **Results, not exceptions.** A motion blocks until it is done and returns what actually happened
  (`Move.reached`, `Pick.success`, ...); a mask that was not found is a result too. Only misuse raises.
- **A finished episode stops everything.** Once the benchmark's success check fires, `robo.done` is true and every
  further motion is refused. RoboDojo judges only the final scene: its robot has no `done`, and RoboDojo's client
  ends the episode itself when the task is solved.
- **Every call is recorded**, with its arguments, its outcome and the simulator steps it took.

Each [robot page](../robots/index.md) lists that robot's primitives and the full API reference.

## What the model reads

1. The contract: what to do, how the code is run, how the episode is judged.
2. The API reference of `robo`, generated from the robot class itself, so it cannot drift from the code
   (`pyrualean api` prints LIBERO-PRO's; each robot page shows its own).
3. The robot's knowledge text: frames, units, how the primitives behave, what to watch out for. The same for every
   task; no task recipes.
4. A task card: the instruction, the first camera images and the robot's starting state. Nothing privileged.

A `play` episode saves the whole prompt its model read as `prompt.txt` in the run directory; `pyrualean prompt`
prints the prompt of a single shot (`generate`).
