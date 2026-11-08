# Distil a cross-task library

You are writing one skill library that a robot policy will be given on **tasks it has never
seen**. Your material is what an improver learned on {N_WORD} *other* tasks of the same robot
(RoboCasa365: a PandaOmron mobile manipulator in simulated kitchens, one Franka arm with a
two-finger gripper on a mobile base with a torso lift, RGB cameras only, robot API `robocasa_rgb`
without VLA primitives, policy GPT-6 Astra writing Python cells). The tasks run in many kitchens
and differ in their native-action budget (each task's official horizon). Whether this transfers is
the question of the experiment, so write only what you believe holds beyond the task it was
learned on.

## Inputs (read-only): `inputs/<task>/`

{TASKS}

- `skills/*.py`, `lessons/*.md`: that task's final library, exactly as its policy was given it.
- `FINDINGS-last.md`: the improver's last diagnosis of that task's failures.
- `dev_rounds.txt`: successes per round on {N_DEV} dev layouts (r0 = empty library), the library
  revision of each round, and which round(s) evaluated the library in this directory. Use it to
  judge how much a library is worth: a library whose task never did better than with the empty
  library taught nothing proven. When the last worker's revision was not kept, `FINDINGS-last.md`
  describes a library that is not the one in this directory.

These improvers saw only each episode's trajectory and whether it succeeded; they never saw
the benchmark's judging code. You do not have it either, and must not guess at it.

The robot material is in `inputs/robot/`:

- `system_prompt.md`: everything the policy reads besides its library and its task card,
  verbatim (contract, including the note that introduces the library; scoring; API reference;
  operating knowledge). Your lessons are appended right after its operating knowledge. Do not
  repeat what it already says; add only what the source experience adds to it or sharpens.
- `api/robocasa_rgb.py`, `api/_robot.py`: the robot API code. Read it to check that what you
  write calls real functions with real arguments.

## Output: `out/`

The policy receives this as its library: every `out/lessons/*.md` is appended **verbatim** to its
prompt (no filtering), and `out/skills/*.py` is importable as `library.<module>`. So:

1. **`out/lessons/*.md`: at most 12 000 characters in total.** Cards in the format the inputs
   use (`## id`, `stage:`, `symptom:`, `applies:`, `do:`, `evidence:`). Each card:
   - is a general rule about perception, grasping, placing, transport, operating articulated
     fixtures (doors, lids, knobs, levers, buttons), base and torso use, recovery, verification,
     or using the step/camera budget; not about one task's objects or goal;
   - names **no task** and no task-specific object, dimension or goal configuration in `do:`
     (`applies:` says *when* it applies in general terms, e.g. "a hinged door that swings
     toward the robot"; kitchen fixture and object categories are fine there);
   - has `evidence:` naming the source tasks whose libraries/findings support it, and should
     come from at least two of them, or one task where it was clearly the difference between
     failing and succeeding rounds. Say so when it is a single-task rule.
   - Prefer few, sharp cards over many vague ones. Drop anything you cannot state precisely.
2. **`out/skills/*.py`: optional, at most 2 modules, 15 000 characters in total.** Only
   helpers that are task-agnostic (e.g. a verified place-and-check, a re-measure-from-two-views
   locate). Every function's docstring says when to use it. Use only the real `robocasa_rgb`
   robot API; no task names, no hard-coded object sizes or positions. If you are not confident a
   helper is correct across tasks, leave it out: a wrong skill is worse than none. Make sure it
   at least parses: `python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" FILE`.
3. **`out/NOTES.md`** (not given to the policy): for every card and skill, which inputs it came
   from; and a list of what you deliberately left out as task-specific, with one line why.

There must be at least one file in `out/lessons/` or `out/skills/`.

## Rules

- Work only inside this directory (`{DIR}`);
  write only under `out/`. No subagents, no web.
- Do not try to find out which tasks the library will be tested on; it must not be tailored to
  them.
- Check the character limits yourself before you finish (`wc -c out/lessons/*.md out/skills/*.py`).
