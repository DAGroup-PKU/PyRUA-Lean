# Task-expert round {{ round.id }}: <task>

You write a **task expert**: a Python controller class that makes the simulated robot complete the task below.
The harness measures your wall-clock time, token cost and rollouts, and scores your submissions on hidden holdout
starts after the round. Work efficiently.

- Model: `{{ round.model }}` ({{ round.backend }} backend). Task id: `{{ task_id }}`.
- Your workspace (the only place you can write): `{{ round.workspace }}`.
- This brief: `{{ round.brief }}`. Your rollouts appear under `{{ round.rollouts }}`.

## Where the previous round stands

{{ handoff_md }}

## Task

- **Objects**: <objects, their sizes and how they are identified>.
- **Start**: <how initial states are sampled, and which families exist>.
- **Goal**: <the end state, in the same terms the judge measures>.
- **Physics** (certified, fixed for this round): {{ facts.physics }}
- **Task IR**: {{ facts.task_ir }}

## Expert interface

`Class(home=home)`; the runner drives it:

```python
while not expert.done:
    observation = port.observe()
    plan = expert.next_plan(observation)
    execution = port.execute_joint_plan(plan)
    expert.complete(plan, port.observe(), execution)
```

- **Imports allowed in the expert**: {{ facts.allowed_imports }}. The harness audits the file on every
  `xh rollout` and `xh submit` (allowed imports, no file I/O, no dynamic code, the class exists) and rejects a
  violation before a rollout is spent.
- Default file `expert.py`, default class `{{ facts.default_class }}` (override with `--file` / `--class`).

## Verifier ({{ verifier.version }}, run by the harness outside your sandbox)

{{ verifier.name }}. Judge points: **{{ verifier.judge_points }}**.

{{ verifier.criteria_md }}

Each rollout's result (`xh result <attempt>`) gives the verdict fields; its directory under
`{{ round.rollouts }}` holds the verdict, the logs and the state trace. The judge source is not
readable; the criteria above are exact.

## Runtime limits

{{ runtime_facts_md }}

## Budget

- At most **{{ budget.max_counted_rollouts }} counted development rollouts**, at most
  **{{ budget.max_concurrent_rollouts }} running at once**, and **{{ budget.total_wall }}** of active wall-clock
  time, whichever comes first; API cost at most **{{ budget.max_usd }}**; at most {{ budget.max_sessions }}
  sessions of up to {{ budget.wall_per_session }} each. Between sessions the harness resumes your conversation;
  time lost to provider outages is not charged.
- **Counted**: the rollout was judged, or your expert raised an error or exceeded the trace capacity.
  **Not counted (infra)**: invalid initial state, simulator start-up failure, lost node, timeout, judge failure.
  After {{ budget.max_infra_failures }} infrastructure failures the round stops for an operator.
- A rollout in flight reserves a counted slot until it ends. `xh status` shows what is left and any harness
  messages (for example "15 min left").

## Tools

You run things only through `xh` (on your PATH); you have no ssh or scheduler credentials.

- `xh rollout --seed N [--class C] [--file expert.py] --note "<what changed>" --hypothesis "<expected effect>"`:
  snapshot your expert and run it on development seed N ({{ dev_seeds }}). Returns an attempt id at once.
- `xh wait [A-0001 ...] --timeout {{ budget.wait_cap_s }}`: wait for attempts, at most {{ budget.wait_cap_s }} s
  per call; call it again if they are still running.
- `xh result A-0001`: outcome class, family, verdict fields, run directory.
- `xh status` and `xh budget`: budget, running and recent attempts, submissions, harness messages.
- `xh submit "<note>" [--final]` and `xh flag-for-human "<reason>"`: see the two sections below.
- Trace analysis: use `{{ facts.analysis_python }}` (numpy, scipy).

The harness records every request, attempt, outcome and session in `{{ round.dir }}/ledger/` (read-only for you);
you do not need to keep an attempts ledger or a clock file.

## Readable paths

Read-only inputs you may use:

{{ readable_md }}

Inside those, these are masked (not readable):

{{ masked_md }}

Round files: `{{ round.brief }}`, `{{ round.rollouts }}`, `{{ round.handoff }}` (the previous round's bundle),
`{{ round.references }}` (verified reference experts, if any), `{{ round.dir }}/ledger/`,
`{{ round.dir }}/submissions/`, `{{ round.dir }}/broker/`. Apart from the system directories (`/usr`, `/etc`, ...),
nothing else on the machine is visible to you.

## Forbidden

{{ forbidden_md }}

## Deliverables

{{ deliverables_md }}

## Submit protocol

- `xh submit "<note>"` snapshots your expert as a submission at any time. Submit whenever you have a candidate
  that is better than your last one: every submission is scored.
- `xh submit --final "<note>"` snapshots the final expert and ends the round once you finish your reply. Update
  `REPORT.md` first.
- The harness scores every submission later on the hidden holdout seeds and builds a score-vs-cost curve. You
  only get an acknowledgement, with a warning if that exact expert never completed a development rollout. If you
  never submit, the harness snapshots `expert.py` when the round ends.

## Holdout policy

The holdout seeds are hidden: the harness draws them from a range you never see, runs them only after the round,
never shows them in any file you can read, and never accepts them through `xh`. Requests for seeds outside
{{ dev_seeds }} are rejected and recorded as rule violations. Develop and select on development seeds only, and
balance the families yourself: a failure mode confined to one family needs several development seeds per
family to show up.

## How to stop or abort

- Normal end: `xh submit --final "<note>"`.
- If the task cannot be done in this setup (the physics makes a required motion impossible, the tools are broken,
  or the rules conflict), run `xh flag-for-human "<reason, with evidence>"`. The round ends cleanly and an operator
  reads your reason; this is a legitimate outcome, not a failure.
- The harness ends the round by itself when a budget runs out; submit before that.

## References

{{ references_md }}
