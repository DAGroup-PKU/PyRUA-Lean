# Registration, background persistence and recovery

A request for a complete pipeline, or for a cohort to go from stage X to the end, first registers through `tools/datafarm_runs.py` the
original request text, the authorization basis, the stage, the fixed sample denominator, the code version and the output directory. A repair
is a child run of the original run. Do not bypass the ledger to start a loop with no provenance.

A formal run requires traceable, clean, committed code. Do not overwrite user changes to clean the state; open a separate isolated worktree
and commit only this task's changes. The budget scope follows the existing authorization; only exceeding the deployment's GPU allowance or
using an unauthorized paid service needs separate handling. Read credentials from a protected file; they must not appear in argv, logs or manifests.

Host through ledger launch. After the launcher exits, confirm the real worker's PID, PPID, session, command, and log/state; for GPUs also check
the job or lease against the actual worker. After an interval long enough to expose exit problems, do a second check in a new tool call: besides
being alive, the worker's log, counters, outputs or check times must have advanced. Waiting may be segmented, and a single block must not exceed
60 seconds so that communication continues; the total observation window follows the task's needs.

Only when both checks pass may you say "running persistently in the background", and give the run/job, update time and progress with it. A
sentinel heartbeat only proves the sentinel is alive. When the remote shows no change, also prove that the poller's check time is updating and
that the remote task ID is valid.

A failed lease or PID query is unknown, not a lost task. Prefer reusing the task or job ID and breakpoints to avoid duplicate cost. Releasing a
lease does not mean the process stopped; after cancelling, verify the actual worker. Record state changes as ledger events and bind the verification
report at finalize. Report any gaps explicitly, and never shrink the denominator.

New collection or replanning obeys `configs/joint-margin-requirements.json`: the source of the real limits for every joint, the actual readback of
the whole segment (including home, wait and recovery), next-state and connection, and margin and dwell statistics. Validate the task boundaries with a
small sample before expanding the batch. Candidate margins are candidates awaiting validation, not new hardware limits. Never crop only the labels
without recapturing the corresponding images.
