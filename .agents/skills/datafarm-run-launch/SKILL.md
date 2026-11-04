---
name: datafarm-run-launch
description: Register, launch, independently persistence-check, recover and close a DataFarm pipeline run.
---

# datafarm-run-launch

Read `prompts/instructions/operations.md` and `docs/reference/run-ledger.md`. Full runs and cohort stage-to-end requests must use `tools/datafarm_runs.py`; recovery uses parent/child lineage.

1. Preserve valid user authorization for GPU use and paid providers. Do not seek repeated approval within scope. Resolve render mode and real references separately before submitting render work.
2. Use a clean traceable commit without overwriting user changes; register request, authorization source, phase range, fixed IDs, task/cohort and artifacts.
3. Launch through ledger sentinel. Check actual worker after launcher exit, including PID/PPID/session/log and GPU job/lease where applicable.
4. In a new tool call after an appropriate interval, recheck worker plus observable progress. Split waits at 60 seconds to keep communication responsive. Only after both checks report persistent running with IDs and timestamps.
5. Query failures are unknown, not permission to resubmit. Reuse existing task IDs/checkpoints, record interventions and bound retries. Sentinel heartbeat alone is insufficient.
6. Apply joint-margin requirements before scaling newly collected/replanned trajectories; never clip action labels alone. Preserve existing task camera/qualification requirements.
7. Finalize against evidence and report gaps explicitly; maintain attempts, rejections, replacements and costs.

Deployment scope: GPU and paid-provider authorization comes from the current deployment and session. Obtaining the repository grants none, and permission already given is not asked for again.
