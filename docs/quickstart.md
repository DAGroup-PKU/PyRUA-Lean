# Quickstart

Use Python 3.11 or newer and Git in a full source checkout. The runtime needs only the standard library. The image pipeline needs Pillow, and the full CPU test suite needs the `dev` and `workflow` extras (`pip install -e '.[dev,workflow]'`). The simulator and training environments are deployed separately; see [deployment](deployment.md).

Install a coding agent. To use SoL-Pi or Claude Code, see [examples/runtime](../examples/runtime/README.md) and use its config in the commands below. For the built-in driver, install Codex and log in yourself, then confirm `codex exec --help`, `codex exec resume --help` and `codex login status`. The runtime inherits the login and does not read or copy credentials. If the model is omitted, the deployment setting applies; test any custom model or provider in your deployment.

## Run a task

Save the user's request to a file, then prepare the task:

```bash
python tools/pipeline_agent.py prepare --episode /data/episode.mp4 \
  --request-file /data/request.txt --target-count 5 --stop-after white --output /data/task
python tools/pipeline_agent.py context --work /data/task --role orchestrator --output /data/task/context
python tools/pipeline_agent.py supervise --work /data/task --config examples/runtime/native-codex.json
```

`prepare` pins the source code, the input and the cost policy. To use another mode, add `--cost-policy examples/runtime/budget-disabled.json` or your own policy. The enforced example cannot be paired with native Codex. See [cost control](reference/cost-control.md).

Later commands with `--work` run the pinned source, and the pin does not change automatically.

## Run phase by phase

To control cost, [run a task phase by phase](reference/phased-runs.md): each phase is one fresh agent session with a USD cap the host enforces, and the flow ends at the white delivery.

## Run under the ledger

For a continuous run, commit clean source and create a run per the [run ledger](reference/run-ledger.md), registering the request, authorization, input and fixed target denominator. Then host the same entry point:

```bash
tools/datafarm_runs.py launch PRUN-... --role supervisor -- \
  python tools/pipeline_agent.py supervise --work /data/task --config examples/runtime/native-codex.json
tools/datafarm_runs.py check PRUN-...
python tools/pipeline_agent.py runtime-status --work /data/task
```

After launch, check the actual worker, PID, session and progress. After an interval, check again and observe the logs, check times or output advancing. A sentinel heartbeat does not stand in for the worker.

## Stop, resume, verify

```bash
python tools/pipeline_agent.py stop --work /data/task --reason 'operator stop'
python tools/pipeline_agent.py runtime-status --work /data/task
python tools/pipeline_agent.py supervise --work /data/task --config examples/runtime/native-codex.json --resume
python tools/pipeline_agent.py verify --work /data/task
```

`stop` terminates the current model process group, keeps the external task IDs and does not cancel remote GPU or provider jobs, so recovery starts by checking the original job status. `--resume` clears the stop request and keeps the budget, the session count, the retry deadline and the thread ID. If more sessions are needed, raise the cap in the configuration; do not delete the control file.

A task is complete only when `verify` passes on its delivery. A process exit of 0 or a `complete` state does not replace it.
