## Environment

- You run non-interactively. Never ask questions or wait for input: decide, record the rationale and continue. Never print, copy or search for credentials or API keys.
- Task directory: `{T}`. Workspace W = `{W}`, prepared by `tools/pipeline_agent.py prepare` (`input.json`, `state.json`, `task/`, `evidence/`).
- Pinned general source, read-only: `{SOURCE}`.
- Mandatory rules: `{SOURCE}/prompts/phases/protocol.md`. User intent: `W/input.json` (`request`). Also do not open: {FORBIDDEN}.
- Spend: this task has a hard cap of ${TASK_CAP} across all phases, and this phase a hard cap of ${PCAP}. The host kills the session at either. Check the running total with `{COST_COMMAND}`. If you approach a cap, write your outputs and stop.
- Economy: summarize tool outputs and never dump large files or images in bulk.
- Do not start other model or agent sessions yourself. The host launches independent reviews as separate sessions. If your phase needs a review, write the request file and stop.
{ENVIRONMENT}
