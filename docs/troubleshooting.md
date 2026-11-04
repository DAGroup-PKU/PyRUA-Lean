# Troubleshooting

| Symptom | Check and recovery |
|---|---|
| cost-budget-blocked | Run `cost-status`. Enforced mode needs a real cap and complete receipts, and the native Codex driver is refused for it. Do not delete the ledger or rewrite the pin. A new task may choose advisory mode. |
| Native dollar cost unknown | Check `native-result.json` and the JSONL token evidence. This tool does not confirm a per-task dollar amount for a ChatGPT login, and missing usage is not zero. |
| capacity/429/502/503 | `supervise` applies bounded exponential backoff and persists `retry_at` and the retry count. Reaching the cap reports failure and keeps the same thread ID and all attempts. |
| session-budget-exhausted | Check the cumulative author, review and retry records. If necessary, raise the total cap in the configuration by hand, then start. Do not delete the control state. |
| waiting-external-jobs | Check the real status of the original `external_jobs` ID, which the job supervisor updates in state. A heartbeat alone does not wake the model. |
| waiting-worker | Check the process identity in `runtime-status` and the session logs. Do not start again while the worker is alive. |
| recovery-required | The last launch did not save a thread ID. First check the process, logs and the original external ID. Do not guess or submit anew after a failure. |
| Lock conflict | A `drive` or `supervise` controller already exists. Check the real PID: the existence of the file does not mean the lock is still held. Do not delete the file to bypass the lock. |
| Review not resumed | `state.review_requests` must have a unique ID, instructions and hashes of the actual subjects. Check the independent reviewer execution and its report, and use a new ID for a revision. |
| `complete` rejected by `verify` | `verify` needs a fully source-bound delivery and an independent review. Exit 0, model text or a state string does not certify success. |
| GPU/provider jobs remain after stop | `stop` only stops the local model group. Remote jobs continue to be monitored under their original IDs. Check them before resuming and do not resubmit. |
| ImportError/numpy/cv2 | The main runtime does not need these packages. The full tests and the image stages need the corresponding extras installed in a virtual environment with Python 3.11 or newer. |

Continuous running needs two ledger checks separated in time, together with evidence that the worker is actually alive and making progress. If a log does not advance for a long time, check the worker rather than trusting a sentinel. Quality acceptance also needs independent physical and visual evidence.

When the state JSON written by the model is invalid, the runtime stops and saves the error, and does not treat the model's exit 0 as success. First save the invalid original file and the repair record, fix only the control metadata and revalidate the JSON, then resume the original thread. The original actions, images and metrics must not be changed.
