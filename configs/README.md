# Configuration

General policy files apply to every task. Templates are copied into a task's run workspace and filled in by the
agent from that task's episode and request; the user does not fill them in. Numbers in a template are illustrations
that give the expected form, not values measured for your robot or task.

| File | Job |
|---|---|
| `agent-instructions.json` | Which instructions and skills each agent role loads |
| `standing-rules.json` | Rules that prompts, reviewers and verifier specs must follow; `tools/lint_standing_rules.py` checks those that declare patterns or required values |
| `task-cost-policy.json` | Default whole-task cost policy |
| `joint-margin-requirements.json` | Review contract for near-limit joint occupancy |
| `external-assets.json` | External weights and repositories, with hashes, checked by `datafarm doctor` |
| `batch/template.json` | Batch supervisor task |
| `expert-harness/template.json`, `template-brief.md` | Expert-harness adapter and the brief it renders |
| `expert-harness/prices.json` | Token prices for sessions that report no cost |
| `render-verifier/template.json` | Render verifier spec |
| `training/template.json` | pi0.5 training run |
| `render/render-intent.template.json` | Render intent. It stays uncompilable until the agent resolves its mode and fields |
| `render/overlay.template.json` | Per-task slots of the render prompt |
| `render/h3-control-hybrid-v1.json` | Default H3 render recipe |
| `render/h3-ood-restyle-v1.json` | Out-of-distribution tabletop and light edit of an accepted render |
