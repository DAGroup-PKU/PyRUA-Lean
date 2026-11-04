# DataFarm documentation

Start with [quickstart](quickstart.md), [deployment](deployment.md) and [architecture](architecture.md).
[Troubleshooting](troubleshooting.md) maps runtime statuses to recovery steps, and [development](development.md) lists the rules for changing this code.

## Reference

One note per component. Each states what the code guarantees and what it does not, with its records, entry points and configuration.

| Component | Note | Code |
|---|---|---|
| Capability layer, context compiler, instruction audit, source pinning | [capability-layer](reference/capability-layer.md) | `src/datafarm/agent/context.py`, `tools/audit_instruction_layers.py`, `src/datafarm/evolution/integration.py` |
| Agent runtime: prepare, drive, supervise, review queue, recovery | [agent-runtime](reference/agent-runtime.md) | `tools/pipeline_agent.py`, `src/datafarm/agent/runtime.py`, `src/datafarm/agent/native_codex.py` |
| Phased runs: fresh sessions with USD caps | [phased-runs](reference/phased-runs.md) | `tools/pipeline_phases.py`, `src/datafarm/agent/phases.py` |
| Model-call ledger and VLM reviewer calibration | [agent-calls](reference/agent-calls.md) | `src/datafarm/agent/call_log.py`, `src/datafarm/verify/reviewer_calibration.py` |
| Run ledger | [run-ledger](reference/run-ledger.md) | `src/datafarm/ledger/` |
| Batch supervisor | [batch-supervisor](reference/batch-supervisor.md) | `src/datafarm/batch/` |
| Grounding and the physical-quantity ledger | [grounding](reference/grounding.md) | `src/datafarm/core/grounding.py` |
| Object assets (optional) | [usdcraft](reference/usdcraft.md) | `src/datafarm/simulation/usdcraft.py` |
| Simulator qualification | [simulator-qualification](reference/simulator-qualification.md) | `src/datafarm/simulation/sim_qualification.py` |
| Task contract, frozen judge, expert harness | [expert-harness](reference/expert-harness.md) | `src/datafarm/expert/` |
| Render contract | [render-contract](reference/render-contract.md) | `src/datafarm/render/policy.py` |
| H3 control hybrid and Seedance | [rendering](reference/rendering.md) | `src/datafarm/render/h3/`, `src/datafarm/render/seedance/` |
| Render verifier | [render-verifier](reference/render-verifier.md) | `src/datafarm/verify/` |
| Review, delivery and admission | [delivery-and-admission](reference/delivery-and-admission.md) | `src/datafarm/task/contract.py`, `src/datafarm/task/white_delivery.py`, `src/datafarm/core/admission.py` |
| Policy training (pi0.5) | [policy-training](reference/policy-training.md) | `tools/launch_pi05_training.py`, `src/datafarm/training/` |
| Whole-task cost control | [cost-control](reference/cost-control.md) | `src/datafarm/agent/cost_budget.py` |
| Capability evolution | [capability-evolution](reference/capability-evolution.md) | `src/datafarm/evolution/` |

## What the agents read

The rules loaded into agent sessions are not documentation pages. They live in [`prompts/instructions/`](../prompts/instructions), [`prompts/roles/`](../prompts/roles) and [`.agents/skills/`](../.agents/skills), and
[`configs/agent-instructions.json`](../configs/agent-instructions.json) lists what each role loads. Start from [`AGENTS.md`](../AGENTS.md). [`configs/README.md`](../configs/README.md) maps the configuration directory.
