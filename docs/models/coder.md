# PyRUA-Lean Coder v0.1-4B

[:simple-huggingface: DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B](https://huggingface.co/DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B){ .md-button .md-button--primary }

**PyRUA-Lean Coder v0.1-4B** is Qwen3.5-4B trained with GRPO in the PyRUA-Lean framework, on 45 LIBERO-PRO task
instances, as its planner: it writes the cells, the primitives do the rest. On 655 task instances it never saw in
training it goes from **18.8% to 44.1%**, above the Qwen3.5-27B base model (29.2%).

## Results

Success rate (%), PyRUA-Lean framework:

| Planner | Trained on (45) | LIBERO-PRO other (155) | RoboTwin 2.0 (250) | RC365 atomic (90) | RC365 composite (160) | All unseen (655) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 37.8 | 13.5 | 14.0 | 52.2 | 12.5 | 18.8 |
| Qwen3.5-27B | 64.4 | **38.1** | 20.8 | 65.6 | 13.1 | 29.2 |
| **PyRUA-Lean Coder v0.1-4B** | **71.1** | 37.4 | **52.4** | **76.7** | **19.4** | **44.1** |

RoboTwin 2.0 and both RoboCasa365 sets are benchmarks it was not trained on at all; the LIBERO-PRO column of the
unseen part holds that benchmark's other tasks.

## Running the model

An open model runs behind any OpenAI-compatible endpoint (a local SGLang or vLLM server, for example) and is driven
by the in-process loop of `pyrualean play --runtime openai` ([Agent Configuration](../guide/agents.md)). With SGLang:

```bash
python -m sglang.launch_server --model-path DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B --port 30000 \
    --context-length 65536 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --constrained-json-disable-any-whitespace
```

Then one LIBERO-PRO episode, on a task the model never saw in training (RPent's LIBERO-PRO stack installed as the
[Quick Start](../getting-started/quickstart.md) describes):

```bash
export RPENT_ROOT=/path/to/rpent OPENAI_API_KEY=none
pyrualean play --robot libero --arm cells --suite libero_goal_swap --task 3 --seed 0 \
    --runtime openai --base-url http://127.0.0.1:30000/v1 --api-key-env OPENAI_API_KEY \
    --model DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B \
    --openai-max-tokens 8192 --openai-tool-choice required --openai-temperature 1.0 \
    --openai-extra-body '{"top_p": 0.95, "top_k": 20}' \
    --images on-demand --feedback pure --guides --max-turns 40 --budget-s 5400 \
    --out runs/coder-demo --cuda-device 0
pyrualean summarize runs/coder-demo
```

These are the settings of the paper's evaluation: a reply of at most 8,192 tokens, a context of 65,536 tokens (an
episode that outgrows it ends there), at most 40 code cells and 90 minutes.
