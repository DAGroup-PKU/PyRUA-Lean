# PyRUA-Lean Coder v0.1-4B

!!! info "Weights coming soon"

    The weights will be released on Hugging Face. This page will then give the model card and a tested command.

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

## Running an open model

An open model runs behind any OpenAI-compatible endpoint (a local SGLang or vLLM server, for example) and is driven
by the in-process loop of `pyrualean play --runtime openai` ([Agent Configuration](../guide/agents.md)).
