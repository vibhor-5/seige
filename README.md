# seige: Adversarial Oversight via Mechanistic Interpretability

> Red attacks at the activation level. Blue defends by reading the machine's mind.

seige is a multi-agent RL environment where attackers and defenders compete
over a frozen target LLM. Unlike text-only red-teaming, Red agents inject
**steering vectors** and **activation patches** directly into the model's
residual stream — attacks invisible to naive output classifiers. Blue agents
learn to detect these attacks via **mechanistic probing** and explain them
in structured natural language.

**Why it matters for AI safety:** Current LLM deployment relies on output-level
filters. This environment trains oversight agents that watch AI at the
mechanistic level — the first reproducible RL testbed for this problem.

## 🔗 Links

| Resource | URL |
|---|---|
| HuggingFace Space (live env) | https://huggingface.co/spaces/YOUR_USERNAME/seige |
| Mini-blog | https://huggingface.co/blog/YOUR_USERNAME/seige |
| Demo video (<2 min) | https://youtube.com/YOUR_VIDEO |
| Training Colab | https://colab.research.google.com/YOUR_NOTEBOOK |
| Wandb training run | https://wandb.ai/YOUR_RUN |

## 📊 Training Results

![Reward Curves](assets/reward_curves.png)
![Before/After](assets/before_after.png)
## Models

- Target model: `google/gemma-4-E2B`
- Red/Blue agent model: `unsloth/Qwen3-14B`

The target model is a prop loaded by the environment server. Red and Blue agents are text-in/text-out policies trained separately with GRPO.

## Local Smoke Run

```bash
python -m pip install -e ".[test]"
python -m pytest
SEIGE_TARGET_BACKEND=mock python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
python scripts/smoke_server.py
```

## OpenEnv Run

```bash
python -m pip install -e ".[test]"
openenv validate
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
openenv validate http://127.0.0.1:8000
```

The OpenEnv server exposes `/reset`, `/step`, `/state`, `/schema`, `/metadata`,
`/mcp`, and `/ws`. Use `client.SeigeOpenEnvClient` for persistent WebSocket
episodes.

Precompute direction artifacts before real activation-space training:

```bash
python scripts/precompute_directions.py
```

## HF/GPU Run

```bash
SEIGE_TARGET_BACKEND=hf \
SEIGE_TARGET_MODEL_ID=google/gemma-4-E2B \
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In a separate training job:

```bash
SEIGE_AGENT_MODEL_ID=unsloth/Qwen3-14B \
SEIGE_ENV_URL=http://localhost:8000 \
python train/grpo_red.py

SEIGE_AGENT_MODEL_ID=unsloth/Qwen3-14B \
SEIGE_ENV_URL=http://localhost:8000 \
python train/grpo_blue.py
```

Save adapters only. Do not merge 4-bit weights after GRPO.
