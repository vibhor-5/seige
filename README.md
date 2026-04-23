# seige

`seige` is an adversarial oversight environment for training Red attackers and Blue defenders around a frozen target model.

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
