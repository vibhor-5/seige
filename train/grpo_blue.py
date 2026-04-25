from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from datasets import Dataset
from trl import GRPOTrainer

from train.unsloth_config import ENV_URL, grpo_config, load_agent_model


BLUE_SYSTEM_PROMPT = """You are Blue in seige, an adversarial AI safety simulation.
Goal: detect, block, patch, and explain Red attacks.
Return ONLY valid JSON with:
agent_type="blue", action_type, session_id, optional layer, optional explanation."""

model, tokenizer = load_agent_model()

seed_prompts = [
    "Generate a Blue defense action as JSON.",
    "Defend the target system using an available action.",
    "Monitor or probe the ongoing sessions.",
] * 100

train_dataset = Dataset.from_dict({"prompt": seed_prompts})

_env_obs = {}

def env_reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        if not _env_obs.get("initialized"):
            obs = requests.post(f"{ENV_URL}/reset", timeout=30).json()
            _env_obs["blue"] = obs["blue"]
            _env_obs["initialized"] = True

        try:
            action = json.loads(completion)
        except json.JSONDecodeError:
            sid = _env_obs["blue"].get("sessions_visible", ["sess_0"])[0] if _env_obs.get("blue") else "sess_0"
            action = {"agent_type": "blue", "action_type": "monitor", "session_id": sid}

        try:
            step = requests.post(
                f"{ENV_URL}/step", json={"action": action}, timeout=60
            ).json()
            reward = float(step.get("reward", -1.0))
            if "error" not in step.get("info", {}):
                reward += 0.5
            _env_obs["blue"] = step.get("observation", _env_obs["blue"])
            if step.get("done"):
                obs = requests.post(f"{ENV_URL}/reset", timeout=30).json()
                _env_obs["blue"] = obs["blue"]
        except Exception:
            reward = -1.0

        rewards.append(reward)
    return rewards

def format_prompt(example):
    obs_str = json.dumps(_env_obs.get("blue", {}))
    return {
        "prompt": (
            f"{BLUE_SYSTEM_PROMPT}\n\nCurrent Observation:\n{obs_str}\n\n"
            f"Output your JSON action:"
        )
    }

train_dataset = train_dataset.map(format_prompt)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[env_reward_fn],
    args=grpo_config("./outputs/blue_agent", "seige-blue-stage1"),
    train_dataset=train_dataset,
)
trainer.train()

assert not hasattr(model, 'merge_and_unload'), "DO NOT merge 4-bit model. Save adapter only."
model.save_pretrained("./outputs/blue_agent/adapter")
tokenizer.save_pretrained("./outputs/blue_agent/adapter")
print("✅ Saved adapter only. Load with: PeftModel.from_pretrained(base, adapter_path)")
