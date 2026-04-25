from __future__ import annotations

import os


AGENT_MODEL_ID = os.getenv("SEIGE_AGENT_MODEL_ID", "unsloth/Qwen3-14B")
TARGET_MODEL_ID = os.getenv("SEIGE_TARGET_MODEL_ID", "google/gemma-4-E2B")
ENV_URL = os.getenv("SEIGE_ENV_URL", "http://localhost:8000")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "seige")

MAX_SEQ_LENGTH = int(os.getenv("SEIGE_AGENT_MAX_SEQ_LENGTH", "4096"))
LOAD_IN_4BIT = os.getenv("SEIGE_LOAD_IN_4BIT", "1") == "1"
LORA_R = int(os.getenv("SEIGE_LORA_R", "16"))
LORA_ALPHA = int(os.getenv("SEIGE_LORA_ALPHA", "32"))


def grpo_config(output_dir: str, run_name: str):
    from trl import GRPOConfig

    return GRPOConfig(
        num_train_epochs=int(os.getenv("SEIGE_GRPO_EPOCHS", "3")),
        per_device_train_batch_size=int(os.getenv("SEIGE_GRPO_BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.getenv("SEIGE_GRPO_GRAD_ACCUM", "4")),
        learning_rate=float(os.getenv("SEIGE_GRPO_LR", "1e-5")),
        logging_steps=int(os.getenv("SEIGE_GRPO_LOGGING_STEPS", "10")),
        output_dir=output_dir,
        report_to=os.getenv("SEIGE_REPORT_TO", "wandb"),
        run_name=run_name,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=256,
        temperature=0.8,
        beta=0.04,
        use_vllm=False,
        reward_weights=None,
        save_steps=50,
        eval_steps=50,
    )


def load_agent_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=AGENT_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return model, tokenizer
