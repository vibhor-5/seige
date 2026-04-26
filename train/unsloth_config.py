from __future__ import annotations

import os
import torch

def _patch_torch_inductor_for_unsloth() -> None:
    # Some Torch builds (common on RunPod images) do not expose torch._inductor.config.
    # Unsloth expects it during import and crashes otherwise.
    if hasattr(torch, "_inductor") and not hasattr(torch._inductor, "config"):
        class _CompatInductorConfig:  # pragma: no cover - import-time compatibility shim
            pass
        torch._inductor.config = _CompatInductorConfig

def _patch_torch_dtypes_for_torchao() -> None:
    # Some TorchAO builds expect experimental dtypes (torch.int1, etc.) that
    # are absent in older Torch wheels. Map them to nearest available types so
    # imports succeed (these paths are not used in our 4-bit LoRA flow).
    fallback_map = {
        "int1": torch.int8,
        "int2": torch.int8,
        "int3": torch.int8,
        "int4": torch.int8,
        "int5": torch.int8,
        "int6": torch.int8,
        "int7": torch.int8,
        "uint1": torch.uint8,
        "uint2": torch.uint8,
        "uint3": torch.uint8,
        "uint4": torch.uint8,
        "uint5": torch.uint8,
        "uint6": torch.uint8,
        "uint7": torch.uint8,
    }
    for name, fallback in fallback_map.items():
        if not hasattr(torch, name):
            setattr(torch, name, fallback)


AGENT_MODEL_ID = os.getenv("SEIGE_AGENT_MODEL_ID", "google/gemma-4-E4B")
TARGET_MODEL_ID = os.getenv("SEIGE_TARGET_MODEL_ID", "google/gemma-4-E2B")
ENV_URL = os.getenv("SEIGE_ENV_URL", "http://localhost:8000")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "seige")

MAX_SEQ_LENGTH = int(os.getenv("SEIGE_AGENT_MAX_SEQ_LENGTH", "4096"))
LOAD_IN_4BIT = os.getenv("SEIGE_LOAD_IN_4BIT", "1") == "1"
LORA_R = int(os.getenv("SEIGE_LORA_R", "8"))
LORA_ALPHA = int(os.getenv("SEIGE_LORA_ALPHA", "16"))


def lora_target_modules() -> list[str]:
    raw = (os.getenv("SEIGE_LORA_TARGET_MODULES", "") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return ["q_proj", "v_proj"]


def grpo_config(output_dir: str, run_name: str):
    from trl import GRPOConfig

    return GRPOConfig(
        num_train_epochs=int(os.getenv("SEIGE_GRPO_EPOCHS", "3")),
        per_device_train_batch_size=int(os.getenv("SEIGE_GRPO_PER_DEVICE_BATCH", "2")),
        gradient_accumulation_steps=int(os.getenv("SEIGE_GRPO_GRAD_ACCUM", "4")),
        learning_rate=float(os.getenv("SEIGE_GRPO_LR", "1e-5")),
        logging_steps=int(os.getenv("SEIGE_GRPO_LOGGING_STEPS", "10")),
        output_dir=output_dir,
        report_to=os.getenv("SEIGE_REPORT_TO", "wandb"),
        run_name=run_name,
        num_generations=int(os.getenv("SEIGE_GRPO_NUM_GENERATIONS", "6")),
        max_prompt_length=int(os.getenv("SEIGE_GRPO_MAX_PROMPT_LENGTH", "256")),
        max_completion_length=int(os.getenv("SEIGE_GRPO_MAX_COMPLETION_LENGTH", "64")),
        temperature=float(os.getenv("SEIGE_GRPO_TEMPERATURE", "0.8")),
        beta=0.04,
        use_vllm=False,
        reward_weights=None,
        save_steps=50,
        eval_steps=50,
    )


def load_agent_model():
    os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")
    _patch_torch_inductor_for_unsloth()
    _patch_torch_dtypes_for_torchao()
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
        target_modules=lora_target_modules(),
    )
    return model, tokenizer
