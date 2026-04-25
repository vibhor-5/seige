# Seige: Adversarial Oversight Environment

## 1. Project Overview
**Seige** is an adversarial multi-agent environment designed to test and train Mechanistic Interpretability-level AI oversight at scale. 

The core loop involves an escalating arms race between two AI agents operating on a frozen target LLM (the environment):
*   **Red Agent (Attacker):** Attempts to extract hidden secrets or bypass the policy of the target system using strategies like steering vectors, activation patching, persona manipulation, and jailbreaks.
*   **Blue Agent (Defender):** Attempts to monitor the system, detect anomalies, block malicious payloads, and apply patches to secure the target system.

Both agents output structured JSON payloads to interact with the environment API.

## 2. Agent Architecture & Infrastructure
To ensure extreme memory efficiency and fast iteration on consumer GPUs / standard cloud instances (RunPod, HF Compute):
*   **Base Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
*   **Agent Instantiation:** Instead of loading two massive models into VRAM, the project uses a single base model instance loaded in 4-bit quantization. 
*   **Role Separation:** The "minds" of the agents are separated physically via completely independent LoRA (Low-Rank Adaptation) adapters.

## 3. Training Pipeline
The training process follows a three-phase pipeline:

### Phase 1: Synthetic Data Generation
*   **Tool:** `scripts/generate_sft_dataset.py` (or `colab_sft.ipynb`)
*   **Process:** Uses a powerful teacher model (`nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter) to generate highly diverse and creative JSON payloads for both Red attacks and Blue defenses.
*   **Output:** `sft_dataset.jsonl`

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Tool:** `colab_sft.ipynb`
*   **Process:** The base Qwen 3B model is fine-tuned using Unsloth on the combined synthetic dataset. 
*   **Output:** A unified `sft_adapter` that understands the basic JSON schema and role requirements for both Red and Blue tasks.

### Phase 3: Generative Reward Policy Optimization (GRPO)
*   **Tool:** `scripts/run_grpo.py`
*   **Process:** To ensure Red and Blue reward signals do not interfere, the unified `sft_adapter` is duplicated and trained independently.
    *   **Red Training:** Runs GRPO specifically on Red system prompts. The reward function sends the generated attack JSON to the live environment (`localhost:8000`) and optimizes based on the environment's attack success score.
    *   **Blue Training:** Runs GRPO specifically on Blue system prompts, optimizing based on defensive success.
*   **Output:** Two physically isolated LoRA adapters: `grpo_red/final_adapter` and `grpo_blue/final_adapter`.

## 4. Evaluation Pipeline & Swarm Logic
*   **Tool:** `scripts/evaluate.py`
*   **Process:** Evaluates the agents in a live, turn-based interaction against the environment.
*   **Swarm Swapping:** Uses Hugging Face `peft` to load both `grpo_red` and `grpo_blue` adapters into VRAM simultaneously. When it is Red's turn in the environment, the script dynamically activates the Red LoRA weights (`model.set_adapter("red")`). When it is Blue's turn, it instantly swaps to the Blue weights. This allows for seamless, fast multi-agent self-play on a single GPU.

## 5. Codebase Map
*   **`openenv.yaml`**: The configuration defining the Seige target environment.
*   **`client.py`**: Contains `SeigeClient`, the wrapper that handles HTTP communication with the live environment API.
*   **`colab_sft.ipynb`**: The notebook containing Data Generation and the SFT phase.
*   **`scripts/run_grpo.py`**: The independent RLHF/GRPO training script for Red and Blue agents using live environment rewards.
*   **`scripts/evaluate.py`**: The multi-agent evaluation script utilizing dynamic LoRA swapping.

## 6. Future Context for Agents
If an AI agent is reading this file to gain context on the repository:
1. Always ensure actions are formatted as valid JSON matching the schema definitions found in the system prompts of `run_grpo.py`.
2. When running training loops, ensure the target environment server (`uvicorn server.app:app`) is running independently, as the reward functions rely on live HTTP requests.
3. The environment operates via `client.py` - look there to understand how actions are passed to and observations received from the environment.
