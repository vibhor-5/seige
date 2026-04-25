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

### Phase 3: Generative Reward Policy Optimization (GRPO) & Iterated Best Response
*   **Tools:** `run_pipeline.sh`, `scripts/run_grpo.py`, `scripts/opponent_server.py`
*   **Process:** To solve multi-agent training collapse, the project uses an **Iterated Best Response** (Ladder) approach. Agents are trained sequentially rather than simultaneously.
    *   **The Frozen Opponent:** During training, the agent *not* being trained is hosted on an independent, frozen API server (`opponent_server.py` on port 8001). This provides a stable baseline for the trainee to learn against.
    *   **Chain-of-Thought (CoT) Joint Training:** Instead of complex hierarchical orchestrator models, the agents use CoT reasoning. The trainee is forced to generate a `<strategy>` block containing its high-level plan *before* outputting the final JSON payload. GRPO assigns the environment's reward to the entire sequence, mathematically training the agent to generate better meta-strategies and better execution payloads simultaneously.
    *   **Pipeline Execution:** `run_pipeline.sh` automates the entire process: starting the environment, booting the frozen opponent server, running the GRPO optimization loop, and automatically triggering a final evaluation.
*   **Output:** Isolated LoRA adapters (e.g., `grpo_red/final_adapter`) that climb the capability ladder generation by generation.

## 4. Evaluation Pipeline & Swarm Logic
*   **Tool:** `scripts/evaluate.py`
*   **Process:** Evaluates the agents in a live, turn-based interaction against the environment.
*   **Swarm Swapping:** Uses Hugging Face `peft` to load both `grpo_red` and `grpo_blue` adapters into VRAM simultaneously. When it is Red's turn in the environment, the script dynamically activates the Red LoRA weights (`model.set_adapter("red")`). When it is Blue's turn, it instantly swaps to the Blue weights. This allows for seamless, fast multi-agent self-play on a single GPU.

## 5. Codebase Map
*   **`openenv.yaml` & `server/`**: The configuration and FastAPI server for the core Seige target environment.
*   **`client.py`**: Contains `SeigeClient`, handling HTTP communication with the live environment API.
*   **`scripts/opponent_server.py`**: A FastAPI wrapper that hosts a frozen adapter to act as the baseline opponent during GRPO training.
*   **`scripts/run_grpo.py`**: The independent RLHF/GRPO script incorporating live environment rewards and calling the opponent API to resolve multi-turn clashes.
*   **`scripts/evaluate.py`**: The multi-agent evaluation script utilizing dynamic LoRA swapping.
*   **`run_pipeline.sh`**: The master automation script for deploying the servers, orchestrating the generational training run, and tracking it with Weights & Biases.

## 6. Future Context for Agents
If an AI agent is reading this file to gain context on the repository:
1. Always ensure actions are formatted as valid JSON matching the schema definitions.
2. If modifying GRPO or the reward function, remember the agents now utilize a `<strategy>` CoT reasoning block before the JSON. The reward function must parse or strip this appropriately before sending the JSON to the environment.
3. The training loop requires *both* `localhost:8000` (the environment) and `localhost:8001` (the frozen opponent) to be healthy and running. Do not run `run_grpo.py` in isolation; always use `run_pipeline.sh` or replicate its background server initialization.
