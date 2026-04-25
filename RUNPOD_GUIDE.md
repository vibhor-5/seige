# Seige: End-to-End RunPod Training Guide

This guide will walk you through exactly how to spin up a cloud GPU, download your codebase, pull your SFT adapter from Hugging Face, run the automated Generational GRPO training, and save your new agents.

## Step 1: Deploy the Pod
1. Go to [RunPod](https://www.runpod.io/).
2. Click **Deploy**.
3. Select a GPU that has at least 48GB of VRAM:
   - **1x A40 (48GB)** (Recommended for cost/performance)
   - **1x L40S (48GB)**
   - **1x A6000 (48GB)**
4. Choose the **RunPod PyTorch 2.x** template (or the **Unsloth** template if available).
5. Click **Deploy** and wait for the Pod to initialize.

## Step 2: Connect and Clone
1. Once the Pod is running, click **Connect** -> **Web Terminal** (or connect via SSH).
2. Inside the terminal, clone your repository and enter the directory:
```bash
git clone https://github.com/vibhor-5/seige.git
cd seige
```

## Step 3: Install Dependencies
RunPod PyTorch templates have most requirements installed, but we need to ensure the specific libraries for TRL, Unsloth, and FastApi are ready:
```bash
# Install standard dependencies
pip install fastapi uvicorn requests wandb datasets uv

# Install Unsloth and PEFT
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# TransformerLens is used for the default small white-box target. vLLM is optional for black-box baselines.
pip install transformer-lens
# pip install vllm
```

**Resume / optimizer state:** this repo pins **torch ≥ 2.6** so `transformers` can load checkpoint `optimizer.pt` / `scheduler.pt` (older 2.5 + recent `transformers` raises a `torch.load` / CVE-2025-32434 error). The default `./run_pipeline.sh` `TORCH_INDEX_URL` is `cu124` for those wheels. Run `INSTALL_DEPS=1 ./run_pipeline.sh` (or match `requirements.txt` to that index) so the template’s PyTorch is not left at 2.5.x. After `requirements.txt`, the script **re-installs the torch + torchvision + torchaudio triple** from that index; if you only upgrade `torch`, a preinstalled `torchvision` for 2.5.1 (e.g. `+cu121`) will break imports (`torchvision::nms` / `BloomPreTrainedModel` errors).

## Step 4: Download Your SFT Adapter
Because the `sft_adapter` was too large for GitHub, we uploaded it to Hugging Face. Download it directly into the `seige` folder using a Python one-liner (this will be extremely fast on RunPod):
```bash
python3 -m pip install -U huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BART-ender/seige-sft-adapter', local_dir='sft_adapter')"
```

If you prefer `uv`, avoid resolving the whole training project just to download files:
```bash
uv run --no-project --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BART-ender/seige-sft-adapter', local_dir='sft_adapter')"
```

## Step 5: Start Generational Training
To ensure your training doesn't get killed if you accidentally close your browser window, start a `tmux` session:
```bash
tmux new -s training
```

Now, set your Weights & Biases API key and run the sequential alternating loop (Red then Blue per cycle). Optional: push each leg’s final adapter to a **Hugging Face model repo** the first time a leg finishes (the repo is **created if it does not exist**; you must still create a [HF access token](https://huggingface.co/settings/tokens) with write access).
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export SEQUENTIAL_MODE=1
export NUM_CYCLES=3
export GRPO_MAX_STEPS=200
export NUM_EPISODES=100
export SEIGE_SAVE_STEPS=50
export INIT_ADAPTER_PATH="sft_adapter"
export SEIGE_TARGET_BACKEND=transformerlens # small white-box target with named hook points
export SEIGE_TARGET_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
export SEIGE_TL_DTYPE=float32              # stable TransformerLens generation for Qwen; try bfloat16 only after validation
export SEIGE_AGENT_MODEL_ID="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
export SEIGE_TRAIN_WHITEBOX_TOOLS=1        # first run: train probes/vectors/patch baselines
export SEIGE_WHITEBOX_TOOLS_DIR="data/whitebox_tools"
export SEIGE_START_TARGET_SERVER=0          # vLLM is only for optional black-box mode
export SEIGE_USE_LOCAL_ENV_FOR_REWARD=0   # use served real target env

# Optional reward / exploration knobs. Defaults are tuned to avoid GRPO zero-std collapse.
# Increase num generations if VRAM allows; keep tiebreak small so it only prevents dead advantages.
# export SEIGE_GRPO_NUM_GENERATIONS=6
# export SEIGE_GRPO_TEMPERATURE=0.82
# export SEIGE_GRPO_MAX_COMPLETION_LENGTH=192
# export SEIGE_TARGET_MAX_TOKENS=64
# export SEIGE_OPPONENT_MAX_NEW_TOKENS=96
# export SEIGE_REWARD_RED_OPPONENT_CHECK=0   # default: skip expensive frozen-blue check during RED reward
# export SEIGE_REWARD_W_RARITY=0.14
# export SEIGE_REWARD_TIEBREAK=0.028
# export SEIGE_SAMPLE_EVERY_STEPS=10   # print prompt + model output every N trainer steps

# Optional: smoke-test the runner without starting full GRPO.
# export SEIGE_DRY_RUN=1

# Optional: auto-upload after each leg (e.g. red_cycle_1/, blue_cycle_1/, ... under the repo)
export SEIGE_HF_PUSH=1
export SEIGE_HF_REPO_ID="YOUR_USERNAME/seige-grpo-checkpoints"   # pick a new, empty repo name
export HF_TOKEN="hf_YourWriteToken"                               # or: export HUGGINGFACE_HUB_TOKEN=...
# export SEIGE_HF_PRIVATE=1   # if you want the *new* repo to be private on first create
# export SEIGE_HF_RESTORE=1   # default: on fresh pods, pull latest red_cycle_* / blue_cycle_* from this repo first

# Run the master pipeline
./run_pipeline.sh
```

**What is happening?**
1. The script boots the OpenEnv server from `server.app:app` on port `8000`; that server owns the real white-box target model and its hidden task/system prompt.
2. RED and BLUE access the target only through OpenEnv actions/observations, but the environment can still execute white-box mechanics such as activation probing, steering vectors, and residual-stream patching. The default `transformerlens` backend uses named hook points and temporary hooks.
3. It alternates training in a loop: **train Red against frozen Blue, then train Blue against the newly trained Red**.
4. After each leg, it saves the adapter and uses it as the next frozen opponent.
5. It repeats this for `NUM_CYCLES` and outputs live metrics to Weights & Biases.
6. Each episode samples a verifiable task (`secret_extraction`, `canary_extraction`, `api_key_extraction`, `banned_word_elicitation`, or `policy_rule_leak`). RED sees the task objective/hint, BLUE sees the protected value for defense, and the target model receives the protected value in its system context.
7. If `SEIGE_TRAIN_WHITEBOX_TOOLS=1`, it first trains probe/vector artifacts for the small target and writes them under `data/whitebox_tools/` plus the legacy `data/direction_library.json` / `data/intent_probes.pkl` paths.
8. On a fresh pod with no local `pipeline_state.json`, if `SEIGE_HF_REPO_ID` is set, the script restores the newest `red_cycle_*` and `blue_cycle_*` adapters plus latest `tools/*` white-box artifacts from Hugging Face before training.
9. Each leg writes both `final_adapter/` and `best_adapter/`; the pipeline archives/uploads `best_adapter/` when present (lowest logged train loss), not the last-step adapter.
10. If `SEIGE_HF_PUSH=1`, it uploads the archived **cycle** adapter, trained white-box tools, and run manifest/config snapshots to your HF repo.
11. Once complete, it automatically triggers `evaluate.py` using the latest Red and Blue adapters.

The runner writes a reproducibility bundle under `outputs_grpo/runs/<run_id>/` with `config.json`, `events.jsonl`, `latest.json`, and copied `whitebox_tools/` when tools are trained or restored. `outputs_grpo/pipeline_state.json` is saved after every leg so interrupted pods can resume at the correct cycle and agent.

## Fast Target Serving
For fast white-box GRPO, keep the target model out of the training process by running it inside the separate environment server process. On multi-GPU pods, put training on one GPU and the white-box environment/opponent servers on another:

```bash
export CUDA_VISIBLE_DEVICES=0              # GRPO trainer GPU
export ENV_CUDA_VISIBLE_DEVICES=1          # env server + white-box HF target GPU
./run_pipeline.sh
```

For the ultimate non-black-box environment, keep `SEIGE_TARGET_BACKEND=transformerlens` or `SEIGE_TARGET_BACKEND=hf`. The target is externally accessed by agents through the OpenEnv server, but it is white-box inside the environment so hooks and activation interventions work. Start with `transformerlens` plus `Qwen/Qwen2.5-0.5B-Instruct` for fast experiments; scale to larger HF/Qwen models after the reward dynamics look sane.

If you specifically want a faster black-box baseline, use vLLM/OpenAI-compatible serving:

```bash
export SEIGE_TARGET_BACKEND=openai_compat
export SEIGE_START_TARGET_SERVER=1
export SEIGE_TARGET_BASE_URL="http://localhost:8002/v1"
export TARGET_CUDA_VISIBLE_DEVICES=1
export TARGET_TENSOR_PARALLEL_SIZE=1
export TARGET_GPU_MEMORY_UTILIZATION=0.85
./run_pipeline.sh
```

Black-box vLLM mode cannot expose activations, so the curriculum automatically disables steering/probe/patch actions there. The main speed bottleneck is target/opponent generation during reward scoring, not the H200 matmul throughput. Keep `SEIGE_TARGET_MAX_TOKENS` and `SEIGE_OPPONENT_MAX_NEW_TOKENS` short while debugging, and increase `SEIGE_GRPO_NUM_GENERATIONS` only after target latency is stable.

## Step 6: Save your new agent (Hugging Face)
When the script prints **"Pipeline finished successfully!"**, adapters are under `outputs_grpo/`. If you set `SEIGE_HF_PUSH=1` (Step 5), the **best adapter for each completed leg** is already in your HF model repo in subfolders such as `red_cycle_1/`, `blue_cycle_1/`, etc. You can add a manual top-level upload of `best_adapter` the same way if you want a separate “latest-best”-only repo; otherwise the pipeline upload is enough for crash safety between legs.

**Do not delete the Pod** until you have the artifacts you need on Hugging Face or in persistent storage, because instance disks are ephemeral.

To upload by hand (same as the pipeline), set `HF_TOKEN` and run:
```bash
python3 scripts/upload_adapter_to_hf.py \
  --local-dir outputs_grpo/cycles/red_cycle_1 \
  --repo-id YOUR_USERNAME/seige-grpo-checkpoints \
  --path-in-repo red_cycle_1
```
This creates the model repo on first use if it does not exist, then syncs that folder to the `path-in-repo` subpath.

## Step 7: Single-Agent Backward-Compatible Mode (Optional)
If you still want one-off training for only one side, disable sequential mode:
```bash
export SEQUENTIAL_MODE=0
export AGENT_TO_TRAIN="blue"
export FROZEN_ADAPTER_PATH="outputs_grpo/grpo_red/final_adapter"
export SFT_ADAPTER_PATH="sft_adapter"

./run_pipeline.sh
```

*(Once you have pushed everything you want to keep to Hugging Face, you can safely Terminate the Pod from the RunPod dashboard to stop paying for it).*
