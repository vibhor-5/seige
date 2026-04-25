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
```

**Resume / optimizer state:** this repo pins **torch ≥ 2.6** so `transformers` can load checkpoint `optimizer.pt` / `scheduler.pt` (older 2.5 + recent `transformers` raises a `torch.load` / CVE-2025-32434 error). The default `./run_pipeline.sh` `TORCH_INDEX_URL` is `cu124` for those wheels. Run `INSTALL_DEPS=1 ./run_pipeline.sh` (or match `requirements.txt` to that index) so the template’s PyTorch is not left at 2.5.x. After `requirements.txt`, the script **re-installs the torch + torchvision + torchaudio triple** from that index; if you only upgrade `torch`, a preinstalled `torchvision` for 2.5.1 (e.g. `+cu121`) will break imports (`torchvision::nms` / `BloomPreTrainedModel` errors).

## Step 4: Download Your SFT Adapter
Because the `sft_adapter` was too large for GitHub, we uploaded it to Hugging Face. Download it directly into the `seige` folder using a Python one-liner (this will be extremely fast on RunPod):
```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BART-ender/seige-sft-adapter', local_dir='sft_adapter')"
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
export INIT_ADAPTER_PATH="sft_adapter"

# Optional: auto-upload after each leg (e.g. red_cycle_1/, blue_cycle_1/, ... under the repo)
export SEIGE_HF_PUSH=1
export SEIGE_HF_REPO_ID="YOUR_USERNAME/seige-grpo-checkpoints"   # pick a new, empty repo name
export HF_TOKEN="hf_YourWriteToken"                               # or: export HUGGINGFACE_HUB_TOKEN=...
# export SEIGE_HF_PRIVATE=1   # if you want the *new* repo to be private on first create

# Run the master pipeline
./run_pipeline.sh
```

**What is happening?**
1. The script will boot the `seige` environment server on port `8000`.
2. It alternates training in a loop: **train Red against frozen Blue, then train Blue against the newly trained Red**.
3. After each leg, it saves the adapter and uses it as the next frozen opponent.
4. It repeats this for `NUM_CYCLES` and outputs live metrics to Weights & Biases.
5. If `SEIGE_HF_PUSH=1`, it uploads the archived **cycle** adapter to your HF repo (e.g. `red_cycle_1/`, `blue_cycle_1/`, …) so a lost pod does not lose the last full leg.
6. Once complete, it automatically triggers `evaluate.py` using the latest Red and Blue adapters.

## Step 6: Save your new agent (Hugging Face)
When the script prints **"Pipeline finished successfully!"**, adapters are under `outputs_grpo/`. If you set `SEIGE_HF_PUSH=1` (Step 5), the **last completed leg of each type** is already in your HF model repo in subfolders such as `red_cycle_1/`, `blue_cycle_1/`, etc. You can add a manual top-level upload of `final_adapter` the same way if you want a separate “latest”-only repo; otherwise the pipeline upload is enough for crash safety between legs.

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
