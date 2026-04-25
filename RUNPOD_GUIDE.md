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

Now, set your Weights & Biases API key and run the sequential alternating loop (Red then Blue per cycle):
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export SEQUENTIAL_MODE=1
export NUM_CYCLES=3
export GRPO_MAX_STEPS=200
export NUM_EPISODES=100
export INIT_ADAPTER_PATH="sft_adapter"

# Run the master pipeline
./run_pipeline.sh
```

**What is happening?**
1. The script will boot the `seige` environment server on port `8000`.
2. It alternates training in a loop: **train Red against frozen Blue, then train Blue against the newly trained Red**.
3. After each leg, it saves the adapter and uses it as the next frozen opponent.
4. It repeats this for `NUM_CYCLES` and outputs live metrics to Weights & Biases.
5. Once complete, it automatically triggers `evaluate.py` using the latest Red and Blue adapters.

## Step 6: Save Your New Agent!
When the script prints **"Pipeline finished successfully!"**, your brand new agent adapter has been saved to `outputs_grpo/grpo_red/final_adapter`.

**Do not delete the Pod yet!** RunPod instances are ephemeral. If you terminate the Pod, your new model is destroyed.

You must upload the newly trained model back to Hugging Face to save it:
```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('BART-ender/seige-grpo-red', exist_ok=True)
api.upload_folder(folder_path='outputs_grpo/grpo_red/final_adapter', repo_id='BART-ender/seige-grpo-red')
print('Upload complete!')
"
```

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
