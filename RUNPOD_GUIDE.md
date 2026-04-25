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

Now, set your Weights & Biases API key so you can monitor the training live on your dashboard, set the agent you want to train (Red or Blue), and execute the master pipeline:
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export AGENT_TO_TRAIN="red"

# Run the master pipeline
./run_pipeline.sh
```

**What is happening?**
1. The script will boot the `seige` environment server on port `8000`.
2. It will boot the `opponent_server.py` on port `8001` (loading the SFT adapter as the frozen target).
3. It will wait for the models to load into the 48GB VRAM.
4. It will run `run_grpo.py` and output live metrics to Weights & Biases.
5. Once complete, it automatically triggers `evaluate.py` to test the new Red agent against the frozen Blue agent.

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

## Step 7: Train the Next Generation (Optional)
To immediately start Generation 2 and train Blue to defeat your newly trained Red agent, simply swap the variables and run it again:
```bash
export AGENT_TO_TRAIN="blue"
export FROZEN_ADAPTER_PATH="outputs_grpo/grpo_red/final_adapter"

./run_pipeline.sh
```

*(Once you have pushed everything you want to keep to Hugging Face, you can safely Terminate the Pod from the RunPod dashboard to stop paying for it).*
