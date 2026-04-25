#!/bin/bash
set -e

# Default values if not provided via environment variables
AGENT_TO_TRAIN=${AGENT_TO_TRAIN:-"red"}
FROZEN_ADAPTER_PATH=${FROZEN_ADAPTER_PATH:-"sft_adapter"}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-"sft_adapter"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs_grpo"}
PORT_ENV=${PORT_ENV:-8000}
PORT_OPPONENT=${PORT_OPPONENT:-8001}

echo "=== Seige Generational GRPO Pipeline ==="
echo "Training Agent: $AGENT_TO_TRAIN"
echo "Frozen Opponent Adapter: $FROZEN_ADAPTER_PATH"
echo "Starting SFT Adapter: $SFT_ADAPTER_PATH"
echo "Output Directory: $OUTPUT_DIR"
if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB Logging: ENABLED"
else
    echo "WandB Logging: DISABLED (Set WANDB_API_KEY to enable)"
fi
echo "========================================"

# Determine opponent type
if [ "$AGENT_TO_TRAIN" == "red" ]; then
    OPPONENT_TYPE="blue"
else
    OPPONENT_TYPE="red"
fi

# Cleanup function to kill background processes on exit
cleanup() {
    echo "Shutting down background servers..."
    kill $ENV_PID 2>/dev/null || true
    kill $OPP_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# 1. Start the Environment Server
echo "[1/4] Starting Target Environment Server on port $PORT_ENV..."
uvicorn server.app:app --host 0.0.0.0 --port $PORT_ENV > env_server.log 2>&1 &
ENV_PID=$!
sleep 5 # Give FastAPI a moment to bind to the port

# 2. Start the Frozen Opponent Server
echo "[2/4] Starting Frozen Opponent ($OPPONENT_TYPE) on port $PORT_OPPONENT..."
python scripts/opponent_server.py \
    --adapter_path "$FROZEN_ADAPTER_PATH" \
    --agent_type "$OPPONENT_TYPE" \
    --port $PORT_OPPONENT > opponent_server.log 2>&1 &
OPP_PID=$!

echo "Waiting for opponent server to load model into VRAM (check opponent_server.log)..."
sleep 25 

# 3. Start GRPO Training
echo "[3/4] Starting GRPO Training for $AGENT_TO_TRAIN..."
python scripts/run_grpo.py \
    --agent_type "$AGENT_TO_TRAIN" \
    --sft_adapter "$SFT_ADAPTER_PATH" \
    --env_url "http://localhost:$PORT_ENV" \
    --output_dir "$OUTPUT_DIR"

echo "Training Phase Completed!"

# 4. Run Evaluation to check progress
echo "[4/4] Running automatic evaluation..."
# Use the newly trained adapter for the trainee, and the frozen adapter for the opponent
if [ "$AGENT_TO_TRAIN" == "red" ]; then
    RED_EVAL="$OUTPUT_DIR/grpo_red/final_adapter"
    BLUE_EVAL="$FROZEN_ADAPTER_PATH"
else
    RED_EVAL="$FROZEN_ADAPTER_PATH"
    BLUE_EVAL="$OUTPUT_DIR/grpo_blue/final_adapter"
fi

python scripts/evaluate.py \
    --red_adapter "$RED_EVAL" \
    --blue_adapter "$BLUE_EVAL" \
    --env_url "http://localhost:$PORT_ENV" \
    --episodes 5

echo "Pipeline finished successfully!"
