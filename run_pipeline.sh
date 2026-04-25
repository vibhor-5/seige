#!/bin/bash
set -e

# Default values if not provided via environment variables.
# SEQUENTIAL_MODE=1 runs alternating Red/Blue training in a loop.
SEQUENTIAL_MODE=${SEQUENTIAL_MODE:-"1"}
NUM_CYCLES=${NUM_CYCLES:-"1"}
GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-"200"}
NUM_EPISODES=${NUM_EPISODES:-"100"}
INSTALL_DEPS=${INSTALL_DEPS:-"1"}
UNINSTALL_TORCHAO=${UNINSTALL_TORCHAO:-"1"}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-"https://download.pytorch.org/whl/cu121"}

# Backward-compatible single-agent values.
AGENT_TO_TRAIN=${AGENT_TO_TRAIN:-"red"}
FROZEN_ADAPTER_PATH=${FROZEN_ADAPTER_PATH:-"sft_adapter"}
INIT_ADAPTER_PATH=${INIT_ADAPTER_PATH:-"sft_adapter"}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-"$INIT_ADAPTER_PATH"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs_grpo"}
CYCLE_ARCHIVE_DIR=${CYCLE_ARCHIVE_DIR:-"$OUTPUT_DIR/cycles"}
PORT_ENV=${PORT_ENV:-8000}
PORT_OPPONENT=${PORT_OPPONENT:-8001}

echo "=== Seige Sequential GRPO Pipeline ==="
echo "Sequential Mode: $SEQUENTIAL_MODE"
echo "Cycles: $NUM_CYCLES"
echo "GRPO Max Steps / leg: $GRPO_MAX_STEPS"
echo "Proxy Episodes / leg: $NUM_EPISODES"
echo "Init Adapter: $SFT_ADAPTER_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Cycle Archive Dir: $CYCLE_ARCHIVE_DIR"
echo "Install Dependencies: $INSTALL_DEPS"
echo "Uninstall TorchAO: $UNINSTALL_TORCHAO"
echo "Torch Index URL: $TORCH_INDEX_URL"
if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB Logging: ENABLED"
else
    echo "WandB Logging: DISABLED (Set WANDB_API_KEY to enable)"
fi
echo "========================================"

RED_LATEST="$SFT_ADAPTER_PATH"
BLUE_LATEST="$SFT_ADAPTER_PATH"
ENV_PID=""
OPP_PID=""

# Cleanup function to kill background processes on exit
cleanup() {
    echo "Shutting down background servers..."
    kill ${ENV_PID:-} 2>/dev/null || true
    kill ${OPP_PID:-} 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No python interpreter found (expected python3 or python)."
    exit 1
fi

export TRANSFORMERS_NO_TORCHAO=1

install_dependencies() {
    echo "[0/4] Bootstrapping Python dependencies..."
    local PIP_CMD="$PYTHON_BIN -m pip install"
    echo "Using installer: $PYTHON_BIN -m pip"

    if [ "$UNINSTALL_TORCHAO" == "1" ]; then
        echo "Removing torchao to avoid torch/torchao incompatibility..."
        "$PYTHON_BIN" -m pip uninstall -y torchao >/dev/null 2>&1 || true
    fi

    echo "Ensuring torch is installed..."
    if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
        echo "Torch not found. Installing torch/torchvision/torchaudio via pip from $TORCH_INDEX_URL"
        "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
        "$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
    fi

    # Hard verification + fallback path when uv/pip environment wiring is inconsistent.
    if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        echo "Torch import still failing after pip install. Retrying with no cache..."
        "$PYTHON_BIN" -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
    fi

    if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        echo "ERROR: torch is not importable by $PYTHON_BIN after installation attempts."
        echo "Python executable: $PYTHON_BIN"
        "$PYTHON_BIN" -c "import sys; print('sys.executable=', sys.executable); print('sys.path=', sys.path)"
        exit 1
    else
        echo "Torch is available."
    fi

    if [ ! -f "requirements.txt" ]; then
        echo "ERROR: requirements.txt not found in $(pwd)"
        exit 1
    fi

    echo "Installing Python requirements from requirements.txt..."
    $PIP_CMD -r requirements.txt

    # Some dependency graphs can re-introduce torchao transitively. Remove it
    # again after full requirements installation and verify it's gone.
    if [ "$UNINSTALL_TORCHAO" == "1" ]; then
        echo "Re-checking torchao after requirements install..."
        "$PYTHON_BIN" -m pip uninstall -y torchao >/dev/null 2>&1 || true
        if "$PYTHON_BIN" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torchao') is None else 1)"; then
            echo "TorchAO is not present."
        else
            echo "ERROR: torchao still present after uninstall attempts."
            "$PYTHON_BIN" -m pip show torchao || true
            exit 1
        fi
    fi

    # Validate critical imports before starting servers/training.
    "$PYTHON_BIN" -c "import datasets, requests, fastapi, uvicorn, trl, peft, wandb"

    echo "Dependency bootstrap complete."
}

start_opponent_server() {
    local opponent_type=$1
    local adapter_path=$2

    kill ${OPP_PID:-} 2>/dev/null || true

    echo "Starting Frozen Opponent ($opponent_type) on port $PORT_OPPONENT with $adapter_path..."
    "$PYTHON_BIN" scripts/opponent_server.py \
        --adapter_path "$adapter_path" \
        --agent_type "$opponent_type" \
        --port $PORT_OPPONENT > opponent_server.log 2>&1 &
    OPP_PID=$!

    echo "Waiting for opponent model to load..."
    sleep 25
}

train_leg() {
    local trainee=$1
    local init_adapter=$2
    local frozen_type=$3
    local frozen_adapter=$4
    local cycle=$5

    start_opponent_server "$frozen_type" "$frozen_adapter"

    echo "Training $trainee (cycle $cycle) | init=$init_adapter | frozen $frozen_type=$frozen_adapter"
    "$PYTHON_BIN" scripts/run_grpo.py \
        --agent_type "$trainee" \
        --init_adapter "$init_adapter" \
        --env_url "http://localhost:$PORT_ENV" \
        --output_dir "$OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$GRPO_MAX_STEPS" \
        --run_name "seige-grpo-${trainee}-cycle${cycle}"
}

# 0. Install dependencies first (RunPod self-healing bootstrap)
if [ "$INSTALL_DEPS" == "1" ]; then
    install_dependencies
fi

# 1. Start the Environment Server
echo "[1/4] Starting Target Environment Server on port $PORT_ENV..."
"$PYTHON_BIN" -m uvicorn server.app:app --host 0.0.0.0 --port $PORT_ENV > env_server.log 2>&1 &
ENV_PID=$!
sleep 5 # Give FastAPI a moment to bind to the port

if [ "$SEQUENTIAL_MODE" == "1" ]; then
    echo "[2/4] Running sequential alternating training..."
    mkdir -p "$CYCLE_ARCHIVE_DIR"
    for ((cycle=1; cycle<=NUM_CYCLES; cycle++)); do
        echo "----- Cycle $cycle / $NUM_CYCLES: Train RED then BLUE -----"

        train_leg "red" "$RED_LATEST" "blue" "$BLUE_LATEST" "$cycle"
        RED_LATEST="$OUTPUT_DIR/grpo_red/final_adapter"
        RED_ARCHIVE="$CYCLE_ARCHIVE_DIR/red_cycle_${cycle}"
        rm -rf "$RED_ARCHIVE"
        cp -R "$RED_LATEST" "$RED_ARCHIVE"
        echo "Saved RED checkpoint: $RED_LATEST"
        echo "Archived RED cycle checkpoint: $RED_ARCHIVE"

        train_leg "blue" "$BLUE_LATEST" "red" "$RED_LATEST" "$cycle"
        BLUE_LATEST="$OUTPUT_DIR/grpo_blue/final_adapter"
        BLUE_ARCHIVE="$CYCLE_ARCHIVE_DIR/blue_cycle_${cycle}"
        rm -rf "$BLUE_ARCHIVE"
        cp -R "$BLUE_LATEST" "$BLUE_ARCHIVE"
        echo "Saved BLUE checkpoint: $BLUE_LATEST"
        echo "Archived BLUE cycle checkpoint: $BLUE_ARCHIVE"
    done
else
    echo "[2/4] Backward-compatible single-agent mode..."
    if [ "$AGENT_TO_TRAIN" == "red" ]; then
        OPPONENT_TYPE="blue"
    else
        OPPONENT_TYPE="red"
    fi

    start_opponent_server "$OPPONENT_TYPE" "$FROZEN_ADAPTER_PATH"
    "$PYTHON_BIN" scripts/run_grpo.py \
        --agent_type "$AGENT_TO_TRAIN" \
        --init_adapter "$SFT_ADAPTER_PATH" \
        --env_url "http://localhost:$PORT_ENV" \
        --output_dir "$OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$GRPO_MAX_STEPS"

    if [ "$AGENT_TO_TRAIN" == "red" ]; then
        RED_LATEST="$OUTPUT_DIR/grpo_red/final_adapter"
        BLUE_LATEST="$FROZEN_ADAPTER_PATH"
    else
        RED_LATEST="$FROZEN_ADAPTER_PATH"
        BLUE_LATEST="$OUTPUT_DIR/grpo_blue/final_adapter"
    fi
fi

echo "Training phase completed."

# 4. Run Evaluation to check progress
echo "[4/4] Running automatic evaluation..."
"$PYTHON_BIN" scripts/evaluate.py \
    --red_adapter "$RED_LATEST" \
    --blue_adapter "$BLUE_LATEST" \
    --env_url "http://localhost:$PORT_ENV" \
    --episodes 5

echo "Pipeline finished successfully!"
