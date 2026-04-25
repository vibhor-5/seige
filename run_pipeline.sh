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
# PyTorch 2.6+ wheels use cu124/cu126; cu121 often tops out at 2.5.x. Driver CUDA 12.x is compatible.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-"https://download.pytorch.org/whl/cu124"}
SEIGE_FAST_INFERENCE=${SEIGE_FAST_INFERENCE:-"0"}
TORCH_MIN_VERSION=${TORCH_MIN_VERSION:-"2.6.0"}
RESUME_PIPELINE=${RESUME_PIPELINE:-"1"}
# Hugging Face: after each leg, push the archived cycle adapter (requires HF_TOKEN and SEIGE_HF_REPO_ID).
SEIGE_HF_PUSH=${SEIGE_HF_PUSH:-"0"}
SEIGE_HF_REPO_ID=${SEIGE_HF_REPO_ID:-""}
SEIGE_HF_PRIVATE=${SEIGE_HF_PRIVATE:-"0"}

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
echo "SEIGE_FAST_INFERENCE: $SEIGE_FAST_INFERENCE"
echo "TORCH_MIN_VERSION: $TORCH_MIN_VERSION"
echo "Resume Pipeline: $RESUME_PIPELINE"
if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB Logging: ENABLED"
else
    echo "WandB Logging: DISABLED (Set WANDB_API_KEY to enable)"
fi
if [ "$SEIGE_HF_PUSH" = "1" ]; then
    echo "Hugging Face upload: ENABLED -> repo ${SEIGE_HF_REPO_ID:-?} (per-cycle subfolders, repo created if missing)"
else
    echo "Hugging Face upload: DISABLED (set SEIGE_HF_PUSH=1 and SEIGE_HF_REPO_ID to enable after each leg)"
fi
echo "========================================"

RED_LATEST="$SFT_ADAPTER_PATH"
BLUE_LATEST="$SFT_ADAPTER_PATH"
ENV_PID=""
OPP_PID=""
STATE_FILE="$OUTPUT_DIR/pipeline_state.json"

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
export SEIGE_FAST_INFERENCE

DEBUG_LOG_PATH="/Users/vibhorkumar/Desktop/codes/seige/.cursor/debug-712a1a.log"
DEBUG_RUN_ID="${DEBUG_RUN_ID:-run_pipeline_debug}"
DEBUG_SESSION_ID="712a1a"

debug_log() {
    local hypothesis_id="$1"
    local location="$2"
    local message="$3"
    local data_json="$4"
    DEBUG_HYPOTHESIS_ID="$hypothesis_id" \
    DEBUG_LOCATION="$location" \
    DEBUG_MESSAGE="$message" \
    DEBUG_DATA_JSON="$data_json" \
    DEBUG_LOG_PATH="$DEBUG_LOG_PATH" \
    DEBUG_RUN_ID="$DEBUG_RUN_ID" \
    DEBUG_SESSION_ID="$DEBUG_SESSION_ID" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import time

payload = {
    "sessionId": os.environ["DEBUG_SESSION_ID"],
    "runId": os.environ["DEBUG_RUN_ID"],
    "hypothesisId": os.environ["DEBUG_HYPOTHESIS_ID"],
    "location": os.environ["DEBUG_LOCATION"],
    "message": os.environ["DEBUG_MESSAGE"],
    "timestamp": int(time.time() * 1000),
}
raw_data = os.environ.get("DEBUG_DATA_JSON", "{}")
try:
    payload["data"] = json.loads(raw_data)
except Exception:
    payload["data"] = {"raw": raw_data}
try:
    log_path = os.environ["DEBUG_LOG_PATH"]
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
except Exception:
    # Never fail the training pipeline due to debug logging issues.
    pass
PY
}

install_dependencies() {
    echo "[0/4] Bootstrapping Python dependencies..."
    local PIP_CMD="$PYTHON_BIN -m pip install"
    echo "Using installer: $PYTHON_BIN -m pip"
    # #region agent log
    debug_log "H1" "run_pipeline.sh:install_dependencies:start" "Dependency bootstrap start" "{\"python_bin\":\"$PYTHON_BIN\",\"pwd\":\"$(pwd)\",\"path\":\"$PATH\"}"
    # #endregion

    if [ "$UNINSTALL_TORCHAO" == "1" ]; then
        echo "Removing torchao to avoid torch/torchao incompatibility..."
        "$PYTHON_BIN" -m pip uninstall -y torchao >/dev/null 2>&1 || true
    fi

    echo "Ensuring torch is installed..."
    if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
        echo "Torch not found. Installing torch/torchvision/torchaudio via pip from $TORCH_INDEX_URL"
        "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
        if ! "$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX_URL" "torch>=$TORCH_MIN_VERSION" torchvision torchaudio; then
            echo "Torch index install failed. Retrying from default pip index..."
            "$PYTHON_BIN" -m pip install "torch>=$TORCH_MIN_VERSION" torchvision torchaudio
        fi
    fi

    # Hard verification + fallback path when uv/pip environment wiring is inconsistent.
    if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        echo "Torch import still failing after pip install. Retrying with no cache..."
        if ! "$PYTHON_BIN" -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch>=$TORCH_MIN_VERSION" torchvision torchaudio; then
            "$PYTHON_BIN" -m pip install --no-cache-dir "torch>=$TORCH_MIN_VERSION" torchvision torchaudio
        fi
    fi

    if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        echo "ERROR: torch is not importable by $PYTHON_BIN after installation attempts."
        echo "Python executable: $PYTHON_BIN"
        "$PYTHON_BIN" -c "import sys; print('sys.executable=', sys.executable); print('sys.path=', sys.path)"
        exit 1
    else
        echo "Torch is available."
    fi
    # #region agent log
    debug_log "H6" "run_pipeline.sh:install_dependencies:torch_api_compat" "Torch API compatibility probe" "$("$PYTHON_BIN" - <<'PY'
import json
import torch
print(json.dumps({
    "torch_version": torch.__version__,
    "has_set_submodule": hasattr(torch.nn.Module, "set_submodule"),
}))
PY
)"
    # #endregion

    # Transformers 5.x quantization path requires torch.nn.Module.set_submodule.
    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if hasattr(torch.nn.Module, 'set_submodule') else 1)"; then
        echo "Torch API incompatible (missing nn.Module.set_submodule). Upgrading torch stack..."
        if ! "$PYTHON_BIN" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" "torch>=$TORCH_MIN_VERSION" torchvision torchaudio; then
            "$PYTHON_BIN" -m pip install --upgrade "torch>=$TORCH_MIN_VERSION" torchvision torchaudio
        fi
    fi

    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if hasattr(torch.nn.Module, 'set_submodule') else 1)"; then
        echo "ERROR: Torch is installed but still lacks nn.Module.set_submodule."
        "$PYTHON_BIN" -c "import torch, sys; print('torch=', torch.__version__); print('executable=', sys.executable)"
        exit 1
    fi
    # #region agent log
    debug_log "H2" "run_pipeline.sh:install_dependencies:torch_check" "Torch verification complete" "$("$PYTHON_BIN" - <<'PY'
import json
import sys
try:
    import torch
    payload = {"ok": True, "torch_version": torch.__version__, "executable": sys.executable, "prefix": sys.prefix}
except Exception as exc:
    payload = {"ok": False, "error": repr(exc), "executable": sys.executable, "prefix": sys.prefix}
print(json.dumps(payload))
PY
)"
    # #endregion

    if [ ! -f "requirements.txt" ]; then
        echo "ERROR: requirements.txt not found in $(pwd)"
        exit 1
    fi

    echo "Installing Python requirements from requirements.txt..."
    $PIP_CMD -r requirements.txt
    # #region agent log
    debug_log "H3" "run_pipeline.sh:install_dependencies:post_requirements" "Post requirements install module probe" "$("$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import sys
mods = ["datasets", "trl", "peft", "wandb", "torchao"]
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "modules": {m: bool(importlib.util.find_spec(m)) for m in mods},
}))
PY
)"
    # #endregion

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
    # #region agent log
    debug_log "H4" "run_pipeline.sh:install_dependencies:critical_imports" "Critical import gate passed" "$("$PYTHON_BIN" - <<'PY'
import json
import sys
print(json.dumps({"ok": True, "executable": sys.executable, "prefix": sys.prefix}))
PY
)"
    # #endregion

    echo "Dependency bootstrap complete."
}

preflight_runtime_checks() {
    echo "Running runtime preflight checks..."
    if ! "$PYTHON_BIN" -c "import torch, datasets, requests, fastapi, uvicorn, trl, peft, wandb" >/dev/null 2>&1; then
        echo "ERROR: Missing required runtime dependencies for $PYTHON_BIN."
        echo "Run with INSTALL_DEPS=1 (default) or install from requirements.txt manually."
        "$PYTHON_BIN" -c "import sys; print('python=', sys.executable); print('prefix=', sys.prefix)"
        exit 1
    fi
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
    # #region agent log
    debug_log "H5" "run_pipeline.sh:train_leg:before_run_grpo" "About to launch run_grpo.py" "{\"trainee\":\"$trainee\",\"cycle\":\"$cycle\",\"python_bin\":\"$PYTHON_BIN\",\"env_url\":\"http://localhost:$PORT_ENV\"}"
    # #endregion
    "$PYTHON_BIN" scripts/run_grpo.py \
        --agent_type "$trainee" \
        --init_adapter "$init_adapter" \
        --env_url "http://localhost:$PORT_ENV" \
        --output_dir "$OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$GRPO_MAX_STEPS" \
        --run_name "seige-grpo-${trainee}-cycle${cycle}" \
        --resume_if_possible
}

write_state() {
    local cycle=$1
    local next_agent=$2
    local red_latest=$3
    local blue_latest=$4
    mkdir -p "$OUTPUT_DIR"
    cat > "$STATE_FILE" <<EOF
{"cycle":$cycle,"next_agent":"$next_agent","red_latest":"$red_latest","blue_latest":"$blue_latest","num_cycles":$NUM_CYCLES}
EOF
}

hf_upload_if_enabled() {
    local path_in_repo=$1
    local local_dir=$2
    if [ "$SEIGE_HF_PUSH" != "1" ]; then
        return 0
    fi
    if [ -z "$SEIGE_HF_REPO_ID" ]; then
        echo "ERROR: SEIGE_HF_PUSH=1 but SEIGE_HF_REPO_ID is not set" >&2
        exit 1
    fi
    if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
        echo "ERROR: SEIGE_HF_PUSH=1 but HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set" >&2
        exit 1
    fi
    if [ ! -d "$local_dir" ]; then
        echo "ERROR: upload source is not a directory: $local_dir" >&2
        exit 1
    fi
    PRIV=()
    if [ "$SEIGE_HF_PRIVATE" = "1" ]; then
        PRIV=(--private)
    fi
    echo "Uploading adapter to Hugging Face: $SEIGE_HF_REPO_ID (path: $path_in_repo) from $local_dir"
    "$PYTHON_BIN" scripts/upload_adapter_to_hf.py \
        --local-dir "$local_dir" \
        --repo-id "$SEIGE_HF_REPO_ID" \
        --path-in-repo "$path_in_repo" \
        "${PRIV[@]}"
}

load_state_if_requested() {
    if [ "$RESUME_PIPELINE" != "1" ]; then
        return
    fi
    if [ ! -f "$STATE_FILE" ]; then
        return
    fi
    echo "Found existing pipeline state at $STATE_FILE. Resuming..."
    local loaded
    loaded=$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$STATE_FILE")
data = json.loads(p.read_text())
print(data.get("cycle", 1))
print(data.get("next_agent", "red"))
print(data.get("red_latest", "$RED_LATEST"))
print(data.get("blue_latest", "$BLUE_LATEST"))
PY
)
    RESUME_CYCLE=$(echo "$loaded" | sed -n '1p')
    RESUME_NEXT_AGENT=$(echo "$loaded" | sed -n '2p')
    RED_LATEST=$(echo "$loaded" | sed -n '3p')
    BLUE_LATEST=$(echo "$loaded" | sed -n '4p')
}

# 0. Install dependencies first (RunPod self-healing bootstrap)
if [ "$INSTALL_DEPS" == "1" ]; then
    install_dependencies
fi
preflight_runtime_checks
load_state_if_requested

# 1. Start the Environment Server
echo "[1/4] Starting Target Environment Server on port $PORT_ENV..."
"$PYTHON_BIN" -m uvicorn server.app:app --host 0.0.0.0 --port $PORT_ENV > env_server.log 2>&1 &
ENV_PID=$!
sleep 5 # Give FastAPI a moment to bind to the port

if [ "$SEQUENTIAL_MODE" == "1" ]; then
    echo "[2/4] Running sequential alternating training..."
    mkdir -p "$CYCLE_ARCHIVE_DIR"
    cycle=${RESUME_CYCLE:-1}
    next_agent=${RESUME_NEXT_AGENT:-red}
    while [ "$cycle" -le "$NUM_CYCLES" ]; do
        echo "----- Cycle $cycle / $NUM_CYCLES | Next agent: $next_agent -----"

        if [ "$next_agent" = "red" ]; then
            train_leg "red" "$RED_LATEST" "blue" "$BLUE_LATEST" "$cycle"
            RED_LATEST="$OUTPUT_DIR/grpo_red/final_adapter"
            RED_ARCHIVE="$CYCLE_ARCHIVE_DIR/red_cycle_${cycle}"
            rm -rf "$RED_ARCHIVE"
            cp -R "$RED_LATEST" "$RED_ARCHIVE"
            echo "Saved RED checkpoint: $RED_LATEST"
            echo "Archived RED cycle checkpoint: $RED_ARCHIVE"
            hf_upload_if_enabled "red_cycle_${cycle}" "$RED_ARCHIVE"
            next_agent="blue"
            write_state "$cycle" "$next_agent" "$RED_LATEST" "$BLUE_LATEST"
            continue
        fi

        train_leg "blue" "$BLUE_LATEST" "red" "$RED_LATEST" "$cycle"
        BLUE_LATEST="$OUTPUT_DIR/grpo_blue/final_adapter"
        BLUE_ARCHIVE="$CYCLE_ARCHIVE_DIR/blue_cycle_${cycle}"
        rm -rf "$BLUE_ARCHIVE"
        cp -R "$BLUE_LATEST" "$BLUE_ARCHIVE"
        echo "Saved BLUE checkpoint: $BLUE_LATEST"
        echo "Archived BLUE cycle checkpoint: $BLUE_ARCHIVE"
        hf_upload_if_enabled "blue_cycle_${cycle}" "$BLUE_ARCHIVE"
        cycle=$((cycle + 1))
        next_agent="red"
        write_state "$cycle" "$next_agent" "$RED_LATEST" "$BLUE_LATEST"
    done
    rm -f "$STATE_FILE"
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
        hf_upload_if_enabled "grpo_red_single" "$RED_LATEST"
    else
        RED_LATEST="$FROZEN_ADAPTER_PATH"
        BLUE_LATEST="$OUTPUT_DIR/grpo_blue/final_adapter"
        hf_upload_if_enabled "grpo_blue_single" "$BLUE_LATEST"
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
