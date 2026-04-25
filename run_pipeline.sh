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
# Mismatched triples (e.g. torch 2.6 + torchvision 0.20 for 2.5) cause import/runtime errors (e.g. torchvision::nms).
TORCH_VERSION=${TORCH_VERSION:-"2.6.0"}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.21.0"}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-"2.6.0"}
RESUME_PIPELINE=${RESUME_PIPELINE:-"1"}
SEIGE_TARGET_BACKEND=${SEIGE_TARGET_BACKEND:-"transformerlens"}
SEIGE_TARGET_MODEL_ID=${SEIGE_TARGET_MODEL_ID:-"Qwen/Qwen2.5-0.5B-Instruct"}
SEIGE_AGENT_MODEL_ID=${SEIGE_AGENT_MODEL_ID:-"unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"}
SEIGE_TRAIN_WHITEBOX_TOOLS=${SEIGE_TRAIN_WHITEBOX_TOOLS:-"0"}
SEIGE_WHITEBOX_TOOLS_DIR=${SEIGE_WHITEBOX_TOOLS_DIR:-"data/whitebox_tools"}
PORT_TARGET=${PORT_TARGET:-8002}
SEIGE_TARGET_BASE_URL=${SEIGE_TARGET_BASE_URL:-"http://localhost:${PORT_TARGET}/v1"}
SEIGE_TARGET_API_KEY=${SEIGE_TARGET_API_KEY:-"EMPTY"}
SEIGE_START_TARGET_SERVER=${SEIGE_START_TARGET_SERVER:-"0"}
ENV_CUDA_VISIBLE_DEVICES=${ENV_CUDA_VISIBLE_DEVICES:-""}
TARGET_CUDA_VISIBLE_DEVICES=${TARGET_CUDA_VISIBLE_DEVICES:-""}
TARGET_TENSOR_PARALLEL_SIZE=${TARGET_TENSOR_PARALLEL_SIZE:-"1"}
TARGET_GPU_MEMORY_UTILIZATION=${TARGET_GPU_MEMORY_UTILIZATION:-"0.85"}
SEIGE_USE_LOCAL_ENV_FOR_REWARD=${SEIGE_USE_LOCAL_ENV_FOR_REWARD:-"0"}
# Hugging Face: after each leg, push the archived cycle adapter (requires HF_TOKEN and SEIGE_HF_REPO_ID).
SEIGE_HF_PUSH=${SEIGE_HF_PUSH:-"0"}
SEIGE_HF_REPO_ID=${SEIGE_HF_REPO_ID:-""}
SEIGE_HF_PRIVATE=${SEIGE_HF_PRIVATE:-"0"}
SEIGE_HF_RESTORE=${SEIGE_HF_RESTORE:-"1"}

# Backward-compatible single-agent values.
AGENT_TO_TRAIN=${AGENT_TO_TRAIN:-"red"}
FROZEN_ADAPTER_PATH=${FROZEN_ADAPTER_PATH:-"sft_adapter"}
INIT_ADAPTER_PATH=${INIT_ADAPTER_PATH:-"sft_adapter"}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-"$INIT_ADAPTER_PATH"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs_grpo"}
CYCLE_ARCHIVE_DIR=${CYCLE_ARCHIVE_DIR:-"$OUTPUT_DIR/cycles"}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"$OUTPUT_DIR/runs/$RUN_ID"}
SEIGE_DRY_RUN=${SEIGE_DRY_RUN:-"0"}
PORT_ENV=${PORT_ENV:-8000}
PORT_OPPONENT=${PORT_OPPONENT:-8001}

echo "=== Seige Sequential GRPO Pipeline ==="
echo "Sequential Mode: $SEQUENTIAL_MODE"
echo "Cycles: $NUM_CYCLES"
echo "GRPO Max Steps / leg: $GRPO_MAX_STEPS"
echo "Proxy Episodes / leg: $NUM_EPISODES"
echo "Init Adapter: $SFT_ADAPTER_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Run ID: $RUN_ID"
echo "Run Directory: $RUN_DIR"
echo "Cycle Archive Dir: $CYCLE_ARCHIVE_DIR"
echo "Install Dependencies: $INSTALL_DEPS"
echo "Uninstall TorchAO: $UNINSTALL_TORCHAO"
echo "Torch Index URL: $TORCH_INDEX_URL"
echo "SEIGE_FAST_INFERENCE: $SEIGE_FAST_INFERENCE"
echo "TORCH_MIN_VERSION: $TORCH_MIN_VERSION (pinned stack: $TORCH_VERSION / torchvision $TORCHVISION_VERSION / torchaudio $TORCHAUDIO_VERSION)"
echo "Resume Pipeline: $RESUME_PIPELINE"
echo "Target Backend: $SEIGE_TARGET_BACKEND ($SEIGE_TARGET_MODEL_ID)"
echo "Agent Base Model: $SEIGE_AGENT_MODEL_ID"
echo "Train white-box tools: $SEIGE_TRAIN_WHITEBOX_TOOLS ($SEIGE_WHITEBOX_TOOLS_DIR)"
echo "Target Base URL: $SEIGE_TARGET_BASE_URL"
echo "Env server CUDA_VISIBLE_DEVICES: ${ENV_CUDA_VISIBLE_DEVICES:-inherited}"
echo "Start vLLM Target Server: $SEIGE_START_TARGET_SERVER (port $PORT_TARGET, CUDA_VISIBLE_DEVICES=${TARGET_CUDA_VISIBLE_DEVICES:-all})"
echo "Local reward env: $SEIGE_USE_LOCAL_ENV_FOR_REWARD"
echo "Dry run: $SEIGE_DRY_RUN"
echo "Restore latest HF adapters on fresh start: $SEIGE_HF_RESTORE"
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
TARGET_PID=""
ENV_PID=""
OPP_PID=""
STATE_FILE="$OUTPUT_DIR/pipeline_state.json"

# Cleanup function to kill background processes on exit
cleanup() {
    echo "Shutting down background servers..."
    kill ${TARGET_PID:-} 2>/dev/null || true
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
export SEIGE_TARGET_BACKEND
export SEIGE_TARGET_MODEL_ID
export SEIGE_AGENT_MODEL_ID
export SEIGE_WHITEBOX_TOOLS_DIR
export SEIGE_TARGET_BASE_URL
export SEIGE_TARGET_API_KEY
export SEIGE_USE_LOCAL_ENV_FOR_REWARD
export RUN_ID
export RUN_DIR
export OUTPUT_DIR
export NUM_CYCLES
export GRPO_MAX_STEPS
export NUM_EPISODES
export WANDB_PROJECT
export SEIGE_HF_REPO_ID

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

# Install/upgrade the PyTorch *triple* from the same index so we never mix template cu121 2.5 vision with torch 2.6.
pip_install_pytorch_stack() {
    if ! "$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX_URL" "$@" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION"
    then
        echo "PyTorch index install failed. Retrying torch/vision/audio from default PyPI index..."
        "$PYTHON_BIN" -m pip install "$@" \
            "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" || return 1
    fi
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
        echo "Torch not found. Installing torch/vision/audio ($TORCH_VERSION) via pip from $TORCH_INDEX_URL"
        "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
        pip_install_pytorch_stack
    fi

    # Hard verification + fallback path when uv/pip environment wiring is inconsistent.
    if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        echo "Torch import still failing after pip install. Retrying with no cache..."
        pip_install_pytorch_stack --no-cache-dir --upgrade
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
        pip_install_pytorch_stack --upgrade
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

    # requirements.txt can bump torch while leaving a template torchvision/torchaudio built for 2.5.1; resync the triple.
    echo "Re-syncing torch, torchvision, torchaudio to a single CUDA build from $TORCH_INDEX_URL..."
    pip_install_pytorch_stack --upgrade
    # #region agent log
    debug_log "H3" "run_pipeline.sh:install_dependencies:post_requirements" "Post requirements install module probe" "$("$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import sys
mods = ["datasets", "trl", "peft", "wandb", "openenv", "openenv.core", "transformer_lens", "torchao"]
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
    "$PYTHON_BIN" -c "import torch, torchvision, torchaudio, datasets, requests, fastapi, uvicorn, trl, peft, wandb, openenv, openenv.core, transformer_lens"
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
    if ! "$PYTHON_BIN" -c "import torch, torchvision, torchaudio, datasets, requests, fastapi, uvicorn, trl, peft, wandb, openenv, openenv.core" >/dev/null 2>&1; then
        echo "ERROR: Missing required runtime dependencies for $PYTHON_BIN."
        echo "Run with INSTALL_DEPS=1 (default) or install from requirements.txt manually."
        "$PYTHON_BIN" -c "import sys; print('python=', sys.executable); print('prefix=', sys.prefix)"
        exit 1
    fi
    if [ "$SEIGE_TARGET_BACKEND" = "transformerlens" ] || [ "$SEIGE_TARGET_BACKEND" = "transformer_lens" ] || [ "$SEIGE_TARGET_BACKEND" = "tl" ]; then
        if ! "$PYTHON_BIN" -c "import transformer_lens" >/dev/null 2>&1; then
            echo "ERROR: SEIGE_TARGET_BACKEND=$SEIGE_TARGET_BACKEND requires transformer-lens."
            echo "Run with INSTALL_DEPS=1 or install it with: $PYTHON_BIN -m pip install transformer-lens"
            exit 1
        fi
    fi
}

write_run_config() {
    mkdir -p "$RUN_DIR"
    "$PYTHON_BIN" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
def git_value(args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None

config = {
    "run_id": os.environ.get("RUN_ID"),
    "git_commit": git_value(["rev-parse", "HEAD"]),
    "git_dirty": bool(git_value(["status", "--short"])),
    "target_backend": os.environ.get("SEIGE_TARGET_BACKEND"),
    "target_model_id": os.environ.get("SEIGE_TARGET_MODEL_ID"),
    "agent_model_id": os.environ.get("SEIGE_AGENT_MODEL_ID"),
    "whitebox_tools_dir": os.environ.get("SEIGE_WHITEBOX_TOOLS_DIR"),
    "output_dir": os.environ.get("OUTPUT_DIR", "outputs_grpo"),
    "num_cycles": os.environ.get("NUM_CYCLES"),
    "grpo_max_steps": os.environ.get("GRPO_MAX_STEPS"),
    "num_episodes": os.environ.get("NUM_EPISODES"),
    "wandb_project": os.environ.get("WANDB_PROJECT"),
    "hf_repo_id": os.environ.get("SEIGE_HF_REPO_ID"),
}
(run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
(run_dir / "events.jsonl").touch()
print(f"Wrote run config: {run_dir / 'config.json'}")
PY
}

log_event() {
    local event="$1"
    local data_json="${2:-{}}"
    EVENT_NAME="$event" EVENT_DATA_JSON="$data_json" "$PYTHON_BIN" - <<'PY'
import json
import os
import time
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
try:
    data = json.loads(os.environ.get("EVENT_DATA_JSON", "{}"))
except Exception:
    data = {"raw": os.environ.get("EVENT_DATA_JSON", "")}
payload = {"time": time.time(), "event": os.environ["EVENT_NAME"], "data": data}
with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
PY
}

train_whitebox_tools_if_requested() {
    if [ "$SEIGE_TRAIN_WHITEBOX_TOOLS" != "1" ]; then
        return
    fi
    if [ "$SEIGE_TARGET_BACKEND" != "transformerlens" ] && [ "$SEIGE_TARGET_BACKEND" != "transformer_lens" ] && [ "$SEIGE_TARGET_BACKEND" != "tl" ]; then
        echo "Skipping white-box tool training because SEIGE_TARGET_BACKEND=$SEIGE_TARGET_BACKEND"
        return
    fi
    echo "Training white-box probes, steering vectors, and patch baselines..."
    "$PYTHON_BIN" scripts/train_whitebox_tools.py \
        --model-id "$SEIGE_TARGET_MODEL_ID" \
        --output-dir "$SEIGE_WHITEBOX_TOOLS_DIR"
    mkdir -p "$RUN_DIR"
    if [ -d "$SEIGE_WHITEBOX_TOOLS_DIR" ]; then
        rm -rf "$RUN_DIR/whitebox_tools"
        cp -R "$SEIGE_WHITEBOX_TOOLS_DIR" "$RUN_DIR/whitebox_tools"
    fi
    hf_upload_if_enabled "tools/${RUN_ID}" "$SEIGE_WHITEBOX_TOOLS_DIR"
}

start_target_model_server_if_requested() {
    if [ "$SEIGE_START_TARGET_SERVER" != "1" ]; then
        echo "Using external target server at $SEIGE_TARGET_BASE_URL"
        return
    fi
    if [ "$SEIGE_TARGET_BACKEND" != "openai_compat" ] && [ "$SEIGE_TARGET_BACKEND" != "openai" ] && [ "$SEIGE_TARGET_BACKEND" != "vllm" ]; then
        echo "SEIGE_TARGET_BACKEND=$SEIGE_TARGET_BACKEND does not use the vLLM target server."
        return
    fi
    echo "Starting target model server (vLLM OpenAI API) on port $PORT_TARGET..."
    if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
        echo "ERROR: vLLM is not installed but SEIGE_START_TARGET_SERVER=1."
        echo "Install vLLM or set SEIGE_START_TARGET_SERVER=0 and point SEIGE_TARGET_BASE_URL at an existing OpenAI-compatible server."
        exit 1
    fi
    TARGET_ENV=()
    if [ -n "$TARGET_CUDA_VISIBLE_DEVICES" ]; then
        TARGET_ENV=(CUDA_VISIBLE_DEVICES="$TARGET_CUDA_VISIBLE_DEVICES")
    fi
    env "${TARGET_ENV[@]}" "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port "$PORT_TARGET" \
        --model "$SEIGE_TARGET_MODEL_ID" \
        --tensor-parallel-size "$TARGET_TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$TARGET_GPU_MEMORY_UTILIZATION" \
        > target_server.log 2>&1 &
    TARGET_PID=$!
    wait_for_target_server
}

wait_for_target_server() {
    echo "Waiting for target model server at $SEIGE_TARGET_BASE_URL..."
    for _ in $(seq 1 240); do
        if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("$SEIGE_TARGET_BASE_URL/models", timeout=2).read()
PY
        then
            echo "Target model server is healthy."
            return
        fi
        sleep 2
    done
    echo "ERROR: Target model server did not become healthy. Recent target_server.log:"
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path("target_server.log")
if p.exists():
    print("\n".join(p.read_text(errors="replace").splitlines()[-100:]))
PY
    exit 1
}

start_opponent_server() {
    local opponent_type=$1
    local adapter_path=$2

    kill ${OPP_PID:-} 2>/dev/null || true

    echo "Starting Frozen Opponent ($opponent_type) on port $PORT_OPPONENT with $adapter_path..."
    "$PYTHON_BIN" scripts/opponent_server.py \
        --base_model "$SEIGE_AGENT_MODEL_ID" \
        --adapter_path "$adapter_path" \
        --agent_type "$opponent_type" \
        --port $PORT_OPPONENT > opponent_server.log 2>&1 &
    OPP_PID=$!

    echo "Waiting for opponent model to load..."
    sleep 25
}

wait_for_env_server() {
    echo "Waiting for target environment server to become healthy..."
    for _ in $(seq 1 120); do
        if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://localhost:$PORT_ENV/health", timeout=2).read()
PY
        then
            echo "Target environment server is healthy."
            return
        fi
        sleep 2
    done
    echo "ERROR: Target environment server did not become healthy. Recent env_server.log:"
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path("env_server.log")
if p.exists():
    print("\n".join(p.read_text(errors="replace").splitlines()[-80:]))
PY
    exit 1
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
        --base_model "$SEIGE_AGENT_MODEL_ID" \
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
    write_latest_manifest "$cycle" "$next_agent" "$red_latest" "$blue_latest"
}

write_latest_manifest() {
    local cycle=$1
    local next_agent=$2
    local red_latest=$3
    local blue_latest=$4
    mkdir -p "$RUN_DIR"
    CYCLE="$cycle" NEXT_AGENT="$next_agent" RED_LATEST_MANIFEST="$red_latest" BLUE_LATEST_MANIFEST="$blue_latest" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
manifest = {
    "run_id": os.environ.get("RUN_ID"),
    "cycle": int(os.environ["CYCLE"]),
    "next_agent": os.environ["NEXT_AGENT"],
    "red_latest": os.environ["RED_LATEST_MANIFEST"],
    "blue_latest": os.environ["BLUE_LATEST_MANIFEST"],
    "whitebox_tools_dir": os.environ.get("SEIGE_WHITEBOX_TOOLS_DIR"),
    "hf_repo_id": os.environ.get("SEIGE_HF_REPO_ID"),
}
(run_dir / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
}

hf_upload_if_enabled() {
    local path_in_repo=$1
    local local_dir=$2
    if [ "$SEIGE_HF_PUSH" != "1" ]; then
        return 0
    fi
    if [ -z "$SEIGE_HF_REPO_ID" ]; then
        echo "WARNING: SEIGE_HF_PUSH=1 but SEIGE_HF_REPO_ID is not set; recording pending upload." >&2
        log_event "hf_upload_pending" "{\"path_in_repo\":\"$path_in_repo\",\"local_dir\":\"$local_dir\",\"reason\":\"missing_repo_id\"}"
        return 0
    fi
    if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
        echo "WARNING: SEIGE_HF_PUSH=1 but HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set; recording pending upload." >&2
        log_event "hf_upload_pending" "{\"path_in_repo\":\"$path_in_repo\",\"local_dir\":\"$local_dir\",\"reason\":\"missing_token\"}"
        return 0
    fi
    if [ ! -d "$local_dir" ]; then
        echo "WARNING: upload source is not a directory: $local_dir" >&2
        log_event "hf_upload_pending" "{\"path_in_repo\":\"$path_in_repo\",\"local_dir\":\"$local_dir\",\"reason\":\"missing_local_dir\"}"
        return 0
    fi
    PRIV=()
    if [ "$SEIGE_HF_PRIVATE" = "1" ]; then
        PRIV=(--private)
    fi
    echo "Uploading adapter to Hugging Face: $SEIGE_HF_REPO_ID (path: $path_in_repo) from $local_dir"
    if "$PYTHON_BIN" scripts/upload_adapter_to_hf.py \
        --local-dir "$local_dir" \
        --repo-id "$SEIGE_HF_REPO_ID" \
        --path-in-repo "$path_in_repo" \
        "${PRIV[@]}"; then
        log_event "hf_upload_complete" "{\"path_in_repo\":\"$path_in_repo\",\"local_dir\":\"$local_dir\"}"
    else
        echo "WARNING: HF upload failed for $local_dir; local training will continue." >&2
        log_event "hf_upload_pending" "{\"path_in_repo\":\"$path_in_repo\",\"local_dir\":\"$local_dir\",\"reason\":\"upload_failed\"}"
    fi
}

select_agent_adapter() {
    local agent=$1
    local best="$OUTPUT_DIR/grpo_${agent}/best_adapter"
    local final="$OUTPUT_DIR/grpo_${agent}/final_adapter"
    if [ -d "$best" ]; then
        echo "$best"
    else
        echo "$final"
    fi
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

restore_latest_hf_adapters_if_requested() {
    if [ "$SEIGE_HF_RESTORE" != "1" ]; then
        return
    fi
    if [ -f "$STATE_FILE" ]; then
        echo "Local pipeline state exists; skipping HF adapter restore."
        return
    fi
    if [ -z "$SEIGE_HF_REPO_ID" ]; then
        echo "No SEIGE_HF_REPO_ID set; skipping HF adapter restore."
        return
    fi
    echo "Checking Hugging Face for latest saved adapters in $SEIGE_HF_REPO_ID..."
    local restored
    restored=$(SEIGE_HF_REPO_ID="$SEIGE_HF_REPO_ID" OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

repo_id = os.environ["SEIGE_HF_REPO_ID"]
output_dir = Path(os.environ["OUTPUT_DIR"])
restore_root = output_dir / "hf_restore"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

try:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    latest = {}
    for path in files:
        top = path.split("/", 1)[0]
        match = re.fullmatch(r"(red|blue)_cycle_(\d+)", top)
        if match:
            agent, cycle = match.group(1), int(match.group(2))
            if agent not in latest or cycle > latest[agent][0]:
                latest[agent] = (cycle, top)

    restore_root.mkdir(parents=True, exist_ok=True)
    for agent in ("red", "blue"):
        if agent not in latest:
            print(f"{agent.upper()}=")
            continue
        _, prefix = latest[agent]
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{prefix}/*"],
            local_dir=str(restore_root),
            token=token,
        )
        print(f"{agent.upper()}={restore_root / prefix}")
except Exception as exc:
    print(f"WARN={exc!r}")
    print("RED=")
    print("BLUE=")
PY
)
    local warn
    warn=$(echo "$restored" | sed -n 's/^WARN=//p')
    if [ -n "$warn" ]; then
        echo "HF adapter restore warning: $warn"
    fi
    local red_restore
    local blue_restore
    red_restore=$(echo "$restored" | sed -n 's/^RED=//p' | tail -n 1)
    blue_restore=$(echo "$restored" | sed -n 's/^BLUE=//p' | tail -n 1)
    if [ -n "$red_restore" ] && [ -d "$red_restore" ]; then
        RED_LATEST="$red_restore"
        echo "Restored latest RED adapter from HF: $RED_LATEST"
    fi
    if [ -n "$blue_restore" ] && [ -d "$blue_restore" ]; then
        BLUE_LATEST="$blue_restore"
        echo "Restored latest BLUE adapter from HF: $BLUE_LATEST"
    fi
}

restore_latest_hf_tools_if_requested() {
    if [ "$SEIGE_HF_RESTORE" != "1" ]; then
        return
    fi
    if [ -z "$SEIGE_HF_REPO_ID" ]; then
        return
    fi
    if [ -d "$SEIGE_WHITEBOX_TOOLS_DIR" ] && [ -n "$(ls -A "$SEIGE_WHITEBOX_TOOLS_DIR" 2>/dev/null)" ]; then
        echo "Local white-box tools exist at $SEIGE_WHITEBOX_TOOLS_DIR; skipping HF tool restore."
        return
    fi
    echo "Checking Hugging Face for latest white-box tools in $SEIGE_HF_REPO_ID..."
    local restored_tools
    restored_tools=$(SEIGE_HF_REPO_ID="$SEIGE_HF_REPO_ID" OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

repo_id = os.environ["SEIGE_HF_REPO_ID"]
output_dir = Path(os.environ["OUTPUT_DIR"])
restore_root = output_dir / "hf_restore" / "tools"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

try:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefixes = sorted({"/".join(path.split("/")[:2]) for path in files if path.startswith("tools/") and len(path.split("/")) > 2})
    if not prefixes:
        print("TOOLS=")
    else:
        prefix = prefixes[-1]
        restore_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{prefix}/*"],
            local_dir=str(restore_root),
            token=token,
        )
        print(f"TOOLS={restore_root / prefix}")
except Exception as exc:
    print(f"WARN={exc!r}")
    print("TOOLS=")
PY
)
    local warn
    warn=$(echo "$restored_tools" | sed -n 's/^WARN=//p')
    if [ -n "$warn" ]; then
        echo "HF tool restore warning: $warn"
    fi
    local tools_restore
    tools_restore=$(echo "$restored_tools" | sed -n 's/^TOOLS=//p' | tail -n 1)
    if [ -n "$tools_restore" ] && [ -d "$tools_restore" ]; then
        mkdir -p "$(dirname "$SEIGE_WHITEBOX_TOOLS_DIR")"
        rm -rf "$SEIGE_WHITEBOX_TOOLS_DIR"
        cp -R "$tools_restore" "$SEIGE_WHITEBOX_TOOLS_DIR"
        echo "Restored latest white-box tools from HF: $SEIGE_WHITEBOX_TOOLS_DIR"
        log_event "hf_tools_restored" "{\"path\":\"$SEIGE_WHITEBOX_TOOLS_DIR\"}"
    fi
}

# 0. Install dependencies first (RunPod self-healing bootstrap)
write_run_config
if [ "$INSTALL_DEPS" == "1" ]; then
    install_dependencies
fi
preflight_runtime_checks
log_event "preflight_complete"
load_state_if_requested
restore_latest_hf_adapters_if_requested
restore_latest_hf_tools_if_requested
train_whitebox_tools_if_requested
log_event "whitebox_tools_ready" "{\"path\":\"$SEIGE_WHITEBOX_TOOLS_DIR\"}"

if [ "$SEIGE_DRY_RUN" = "1" ]; then
    echo "SEIGE_DRY_RUN=1: running a one-episode mock smoke test and exiting."
    SEIGE_TARGET_BACKEND=mock "$PYTHON_BIN" - <<'PY'
from environment.env import SeigeEnv

env = SeigeEnv()
obs = env.reset()
red = env.step({
    "agent_type": "red",
    "strategy": obs["red"]["available_strategies"][0],
    "sub_strategy": "dry_run",
    "payload": "Please discuss the protected task.",
})
sid = red["observation"]["pending_inference"]["session_id"]
blue = env.step({"agent_type": "blue", "action_type": "monitor", "session_id": sid})
print({"red_info": red["info"], "blue_info": blue["info"]})
PY
    log_event "dry_run_complete"
    exit 0
fi

# 1. Start the target model and OpenEnv Environment Server
start_target_model_server_if_requested
echo "[1/4] Starting OpenEnv Target Environment Server on port $PORT_ENV..."
ENV_SERVER_ENV=()
if [ -n "$ENV_CUDA_VISIBLE_DEVICES" ]; then
    ENV_SERVER_ENV=(CUDA_VISIBLE_DEVICES="$ENV_CUDA_VISIBLE_DEVICES")
fi
env "${ENV_SERVER_ENV[@]}" "$PYTHON_BIN" -m uvicorn server.app:app --host 0.0.0.0 --port $PORT_ENV > env_server.log 2>&1 &
ENV_PID=$!
wait_for_env_server
log_event "env_server_ready"

if [ "$SEQUENTIAL_MODE" == "1" ]; then
    echo "[2/4] Running sequential alternating training..."
    mkdir -p "$CYCLE_ARCHIVE_DIR"
    cycle=${RESUME_CYCLE:-1}
    next_agent=${RESUME_NEXT_AGENT:-red}
    while [ "$cycle" -le "$NUM_CYCLES" ]; do
        echo "----- Cycle $cycle / $NUM_CYCLES | Next agent: $next_agent -----"

        if [ "$next_agent" = "red" ]; then
            train_leg "red" "$RED_LATEST" "blue" "$BLUE_LATEST" "$cycle"
            RED_LATEST="$(select_agent_adapter red)"
            RED_ARCHIVE="$CYCLE_ARCHIVE_DIR/red_cycle_${cycle}"
            rm -rf "$RED_ARCHIVE"
            cp -R "$RED_LATEST" "$RED_ARCHIVE"
            echo "Saved RED checkpoint: $RED_LATEST"
            echo "Archived RED cycle checkpoint: $RED_ARCHIVE"
            hf_upload_if_enabled "red_cycle_${cycle}" "$RED_ARCHIVE"
            log_event "red_leg_complete" "{\"cycle\":$cycle,\"adapter\":\"$RED_LATEST\"}"
            next_agent="blue"
            write_state "$cycle" "$next_agent" "$RED_LATEST" "$BLUE_LATEST"
            continue
        fi

        train_leg "blue" "$BLUE_LATEST" "red" "$RED_LATEST" "$cycle"
        BLUE_LATEST="$(select_agent_adapter blue)"
        BLUE_ARCHIVE="$CYCLE_ARCHIVE_DIR/blue_cycle_${cycle}"
        rm -rf "$BLUE_ARCHIVE"
        cp -R "$BLUE_LATEST" "$BLUE_ARCHIVE"
        echo "Saved BLUE checkpoint: $BLUE_LATEST"
        echo "Archived BLUE cycle checkpoint: $BLUE_ARCHIVE"
        hf_upload_if_enabled "blue_cycle_${cycle}" "$BLUE_ARCHIVE"
        log_event "blue_leg_complete" "{\"cycle\":$cycle,\"adapter\":\"$BLUE_LATEST\"}"
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
        --base_model "$SEIGE_AGENT_MODEL_ID" \
        --init_adapter "$SFT_ADAPTER_PATH" \
        --env_url "http://localhost:$PORT_ENV" \
        --output_dir "$OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$GRPO_MAX_STEPS"

    if [ "$AGENT_TO_TRAIN" == "red" ]; then
        RED_LATEST="$(select_agent_adapter red)"
        BLUE_LATEST="$FROZEN_ADAPTER_PATH"
        hf_upload_if_enabled "grpo_red_single" "$RED_LATEST"
        log_event "single_red_complete" "{\"adapter\":\"$RED_LATEST\"}"
    else
        RED_LATEST="$FROZEN_ADAPTER_PATH"
        BLUE_LATEST="$(select_agent_adapter blue)"
        hf_upload_if_enabled "grpo_blue_single" "$BLUE_LATEST"
        log_event "single_blue_complete" "{\"adapter\":\"$BLUE_LATEST\"}"
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
log_event "evaluation_complete"
hf_upload_if_enabled "runs/${RUN_ID}" "$RUN_DIR"

echo "Pipeline finished successfully!"
