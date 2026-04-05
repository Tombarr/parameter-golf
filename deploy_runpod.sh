#!/usr/bin/env bash
# =============================================================================
# deploy_runpod.sh �� Upload, configure, and run HESTIA ternary training on RunPod
#
# Usage:
#   ./deploy_runpod.sh <SSH_HOST> [OPTIONS]
#
# Required:
#   SSH_HOST    RunPod SSH address (e.g., ssh.runpod.io or user@host:port)
#
# Options:
#   --seed SEED         Training seed (default: 42)
#   --run-id NAME       Run identifier (default: hestia_run)
#   --max-seconds SEC   Max wallclock seconds (default: 599)
#   --skip-setup        Skip repo clone and dataset download
#   --skip-upload       Skip uploading training script
#   --download-only     Just download results from a previous run
#   --dry-run           Print commands without executing
#   --hestia 0|1        Enable HESTIA (default: 1)
#   --structured-24 0|1 Enable 2:4 packing (default: 1)
#   --gptq 0|1          Enable GPTQ-ternary (default: 1)
#   --eggroll 0|1       Enable EGGROLL (default: 1)
#   --gpus N            Number of GPUs (default: 8)
#   --validate          Quick validation run: 2L/128d, 200 steps, 1 GPU, ~2 min
#                       Use on a cheap instance ($0.40/hr) to verify everything works
#
# Examples:
#   # Quick validation on cheap 1xGPU (~$0.40/hr, takes ~2 min)
#   ./deploy_runpod.sh root@203.0.113.1 -p 22 --validate
#
#   # Full run on 8xH100
#   ./deploy_runpod.sh root@203.0.113.1 -p 22 --seed 42
#
#   # Quick test with HESTIA only, no GPTQ/EGGROLL
#   ./deploy_runpod.sh root@host --gptq 0 --eggroll 0
#
#   # Download results from previous run
#   ./deploy_runpod.sh root@host --download-only --run-id hestia_run
# =============================================================================
set -euo pipefail

# Defaults
SEED=42
RUN_ID="hestia_run"
MAX_SECONDS=599
SKIP_SETUP=0
SKIP_UPLOAD=0
DOWNLOAD_ONLY=0
DRY_RUN=0
HESTIA=1
STRUCTURED_24=1
GPTQ=1
EGGROLL=1
GPUS=8
VALIDATE=0
SSH_PORT=""
SSH_HOST=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2;;
        --run-id) RUN_ID="$2"; shift 2;;
        --max-seconds) MAX_SECONDS="$2"; shift 2;;
        --skip-setup) SKIP_SETUP=1; shift;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        --download-only) DOWNLOAD_ONLY=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --hestia) HESTIA="$2"; shift 2;;
        --structured-24) STRUCTURED_24="$2"; shift 2;;
        --gptq) GPTQ="$2"; shift 2;;
        --eggroll) EGGROLL="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --validate) VALIDATE=1; shift;;
        -p) SSH_PORT="$2"; shift 2;;
        -*) echo "Unknown option: $1"; exit 1;;
        *) SSH_HOST="$1"; shift;;
    esac
done

# Apply --validate overrides (cheap quick test)
VALIDATE_SHARDS=80
if [[ $VALIDATE -eq 1 ]]; then
    RUN_ID="${RUN_ID:-validate_run}"
    GPUS=1
    MAX_SECONDS=120
    VALIDATE_SHARDS=1
fi

if [[ -z "$SSH_HOST" ]]; then
    echo "Usage: $0 <SSH_HOST> [OPTIONS]"
    echo "Run with --help or read the script header for options."
    exit 1
fi

# Build SSH/SCP commands
SSH_CMD="ssh -t"
SCP_CMD="scp -O"
if [[ -n "$SSH_PORT" ]]; then
    SSH_CMD="ssh -t -p $SSH_PORT"
    SCP_CMD="scp -O -P $SSH_PORT"
fi

REMOTE_DIR="/workspace/parameter-golf"
SCRIPT_NAME="train_gpt_hestia_ternary.py"
LOCAL_SCRIPT="records/track_10min_16mb/2026-04-05_HESTIA_Ternary_24Sparse_GPTQ_EGGROLL/$SCRIPT_NAME"

run_remote() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] $SSH_CMD $SSH_HOST '...'"
    else
        # Append exit to prevent hanging interactive session on RunPod
        $SSH_CMD "$SSH_HOST" "$1; exit 0"
    fi
}

run_scp_up() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] upload $1 -> $SSH_HOST:$2"
    else
        # Try scp first, fall back to ssh+cat pipe
        $SCP_CMD "$1" "$SSH_HOST:$2" 2>/dev/null || {
            echo "  scp failed, using ssh pipe fallback..."
            cat "$1" | ssh ${SSH_PORT:+-p $SSH_PORT} "$SSH_HOST" "cat > $2"
        }
    fi
}

run_scp_down() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] download $SSH_HOST:$1 -> $2"
    else
        $SCP_CMD "$SSH_HOST:$1" "$2" 2>/dev/null || {
            echo "  scp failed, using ssh pipe fallback..."
            ssh ${SSH_PORT:+-p $SSH_PORT} "$SSH_HOST" "cat $1" > "$2"
        }
    fi
}

echo "============================================================"
echo "  HESTIA Ternary — RunPod Deployment"
echo "============================================================"
echo "  Host:       $SSH_HOST"
echo "  Run ID:     $RUN_ID"
echo "  Seed:       $SEED"
echo "  GPUs:       $GPUS"
echo "  Max time:   ${MAX_SECONDS}s"
echo "  HESTIA:     $HESTIA"
echo "  2:4 sparse: $STRUCTURED_24"
echo "  GPTQ:       $GPTQ"
echo "  EGGROLL:    $EGGROLL"
echo "============================================================"

# ── Download only mode ──────────────────────────────────────────
if [[ $DOWNLOAD_ONLY -eq 1 ]]; then
    echo ""
    echo "==> Downloading results..."
    mkdir -p "results/${RUN_ID}"
    run_scp_down "$REMOTE_DIR/logs/cuda/${RUN_ID}.txt" "results/${RUN_ID}/"
    run_scp_down "$REMOTE_DIR/final_model.hestia.ptz" "results/${RUN_ID}/" || true
    echo "==> Results saved to results/${RUN_ID}/"
    exit 0
fi

# ── Step 1: Setup (clone repo + download dataset) ──────────────
if [[ $SKIP_SETUP -eq 0 ]]; then
    echo ""
    echo "==> Step 1: Setting up remote environment..."
    run_remote "
        cd /workspace
        if [ ! -d parameter-golf ]; then
            git clone https://github.com/openai/parameter-golf.git
        fi
        cd parameter-golf
        git pull --ff-only 2>/dev/null || true

        # Install Flash Attention 3 if not present
        python3 -c 'from flash_attn_interface import flash_attn_func; print(\"FA3 OK\")' 2>/dev/null || {
            echo 'Installing Flash Attention 3...'
            pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>/dev/null || \
            pip install flash-attn --no-build-isolation 2>/dev/null || \
            echo 'WARNING: Could not install FA3 — may need manual install'
        }

        # Install sentencepiece if needed
        pip install --break-system-packages sentencepiece 2>/dev/null || pip install sentencepiece 2>/dev/null

        # Download dataset (sp8192 variant)
        if [ ! -d data/datasets/fineweb10B_sp8192 ]; then
            echo 'Downloading sp8192 dataset...'
            python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards ${VALIDATE_SHARDS:-80}
        else
            echo 'Dataset already present.'
        fi
        echo 'Setup complete.'
    "
else
    echo "==> Skipping setup (--skip-setup)"
fi

# ── Step 2: Upload training script ─────────────────────────────
if [[ $SKIP_UPLOAD -eq 0 ]]; then
    echo ""
    echo "==> Step 2: Uploading training script..."
    if [[ ! -f "$LOCAL_SCRIPT" ]]; then
        echo "ERROR: $LOCAL_SCRIPT not found. Run from the parameter-golf root directory."
        exit 1
    fi
    run_scp_up "$LOCAL_SCRIPT" "$REMOTE_DIR/$SCRIPT_NAME"
    echo "  Uploaded $SCRIPT_NAME"
else
    echo "==> Skipping upload (--skip-upload)"
fi

# ── Step 3: Run training ───────────────────────────────────────
echo ""
echo "==> Step 3: Starting training..."
echo "  This will take ~10 minutes. Logs stream below."
echo "------------------------------------------------------------"

# Validation mode: tiny model, few steps, 1 GPU, 1 data shard
VALIDATE_OVERRIDES=""
if [[ $VALIDATE -eq 1 ]]; then
    VALIDATE_OVERRIDES="NUM_LAYERS=2 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 EMBED_DIM=0 ITERATIONS=200 WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 GPTQ_CALIB_SEQS=4 GPTQ_CALIB_LEN=256 \
EGGROLL_SECONDS=10 EGGROLL_CANDIDATES=128 SLIDING_BATCH_SIZE=32"
fi

TRAIN_CMD="cd $REMOTE_DIR && \
RUN_ID=${RUN_ID} \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
SEED=${SEED} \
NUM_LAYERS=10 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
EMBED_DIM=254 \
BITNET_GROUP_SIZE=128 \
ACTIVATION=relu2 \
SOFTCAP_TYPE=poly \
LOGIT_SOFTCAP=10 \
QK_GAIN_INIT=2.25 \
ROPE_TYPE=yarn \
YARN_MAX_LEN=2048 \
ROPE_BASE=5000 \
TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
WARMUP_STEPS=5 \
WARMDOWN_FRACTION=0.2 \
MAX_WALLCLOCK_SECONDS=${MAX_SECONDS} \
ITERATIONS=10000 \
MATRIX_OPTIMIZER=muon \
MATRIX_LR=0.04 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.02 \
MUON_BACKEND_STEPS=3 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_WD=0.0 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
CHURN_LOG_EVERY=0 \
FP_STORAGE=FP8 \
HESTIA=${HESTIA} \
HESTIA_TAU_INIT=0.3 \
HESTIA_PRESSURE_RAMP=0.2 \
HESTIA_ALPHA=0.4 \
STRUCTURED_2_4=${STRUCTURED_24} \
SPARSITY_REG=0.001 \
GPTQ_TERNARY=${GPTQ} \
GPTQ_CALIB_SEQS=64 \
GPTQ_CALIB_LEN=2048 \
GPTQ_CALIB_TEMP=0.8 \
EGGROLL=${EGGROLL} \
EGGROLL_SECONDS=60 \
EGGROLL_CANDIDATES=1024 \
SLIDING_EVAL=1 \
SLIDING_EVAL_STRIDE=16 \
SLIDING_BATCH_SIZE=256 \
TEMP_SCALING=1 \
COMPILE_MODE=default \
OMP_NUM_THREADS=1 \
${VALIDATE_OVERRIDES} \
torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT_NAME} 2>&1 | tee logs/cuda/${RUN_ID}_live.txt"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would execute:"
    echo "$TRAIN_CMD"
else
    $SSH_CMD "$SSH_HOST" "$TRAIN_CMD"
fi

echo "------------------------------------------------------------"
echo ""

# ── Step 4: Download results ───────────────────────────────────
echo "==> Step 4: Downloading results..."
mkdir -p "results/${RUN_ID}"
run_scp_down "$REMOTE_DIR/logs/cuda/${RUN_ID}.txt" "results/${RUN_ID}/" || true
run_scp_down "$REMOTE_DIR/logs/cuda/${RUN_ID}_live.txt" "results/${RUN_ID}/" || true
run_scp_down "$REMOTE_DIR/final_model.hestia.ptz" "results/${RUN_ID}/" || true

echo ""
echo "============================================================"
echo "  COMPLETE"
echo "============================================================"
echo "  Results: results/${RUN_ID}/"
echo "  Log:     results/${RUN_ID}/${RUN_ID}.txt"
echo "  Model:   results/${RUN_ID}/final_model.hestia.ptz"
echo ""
echo "  To re-download: $0 $SSH_HOST --download-only --run-id $RUN_ID"
echo "============================================================"
