#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Collect Model Activations Script
# This script runs collect_activations.py to collect activations from a dataset

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Collect Model Activations"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
MODEL="/pscratch/sd/r/ritesh11/temp/Qwen3-30B-A3B"
DATASET="prompts"
DATASET_SPLIT="eval"
TEXT_COLUMN="text"
ACTIVATION_STORE_DIR="model_activations"

# Layer selection - choose one or define custom
# Single layer
LAYERS=(22)

# Multiple specific layers (uncomment to use)
# LAYERS=(10 15 20 25 30)

# Range of layers (uncomment to use)
# LAYERS=($(seq 10 5 30))  # Layers 10, 15, 20, 25, 30

# All middle layers (uncomment to use)
# LAYERS=($(seq 10 40))  # Layers 10 through 40

# First, middle, and last layers for a 48-layer model (uncomment to use)
# LAYERS=(0 12 24 36 47)

BATCH_SIZE=1
CONTEXT_LEN=3008
NUM_SAMPLES=1000000
MAX_TOKENS=100000000
DTYPE="float16"
RANDOM_SEED=42
STORE_TOKENS=true
OVERWRITE=false
DISABLE_MULTIPROCESSING=false

# Display configuration
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET (split: $DATASET_SPLIT)"
echo "  Layers: ${LAYERS[@]} (${#LAYERS[@]} layers)"
echo "  Batch size: $BATCH_SIZE"
echo "  Context length: $CONTEXT_LEN"
echo "  Output: $ACTIVATION_STORE_DIR"
echo ""

# Execute the collection script
python3 evalawareness_techniques/steering/collect_activations.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  --activation-store-dir "$ACTIVATION_STORE_DIR" \
  --layers ${LAYERS[@]} \
  --batch-size $BATCH_SIZE \
  --context-len $CONTEXT_LEN \
  --num-samples $NUM_SAMPLES \
  --max-tokens $MAX_TOKENS \
  --dtype "$DTYPE" \
  --random-seed $RANDOM_SEED \
  $([ "$STORE_TOKENS" = true ] && echo "--store-tokens" || echo "--no-store-tokens") \
  $([ "$OVERWRITE" = true ] && echo "--overwrite" || echo "") \
  $([ "$DISABLE_MULTIPROCESSING" = true ] && echo "--disable-multiprocessing" || echo "")

echo ""
echo "=========================================="
echo "Activation collection completed!"
echo "=========================================="

