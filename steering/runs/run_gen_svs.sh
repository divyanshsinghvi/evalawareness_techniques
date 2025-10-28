#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Generate Steering Vectors Script
# This script runs generate_simple_svs.py to compute steering vectors from activations

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Generate Steering Vectors"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
MODEL_NAME="nvidia_llama-3.3-nemotron-super-49b-v1.5"
EVAL_DIR="/pscratch/sd/r/ritesh11/temp/steering_experiments/nemotron-49B/nemo-anthropic-modelacts/eval"
DEPLOY_DIR="/pscratch/sd/r/ritesh11/temp/steering_experiments/nemotron-49B/nemo-anthropic-modelacts/deploy"
BASE_OUTPUT_DIR="/pscratch/sd/r/ritesh11/temp/working/steering-vecs"
CATEGORY="anthropic"  # or "anthropic"

# Layer range - choose one or define custom
# Specific range
START_LAYER=9
END_LAYER=31

# Alternative: Middle layers only (uncomment to use)
# START_LAYER=20
# END_LAYER=40

# Alternative: Last quarter of a 48-layer model (uncomment to use)
# START_LAYER=36
# END_LAYER=47

# Alternative: All layers (uncomment to use)
# START_LAYER=0
# END_LAYER=47

# Construct final output path
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME/$CATEGORY"

# Display configuration
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Eval dir: $EVAL_DIR"
echo "  Deploy dir: $DEPLOY_DIR"
echo "  Category: $CATEGORY"
echo "  Layers: $START_LAYER to $END_LAYER ($(($END_LAYER - $START_LAYER + 1)) layers)"
echo "  Base output: $BASE_OUTPUT_DIR"
echo "  Final output: $FINAL_OUTPUT_DIR"
echo ""

# Check if input directories exist
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Eval directory not found: $EVAL_DIR"
    exit 1
fi

if [ ! -d "$DEPLOY_DIR" ]; then
    echo "ERROR: Deploy directory not found: $DEPLOY_DIR"
    exit 1
fi

# Execute the generation script
python generate_simple_svs.py \
  --model_name "$MODEL_NAME" \
  --eval_dir "$EVAL_DIR" \
  --deploy_dir "$DEPLOY_DIR" \
  --category "$CATEGORY" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --start_layer $START_LAYER \
  --end_layer $END_LAYER

echo ""
echo "=========================================="
echo "Steering vector generation completed!"
echo "Saved $(($END_LAYER - $START_LAYER + 1)) vectors to $FINAL_OUTPUT_DIR"
echo "=========================================="