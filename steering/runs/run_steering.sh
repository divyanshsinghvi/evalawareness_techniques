#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Run steering model script
# This script demonstrates how to run steer_model.py with all required arguments

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Steering Model Execution Script"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Set up model and paths
MODEL="qwen_qwen3-30b-a3b-thinking-2507"
MODEL_PATH="/pscratch/sd/r/ritesh11/temp/models/Qwen3-30B-A3B-Thinking-2507"

# Base directories (script will construct full paths automatically)
BASE_PROMPTS_DIR="/pscratch/sd/r/ritesh11/temp/working/steer_formatted_prompts"  # Will use: BASE_PROMPTS_DIR/MODEL
BASE_OUTPUT_DIR="/pscratch/sd/r/ritesh11/temp/working/steered-outs"              # Will create: BASE_OUTPUT_DIR/MODEL/PRIORITY/VEC_TYPE/MODE
STEERING_VEC_DIR="/pscratch/sd/r/ritesh11/temp/working/steering-vecs/qwen_qwen-3-30b-a3b-thinking-2507/tim"

# Configuration
MODE="eval"  # or "deploy"
PRIORITY="high_awareness_bc"
VEC_TYPE="tim"  # or your vector type

# Model dimensions (adjust based on your model)
D_MODEL=2048
MODEL_LEN=48

# Steering configuration
STEER_ON_USER=true
STEER_ON_THINKING=true
STEER_ON_SYSTEM=false

# Generation settings
BATCH_SIZE=128
MAX_NEW_TOKENS=3000
TEMPERATURE=0.7
TOP_P=1
DTYPE="bfloat16"
SEED=0

# Steering parameters
LAYER_RANGE=(5 20)  # First and last layer to consider
NUM_LAYERS=(4)    # Number of layers to steer
STRENGTH=(3.5)  # Positive for eval (increase awareness), negative for deploy (decrease awareness)

# Construct full paths that will be used
FULL_PROMPTS_DIR="$BASE_PROMPTS_DIR/$MODEL"
FULL_OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL/$PRIORITY/$VEC_TYPE/$MODE"

# Check if prompts directory exists
if [ ! -d "$FULL_PROMPTS_DIR" ]; then
    echo "Error: Prompts directory not found: $FULL_PROMPTS_DIR"
    echo "Expected structure: $BASE_PROMPTS_DIR/$MODEL/$PRIORITY/"
    exit 1
fi

# Check if steering vector directory exists
if [ ! -d "$STEERING_VEC_DIR" ]; then
    echo "Warning: Steering vector directory not found: $STEERING_VEC_DIR"
    echo "Creating directory..."
    mkdir -p "$STEERING_VEC_DIR"
fi

echo "Output will be saved to: $FULL_OUTPUT_DIR/N{n}_S{s}/"

# Build the base command
python3 evalawareness_techniques/steering/steer_model.py \
  --model "$MODEL" \
  --model_path "$MODEL_PATH" \
  --vec_type "$VEC_TYPE" \
  --mode "$MODE" \
  --priority "$PRIORITY" \
  --steering_vec_dir "$STEERING_VEC_DIR" \
  --out_dir "$BASE_OUTPUT_DIR" \
  --prompts_dir "$BASE_PROMPTS_DIR" \
  --dtype "$DTYPE" \
  --seed $SEED \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --d_model $D_MODEL \
  --model_len $MODEL_LEN \
  --layer_range ${LAYER_RANGE[@]} \
  --num_layers ${NUM_LAYERS[@]} \
  --strength ${STRENGTH[@]} \
  $([ "$STEER_ON_USER" = true ] && echo "--steer_on_user" || echo "") \
  $([ "$STEER_ON_THINKING" = true ] && echo "--steer_on_thinking" || echo "") \
  $([ "$STEER_ON_SYSTEM" = true ] && echo "--steer_on_system" || echo "")

echo ""
echo "=========================================="
echo "Steering execution completed!"
echo "=========================================="

