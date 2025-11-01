#!/bin/bash
# Run check_awareness_followup.py for all experiment variants
# Usage: ./run_awareness_followup_all.sh [--skip EXPERIMENT_NAME] [--only EXPERIMENT_NAME] [--mode binary|ternary] [--verbose] [--concurrency N] [--model MODEL]

set -e

EVALUATED_MODEL="qwen_qwen3-32b"  # The model being evaluated (filesystem-safe name for directory paths) - can be overridden with --model
JUDGE_MODEL="qwen/qwen3-32b"  # The model that answers the follow-up question (API format, always fixed)
MODE="ternary"  # binary (yes/no) or ternary (yes/no/unsure) - can be overridden with --mode
QUESTION_TYPE="awareness"  # awareness or counterfactual - can be overridden with --question-type
LIMIT_SEEDS=29
CONCURRENCY=100
VERBOSE=""  # Set to "--verbose" to show detailed output

# Parse command line arguments
SKIP_EXPERIMENTS=()
ONLY_EXPERIMENTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip)
            SKIP_EXPERIMENTS+=("$2")
            shift 2
            ;;
        --only)
            ONLY_EXPERIMENTS+=("$2")
            shift 2
            ;;
        --mode)
            MODE="$2"
            if [[ "$MODE" != "binary" && "$MODE" != "ternary" ]]; then
                echo "Error: --mode must be 'binary' or 'ternary'"
                exit 1
            fi
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift 1
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --model)
            EVALUATED_MODEL="$2"
            shift 2
            ;;
        --question-type)
            QUESTION_TYPE="$2"
            if [[ "$QUESTION_TYPE" != "awareness" && "$QUESTION_TYPE" != "counterfactual" ]]; then
                echo "Error: --question-type must be 'awareness' or 'counterfactual'"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip EXPERIMENT_NAME] [--only EXPERIMENT_NAME] [--mode binary|ternary] [--question-type awareness|counterfactual] [--verbose] [--concurrency N] [--model MODEL]"
            exit 1
            ;;
    esac
done

# Function to check if experiment should be skipped
should_skip() {
    local exp_name="$1"

    # If --only is specified, skip everything except those experiments
    if [ ${#ONLY_EXPERIMENTS[@]} -gt 0 ]; then
        for only in "${ONLY_EXPERIMENTS[@]}"; do
            if [ "$exp_name" == "$only" ]; then
                return 1  # Don't skip
            fi
        done
        return 0  # Skip
    fi

    # Check if in skip list
    for skip in "${SKIP_EXPERIMENTS[@]}"; do
        if [ "$exp_name" == "$skip" ]; then
            return 0  # Skip
        fi
    done

    return 1  # Don't skip
}

# Function to run awareness followup
run_followup() {
    local exp_name="$1"
    local file_type="$2"
    local input_dir="$3"
    local extra_args="${4:-}"

    if should_skip "$exp_name"; then
        echo "⏭️  Skipping: $exp_name"
        return 0
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Running: $exp_name"
    echo "Type: $file_type"
    echo "Input: $input_dir"
    echo "═══════════════════════════════════════════════════════════════"

    python awareness_probe/check_awareness_followup.py \
        --input-dir "$input_dir" \
        --file-type "$file_type" \
        --model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --mode "$MODE" \
        --question-type "$QUESTION_TYPE" \
        --max-seed "$LIMIT_SEEDS" \
        --skip-existing \
        $VERBOSE \
        $extra_args

    echo "✓ Completed: $exp_name"
}

echo "═══════════════════════════════════════════════════════════════"
echo "AWARENESS FOLLOWUP - ALL EXPERIMENTS"
echo "═══════════════════════════════════════════════════════════════"
echo "Evaluated Model: $EVALUATED_MODEL"
echo "Judge Model: $JUDGE_MODEL"
echo "Mode: $MODE"
echo "Limit seeds: $LIMIT_SEEDS"
echo "Concurrency: $CONCURRENCY"
if [ ${#SKIP_EXPERIMENTS[@]} -gt 0 ]; then
    echo "Skipping: ${SKIP_EXPERIMENTS[*]}"
fi
if [ ${#ONLY_EXPERIMENTS[@]} -gt 0 ]; then
    echo "Only running: ${ONLY_EXPERIMENTS[*]}"
fi
echo "═══════════════════════════════════════════════════════════════"

# ============================================================================
# ROLLOUTS WITH PROMPT-LIST FILTERING (1 path) - RUN FIRST
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROLLOUTS (with prompt-list filtering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use high_awareness_bc_seeds.yaml as the prompt list
PROMPT_LIST="working/response_categorization/v1/$EVALUATED_MODEL/high_awareness_bc_seeds.yaml"

if [ ! -f "$PROMPT_LIST" ]; then
    echo "⚠️  Warning: Prompt list not found at $PROMPT_LIST"
    echo "   Skipping rollouts processing."
else
    run_followup \
        "rollouts" \
        "rollout" \
        "working/rollouts/$EVALUATED_MODEL" \
        "--prompt-list $PROMPT_LIST"
fi

# ============================================================================
# SUPPRESSION EXPERIMENTS (12 paths)
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUPPRESSION EXPERIMENTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_followup \
    "sentence_resample_pick_if_not_evaly" \
    "suppression" \
    "working/suppression_experiments/01_chunk_resample/llm_sentence_thresh_3_selfandother_reasoningonly_max_resample_10_temperature_07/$EVALUATED_MODEL"

run_followup \
    "sentence_resample_pick_least_eval_among_5" \
    "suppression" \
    "working/suppression_experiments/06_resample_best_of_n5/llm_sentence_thresh_3_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "remove_only" \
    "suppression" \
    "working/suppression_experiments/02_remove_only/llm_sentence_thresh_3_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "system_prompt" \
    "suppression" \
    "working/suppression_experiments/03_system_prompt_deployment/llm_sentence_thresh_3_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "system_prompt_no_spec" \
    "suppression" \
    "working/suppression_experiments/04_system_prompt_deployment_no_speculation/llm_sentence_thresh_3_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "no_reasoning" \
    "suppression" \
    "working/suppression_experiments/05_no_reasoning/llm_sentence_thresh_3_selfandother_full/$EVALUATED_MODEL"

run_followup \
    "sentence_resample_pick_least_eval_for_reasoning_and_content" \
    "suppression" \
    "working/suppression_experiments/06_resample_best_of_n5_full/llm_sentence_thresh_3_selfandother_full/$EVALUATED_MODEL"

run_followup \
    "only_final_answer" \
    "suppression" \
    "working/suppression_experiments/07_strict_boxed_only/llm_sentence_thresh_3_selfandother_full/$EVALUATED_MODEL"

run_followup \
    "system_prompt_taboo_words" \
    "suppression" \
    "working/suppression_experiments/09_system_prompt_taboo_words/llm_sentence_thresh_3_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "chunk_resample_thresh5_temp07_fixed" \
    "suppression" \
    "working/suppression_experiments/10_chunk_resample_threshold_5_temperature_07_fixed/llm_sentence_thresh_5_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "chunk_resample_exclude_criteria_2_v2" \
    "suppression" \
    "working/suppression_experiments/11_chunk_resample_criteria_exclude_2_v2sentence_temperature_07/llm_sentence_exclude_criteria_2_selfandother_reasoningonly/$EVALUATED_MODEL"

run_followup \
    "resample_best_of_n5_thresh5_temp07" \
    "suppression" \
    "working/suppression_experiments/12_resample_best_of_n5_threshold_5_temperature_07/llm_sentence_thresh_5_selfandother_reasoningonly/$EVALUATED_MODEL"

# ============================================================================
# STEERING EXPERIMENTS (4 paths)
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEERING EXPERIMENTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_followup \
    "tim_N4_S3.5" \
    "steered" \
    "working/steered-outs/$EVALUATED_MODEL/high_awareness_bc/tim/eval/N4_S3.5"

run_followup \
    "anthropic_N4_S1.0" \
    "steered" \
    "working/steered-outs/$EVALUATED_MODEL/high_awareness_bc/anthropic/eval/N4_S1.0"

run_followup \
    "tim_N4_S3.5_random" \
    "steered" \
    "working/steered-outs/$EVALUATED_MODEL/high_awareness_bc/tim/eval/N4_S3.5_random"

run_followup \
    "tim_deploy_N4_S-3.5" \
    "steered" \
    "working/steered-outs/$EVALUATED_MODEL/high_awareness_bc/tim/deploy/N4_S-3.5"

# ============================================================================
# OFF-POLICY EXPERIMENTS (2 paths)
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OFF-POLICY EXPERIMENTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_followup \
    "prefill_explicit" \
    "off-policy" \
    "working/off-policy-intervention/$EVALUATED_MODEL/explicit/prefill/explicit"

run_followup \
    "prefill_explicit1" \
    "off-policy" \
    "working/off-policy-intervention/$EVALUATED_MODEL/explicit1/prefill/explicit"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✓ ALL EXPERIMENTS COMPLETED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Output locations:"
echo "  Suppression: working/awareness_yes_no_unsure/suppression/"
echo "  Steering:    working/awareness_yes_no_unsure/steered/"
echo "  Off-Policy:  working/awareness_yes_no_unsure/off-policy/"
echo "  Rollouts:    working/awareness_yes_no_unsure/rollout/"
echo "═══════════════════════════════════════════════════════════════"
