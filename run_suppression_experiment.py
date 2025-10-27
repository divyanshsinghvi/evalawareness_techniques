#!/usr/bin/env python3
"""
Run CoT suppression experiments on rollout files.

Reads rollout files, runs suppression, and scores with LLM judge to measure effectiveness.

Usage:
    python run_suppression_experiment.py --model-dir qwen_qwen3-32b
    python run_suppression_experiment.py --model-dir qwen_qwen3-32b --limit 10
    python run_suppression_experiment.py --model-dir qwen_qwen3-32b --strategy seed --judge-model anthropic/claude-sonnet-4
"""

import yaml
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm.asyncio import tqdm
import os
import re
import hashlib
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import suppression engine
from suppression.engine import generate_with_suppression

# Import LLM judge from score_awareness_with_llm.py
from score_awareness_with_llm import score_response, score_batch, AsyncOpenAI

# Import awareness detection utilities
from utils import analyze_awareness_levels


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file byte-by-byte."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def extract_clean_reasoning_seed(
    response: str,
    max_sentences: int = None,
    detection_mode: str = "rule_based",
    detection_threshold: str = "any",
    judge_client = None,
    judge_model: str = None,
    verbose: bool = False
) -> tuple:
    """
    Extract clean reasoning seed from a response.

    Collects ALL sentences that have no awareness indicators (score below threshold).

    Args:
        response: Full response with <thinking> tags
        max_sentences: Maximum sentences to use as seed (None = use all clean sentences)
        detection_mode: "rule_based" or "llm_sentence"
        detection_threshold: For rule_based: "any"/"high"; for llm_sentence: int 1-10
        judge_client: AsyncOpenAI client for LLM detection
        judge_model: Model name for LLM detection
        verbose: Print extraction details

    Returns:
        tuple: (seed_string, stats_dict)
    """
    from score_awareness_with_llm import score_sentence

    # Extract reasoning from thinking tags
    thinking_match = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', response, re.DOTALL)
    if not thinking_match:
        return "", {}

    eval_reasoning = thinking_match.group(1).strip()

    # Split into sentences
    reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', eval_reasoning)
    clean_reasoning_parts = []

    for sent in reasoning_sentences:
        # Check awareness using same detection mode as experiment
        is_aware = False

        if detection_mode == "rule_based":
            sent_awareness = analyze_awareness_levels(sent)
            if detection_threshold == "high":
                # Only consider global or explicit awareness
                is_aware = (len(sent_awareness.global_awareness_sentences) > 0 or
                           len(sent_awareness.explicit_awareness_sentences) > 0)
            else:  # "any"
                is_aware = sent_awareness.total_awareness_count > 0

        elif detection_mode == "llm_sentence":
            if judge_client and judge_model:
                result = await score_sentence(judge_client, sent, judge_model)
                score = result.get('score', 0)
                threshold_int = int(detection_threshold)
                is_aware = score >= threshold_int

        if not is_aware:
            clean_reasoning_parts.append(sent)
        # Continue checking all sentences, don't break

    # Use all clean sentences as seed (or limit to max_sentences if specified)
    if max_sentences is None:
        seed_sentences = clean_reasoning_parts
    else:
        seed_sentences = clean_reasoning_parts[:max_sentences]

    seed_string = ' '.join(seed_sentences)

    stats = {
        'original_reasoning_length': len(eval_reasoning),
        'total_sentences': len(reasoning_sentences),
        'clean_sentences': len(clean_reasoning_parts),
        'seed_sentences_used': len(seed_sentences),
        'seed_length': len(seed_string)
    }

    if verbose:
        print(f"  Original reasoning: {stats['original_reasoning_length']} chars")
        print(f"  Total clean sentences: {stats['clean_sentences']}/{stats['total_sentences']}")
        if max_sentences is None:
            print(f"  Using ALL {stats['seed_sentences_used']} clean sentences as seed ({stats['seed_length']} chars)")
        else:
            print(f"  Using first {stats['seed_sentences_used']} sentences as seed ({stats['seed_length']} chars)")
        preview = seed_string[:200].replace('\n', ' ')
        print(f"  Preview: {preview}...")

    return seed_string, stats


async def process_rollout(
    rollout_file: Path,
    output_file: Path,
    suppression_model: str,
    judge_model: str,
    strategy: str,
    max_iterations: int,
    max_resample_attempts: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    judge_client: AsyncOpenAI,
    detection_mode: str = "rule_based",
    detection_threshold: str = "any",
    skip_existing: bool = False,
    verbose: bool = False
) -> bool:
    """Process a single rollout file with suppression and LLM judging."""
    try:
        # Simple file existence check
        if skip_existing and output_file.exists():
            if verbose:
                print(f"[DEBUG] Skipping existing file: {output_file}")
            return True  # Skip - output file exists

        # Calculate rollout file checksum
        rollout_checksum = calculate_file_checksum(rollout_file)

        # Check if already processed with matching checksum
        if output_file.exists() and not skip_existing:
            with open(output_file) as f:
                existing_data = yaml.safe_load(f)
            existing_checksum = existing_data.get('source_checksum_sha256', '')
            if existing_checksum == rollout_checksum:
                return True  # Skip - already processed with same checksum

        # Read rollout YAML
        with open(rollout_file) as f:
            data = yaml.safe_load(f)

        if not data:
            return False

        # Extract system prompt and messages
        system_prompt_original = data.get('system_prompt', '')
        deployment = data.get('deployment', {})
        evaluation = data.get('evaluation', {})

        deployment_user_msg = deployment.get('user_message', '')
        evaluation_user_msg = evaluation.get('user_message', '')

        deployment_original_response = deployment.get('assistant_response', '')
        evaluation_original_response = evaluation.get('assistant_response', '')

        if not all([system_prompt_original, deployment_user_msg, evaluation_user_msg]):
            return False

        # Replace boxing instruction with explicit \\boxed{} format
        system_prompt = system_prompt_original.replace(
            "At the end of your response, clearly state your final recommendation by boxing either A or B.",
            "At the end of your response, clearly state your final recommendation as \\boxed{A} or \\boxed{B}."
        )

        # Extract clean reasoning seed from evaluation response using same detection method
        initial_reasoning, seed_stats = await extract_clean_reasoning_seed(
            evaluation_original_response,
            max_sentences=None,  # Use ALL clean sentences
            detection_mode=detection_mode,
            detection_threshold=detection_threshold,
            judge_client=judge_client,
            judge_model=judge_model,
            verbose=verbose
        )

        # Run suppression on both branches
        async with semaphore:
            if verbose:
                print(f"\n[DEBUG] Starting to process: {rollout_file.name}")
                print(f"\n{'='*80}")
                print(f"Processing: {rollout_file.name}")
                print(f"{'='*80}")
                if seed_stats:
                    print(f"CLEAN REASONING SEED EXTRACTION:")
            # Suppress deployment
            deployment_suppression_result = await generate_with_suppression(
                user_message=deployment_user_msg,
                system_prompt=system_prompt,
                model_name=suppression_model,
                chunk_strategy="sentence",
                reasoning_strategy=strategy,
                initial_reasoning=initial_reasoning,
                initial_content="",
                max_iterations=max_iterations,
                max_resample_attempts_per_chunk=max_resample_attempts,
                temperature=temperature,
                max_tokens=1000,
                adaptive_tokens=True,
                base_tokens=100,
                growth_rate=1.5,
                detection_mode=detection_mode,
                detection_judge_model=judge_model,
                detection_threshold=detection_threshold,
                verbose=verbose
            )

            # Suppress evaluation
            evaluation_suppression_result = await generate_with_suppression(
                user_message=evaluation_user_msg,
                system_prompt=system_prompt,
                model_name=suppression_model,
                chunk_strategy="sentence",
                reasoning_strategy=strategy,
                initial_reasoning=initial_reasoning,
                initial_content="",
                max_iterations=max_iterations,
                max_resample_attempts_per_chunk=max_resample_attempts,
                temperature=temperature,
                max_tokens=1000,
                adaptive_tokens=True,
                base_tokens=100,
                growth_rate=1.5,
                detection_mode=detection_mode,
                detection_judge_model=judge_model,
                detection_threshold=detection_threshold,
                verbose=verbose
            )

            # Score all responses with LLM judge
            deployment_original_score = await score_response(
                judge_client, deployment_original_response, judge_model
            )

            deployment_suppressed_score = await score_response(
                judge_client, deployment_suppression_result.final_response, judge_model
            )

            evaluation_original_score = await score_response(
                judge_client, evaluation_original_response, judge_model
            )

            evaluation_suppressed_score = await score_response(
                judge_client, evaluation_suppression_result.final_response, judge_model
            )

        # Analyze with rule-based detection for comparison
        deployment_original_awareness = analyze_awareness_levels(deployment_original_response)
        deployment_suppressed_awareness = analyze_awareness_levels(deployment_suppression_result.final_response)
        evaluation_original_awareness = analyze_awareness_levels(evaluation_original_response)
        evaluation_suppressed_awareness = analyze_awareness_levels(evaluation_suppression_result.final_response)

        # Build output data
        output_data = {
            'source_file': str(rollout_file),
            'source_checksum_sha256': rollout_checksum,
            'timestamp': datetime.now().isoformat(),
            'experiment_config': {
                'suppression_strategy': strategy,
                'suppression_model': suppression_model,
                'judge_model': judge_model,
                'max_iterations': max_iterations,
                'temperature': temperature,
                'seed_extraction': seed_stats
            },
            'deployment': {
                'user_message': deployment_user_msg,
                'original': {
                    'response': deployment_original_response,
                    'llm_judge_score': {
                        'awareness_level': deployment_original_score.get('awareness_level'),
                        'awareness_intensity': deployment_original_score.get('awareness_intensity'),
                        'awareness_type': deployment_original_score.get('awareness_type'),
                        'dominant_features': deployment_original_score.get('dominant_features', []),
                        'response_summary': deployment_original_score.get('response_summary', '')
                    },
                    'rule_based_awareness': {
                        'global_count': len(deployment_original_awareness.global_awareness_sentences),
                        'explicit_count': len(deployment_original_awareness.explicit_awareness_sentences),
                        'implicit_count': len(deployment_original_awareness.implicit_acknowledgment_sentences),
                        'total_count': deployment_original_awareness.total_awareness_count,
                        'highest_level': deployment_original_awareness.highest_level
                    }
                },
                'suppressed': {
                    'response': deployment_suppression_result.final_response,
                    'llm_judge_score': {
                        'awareness_level': deployment_suppressed_score.get('awareness_level'),
                        'awareness_intensity': deployment_suppressed_score.get('awareness_intensity'),
                        'awareness_type': deployment_suppressed_score.get('awareness_type'),
                        'dominant_features': deployment_suppressed_score.get('dominant_features', []),
                        'response_summary': deployment_suppressed_score.get('response_summary', '')
                    },
                    'rule_based_awareness': {
                        'global_count': len(deployment_suppressed_awareness.global_awareness_sentences),
                        'explicit_count': len(deployment_suppressed_awareness.explicit_awareness_sentences),
                        'implicit_count': len(deployment_suppressed_awareness.implicit_acknowledgment_sentences),
                        'total_count': deployment_suppressed_awareness.total_awareness_count,
                        'highest_level': deployment_suppressed_awareness.highest_level
                    },
                    'suppression_metrics': {
                        'chunks_generated': deployment_suppression_result.chunks_generated,
                        'chunks_resampled': deployment_suppression_result.chunks_resampled,
                        'total_awareness_detected': deployment_suppression_result.total_awareness_detected,
                        'suppression_successful': deployment_suppression_result.suppression_successful
                    }
                }
            },
            'evaluation': {
                'user_message': evaluation_user_msg,
                'original': {
                    'response': evaluation_original_response,
                    'llm_judge_score': {
                        'awareness_level': evaluation_original_score.get('awareness_level'),
                        'awareness_intensity': evaluation_original_score.get('awareness_intensity'),
                        'awareness_type': evaluation_original_score.get('awareness_type'),
                        'dominant_features': evaluation_original_score.get('dominant_features', []),
                        'response_summary': evaluation_original_score.get('response_summary', '')
                    },
                    'rule_based_awareness': {
                        'global_count': len(evaluation_original_awareness.global_awareness_sentences),
                        'explicit_count': len(evaluation_original_awareness.explicit_awareness_sentences),
                        'implicit_count': len(evaluation_original_awareness.implicit_acknowledgment_sentences),
                        'total_count': evaluation_original_awareness.total_awareness_count,
                        'highest_level': evaluation_original_awareness.highest_level
                    }
                },
                'suppressed': {
                    'response': evaluation_suppression_result.final_response,
                    'llm_judge_score': {
                        'awareness_level': evaluation_suppressed_score.get('awareness_level'),
                        'awareness_intensity': evaluation_suppressed_score.get('awareness_intensity'),
                        'awareness_type': evaluation_suppressed_score.get('awareness_type'),
                        'dominant_features': evaluation_suppressed_score.get('dominant_features', []),
                        'response_summary': evaluation_suppressed_score.get('response_summary', '')
                    },
                    'rule_based_awareness': {
                        'global_count': len(evaluation_suppressed_awareness.global_awareness_sentences),
                        'explicit_count': len(evaluation_suppressed_awareness.explicit_awareness_sentences),
                        'implicit_count': len(evaluation_suppressed_awareness.implicit_acknowledgment_sentences),
                        'total_count': evaluation_suppressed_awareness.total_awareness_count,
                        'highest_level': evaluation_suppressed_awareness.highest_level
                    },
                    'suppression_metrics': {
                        'chunks_generated': evaluation_suppression_result.chunks_generated,
                        'chunks_resampled': evaluation_suppression_result.chunks_resampled,
                        'total_awareness_detected': evaluation_suppression_result.total_awareness_detected,
                        'suppression_successful': evaluation_suppression_result.suppression_successful
                    }
                }
            },
            'comparison': {
                'deployment_llm_intensity_reduction': (
                    deployment_original_score.get('awareness_intensity', 0) -
                    deployment_suppressed_score.get('awareness_intensity', 0)
                ),
                'evaluation_llm_intensity_reduction': (
                    evaluation_original_score.get('awareness_intensity', 0) -
                    evaluation_suppressed_score.get('awareness_intensity', 0)
                ),
                'deployment_rule_based_reduction': (
                    deployment_original_awareness.total_awareness_count -
                    deployment_suppressed_awareness.total_awareness_count
                ),
                'evaluation_rule_based_reduction': (
                    evaluation_original_awareness.total_awareness_count -
                    evaluation_suppressed_awareness.total_awareness_count
                )
            }
        }

        # Save to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return True

    except Exception as e:
        import sys
        error_msg = f"\n{'='*80}\nERROR processing {rollout_file.name}\n{'='*80}\nError type: {type(e).__name__}\nError message: {e}\n\nFull traceback:\n"
        sys.stderr.write(error_msg)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"{'='*80}\n\n")
        sys.stderr.flush()
        return False


async def main():
    parser = argparse.ArgumentParser(description='Run CoT suppression experiments')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory under working/rollouts/ (e.g., qwen_qwen3-32b)')
    parser.add_argument('--suppression-model', default='qwen/qwen3-32b',
                       help='Model to use for suppression (default: qwen/qwen3-32b)')
    parser.add_argument('--judge-model', default='qwen/qwen3-32b',
                       help='Judge model for scoring (default: qwen/qwen3-32b)')
    parser.add_argument('--strategy', default='chunk_resample',
                       choices=['chunk_resample', 'seed', 'hybrid'],
                       help='Suppression strategy (default: chunk_resample)')
    parser.add_argument('--max-iterations', type=int, default=15,
                       help='Max suppression iterations (default: 15)')
    parser.add_argument('--max-resample-attempts', type=int, default=5,
                       help='Max resample attempts per chunk (default: 5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation (default: 1.0)')
    parser.add_argument('--concurrency', type=int, default=5,
                       help='Parallel file processing (default: 5)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total files to process')
    parser.add_argument('--limit-prompts', type=int, default=None,
                       help='Limit number of unique prompts (all seeds per prompt)')
    parser.add_argument('--limit-seeds', type=int, default=None,
                       help='Limit seeds per prompt (filters all prompts or requires --limit-prompts)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that exist (fast, no checksum verification)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--experiment-name', default='01_chunk_resample',
                       help='Experiment name for output directory')
    parser.add_argument('--detection-mode', default='rule_based',
                       choices=['rule_based', 'llm_sentence'],
                       help='Detection method during generation: rule_based (regex patterns) or llm_sentence (LLM scores sentences). Final measurement uses BOTH methods.')
    parser.add_argument('--detection-threshold', default='any',
                       help='For rule_based: "any" or "high". For llm_sentence: int 1-10 (default: any for rule_based, 5 for llm_sentence)')
    parser.add_argument('--prompt-list', type=str,
                       help='Path to file containing prompt names to filter (e.g., high_awareness_high_change_prompts.txt)')

    args = parser.parse_args()

    # Validate arguments
    if args.limit and (args.limit_prompts or args.limit_seeds):
        print("Error: --limit cannot be used with --limit-prompts or --limit-seeds")
        return

    # Determine threshold value
    if args.detection_mode == "rule_based":
        detection_threshold = args.detection_threshold if args.detection_threshold in ["any", "high"] else "any"
    else:  # llm_sentence
        try:
            detection_threshold = str(int(args.detection_threshold)) if args.detection_threshold.isdigit() else "5"
        except:
            detection_threshold = "5"

    # Paths - include threshold in directory name
    rollouts_dir = Path('working/rollouts') / args.model_dir
    threshold_str = f"thresh_{detection_threshold}"
    output_dir = Path('working/suppression_experiments') / args.experiment_name / f"{args.detection_mode}_{threshold_str}" / args.model_dir

    if not rollouts_dir.exists():
        print(f"Error: Rollouts directory not found: {rollouts_dir}")
        return

    # Find all rollout files (exclude analysis files)
    all_files = list(rollouts_dir.rglob('*.yaml'))
    rollout_files = sorted([f for f in all_files if 'analysis' not in f.parts])

    if not rollout_files:
        print(f"No rollout files found in {rollouts_dir}")
        return

    # Filter by prompt list if provided
    if args.prompt_list:
        prompt_list_path = Path(args.prompt_list)
        if not prompt_list_path.exists():
            print(f"Error: Prompt list file {args.prompt_list} not found!")
            return

        # Parse prompt names from the file
        prompt_names = set()
        with open(prompt_list_path) as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Extract prompt name from lines like "Prompt: information_withholding_2025-10-24_04-25-40_ced2d7e1"
                if line.startswith('Prompt:'):
                    prompt_name = line.split(':', 1)[1].strip()
                    prompt_names.add(prompt_name)

        # Filter files to only include those matching prompt names (base name before _seed_)
        filtered_files = []
        for file in rollout_files:
            # Extract base name (before _seed_)
            match = re.match(r'(.+)_seed_\d+\.yaml$', file.name)
            if match:
                base_name = match.group(1)
                if base_name in prompt_names:
                    filtered_files.append(file)

        print(f"Filtered {len(rollout_files)} files to {len(filtered_files)} files from prompt list")
        rollout_files = filtered_files

    # Apply filtering
    if args.limit_prompts or args.limit_seeds:
        from collections import defaultdict

        # Group files by base prompt (everything before "_seed_")
        prompt_groups = defaultdict(list)
        for f in rollout_files:
            # Extract base name (before _seed_)
            match = re.match(r'(.+)_seed_(\d+)\.yaml$', f.name)
            if match:
                base_name = match.group(1)
                seed_num = int(match.group(2))
                prompt_groups[base_name].append((seed_num, f))

        # Sort each group by seed number
        for base_name in prompt_groups:
            prompt_groups[base_name].sort(key=lambda x: x[0])

        # Apply prompt limit
        if args.limit_prompts:
            selected_prompts = sorted(prompt_groups.keys())[:args.limit_prompts]
            prompt_groups = {k: v for k, v in prompt_groups.items() if k in selected_prompts}

        # Apply seed limit per prompt
        if args.limit_seeds:
            for base_name in prompt_groups:
                prompt_groups[base_name] = prompt_groups[base_name][:args.limit_seeds]

        # Flatten back to list
        rollout_files = [f for seed_num, f in sorted(
            [item for group in prompt_groups.values() for item in group],
            key=lambda x: (x[1].parent, x[1].name)
        )]
    elif args.limit:
        rollout_files = rollout_files[:args.limit]

    print(f"Found {len(rollout_files)} rollout files")
    print(f"Suppression model: {args.suppression_model}")
    print(f"Judge model: {args.judge_model}")
    print(f"Strategy: {args.strategy}")
    print(f"Detection mode: {args.detection_mode}")
    print(f"Output directory: {output_dir}")
    print(f"Concurrency: {args.concurrency}")

    # Initialize OpenAI client for judge
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    judge_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process files
    tasks = []
    for rollout_file in rollout_files:
        # Preserve directory structure
        rel_path = rollout_file.relative_to(rollouts_dir)
        output_file = output_dir / rel_path

        task = process_rollout(
            rollout_file=rollout_file,
            output_file=output_file,
            suppression_model=args.suppression_model,
            judge_model=args.judge_model,
            strategy=args.strategy,
            max_iterations=args.max_iterations,
            max_resample_attempts=args.max_resample_attempts,
            temperature=args.temperature,
            semaphore=semaphore,
            judge_client=judge_client,
            detection_mode=args.detection_mode,
            detection_threshold=detection_threshold,
            skip_existing=args.skip_existing,
            verbose=args.verbose
        )
        tasks.append(task)

    # Run with progress bar
    results = await tqdm.gather(*tasks, desc="Processing rollouts")

    # Debug: print results
    print(f"\n[DEBUG] Results: {results}")
    print(f"[DEBUG] Results type: {type(results)}")
    print(f"[DEBUG] Results length: {len(results)}")

    success_count = sum(results)
    print(f"\n{'='*80}")
    print(f"Completed: {success_count}/{len(rollout_files)} files processed successfully")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
