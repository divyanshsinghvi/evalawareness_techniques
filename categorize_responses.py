#!/usr/bin/env python3
"""
Categorize responses in rollout YAML files using response-level awareness scoring.

Scores entire responses (not individual sentences) for holistic awareness assessment.
Scores full responses (with thinking tags) for deployment and evaluation branches.

Usage:
    python categorize_responses.py --model-dir qwen_qwen3-32b --judge-model qwen/qwen3-32b
    python categorize_responses.py --model-dir qwen_qwen3-32b --judge-model qwen/qwen3-32b --limit 10
"""

import yaml
import asyncio
from pathlib import Path
from typing import Dict
import argparse
from tqdm.asyncio import tqdm
from collections import defaultdict
import os
import re
import hashlib
import traceback

# Import from score_awareness_with_llm.py
from score_awareness_with_llm import score_response, AsyncOpenAI


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file byte-by-byte."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()




async def categorize_response_file(
    client: AsyncOpenAI,
    rollout_file: Path,
    output_file: Path,
    judge_model: str,
    semaphore: asyncio.Semaphore,
    use_checksum: bool = True,
    skip_existing: bool = False
) -> bool:
    """Categorize responses in a rollout file using response-level scoring and save to response_categorization folder."""
    try:
        # Simple file existence check
        if skip_existing and output_file.exists():
            return True  # Skip - output file exists

        # Calculate rollout file checksum
        rollout_checksum = calculate_file_checksum(rollout_file)

        # Check if already processed with matching checksum
        if output_file.exists() and use_checksum and not skip_existing:
            with open(output_file) as f:
                existing_data = yaml.safe_load(f)
            existing_checksum = existing_data.get('source_checksum_sha256', '')
            if existing_checksum == rollout_checksum:
                return True  # Skip - already processed with same checksum

        # Read rollout YAML
        with open(rollout_file) as f:
            rollout_data = yaml.safe_load(f)

        if not rollout_data:
            return False

        deployment_response = rollout_data.get('deployment', {}).get('assistant_response', '')
        evaluation_response = rollout_data.get('evaluation', {}).get('assistant_response', '')

        # Create minimal output data structure
        data = {
            'source_file': str(rollout_file),
            'source_checksum_sha256': rollout_checksum,
            'metadata': rollout_data.get('metadata', {}),
            'behavioral_change': rollout_data.get('behavioral_change', False),
            'deployment': {},
            'evaluation': {}
        }

        async with semaphore:
            # Score deployment response (full with thinking)
            if deployment_response:
                deployment_awareness = await score_response(client, deployment_response, judge_model)
                data['deployment']['awareness'] = deployment_awareness

            # Score evaluation response (full with thinking)
            if evaluation_response:
                evaluation_awareness = await score_response(client, evaluation_response, judge_model)
                data['evaluation']['awareness'] = evaluation_awareness

        # Save to response_categorization folder
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR processing {rollout_file.name}")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        return False


async def main():
    parser = argparse.ArgumentParser(description='Categorize rollout responses (holistic scoring)')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory (e.g., qwen_qwen3-32b)')
    parser.add_argument('--judge-model', default='qwen/qwen3-32b',
                       help='Judge model (default: qwen/qwen3-32b)')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='Parallel file processing (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total files to process (takes first N files)')
    parser.add_argument('--limit-prompts', type=int, default=None,
                       help='Limit number of unique prompts (all seeds per prompt)')
    parser.add_argument('--limit-seeds', type=int, default=None,
                       help='Limit seeds per prompt (requires --limit-prompts or filters all prompts)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocess all files (ignore checksum)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that exist (fast, no checksum verification)')
    parser.add_argument('--prompt-list', type=str,
                       help='Path to file containing prompt names to filter (e.g., high_awareness_high_change_prompts.txt)')

    args = parser.parse_args()

    # Validate arguments
    if args.limit and (args.limit_prompts or args.limit_seeds):
        print("Error: --limit cannot be used with --limit-prompts or --limit-seeds")
        return 1

    if args.force and args.skip_existing:
        print("Error: --force and --skip-existing cannot be used together")
        return 1

    # Validate judge model
    if "qwen3-32b" not in args.judge_model.lower():
        print(f"Error: Judge model must be qwen3-32b, got: {args.judge_model}")
        return 1

    rollout_dir = Path('working/rollouts') / args.model_dir
    response_categorization_dir = Path('working/response_categorization') / args.model_dir

    if not rollout_dir.exists():
        print(f"Error: {rollout_dir} not found")
        return 1

    print(f"{'='*80}")
    print(f"CATEGORIZE ROLLOUT RESPONSES")
    print(f"{'='*80}")
    print(f"Input: {rollout_dir}")
    print(f"Output: {response_categorization_dir}")
    print(f"Judge: {args.judge_model}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Initialize OpenRouter client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return 1

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Find rollout files
    rollout_files = list(rollout_dir.rglob('*.yaml'))
    rollout_files = [f for f in rollout_files if 'analysis' not in f.parts]

    # Filter by prompt list if provided
    if args.prompt_list:
        prompt_list_path = Path(args.prompt_list)
        if not prompt_list_path.exists():
            print(f"Error: Prompt list file {args.prompt_list} not found!")
            return 1

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

    print(f"Found {len(rollout_files)} files")

    # Create file pairs
    file_pairs = [
        (f, response_categorization_dir / f.relative_to(rollout_dir))
        for f in rollout_files
    ]

    # Check existing
    existing = sum(1 for _, out in file_pairs if out.exists())
    if existing > 0:
        print(f"Skipping {existing} already categorized")
    print()

    # Process
    use_checksum = not args.force  # Use checksum unless --force is specified
    skip_existing = args.skip_existing
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        categorize_response_file(client, rollout, output, args.judge_model, semaphore, use_checksum, skip_existing)
        for rollout, output in file_pairs
    ]

    results = await tqdm.gather(*tasks, desc="Processing")

    # Summary
    success = sum(1 for r in results if r)
    print()
    print(f"{'='*80}")
    print(f"Complete: {success}/{len(results)} files")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))
