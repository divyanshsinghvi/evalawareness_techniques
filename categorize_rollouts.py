#!/usr/bin/env python3
"""
Categorize sentences in rollout YAML files using score_awareness_with_llm.py functions.

Reads rollout files, scores sentences, and saves to categorization folder.

Usage:
    python categorize_rollouts.py --model-dir qwen_qwen3-32b
    python categorize_rollouts.py --model-dir qwen_qwen3-32b --limit 10
"""

import yaml
import asyncio
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm.asyncio import tqdm
from collections import defaultdict
import os
import re
import hashlib
import traceback

# Import from score_awareness_with_llm.py
from score_awareness_with_llm import compute_checksum, score_batch, AsyncOpenAI


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file byte-by-byte."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_thinking_content(text: str) -> str:
    """Extract content from within thinking tags."""
    thinking_matches = re.findall(r'<thinking>(.*?)</thinking>', text, flags=re.DOTALL | re.IGNORECASE)
    think_matches = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)

    all_thinking = thinking_matches + think_matches
    return ' '.join(all_thinking).strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, removing thinking tags."""
    # Remove thinking tags
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Split on sentence boundaries OR newlines
    # First split on punctuation boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Then split each on newlines to catch headings and non-sentence text
    final_sentences = []
    for sent in sentences:
        parts = sent.split('\n')
        final_sentences.extend(parts)

    # Filter out very short sentences and clean up
    final_sentences = [s.strip() for s in final_sentences if len(s.strip()) > 20]

    return final_sentences


async def categorize_rollout_file(
    client: AsyncOpenAI,
    rollout_file: Path,
    output_file: Path,
    judge_model: str,
    batch_size: int,
    semaphore: asyncio.Semaphore,
    use_checksum: bool = True,
    skip_existing: bool = False
) -> bool:
    """Categorize sentences in a rollout file and save to categorization folder."""
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
            data = yaml.safe_load(f)

        if not data:
            return False

        # Add source checksum to data
        data['source_file'] = str(rollout_file)
        data['source_checksum_sha256'] = rollout_checksum

        deployment = data.get('deployment', {})
        evaluation = data.get('evaluation', {})

        deployment_response = deployment.get('assistant_response', '')
        evaluation_response = evaluation.get('assistant_response', '')

        # Extract thinking content (from within <thinking> tags)
        deployment_thinking = extract_thinking_content(deployment_response)
        evaluation_thinking = extract_thinking_content(evaluation_response)

        # Extract sentences from thinking (if exists)
        deployment_thinking_sentences = split_into_sentences(deployment_thinking) if deployment_thinking else []
        evaluation_thinking_sentences = split_into_sentences(evaluation_thinking) if evaluation_thinking else []

        # Extract sentences from visible response (thinking tags removed)
        deployment_sentences = split_into_sentences(deployment_response)
        evaluation_sentences = split_into_sentences(evaluation_response)

        all_sentences = (deployment_thinking_sentences + deployment_sentences +
                        evaluation_thinking_sentences + evaluation_sentences)

        if not all_sentences:
            return False

        # Score with batch processing
        async with semaphore:
            if batch_size > 1:
                batches = [all_sentences[i:i+batch_size] for i in range(0, len(all_sentences), batch_size)]
                batch_results = []
                for batch in batches:
                    result = await score_batch(client, batch, judge_model)
                    batch_results.extend(result)
            else:
                batch_results = []
                for sentence in all_sentences:
                    result = await score_batch(client, [sentence], judge_model)
                    batch_results.extend(result)

        # Split results back (order: deploy_thinking, deploy_visible, eval_thinking, eval_visible)
        idx = 0
        deployment_thinking_results = batch_results[idx:idx+len(deployment_thinking_sentences)]
        idx += len(deployment_thinking_sentences)

        deployment_results = batch_results[idx:idx+len(deployment_sentences)]
        idx += len(deployment_sentences)

        evaluation_thinking_results = batch_results[idx:idx+len(evaluation_thinking_sentences)]
        idx += len(evaluation_thinking_sentences)

        evaluation_results = batch_results[idx:idx+len(evaluation_sentences)]

        # Create categorized lists - thinking sentences first, then visible
        deployment_categorized = []
        for sent, res in zip(deployment_thinking_sentences, deployment_thinking_results):
            entry = {
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'thinking'
            }
            if 'test_reference_flag' in res:
                entry['test_reference_flag'] = res['test_reference_flag']
            deployment_categorized.append(entry)

        for sent, res in zip(deployment_sentences, deployment_results):
            entry = {
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'visible'
            }
            if 'test_reference_flag' in res:
                entry['test_reference_flag'] = res['test_reference_flag']
            deployment_categorized.append(entry)

        evaluation_categorized = []
        for sent, res in zip(evaluation_thinking_sentences, evaluation_thinking_results):
            entry = {
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'thinking'
            }
            if 'test_reference_flag' in res:
                entry['test_reference_flag'] = res['test_reference_flag']
            evaluation_categorized.append(entry)

        for sent, res in zip(evaluation_sentences, evaluation_results):
            entry = {
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'visible'
            }
            if 'test_reference_flag' in res:
                entry['test_reference_flag'] = res['test_reference_flag']
            evaluation_categorized.append(entry)

        deployment['categorized_sentences'] = deployment_categorized
        evaluation['categorized_sentences'] = evaluation_categorized

        # Save to categorization folder
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
    parser = argparse.ArgumentParser(description='Categorize rollout sentences')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory (e.g., qwen_qwen3-32b)')
    parser.add_argument('--judge-model', default='qwen/qwen3-32b',
                       help='Judge model (default: qwen/qwen3-32b)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Sentences per batch (default: 5)')
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

    args = parser.parse_args()

    # Validate arguments
    if args.limit and (args.limit_prompts or args.limit_seeds):
        print("Error: --limit cannot be used with --limit-prompts or --limit-seeds")
        return 1

    if args.force and args.skip_existing:
        print("Error: --force and --skip-existing cannot be used together")
        return 1

    rollout_dir = Path('working/rollouts') / args.model_dir
    categorization_dir = Path('working/categorization') / args.model_dir

    if not rollout_dir.exists():
        print(f"Error: {rollout_dir} not found")
        return 1

    print(f"{'='*80}")
    print(f"CATEGORIZE ROLLOUT SENTENCES")
    print(f"{'='*80}")
    print(f"Input: {rollout_dir}")
    print(f"Output: {categorization_dir}")
    print(f"Judge: {args.judge_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Initialize client
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
        (f, categorization_dir / f.relative_to(rollout_dir))
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
        categorize_rollout_file(client, rollout, output, args.judge_model, args.batch_size, semaphore, use_checksum, skip_existing)
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
