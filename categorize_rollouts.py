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

# Import from score_awareness_with_llm.py
from score_awareness_with_llm import compute_checksum, score_batch, AsyncOpenAI
import os
import re


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
    semaphore: asyncio.Semaphore
) -> bool:
    """Categorize sentences in a rollout file and save to categorization folder."""
    try:
        # Skip if already processed
        if output_file.exists():
            return True

        # Read rollout YAML
        with open(rollout_file) as f:
            data = yaml.safe_load(f)

        if not data:
            return False

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
            deployment_categorized.append({
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'thinking'
            })
        for sent, res in zip(deployment_sentences, deployment_results):
            deployment_categorized.append({
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'visible'
            })

        evaluation_categorized = []
        for sent, res in zip(evaluation_thinking_sentences, evaluation_thinking_results):
            evaluation_categorized.append({
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'thinking'
            })
        for sent, res in zip(evaluation_sentences, evaluation_results):
            evaluation_categorized.append({
                'sentence': sent,
                'score': res['score'],
                'category': res['category'],
                'checksum': compute_checksum(sent),
                'source': 'visible'
            })

        deployment['categorized_sentences'] = deployment_categorized
        evaluation['categorized_sentences'] = evaluation_categorized

        # Save to categorization folder
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return True

    except Exception as e:
        print(f"\nERROR processing {rollout_file.name}: {e}")
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
                       help='Limit files to process')

    args = parser.parse_args()

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

    if args.limit:
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
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        categorize_rollout_file(client, rollout, output, args.judge_model, args.batch_size, semaphore)
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
