#!/usr/bin/env python3
"""
Categorize steered responses using response-level awareness scoring.

Scores entire steered responses (not individual sentences) for holistic awareness assessment.

Usage:
    python categorize_steered_responses.py --steered-dir working/steered-outs/qwen_qwen3-30b-a3b-thinking-2507 --judge-model qwen/qwen3-32b
    python categorize_steered_responses.py --steered-dir working/steered-outs --judge-model qwen/qwen3-32b --concurrency 100
"""

import yaml
import asyncio
from pathlib import Path
from typing import Dict
import argparse
from tqdm.asyncio import tqdm
import os
import hashlib
import traceback
import config

# Import from score_awareness_with_llm.py
from score_awareness_with_llm import score_response, AsyncOpenAI


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file byte-by-byte."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def categorize_steered_file(
    client: AsyncOpenAI,
    steered_file: Path,
    output_file: Path,
    judge_model: str,
    mode: str,
    semaphore: asyncio.Semaphore,
    use_checksum: bool = True,
    skip_existing: bool = False
) -> bool:
    """Categorize a steered response using response-level scoring and save to categorization folder."""
    try:
        # Simple file existence check
        if skip_existing and output_file.exists():
            return True  # Skip - output file exists

        # Calculate steered file checksum
        steered_checksum = calculate_file_checksum(steered_file)

        # Check if already processed with matching checksum
        if output_file.exists() and use_checksum and not skip_existing:
            with open(output_file) as f:
                existing_data = yaml.safe_load(f)
            existing_checksum = existing_data.get('source_checksum_sha256', '')
            if existing_checksum == steered_checksum:
                return True  # Skip - already processed with same checksum

        # Read steered YAML
        with open(steered_file) as f:
            steered_data = yaml.safe_load(f)

        if not steered_data:
            return False

        # Read model_only_response and clean <|endoftext|> tokens
        steered_response_raw = steered_data.get('model_only_response', '')
        if not steered_response_raw:
            return False

        # Replace <|endoftext|> tokens with empty string
        steered_response = steered_response_raw.replace('<|endoftext|>', '')

        # Create minimal output data structure
        data = {
            'source_file': str(steered_file),
            'source_checksum_sha256': steered_checksum,
            'original_source_file': steered_data.get('source_file', ''),
            'original_checksum': steered_data.get('checksum', ''),
            'steered_response': {
                'response': steered_response,  # Store cleaned response
                'awareness': {}
            }
        }

        async with semaphore:
            # Score steered response
            if steered_response:
                awareness = await score_response(client, steered_response, judge_model, mode, verbose=False)
                data['steered_response']['awareness'] = awareness

        # Save to steered categorization folder
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR processing {steered_file.name}")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        return False


async def main():
    parser = argparse.ArgumentParser(description='Categorize steered responses (holistic scoring)')
    parser.add_argument('--steered-dir', type=str, required=True,
                       help='Steered outputs directory (e.g., working/steered-outs/qwen_qwen3-30b-a3b-thinking-2507)')
    parser.add_argument('--judge-model', default=config.DEFAULT_JUDGE_MODEL,
                       help=f'Judge model (default: {config.DEFAULT_JUDGE_MODEL})')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='Parallel file processing (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total files to process (takes first N files)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocess all files (ignore checksum)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that exist (fast, no checksum verification)')
    parser.add_argument('--mode', type=str, required=True, choices=['v0', 'v1'],
                       help='Prompt version to use (v0 or v1)')

    args = parser.parse_args()

    # Validate arguments
    if args.force and args.skip_existing:
        print("Error: --force and --skip-existing cannot be used together")
        return 1

    steered_dir = Path(args.steered_dir)

    # Create output directory path: steered_categorization/{mode} mirrors steered-outs structure
    # steered-outs/model_name/... -> steered_categorization/{mode}/model_name/...
    if 'steered-outs' in steered_dir.parts:
        # Extract parts after 'working'
        parts_after_working = []
        found_working = False
        for part in steered_dir.parts:
            if part == 'working':
                found_working = True
            elif found_working:
                if part == 'steered-outs':
                    parts_after_working.append('steered_categorization')
                    parts_after_working.append(args.mode)
                else:
                    parts_after_working.append(part)

        steered_categorization_dir = Path('working') / Path(*parts_after_working)
    else:
        # Default fallback
        steered_categorization_dir = Path('working/steered_categorization') / args.mode / steered_dir.name

    if not steered_dir.exists():
        print(f"Error: {steered_dir} not found")
        return 1

    print(f"{'='*80}")
    print(f"CATEGORIZE STEERED RESPONSES")
    print(f"{'='*80}")
    print(f"Input: {steered_dir}")
    print(f"Output: {steered_categorization_dir}")
    print(f"Judge: {args.judge_model}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Initialize OpenRouter client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return 1

    client = AsyncOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=api_key
    )

    # Find steered output files
    steered_files = list(steered_dir.rglob('*_steer_out.yaml'))

    if args.limit:
        steered_files = steered_files[:args.limit]

    print(f"Found {len(steered_files)} steered output files")

    # Create file pairs
    file_pairs = [
        (f, steered_categorization_dir / f.relative_to(steered_dir))
        for f in steered_files
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
        categorize_steered_file(client, steered, output, args.judge_model, args.mode, semaphore, use_checksum, skip_existing)
        for steered, output in file_pairs
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
