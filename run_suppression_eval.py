#!/usr/bin/env python3
"""
Run on-policy suppression evaluation on high-awareness seeds.

This script:
1. Loads rollout YAML files
2. Runs on-policy suppression on evaluation responses
3. Compares original vs suppressed responses
4. Computes comprehensive metrics
5. Saves results with caching

Usage:
    # Process all transcripts
    python run_suppression_eval.py \\
        --transcripts working/test_transcripts_suppression.txt \\
        --output-dir working/suppression_result_cot \\
        --chunk-strategy sentence \\
        --max-iterations 10 \\
        --max-resample-attempts 3

    # Limit to 2 prompts for testing
    python run_suppression_eval.py \\
        --transcripts working/test_transcripts_suppression.txt \\
        --output-dir working/suppression_result_cot \\
        --max-prompts 2 \\
        --cache

Output structure:
    working/suppression_result_cot/
    ├── {model}/
    │   ├── {prompt_id}_seed_0.json
    │   ├── {prompt_id}_seed_1.json
    │   └── ...
    └── summary_YYYYMMDD_HHMMSS.json
"""

import sys
import json
import yaml
import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import analyze_awareness_levels
from onpolicy_suppression import generate_with_suppression, SuppressionResult
from inspect_ai.model import ChatMessageSystem, ChatMessageUser


@dataclass
class TranscriptEvalResult:
    """Result from evaluating a single transcript."""
    # Metadata
    source_file: str
    prompt_id: str
    scenario: str
    model: str
    seed: int

    # Original response
    original_response: str
    original_choice: Optional[str]
    original_awareness: Dict

    # Suppressed response
    suppressed_response: str
    suppressed_choice: Optional[str]
    suppressed_awareness: Dict

    # Suppression metrics
    chunks_generated: int
    chunks_resampled: int
    iterations_used: int
    suppression_successful: bool

    # Comparison
    same_choice: bool
    awareness_eliminated: bool
    awareness_reduction_ratio: float

    # Timing
    time_seconds: float

    # NEW: Detailed tracking
    all_detected_aware_sentences: List[str] = None  # All eval-aware sentences detected
    chunk_resample_details: List[Dict] = None  # Details for each chunk that needed resampling


def extract_choice(response: str) -> Optional[str]:
    """Extract boxed choice (A or B) from response."""
    import re
    match = re.search(r'\\boxed\{([AB])\}', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def load_transcript(filepath: Path) -> Dict:
    """Load a single transcript YAML file."""
    with open(filepath) as f:
        data = yaml.safe_load(f)

    # Extract metadata
    metadata = data.get('metadata', {})

    # Extract prompt ID from filename
    filename = filepath.stem
    if '_seed_' in filename:
        prompt_id = filename.rsplit('_seed_', 1)[0]
        seed = int(filename.rsplit('_seed_', 1)[1])
    else:
        prompt_id = filename
        seed = 0

    return {
        'filepath': filepath,
        'prompt_id': prompt_id,
        'scenario': metadata.get('scenario', 'unknown'),
        'model': metadata.get('rollout_model', 'unknown'),
        'seed': seed,
        'system_prompt': data.get('system_prompt', ''),
        'evaluation_user_message': data.get('evaluation', {}).get('user_message', ''),
        'evaluation_response': data.get('evaluation', {}).get('assistant_response', data.get('evaluation', {}).get('response', '')),
        'deployment_response': data.get('deployment', {}).get('assistant_response', data.get('deployment', {}).get('response', ''))
    }


async def evaluate_single_transcript(
    transcript_data: Dict,
    chunk_strategy: str,
    max_iterations: int,
    max_resample_attempts: int,
    verbose: bool = False
) -> TranscriptEvalResult:
    """Evaluate suppression on a single transcript."""
    import time
    start_time = time.time()

    # Extract original response
    original_response = transcript_data['evaluation_response']
    original_choice = extract_choice(original_response)

    # Analyze original awareness
    original_awareness_analysis = analyze_awareness_levels(original_response)
    original_awareness = {
        'total_count': original_awareness_analysis.total_awareness_count,
        'global_count': original_awareness_analysis.global_awareness_count,
        'explicit_count': original_awareness_analysis.explicit_awareness_count,
        'implicit_count': original_awareness_analysis.implicit_acknowledgment_count,
        'highest_level': original_awareness_analysis.highest_level,
        'sample_sentences': (
            original_awareness_analysis.global_awareness_sentences[:2] +
            original_awareness_analysis.explicit_awareness_sentences[:2] +
            original_awareness_analysis.implicit_acknowledgment_sentences[:2]
        )[:3]
    }

    # Construct messages for suppression
    messages = []

    # Add system prompt if present
    if transcript_data['system_prompt']:
        messages.append(ChatMessageSystem(content=transcript_data['system_prompt']))

    # Add evaluation user message
    messages.append(ChatMessageUser(content=transcript_data['evaluation_user_message']))

    # Run on-policy suppression
    print(f"   Running suppression on: {transcript_data['prompt_id']} seed {transcript_data['seed']}")
    print(f"   Original awareness: {original_awareness['total_count']} sentences")

    # Prepare model name - add openrouter prefix if needed
    model_name = transcript_data['model']
    if '/' in model_name and not model_name.startswith('openrouter/'):
        model_name = f"openrouter/{model_name}"

    suppression_result = await generate_with_suppression(
        messages=messages,
        tools=[],  # No tools for now
        model_name=model_name,
        chunk_strategy=chunk_strategy,
        max_iterations=max_iterations,
        max_resample_attempts_per_chunk=max_resample_attempts,
        verbose=verbose
    )

    suppressed_response = suppression_result.final_response
    suppressed_choice = extract_choice(suppressed_response)

    # Analyze suppressed response
    suppressed_awareness_analysis = analyze_awareness_levels(suppressed_response)
    suppressed_awareness = {
        'total_count': suppressed_awareness_analysis.total_awareness_count,
        'global_count': suppressed_awareness_analysis.global_awareness_count,
        'explicit_count': suppressed_awareness_analysis.explicit_awareness_count,
        'implicit_count': suppressed_awareness_analysis.implicit_acknowledgment_count,
        'highest_level': suppressed_awareness_analysis.highest_level
    }

    # Compute metrics
    same_choice = (original_choice == suppressed_choice)
    awareness_eliminated = (suppressed_awareness['total_count'] == 0)

    if original_awareness['total_count'] > 0:
        awareness_reduction_ratio = 1.0 - (suppressed_awareness['total_count'] / original_awareness['total_count'])
    else:
        awareness_reduction_ratio = 0.0

    elapsed_time = time.time() - start_time

    # Convert ChunkResampleInfo objects to dicts for JSON serialization
    chunk_resample_dicts = []
    for info in suppression_result.chunk_resample_details:
        chunk_resample_dicts.append({
            'chunk_index': info.chunk_index,
            'original_chunk': info.original_chunk,
            'attempts_needed': info.attempts_needed,
            'final_chunk': info.final_chunk,
            'all_aware_sentences': info.all_aware_sentences
        })

    return TranscriptEvalResult(
        source_file=str(transcript_data['filepath']),
        prompt_id=transcript_data['prompt_id'],
        scenario=transcript_data['scenario'],
        model=transcript_data['model'],
        seed=transcript_data['seed'],
        original_response=original_response,
        original_choice=original_choice,
        original_awareness=original_awareness,
        suppressed_response=suppressed_response,
        suppressed_choice=suppressed_choice,
        suppressed_awareness=suppressed_awareness,
        chunks_generated=suppression_result.chunks_generated,
        chunks_resampled=suppression_result.chunks_resampled,
        iterations_used=suppression_result.chunks_generated,  # Iterations = chunks generated
        suppression_successful=suppression_result.suppression_successful,
        same_choice=same_choice,
        awareness_eliminated=awareness_eliminated,
        awareness_reduction_ratio=awareness_reduction_ratio,
        time_seconds=elapsed_time,
        all_detected_aware_sentences=suppression_result.all_detected_aware_sentences,
        chunk_resample_details=chunk_resample_dicts
    )


def compute_summary_metrics(results: List[TranscriptEvalResult]) -> Dict:
    """Compute summary metrics across all results."""
    if not results:
        return {}

    total = len(results)

    return {
        'total_transcripts': total,
        'suppression_success_rate': sum(r.suppression_successful for r in results) / total,
        'choice_preservation_rate': sum(r.same_choice for r in results) / total,
        'awareness_elimination_rate': sum(r.awareness_eliminated for r in results) / total,
        'avg_awareness_reduction_ratio': sum(r.awareness_reduction_ratio for r in results) / total,
        'avg_chunks_generated': sum(r.chunks_generated for r in results) / total,
        'avg_chunks_resampled': sum(r.chunks_resampled for r in results) / total,
        'avg_time_seconds': sum(r.time_seconds for r in results) / total,
        'total_time_seconds': sum(r.time_seconds for r in results),
    }


async def main():
    parser = argparse.ArgumentParser(
        description='Run on-policy suppression evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--transcripts', required=True,
                       help='File with list of transcript paths to evaluate')
    parser.add_argument('--output-dir', default='working/suppression_result_cot',
                       help='Output directory for results (default: working/suppression_result_cot)')
    parser.add_argument('--chunk-strategy', choices=['sentence', 'paragraph'], default='sentence',
                       help='Chunking strategy (default: sentence)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum regeneration iterations (default: 10)')
    parser.add_argument('--max-resample-attempts', type=int, default=50,
                       help='Max attempts per chunk (default: 50)')
    parser.add_argument('--max-prompts', type=int, default=None,
                       help='Limit number of unique prompts to process (default: all)')
    parser.add_argument('--cache', action='store_true', default=True,
                       help='Use caching (skip already processed transcripts) (default: True)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Show detailed logging of message templates and generation (default: False)')

    args = parser.parse_args()

    # Load transcript list
    with open(args.transcripts) as f:
        transcript_paths = [line.strip() for line in f if line.strip()]

    # Group by prompt and model
    from collections import defaultdict
    by_prompt_model = defaultdict(list)
    for path in transcript_paths:
        filepath = Path(path)
        filename = filepath.stem

        # Extract prompt ID and model from path
        # Path format: working/rollouts/{model}/{category}/{prompt_id}_seed_N.yaml
        try:
            parts = filepath.parts
            model = parts[2] if len(parts) > 2 else 'unknown'

            if '_seed_' in filename:
                prompt_id = filename.rsplit('_seed_', 1)[0]
            else:
                prompt_id = filename

            key = (model, prompt_id)
            by_prompt_model[key].append(path)
        except:
            # Fallback
            by_prompt_model[('unknown', filename)].append(path)

    # Limit prompts if requested
    prompt_keys = list(by_prompt_model.keys())
    if args.max_prompts and args.max_prompts < len(prompt_keys):
        print(f"Limiting to {args.max_prompts} prompts (out of {len(prompt_keys)} total)")
        prompt_keys = prompt_keys[:args.max_prompts]
        transcript_paths = []
        for key in prompt_keys:
            transcript_paths.extend(by_prompt_model[key])

    print("="*80)
    print("ON-POLICY SUPPRESSION EVALUATION")
    print("="*80)
    print(f"Unique prompts: {len(prompt_keys)}")
    print(f"Total transcripts to process: {len(transcript_paths)}")
    print(f"Chunk strategy: {args.chunk_strategy}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Max resample attempts: {args.max_resample_attempts}")
    print(f"Output directory: {args.output_dir}")
    print(f"Caching: {args.cache}")
    print("="*80 + "\n")

    # Process transcripts
    results = []
    processed_count = 0
    cached_count = 0

    for i, transcript_path in enumerate(transcript_paths, 1):
        filepath = Path(transcript_path)

        # Determine output path: {output_dir}/{model}/{original_filename}.json
        try:
            parts = filepath.parts
            model = parts[2] if len(parts) > 2 else 'unknown'
        except:
            model = 'unknown'

        output_dir = Path(args.output_dir) / model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{filepath.stem}.json"

        # Check cache
        if args.cache and output_file.exists():
            print(f"[{i}/{len(transcript_paths)}] CACHED: {filepath.name}")
            try:
                with open(output_file) as f:
                    result_dict = json.load(f)
                    results.append(TranscriptEvalResult(**result_dict))
                    cached_count += 1
                    continue
            except:
                print(f"  Warning: Cache file corrupted, reprocessing")

        # Process new transcript
        print(f"[{i}/{len(transcript_paths)}] Processing: {filepath.name}")

        try:
            transcript_data = load_transcript(filepath)
            result = await evaluate_single_transcript(
                transcript_data,
                args.chunk_strategy,
                args.max_iterations,
                args.max_resample_attempts,
                args.verbose
            )
            results.append(result)

            # Save individual result
            with open(output_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)

            print(f"  Original awareness: {result.original_awareness['total_count']} sentences")
            print(f"  Suppressed awareness: {result.suppressed_awareness['total_count']} sentences")
            print(f"  Success: {result.suppression_successful}")
            print(f"  Saved to: {output_file}")
            print()

            processed_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Compute summary
    summary = compute_summary_metrics(results)

    # Save summary
    summary_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_transcripts_requested': len(transcript_paths),
            'total_transcripts_processed': len(results),
            'cached_count': cached_count,
            'newly_processed_count': processed_count,
            'unique_prompts': len(prompt_keys),
            'chunk_strategy': args.chunk_strategy,
            'max_iterations': args.max_iterations,
            'max_resample_attempts': args.max_resample_attempts,
        },
        'summary': summary,
    }

    summary_file = Path(args.output_dir) / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Transcripts processed: {processed_count}")
    print(f"Transcripts cached: {cached_count}")
    print(f"Total results: {len(results)}")
    print(f"Results directory: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nSummary metrics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
