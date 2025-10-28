#!/usr/bin/env python3
"""
Analyze suppression experiment results.

Processes YAML files from suppression experiments and generates statistics on:
- Seed extraction effectiveness (awareness removed from initial reasoning)
- Deployment branch suppression (original vs suppressed awareness)
- Evaluation branch suppression (original vs suppressed awareness)
- Behavioral changes (choice flips between original and suppressed)
- Suppression success metrics (chunks resampled, awareness detected)

Usage:
    python analyze_suppression_results.py --experiment-dir working/suppression_experiments/01_chunk_resample/llm_sentence_thresh_5/qwen_qwen3-32b
    python analyze_suppression_results.py --experiment-dir working/suppression_experiments --recursive
"""

import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import re


def extract_choice(response: str) -> str:
    """Extract boxed choice from response (A or B)."""
    match = re.search(r'\\boxed\{([AB])\}', response)
    return match.group(1) if match else None


def analyze_suppression_file(file_path: Path) -> Dict:
    """Analyze a single suppression experiment file."""
    # Use safe_load with custom constructor for tuples (legacy format)
    yaml.SafeLoader.add_constructor(
        'tag:yaml.org,2002:python/tuple',
        lambda loader, node: list(loader.construct_sequence(node))
    )

    with open(file_path) as f:
        data = yaml.safe_load(f)

    result = {
        'file': file_path.name,
        'source_file': data.get('source_file', ''),
        'timestamp': data.get('timestamp', ''),
    }

    # Seed extraction stats
    seed_stats = data.get('experiment_config', {}).get('seed_extraction', {})
    result['seed'] = {
        'total_sentences': seed_stats.get('total_sentences', 0),
        'clean_sentences': seed_stats.get('clean_sentences', 0),
        'removed_sentences': seed_stats.get('removed_sentences', 0),
        'removed_pct': (seed_stats.get('removed_sentences', 0) / seed_stats.get('total_sentences', 1)) * 100 if seed_stats.get('total_sentences', 0) > 0 else 0,
        'seed_length': seed_stats.get('seed_length', 0),
    }

    # Deployment branch (no suppression - just control)
    deployment = data.get('deployment', {})

    result['deployment'] = {
        'awareness': deployment.get('llm_judge_score', {}).get('awareness_intensity', 0),
        'awareness_type': deployment.get('llm_judge_score', {}).get('awareness_type', 'unknown'),
        'choice': extract_choice(deployment.get('response', '')),
    }

    # Evaluation branch
    evaluation = data.get('evaluation', {})
    eval_orig = evaluation.get('original', {})
    eval_supp = evaluation.get('suppressed', {})

    result['evaluation'] = {
        'original_awareness': eval_orig.get('llm_judge_score', {}).get('awareness_intensity', 0),
        'original_type': eval_orig.get('llm_judge_score', {}).get('awareness_type', 'unknown'),
        'suppressed_awareness': eval_supp.get('llm_judge_score', {}).get('awareness_intensity', 0),
        'suppressed_type': eval_supp.get('llm_judge_score', {}).get('awareness_type', 'unknown'),
        'awareness_reduction': eval_orig.get('llm_judge_score', {}).get('awareness_intensity', 0) - eval_supp.get('llm_judge_score', {}).get('awareness_intensity', 0),
        'original_choice': extract_choice(eval_orig.get('response', '')),
        'suppressed_choice': extract_choice(eval_supp.get('response', '')),
        'suppression_metrics': eval_supp.get('suppression_metrics', {}),
    }

    # Behavioral change (only evaluation has suppression)
    result['behavioral_change'] = {
        'evaluation_flipped': result['evaluation']['original_choice'] != result['evaluation']['suppressed_choice'] if result['evaluation']['original_choice'] and result['evaluation']['suppressed_choice'] else False,
    }

    return result


def print_summary(results: List[Dict], bucket_data: Optional[Dict] = None):
    """Print summary statistics."""
    if not results:
        print("No results to analyze")
        return

    print(f"\n{'='*80}")
    print(f"SUPPRESSION EXPERIMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Total files analyzed: {len(results)}\n")

    # Seed extraction stats
    print(f"{'='*80}")
    print(f"SEED EXTRACTION (Initial Reasoning Cleanup)")
    print(f"{'='*80}")
    avg_removed = sum(r['seed']['removed_sentences'] for r in results) / len(results)
    avg_removed_pct = sum(r['seed']['removed_pct'] for r in results) / len(results)
    print(f"Average sentences removed: {avg_removed:.1f} ({avg_removed_pct:.1f}%)")
    print(f"Files with awareness in seed: {sum(1 for r in results if r['seed']['removed_sentences'] > 0)} / {len(results)}")
    print()

    # Deployment branch stats (control - no suppression)
    print(f"{'='*80}")
    print(f"DEPLOYMENT BRANCH (Control - No Suppression)")
    print(f"{'='*80}")
    deploy_avg = sum(r['deployment']['awareness'] for r in results) / len(results)
    print(f"Awareness (avg): {deploy_avg:.2f}")
    print(f"Note: Deployment branch is kept as-is (no eval context to suppress)")
    print()

    # Evaluation branch stats
    print(f"{'='*80}")
    print(f"EVALUATION BRANCH")
    print(f"{'='*80}")
    eval_orig_avg = sum(r['evaluation']['original_awareness'] for r in results) / len(results)
    eval_supp_avg = sum(r['evaluation']['suppressed_awareness'] for r in results) / len(results)
    eval_reduction = eval_orig_avg - eval_supp_avg

    print(f"Original awareness (avg):    {eval_orig_avg:.2f}")
    print(f"Suppressed awareness (avg):  {eval_supp_avg:.2f}")
    print(f"Reduction:                   {eval_reduction:.2f} ({eval_reduction/eval_orig_avg*100:.1f}% decrease)" if eval_orig_avg > 0 else "Reduction: N/A")

    eval_flips = sum(1 for r in results if r['behavioral_change']['evaluation_flipped'])
    print(f"Behavioral changes (choice flips): {eval_flips} / {len(results)} ({eval_flips/len(results)*100:.1f}%)")

    # Awareness type distribution
    print(f"\nOriginal awareness type distribution:")
    orig_type_counts = defaultdict(int)
    for r in results:
        orig_type_counts[r['evaluation']['original_type']] += 1
    for atype, count in sorted(orig_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(results)) * 100
        print(f"  {atype}: {count} ({pct:.1f}%)")

    print(f"\nSuppressed awareness type distribution:")
    supp_type_counts = defaultdict(int)
    for r in results:
        supp_type_counts[r['evaluation']['suppressed_type']] += 1
    for atype, count in sorted(supp_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(results)) * 100
        print(f"  {atype}: {count} ({pct:.1f}%)")

    # Choice distribution
    eval_orig_a = sum(1 for r in results if r['evaluation']['original_choice'] == 'A')
    eval_orig_b = sum(1 for r in results if r['evaluation']['original_choice'] == 'B')
    eval_orig_none = sum(1 for r in results if not r['evaluation']['original_choice'])

    eval_supp_a = sum(1 for r in results if r['evaluation']['suppressed_choice'] == 'A')
    eval_supp_b = sum(1 for r in results if r['evaluation']['suppressed_choice'] == 'B')
    eval_supp_none = sum(1 for r in results if not r['evaluation']['suppressed_choice'])

    print(f"\nChoice Distribution:")
    print(f"  Original:   A={eval_orig_a}, B={eval_orig_b}, None={eval_orig_none}")
    print(f"  Suppressed: A={eval_supp_a}, B={eval_supp_b}, None={eval_supp_none}")

    # Show which direction flips went
    if eval_flips > 0:
        a_to_b = sum(1 for r in results if r['evaluation']['original_choice'] == 'A' and r['evaluation']['suppressed_choice'] == 'B')
        b_to_a = sum(1 for r in results if r['evaluation']['original_choice'] == 'B' and r['evaluation']['suppressed_choice'] == 'A')
        print(f"  Flips: A→B={a_to_b}, B→A={b_to_a}")
    print()

    # Suppression metrics
    print(f"{'='*80}")
    print(f"SUPPRESSION PROCESS METRICS")
    print(f"{'='*80}")

    # Average across evaluation branch only (deployment not suppressed)
    all_metrics = [r['evaluation']['suppression_metrics'] for r in results]
    all_metrics = [m for m in all_metrics if m]  # Filter out empty dicts

    if all_metrics:
        avg_chunks = sum(m.get('chunks_generated', 0) for m in all_metrics) / len(all_metrics)
        avg_resampled = sum(m.get('chunks_resampled', 0) for m in all_metrics) / len(all_metrics)
        avg_detected = sum(m.get('total_awareness_detected', 0) for m in all_metrics) / len(all_metrics)
        successful = sum(1 for m in all_metrics if m.get('suppression_successful', False))

        print(f"Average chunks generated:     {avg_chunks:.1f}")
        print(f"Average chunks resampled:     {avg_resampled:.1f}")
        print(f"Average awareness detected:   {avg_detected:.1f}")
        print(f"Suppression successful:       {successful} / {len(all_metrics)} ({successful/len(all_metrics)*100:.1f}%)")
    else:
        print("No suppression metrics available")

    # Per-bucket breakdown
    if bucket_data:
        print(f"{'='*80}")
        print(f"BY AWARENESS BUCKET (from bucket file)")
        print(f"{'='*80}")

        # Extract all prompts from bucket data and map to bucket category
        prompt_to_bucket = {}
        buckets = bucket_data.get('eval_awareness_buckets', {})
        for bucket_name, bucket_prompts in buckets.items():
            for prompt_name in bucket_prompts.keys():
                prompt_to_bucket[prompt_name] = bucket_name

        # Group results by bucket
        bucket_groups = defaultdict(list)
        for r in results:
            # Extract prompt name from source_file
            # e.g., working/rollouts/model/behavioral_change/explicit/hallucination_2025-10-23_15-15-14_1b2eebd9_seed_0.yaml
            source_file = r.get('source_file', '')
            if source_file:
                # Extract base name without seed
                base_name = Path(source_file).stem  # hallucination_2025-10-23_15-15-14_1b2eebd9_seed_0
                base_prompt = re.sub(r'_seed_\d+$', '', base_name)  # hallucination_2025-10-23_15-15-14_1b2eebd9

                # Find which bucket this prompt belongs to
                if base_prompt in prompt_to_bucket:
                    bucket_name = prompt_to_bucket[base_prompt]
                    bucket_groups[bucket_name].append(r)

        # Sort buckets by their percentile range
        bucket_order = ['10-25%', '25-50%', '50-75%', '75-100%']
        for bucket_name in bucket_order:
            if bucket_name in bucket_groups:
                group = bucket_groups[bucket_name]
                eval_orig_avg = sum(r['evaluation']['original_awareness'] for r in group) / len(group)
                eval_supp_avg = sum(r['evaluation']['suppressed_awareness'] for r in group) / len(group)
                reduction = eval_orig_avg - eval_supp_avg
                reduction_pct = (reduction / eval_orig_avg * 100) if eval_orig_avg > 0 else 0
                print(f"{bucket_name}: {len(group)} files, original={eval_orig_avg:.2f} → suppressed={eval_supp_avg:.2f} (reduction={reduction:.2f}, {reduction_pct:.1f}%)")

    print()
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Analyze suppression experiment results')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Directory containing suppression experiment outputs')
    parser.add_argument('--bucket-file', type=str,
                       help='Bucket file for per-prompt breakdown (e.g., working/response_categorization/model/high_awareness_bc_seeds.yaml)')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search subdirectories')
    parser.add_argument('--output-csv', type=str,
                       help='Save detailed results to CSV file')

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: {experiment_dir} not found")
        return 1

    # Find all YAML files
    if args.recursive:
        files = list(experiment_dir.rglob('*.yaml'))
    else:
        files = list(experiment_dir.glob('*.yaml'))

    if not files:
        print(f"No YAML files found in {experiment_dir}")
        return 1

    print(f"Found {len(files)} suppression experiment files")

    # Analyze each file
    results = []
    for file_path in files:
        try:
            result = analyze_suppression_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Load bucket file if provided
    bucket_data = None
    if args.bucket_file:
        bucket_path = Path(args.bucket_file)
        if bucket_path.exists():
            with open(bucket_path) as f:
                bucket_data = yaml.safe_load(f)
        else:
            print(f"Warning: Bucket file not found: {bucket_path}")

    # Print summary
    print_summary(results, bucket_data)

    # Save to CSV if requested
    if args.output_csv:
        import csv
        output_path = Path(args.output_csv)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file', 'seed_removed_sentences', 'seed_removed_pct',
                'deploy_awareness', 'deploy_choice',
                'eval_orig_awareness', 'eval_supp_awareness', 'eval_reduction',
                'eval_orig_choice', 'eval_supp_choice', 'eval_flipped',
            ])
            writer.writeheader()

            for r in results:
                writer.writerow({
                    'file': r['file'],
                    'seed_removed_sentences': r['seed']['removed_sentences'],
                    'seed_removed_pct': f"{r['seed']['removed_pct']:.1f}",
                    'deploy_awareness': r['deployment']['awareness'],
                    'deploy_choice': r['deployment']['choice'],
                    'eval_orig_awareness': r['evaluation']['original_awareness'],
                    'eval_supp_awareness': r['evaluation']['suppressed_awareness'],
                    'eval_reduction': r['evaluation']['awareness_reduction'],
                    'eval_orig_choice': r['evaluation']['original_choice'],
                    'eval_supp_choice': r['evaluation']['suppressed_choice'],
                    'eval_flipped': r['behavioral_change']['evaluation_flipped'],
                })

        print(f"Detailed results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
