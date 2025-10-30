#!/usr/bin/env python3
"""
Analyze prompt judgment statistics across all judge models.

Reads categorized prompt responses and generates statistical summaries.

Usage:
    python analyze_prompt_judgments.py
    python analyze_prompt_judgments.py --plot-mode  # Generate comparative plots
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import config


def load_judgment_files(model_dir: Path) -> List[Dict]:
    """Load all judgment YAML files from a model directory.
    
    Args:
        model_dir: Path to judge model directory (e.g., categorize_prompts/qwen_qwen3-32b/)
        
    Returns:
        List of judgment data dicts
    """
    judgments = []
    
    # Find all *_cat_response.yaml files
    for yaml_file in model_dir.glob('*_cat_response.yaml'):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            
            if data and 'judgment' in data:
                judgments.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file.name}: {e}")
            continue
    
    return judgments


def calculate_statistics(judgments: List[Dict]) -> Dict:
    """Calculate comprehensive statistics from judgment data.
    
    Args:
        judgments: List of judgment data dicts
        
    Returns:
        Dict with statistics
    """
    if not judgments:
        return {}
    
    # Extract scores
    scenario_realism = []
    cue_integration = []
    cue_constraint = []
    num_cues = []
    
    # Track distributions
    bucket_counts = defaultdict(int)
    model_counts = defaultdict(int)
    source_scenarios = defaultdict(int)  # e.g., "hallucination", "credit_attribution"
    
    # Track scores by bucket
    bucket_scores = defaultdict(lambda: {
        'scenario_realism': [],
        'cue_integration_quality': [],
        'cue_constraint_adherence': [],
        'number_of_cues': []
    })
    
    for item in judgments:
        eval_data = item.get('judgment', {}).get('evaluation', {})
        bucket = item.get('bucket', 'unknown')
        
        # Collect overall scores
        if 'scenario_realism' in eval_data:
            score = eval_data['scenario_realism']
            scenario_realism.append(score)
            bucket_scores[bucket]['scenario_realism'].append(score)
        if 'cue_integration_quality' in eval_data:
            score = eval_data['cue_integration_quality']
            cue_integration.append(score)
            bucket_scores[bucket]['cue_integration_quality'].append(score)
        if 'cue_constraint_adherence' in eval_data:
            score = eval_data['cue_constraint_adherence']
            cue_constraint.append(score)
            bucket_scores[bucket]['cue_constraint_adherence'].append(score)
        if 'number_of_cues' in eval_data:
            score = eval_data['number_of_cues']
            num_cues.append(score)
            bucket_scores[bucket]['number_of_cues'].append(score)
        
        # Track distributions
        bucket_counts[bucket] += 1
        
        # Count models
        for model in item.get('models', []):
            model_counts[model] += 1
        
        # Extract scenario type from source_file
        source_file = item.get('source_file', '')
        if source_file:
            # e.g., "behavioral_change/explicit/hallucination_2025-10-28_05-08-48.yaml"
            filename = Path(source_file).name
            scenario = filename.split('_')[0]  # Get first part before underscore
            source_scenarios[scenario] += 1
    
    # Helper function to compute stats for a list
    def compute_score_stats(scores: List[float], name: str) -> Dict:
        if not scores:
            return {}
        
        return {
            'count': len(scores),
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'p25': float(np.percentile(scores, 25)),
            'p75': float(np.percentile(scores, 75)),
            'p95': float(np.percentile(scores, 95)),
            'distribution': {
                '1-2': sum(1 for s in scores if 1 <= s <= 2),
                '3-4': sum(1 for s in scores if 3 <= s <= 4),
                '5-6': sum(1 for s in scores if 5 <= s <= 6),
                '7-8': sum(1 for s in scores if 7 <= s <= 8),
                '9-10': sum(1 for s in scores if 9 <= s <= 10)
            }
        }
    
    # Calculate bucket-level statistics
    bucket_stats = {}
    for bucket, scores_dict in bucket_scores.items():
        bucket_stats[bucket] = {
            'count': bucket_counts[bucket],
            'scores': {
                'scenario_realism': compute_score_stats(scores_dict['scenario_realism'], 'Scenario Realism'),
                'cue_integration_quality': compute_score_stats(scores_dict['cue_integration_quality'], 'Cue Integration Quality'),
                'cue_constraint_adherence': compute_score_stats(scores_dict['cue_constraint_adherence'], 'Cue Constraint Adherence'),
                'number_of_cues': compute_score_stats(scores_dict['number_of_cues'], 'Number of Cues')
            }
        }
    
    # Compile statistics
    stats = {
        'total_prompts': len(judgments),
        'scores': {
            'scenario_realism': compute_score_stats(scenario_realism, 'Scenario Realism'),
            'cue_integration_quality': compute_score_stats(cue_integration, 'Cue Integration Quality'),
            'cue_constraint_adherence': compute_score_stats(cue_constraint, 'Cue Constraint Adherence'),
            'number_of_cues': compute_score_stats(num_cues, 'Number of Cues')
        },
        'distributions': {
            'by_bucket': dict(bucket_counts),
            'by_model': dict(model_counts),
            'by_scenario': dict(sorted(source_scenarios.items(), key=lambda x: x[1], reverse=True))
        },
        'by_bucket': bucket_stats
    }
    
    return stats


def generate_summary_report(stats: Dict) -> str:
    """Generate a human-readable summary report.
    
    Args:
        stats: Statistics dict
        
    Returns:
        Formatted string report
    """
    if not stats:
        return "No data available."
    
    lines = []
    lines.append("=" * 80)
    lines.append("PROMPT JUDGMENT STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Total Prompts: {stats['total_prompts']}")
    lines.append("")
    
    # Score summaries
    lines.append("Score Summaries (1-10 Scale):")
    lines.append("-" * 80)
    
    for metric, data in stats['scores'].items():
        if not data:
            continue
        
        metric_name = metric.replace('_', ' ').title()
        lines.append(f"\n{metric_name}:")
        lines.append(f"  Mean: {data['mean']:.2f} | Median: {data['median']:.1f} | Std: {data['std']:.2f}")
        lines.append(f"  Range: {data['min']} - {data['max']} | P25: {data['p25']:.1f} | P75: {data['p75']:.1f} | P95: {data['p95']:.1f}")
        
        # Distribution
        dist = data['distribution']
        lines.append(f"  Distribution: 1-2: {dist['1-2']}, 3-4: {dist['3-4']}, 5-6: {dist['5-6']}, 7-8: {dist['7-8']}, 9-10: {dist['9-10']}")
    
    lines.append("")
    lines.append("-" * 80)
    
    # Bucket distribution
    if stats['distributions']['by_bucket']:
        lines.append("\nPrompts by Bucket:")
        for bucket, count in sorted(stats['distributions']['by_bucket'].items()):
            lines.append(f"  {bucket}: {count}")
    
    # Model distribution
    if stats['distributions']['by_model']:
        lines.append("\nPrompts by Model:")
        for model, count in sorted(stats['distributions']['by_model'].items()):
            lines.append(f"  {model}: {count}")
    
    # Scenario distribution
    if stats['distributions']['by_scenario']:
        lines.append("\nPrompts by Scenario (Top 10):")
        for scenario, count in list(stats['distributions']['by_scenario'].items())[:10]:
            lines.append(f"  {scenario}: {count}")
    
    # Bucket-level statistics
    if 'by_bucket' in stats and stats['by_bucket']:
        lines.append("\n" + "-" * 80)
        lines.append("\nBucket-Level Statistics:")
        lines.append("-" * 80)
        
        for bucket in sorted(stats['by_bucket'].keys()):
            bucket_data = stats['by_bucket'][bucket]
            lines.append(f"\n{bucket} (n={bucket_data['count']}):")
            
            for metric, data in bucket_data['scores'].items():
                if not data:
                    continue
                
                metric_name = metric.replace('_', ' ').title()
                lines.append(f"  {metric_name}: Mean={data['mean']:.2f}, Median={data['median']:.1f}, Std={data['std']:.2f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def load_all_statistics(categorize_prompts_dir: Path) -> Dict[str, Dict]:
    """Load pre-computed statistics from all model directories.
    
    Args:
        categorize_prompts_dir: Path to categorize_prompts directory
        
    Returns:
        Dict mapping model_name -> statistics dict
    """
    all_stats = {}
    
    # Find all model directories
    model_dirs = [d for d in categorize_prompts_dir.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        stats_file = model_dir / 'summary' / 'statistics.json'
        
        if not stats_file.exists():
            print(f"Warning: {stats_file} not found, skipping {model_dir.name}")
            continue
        
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            all_stats[model_dir.name] = stats
        except Exception as e:
            print(f"Warning: Failed to load {stats_file}: {e}")
            continue
    
    return all_stats


def create_comparison_plots(all_stats: Dict[str, Dict], output_dir: Path):
    """Create comprehensive comparison plots from statistics.
    
    Args:
        all_stats: Dict mapping model_name -> statistics
        output_dir: Directory to save plots
    """
    if not all_stats:
        print("No statistics to plot!")
        return
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model names
    models = list(all_stats.keys())
    
    # Metrics to plot
    metrics = ['scenario_realism', 'cue_integration_quality', 'cue_constraint_adherence', 'number_of_cues']
    metric_labels = {
        'scenario_realism': 'Scenario Realism',
        'cue_integration_quality': 'Cue Integration Quality',
        'cue_constraint_adherence': 'Cue Constraint Adherence',
        'number_of_cues': 'Number of Cues'
    }
    
    # 1. Overall Comparison Plot (all metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Judge Model Comparison - Overall Scores', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data
        means = []
        medians = []
        stds = []
        model_labels = []
        
        for model in models:
            metric_data = all_stats[model].get('scores', {}).get(metric, {})
            if metric_data:
                means.append(metric_data.get('mean', 0))
                medians.append(metric_data.get('median', 0))
                stds.append(metric_data.get('std', 0))
                model_labels.append(model.replace('_', '\n', 1))  # Break long names
        
        if not means:
            continue
        
        x = np.arange(len(model_labels))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, means, width, label='Mean', alpha=0.8)
        bars2 = ax.bar(x + width/2, medians, width, label='Median', alpha=0.8)
        
        # Add error bars for std
        ax.errorbar(x - width/2, means, yerr=stds, fmt='none', color='black', alpha=0.3, capsize=5)
        
        ax.set_xlabel('Judge Model', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(metric_labels[metric], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'overall_comparison.png'}")
    plt.close()
    
    # 2. Bucket Comparison Plot
    # Get all unique buckets
    all_buckets = set()
    for stats in all_stats.values():
        all_buckets.update(stats.get('by_bucket', {}).keys())
    
    buckets = sorted(all_buckets)
    
    if buckets:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Judge Model Comparison - By Bucket', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data: bucket -> model -> mean
            bucket_data = defaultdict(dict)
            
            for model in models:
                model_label = model.replace('_', ' ')
                by_bucket = all_stats[model].get('by_bucket', {})
                
                for bucket in buckets:
                    if bucket in by_bucket:
                        metric_data = by_bucket[bucket].get('scores', {}).get(metric, {})
                        if metric_data:
                            bucket_data[bucket][model_label] = metric_data.get('mean', 0)
            
            # Plot grouped bars
            x = np.arange(len(buckets))
            width = 0.8 / len(models) if models else 0.35
            
            for i, model in enumerate(models):
                model_label = model.replace('_', ' ')
                values = [bucket_data[bucket].get(model_label, 0) for bucket in buckets]
                offset = (i - len(models)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model_label, alpha=0.8)
            
            ax.set_xlabel('Bucket', fontweight='bold')
            ax.set_ylabel('Mean Score', fontweight='bold')
            ax.set_title(metric_labels[metric], fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(buckets, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'bucket_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'bucket_comparison.png'}")
        plt.close()
    
    # 3. Distribution Comparison (Heatmap style)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for heatmap
        ranges = ['1-2', '3-4', '5-6', '7-8', '9-10']
        heatmap_data = []
        model_labels = []
        
        for model in models:
            metric_data = all_stats[model].get('scores', {}).get(metric, {})
            if metric_data and 'distribution' in metric_data:
                dist = metric_data['distribution']
                row = [dist.get(r, 0) for r in ranges]
                heatmap_data.append(row)
                model_labels.append(model.replace('_', '\n', 1))
        
        if heatmap_data:
            # Convert to percentages
            heatmap_data_pct = []
            for row in heatmap_data:
                total = sum(row)
                if total > 0:
                    heatmap_data_pct.append([x/total*100 for x in row])
                else:
                    heatmap_data_pct.append([0]*len(row))
            
            sns.heatmap(heatmap_data_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=ranges, yticklabels=model_labels,
                       cbar_kws={'label': 'Percentage (%)'}, ax=ax)
            
            ax.set_title(f'{metric_labels[metric]} - Distribution Comparison (%)', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Score Range', fontweight='bold')
            ax.set_ylabel('Judge Model', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'distribution_{metric}.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_dir / f'distribution_{metric}.png'}")
            plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze prompt judgment statistics')
    parser.add_argument('--plot-mode', action='store_true',
                       help='Generate comparative plots from pre-computed statistics')
    
    args = parser.parse_args()
    
    categorize_prompts_dir = config.WORKING_DIR / 'categorize_prompts'
    
    if not categorize_prompts_dir.exists():
        print(f"Error: {categorize_prompts_dir} not found")
        return 1
    
    # PLOT MODE - Generate comparative visualizations
    if args.plot_mode:
        print(f"{'='*80}")
        print(f"PLOT MODE - GENERATE COMPARATIVE VISUALIZATIONS")
        print(f"{'='*80}")
        print(f"Loading statistics from: {categorize_prompts_dir}")
        print()
        
        # Load all pre-computed statistics
        all_stats = load_all_statistics(categorize_prompts_dir)
        
        if not all_stats:
            print("No statistics found! Run without --plot-mode first to compute statistics.")
            return 1
        
        print(f"Loaded statistics for {len(all_stats)} judge models:")
        for model_name in all_stats.keys():
            print(f"  - {model_name}")
        print()
        
        # Create plots
        plot_dir = Path('evalawareness_techniques/plot_stats/prompt_stats')
        print("Generating comparison plots...")
        create_comparison_plots(all_stats, plot_dir)
        
        print()
        print(f"{'='*80}")
        print(f"PLOT MODE COMPLETE")
        print(f"{'='*80}")
        print(f"Plots saved to: {plot_dir}")
        print(f"{'='*80}")
        
        return 0
    
    # STANDARD MODE - Compute statistics from YAML files
    print(f"{'='*80}")
    print(f"ANALYZE PROMPT JUDGMENTS")
    print(f"{'='*80}")
    print(f"Directory: {categorize_prompts_dir}")
    print()
    
    # Find all judge model directories
    model_dirs = [d for d in categorize_prompts_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("No model directories found!")
        return 1
    
    print(f"Found {len(model_dirs)} judge model directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")
    print()
    
    # Process each model directory
    for model_dir in model_dirs:
        print(f"Processing {model_dir.name}...")
        
        # Load all judgment files
        judgments = load_judgment_files(model_dir)
        
        if not judgments:
            print(f"  No judgment files found. Skipping.")
            continue
        
        print(f"  Loaded {len(judgments)} judgments")
        
        # Calculate statistics
        stats = calculate_statistics(judgments)
        
        # Add metadata
        stats['judge_model'] = model_dir.name
        stats['analyzed_files'] = len(judgments)
        
        # Create summary subdirectory
        summary_dir = model_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        # Save overall statistics JSON in summary dir (includes bucket stats)
        stats_file = summary_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  Saved statistics to: {stats_file}")
        
        # Generate and save summary report in summary dir
        summary_report = generate_summary_report(stats)
        summary_file = summary_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        
        print(f"  Saved summary to: {summary_file}")
        print()
    
    print(f"{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(model_dirs)} judge model directories")
    print(f"Output: model/summary/ directory with:")
    print(f"  - statistics.json (overall stats + bucket-level breakdown)")
    print(f"  - summary.txt (human-readable report)")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    exit(main())

