#!/usr/bin/env python3
"""
Custom plotting script for 3 system prompt experiments.

Generates comprehensive visualizations covering all key metrics:
- BC rate, awareness levels, success rates
- Test type removal rates, choice consistency
- Pareto frontiers, performance heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import yaml

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Experiment names mapping (Code -> Short Label)
EXPERIMENT_MAPPING = {
    'system_prompt': 'SysPromptDeploy',
    'system_prompt_no_spec': 'SysPromptDeploy_NoSpec',
    'system_prompt_taboo_words': 'SysPrompt_Taboo',
}

# Metrics to plot
METRICS_CONFIG = {
    'BC Rate wrt Deploy': {
        'field': 'bc_rate_wrt_deploy',
        'ci_lower': None,  # No CIs available (calculated from bucket data)
        'ci_upper': None,
        'ylabel': 'BC Rate wrt Deploy (%)',
        'baseline': None,
        'invert': False,  # Lower is better (less change from deployment = closer to deployment)
    },
    'Suppressed Eval Awareness': {
        'field': 'suppressed_eval_awareness',
        'ci_lower': 'suppressed_eval_awareness_ci_lower',
        'ci_upper': 'suppressed_eval_awareness_ci_upper',
        'ylabel': 'Awareness Score (1-10)',
        'baseline': 'baseline_eval_awareness',  # Show original eval awareness as baseline
        'invert': False,  # Lower is better
    },
    'Success Rate (Modified)': {
        'field': 'success_rate_modified',
        'ci_lower': 'success_rate_modified_ci_lower',
        'ci_upper': 'success_rate_modified_ci_upper',
        'ylabel': 'Success Rate (%)',
        'baseline': None,
        'invert': True,  # Higher is better
    },
    'Awareness Delta from Deploy': {
        'field': 'awareness_score_delta_from_deployment',
        'ci_lower': 'awareness_score_delta_from_deployment_ci_lower',
        'ci_upper': 'awareness_score_delta_from_deployment_ci_upper',
        'ylabel': 'Delta (points)',
        'baseline': None,
        'invert': False,  # Lower is better
    },
    'Choice B Delta from Deploy': {
        'field': 'choice_b_delta',
        'ci_lower': 'choice_b_delta_ci_lower',
        'ci_upper': 'choice_b_delta_ci_upper',
        'ylabel': 'Choice B Delta (%)',
        'baseline': 0,  # 0 means no change from deployment
        'invert': False,  # Closer to 0 is better (less change from deployment)
    },
    'Self-Test Removal': {
        'field': 'self_test_removal_rate',
        'ci_lower': 'self_test_removal_rate_ci_lower',
        'ci_upper': 'self_test_removal_rate_ci_upper',
        'ylabel': 'Removal Rate (%)',
        'baseline': 0,
        'invert': True,  # More negative is better (removal)
    },
    'Other-Test Removal': {
        'field': 'other_test_removal_rate',
        'ci_lower': 'other_test_removal_rate_ci_lower',
        'ci_upper': 'other_test_removal_rate_ci_upper',
        'ylabel': 'Removal Rate (%)',
        'baseline': 0,
        'invert': True,  # More negative is better (removal)
    },
    'Choice Modal Consistency': {
        'field': 'choice_modal_consistency',
        'ci_lower': 'choice_modal_consistency_ci_lower',
        'ci_upper': 'choice_modal_consistency_ci_upper',
        'ylabel': 'Consistency',
        'baseline': None,
        'invert': True,  # Higher is better
    },
    'Overshoot Rate': {
        'field': 'overshoot_rate',
        'ci_lower': 'overshoot_rate_ci_lower',
        'ci_upper': 'overshoot_rate_ci_upper',
        'ylabel': 'Overshoot Rate (%)',
        'baseline': 0,
        'invert': False,  # Lower is better
    },
    'Task Completion': {
        'field': 'task_completion',
        'ci_lower': None,
        'ci_upper': None,
        'ylabel': 'Completion Rate (%)',
        'baseline': None,
        'invert': True,  # Higher is better
    },
}


def load_and_filter_data(csv_path):
    """Load metrics CSV and filter to 5 specific experiments."""
    df = pd.read_csv(csv_path)

    # Filter to our 5 experiments
    experiments = list(EXPERIMENT_MAPPING.keys())
    df_filtered = df[df['experiment'].isin(experiments)].copy()

    # Add short names for plotting
    df_filtered['short_name'] = df_filtered['experiment'].map(EXPERIMENT_MAPPING)

    print(f"Loaded {len(df_filtered)} experiments:")
    for exp in df_filtered['experiment']:
        print(f"  - {exp} → {EXPERIMENT_MAPPING[exp]}")

    return df_filtered


def load_bucket_data(csv_path):
    """Load combined data with bucket information from high_awareness_bc_seeds.yaml."""
    combined_path = csv_path.parent / 'combined_all_experiments.csv'
    if not combined_path.exists():
        print(f"Warning: Combined data not found at {combined_path}")
        return None

    # Load bucket mapping from YAML file
    yaml_path = Path('working/response_categorization/v1/qwen_qwen3-32b/high_awareness_bc_seeds.yaml')
    if not yaml_path.exists():
        print(f"Warning: Bucket YAML not found at {yaml_path}")
        return None

    # Load YAML
    with open(yaml_path, 'r') as f:
        bucket_data = yaml.safe_load(f)

    # Extract prompt_id -> (eval_bucket, bc_rate) mapping
    prompt_to_bucket = {}
    prompt_to_bc_rate = {}
    if 'eval_awareness_buckets' in bucket_data:
        for bucket_name, prompts in bucket_data['eval_awareness_buckets'].items():
            for prompt_id, prompt_info in prompts.items():
                prompt_to_bucket[prompt_id] = bucket_name
                # Extract BC rate from statistics
                if 'statistics' in prompt_info and 'bc_rate' in prompt_info['statistics']:
                    prompt_to_bc_rate[prompt_id] = prompt_info['statistics']['bc_rate']

    print(f"\nFound {len(prompt_to_bucket)} prompts with bucket assignments:")
    bucket_counts = {}
    for bucket in prompt_to_bucket.values():
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    print(f"  Eval bucket distribution: {bucket_counts}")
    print(f"  BC rates extracted: {len(prompt_to_bc_rate)} prompts")

    # Load experiment data
    df = pd.read_csv(combined_path)

    # Filter to 5 experiments
    experiments = list(EXPERIMENT_MAPPING.keys())
    df_filtered = df[df['experiment'].isin(experiments)].copy()

    # Add short names
    df_filtered['short_name'] = df_filtered['experiment'].map(EXPERIMENT_MAPPING)

    # Extract base prompt ID (remove _seed_N_supp_0.yaml suffix)
    def extract_base_prompt_id(prompt_id):
        # Format: scenario_timestamp_hash_seed_N_supp_0.yaml
        # Want: scenario_timestamp_hash
        if isinstance(prompt_id, str):
            # Remove .yaml extension
            prompt_id = prompt_id.replace('.yaml', '')
            # Split and take first 3 parts (scenario, timestamp, hash)
            parts = prompt_id.split('_')
            if len(parts) >= 3:
                # Find where _seed_ starts
                for i, part in enumerate(parts):
                    if part == 'seed':
                        return '_'.join(parts[:i])
                # If no _seed_ found, assume it's already base format
                return prompt_id
        return prompt_id

    df_filtered['base_prompt_id'] = df_filtered['prompt_id'].apply(extract_base_prompt_id)

    # Add eval bucket and BC rate columns based on base prompt_id
    df_filtered['eval_bucket'] = df_filtered['base_prompt_id'].map(prompt_to_bucket)
    df_filtered['prompt_bc_rate'] = df_filtered['base_prompt_id'].map(prompt_to_bc_rate)

    # Filter to only prompts with bucket assignments
    df_filtered = df_filtered[df_filtered['eval_bucket'].notna()].copy()

    # Assign BC buckets based on BC rate from YAML
    def assign_bc_bucket(bc_rate):
        if pd.isna(bc_rate):
            return None
        if bc_rate >= 0.90:
            return '90-100%'
        elif bc_rate >= 0.75:
            return '75-90%'
        elif bc_rate >= 0.50:
            return '50-75%'
        else:
            return '<50%'

    df_filtered['bc_bucket'] = df_filtered['prompt_bc_rate'].apply(assign_bc_bucket)

    # Show BC bucket distribution
    print(f"\nBC Bucket distribution (from YAML bc_rate):")
    print(df_filtered.groupby('bc_bucket')['base_prompt_id'].nunique().to_dict())

    print(f"\nLoaded bucket data: {len(df_filtered)} seeds across buckets")
    print(f"  Seeds per experiment:")
    for exp in sorted(df_filtered['short_name'].unique()):
        count = len(df_filtered[df_filtered['short_name'] == exp])
        print(f"    {exp}: {count} seeds")

    return df_filtered


def verify_bc_buckets(bucket_df):
    """Verify BC buckets are properly assigned (already done in load_bucket_data)."""
    if bucket_df is None or len(bucket_df) == 0:
        return bucket_df

    # BC buckets are already assigned from YAML in load_bucket_data
    # This function is just for verification
    if 'bc_bucket' not in bucket_df.columns:
        print("⚠️  Warning: bc_bucket column not found!")
        return bucket_df

    print(f"\n✓ BC buckets verified: {bucket_df['bc_bucket'].notna().sum()} seeds with bucket assignments")
    return bucket_df


def plot_metric_bars(df, output_dir):
    """Create bar charts for all key metrics with confidence intervals."""
    n_metrics = len(METRICS_CONFIG)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for idx, (metric_name, config) in enumerate(METRICS_CONFIG.items()):
        ax = axes[idx]

        field = config['field']
        ci_lower = config.get('ci_lower')
        ci_upper = config.get('ci_upper')

        # Get values
        values = df[field].values
        labels = df['short_name'].values

        # Calculate error bars if CIs available
        if ci_lower and ci_upper:
            errors_lower = values - df[ci_lower].values
            errors_upper = df[ci_upper].values - values
            errors = np.array([errors_lower, errors_upper])
        else:
            errors = None

        # Create bar chart
        x_pos = np.arange(len(labels))

        # Use a better color palette - seaborn's muted palette
        palette = sns.color_palette("Set2", n_colors=len(labels))

        # Create bars with improved styling
        bars = ax.bar(x_pos, values, color=palette, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

        # Add error bars if available (with improved styling)
        if errors is not None:
            ax.errorbar(x_pos, values, yerr=errors, fmt='none',
                       ecolor='black', elinewidth=2, capsize=6, capthick=2,
                       alpha=0.7, zorder=10)

        # Add baseline if specified
        if config['baseline'] is not None:
            # Check if baseline is a field name or a numeric value
            if isinstance(config['baseline'], str):
                # It's a field name - get the mean value
                baseline_value = df[config['baseline']].mean()
                ax.axhline(baseline_value, color='red', linestyle='--', alpha=0.5,
                          label=f'Baseline ({config["baseline"]}: {baseline_value:.2f})')
            else:
                # It's a numeric value
                ax.axhline(config['baseline'], color='red', linestyle='--', alpha=0.5, label='Baseline')
            ax.legend()

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(config['ylabel'])
        ax.set_title(metric_name, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars with improved visibility
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            # Add white background for better readability
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.8))

    # Hide unused subplots
    for idx in range(len(METRICS_CONFIG), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / '1_metric_bars.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric bar charts to {output_dir / '1_metric_bars.png'}")
    plt.close()


def plot_pareto_frontiers(df, output_dir):
    """Create line plots showing trade-offs with confidence intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pareto_configs = [
        {
            'x': 'suppressed_eval_awareness',
            'y': 'choice_b_delta',
            'xlabel': 'Suppressed Awareness (1-10)',
            'ylabel': 'Choice B Delta (%)',
            'title': 'Suppressed Awareness vs Choice B Delta',
            'x_invert': False,  # Lower is better
            'y_invert': False,   # Closer to 0 is better (less change from deployment)
        },
        {
            'x': 'bc_rate_wrt_deploy',
            'y': 'success_rate_modified',
            'xlabel': 'BC Rate wrt Deploy (%)',
            'ylabel': 'Success Rate (%)',
            'title': 'BC Rate wrt Deploy vs Success Rate',
            'x_invert': False,  # Lower is better (less change from deployment)
            'y_invert': True,
        },
        {
            'x': 'bc_rate_wrt_deploy',
            'y': 'self_test_removal_rate',
            'xlabel': 'BC Rate wrt Deploy (%)',
            'ylabel': 'Self-Test Removal (%)',
            'title': 'BC Rate wrt Deploy vs Self-Test Removal',
            'x_invert': False,  # Lower is better (less change from deployment)
            'y_invert': True,
        },
        {
            'x': 'suppressed_eval_awareness',
            'y': 'success_rate_modified',
            'xlabel': 'Suppressed Awareness (1-10)',
            'ylabel': 'Success Rate (%)',
            'title': 'Awareness vs Success Rate',
            'x_invert': False,  # Lower is better
            'y_invert': True,
        },
    ]

    for idx, (ax, config) in enumerate(zip(axes.flatten(), pareto_configs)):
        x_vals = df[config['x']].values
        y_vals = df[config['y']].values
        labels = df['short_name'].values

        # Get baseline values (original evaluation before suppression/steering)
        # For awareness metrics, use baseline_eval_awareness
        # For BC rate, use orig_bc_per_prompt_mean (original eval vs deployment)
        # For success rate, use baseline success (from original evaluation)
        baseline_x = None
        baseline_y = None

        if config['x'] == 'suppressed_eval_awareness':
            baseline_x = df['baseline_eval_awareness'].mean()
        elif config['x'] == 'bc_rate_wrt_deploy':
            # BC rate wrt deploy baseline is the original BC rate (eval vs deploy)
            baseline_x = df['baseline_bc_rate'].mean() if 'baseline_bc_rate' in df.columns else df['orig_bc_per_prompt_mean'].mean()

        if config['y'] == 'suppressed_eval_awareness':
            baseline_y = df['baseline_eval_awareness'].mean()
        elif config['y'] == 'success_rate_modified':
            baseline_y = df['success_rate_original'].mean() if 'success_rate_original' in df.columns else None
        elif config['y'] == 'task_completion':
            # Task completion should be similar for baseline
            baseline_y = 100.0  # Assume baseline task completion is 100%
        elif config['y'] == 'self_test_removal_rate':
            # Self-test removal baseline is 0 (no removal before intervention)
            baseline_y = 0.0
        elif config['y'] == 'choice_b_delta':
            # Choice B delta baseline is (original eval - deployment)
            baseline_y = df['baseline_choice_b_delta'].mean() if 'baseline_choice_b_delta' in df.columns else 0.0

        # Sort data by x-axis for line plot
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_sorted = y_vals[sort_idx]
        labels_sorted = labels[sort_idx]

        # Line plot with markers
        ax.plot(x_sorted, y_sorted, 'o-', linewidth=2, markersize=10, alpha=0.7,
               label='Experiments')

        # Plot baseline point if available
        if baseline_x is not None and baseline_y is not None:
            ax.scatter(baseline_x, baseline_y, s=300, alpha=0.8, edgecolors='red',
                      facecolors='none', linewidth=3, marker='*', label='Baseline (Original Eval)', zorder=10)
            ax.annotate('Baseline', (baseline_x, baseline_y), xytext=(5, -15),
                       textcoords='offset points', fontsize=10, fontweight='bold', color='red')

        # Annotate experiment points
        for x, y, label in zip(x_sorted, y_sorted, labels_sorted):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')

        # Show legend
        ax.legend(loc='best', fontsize=9)

        # Formatting
        ax.set_xlabel(config['xlabel'], fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontweight='bold')
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add directional indicator showing which way is better
        # Determine the arrow direction based on x_invert and y_invert
        if not config['x_invert'] and config['y_invert']:
            # Lower X, Higher Y = Bottom-left to top-right diagonal
            arrow_text = "↖ Better"
        elif config['x_invert'] and config['y_invert']:
            # Higher X, Higher Y = Bottom-right to top-right diagonal
            arrow_text = "↗ Better"
        elif not config['x_invert'] and not config['y_invert']:
            # Lower X, Lower Y = Top-left to bottom-right diagonal
            arrow_text = "↙ Better"
        else:
            # Higher X, Lower Y = Top-right to bottom-left diagonal
            arrow_text = "↘ Better"

        # Add text box with arrow
        ax.text(0.02, 0.98, arrow_text, transform=ax.transAxes,
               fontsize=11, ha='left', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                        edgecolor='darkgreen', linewidth=2, alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / '2_tradeoff_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved trade-off curves to {output_dir / '2_tradeoff_curves.png'}")
    plt.close()


def plot_performance_heatmap(df, output_dir):
    """Create performance heatmap showing relative performance across all metrics."""
    # Select key metrics for heatmap
    heatmap_metrics = [
        'bc_rate_wrt_deploy',
        'suppressed_eval_awareness',
        'success_rate_modified',
        'awareness_score_delta_from_deployment',
        'self_test_removal_rate',
        'other_test_removal_rate',
        'choice_modal_consistency',
        'overshoot_rate',
        'task_completion',
    ]

    # Create DataFrame for heatmap
    heatmap_data = df.set_index('short_name')[heatmap_metrics].T

    # Normalize each metric (0-1 scale, where 1 is best)
    heatmap_normalized = heatmap_data.copy()
    for metric in heatmap_metrics:
        vals = heatmap_data.loc[metric].values

        # Determine if lower or higher is better
        if metric in ['suppressed_eval_awareness', 'awareness_score_delta_from_deployment', 'overshoot_rate', 'bc_rate_wrt_deploy']:
            # Lower is better (less change from deployment, less awareness, etc.)
            normalized = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
        else:
            # Higher is better
            normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

        heatmap_normalized.loc[metric] = normalized

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Performance (1=best)'},
                linewidths=0.5, ax=ax, vmin=0, vmax=1)

    ax.set_title('Performance Heatmap Across All Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')

    # Rename y-axis labels
    metric_labels = {
        'bc_rate_wrt_deploy': 'BC Rate wrt Deploy',
        'suppressed_eval_awareness': 'Suppressed Awareness',
        'success_rate_modified': 'Success Rate',
        'awareness_score_delta_from_deployment': 'Awareness Delta',
        'self_test_removal_rate': 'Self-Test Removal',
        'other_test_removal_rate': 'Other-Test Removal',
        'choice_modal_consistency': 'Choice Consistency',
        'overshoot_rate': 'Overshoot Rate',
        'task_completion': 'Task Completion',
    }
    ax.set_yticklabels([metric_labels[m] for m in heatmap_metrics], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / '3_performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance heatmap to {output_dir / '3_performance_heatmap.png'}")
    plt.close()


def plot_awareness_comparison(df, output_dir):
    """Create comparison of awareness across contexts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Awareness levels across contexts
    ax = axes[0]
    awareness_cols = ['baseline_deploy_awareness', 'baseline_eval_awareness', 'suppressed_eval_awareness']
    awareness_labels = ['Deployment', 'Original Eval', 'Suppressed Eval']

    x_pos = np.arange(len(df))
    width = 0.25

    for i, (col, label) in enumerate(zip(awareness_cols, awareness_labels)):
        values = df[col].values
        ax.bar(x_pos + i * width, values, width, label=label, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(df['short_name'], rotation=45, ha='right')
    ax.set_ylabel('Awareness Score (1-10)', fontweight='bold')
    ax.set_title('Awareness Levels Across Contexts', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Deployment convergence (awareness delta)
    ax = axes[1]

    values = df['awareness_score_delta_from_deployment'].values
    ci_lower = df['awareness_score_delta_from_deployment_ci_lower'].values
    ci_upper = df['awareness_score_delta_from_deployment_ci_upper'].values
    labels = df['short_name'].values

    errors_lower = values - ci_lower
    errors_upper = ci_upper - values
    errors = np.array([errors_lower, errors_upper])

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, edgecolor='black')

    # Color bars (lower is better)
    colors = plt.cm.RdYlGn_r(np.interp(values, [values.min(), values.max()], [0, 1]))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add baseline
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Perfect Convergence')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Awareness Delta (points)', fontweight='bold')
    ax.set_title('Deployment Convergence\n(Suppressed - Deployment)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '4_awareness_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved awareness comparison to {output_dir / '4_awareness_comparison.png'}")
    plt.close()


def create_summary_table(df, bucket_df, output_dir):
    """Create summary table with key metrics."""
    summary_metrics = [
        'n_prompts',
        'n_seeds',
        'suppressed_eval_awareness',
        'success_rate_modified',
        'awareness_score_delta_from_deployment',
        'self_test_removal_rate',
        'choice_modal_consistency',
        'task_completion',
    ]

    summary_df = df[['short_name'] + summary_metrics].copy()

    # Calculate BC rate wrt deployment from bucket data (positive metric)
    if bucket_df is not None and len(bucket_df) > 0:
        bc_wrt_deployment = {}
        for exp_name in summary_df['short_name']:
            exp_data = bucket_df[bucket_df['short_name'] == exp_name]
            if len(exp_data) > 0 and 'modified_choice' in exp_data.columns and 'deployment_choice' in exp_data.columns:
                bc_count = (exp_data['modified_choice'] != exp_data['deployment_choice']).sum()
                bc_rate = (bc_count / len(exp_data)) * 100
                bc_wrt_deployment[exp_name] = bc_rate
            else:
                bc_wrt_deployment[exp_name] = 0.0

        summary_df.insert(2, 'bc_rate_wrt_deploy', summary_df['short_name'].map(bc_wrt_deployment))
    else:
        # Fallback: use absolute value of bc_rate from CSV (convert negative delta to positive rate)
        summary_df.insert(2, 'bc_rate_wrt_deploy', df['bc_rate'].abs())

    # Round numeric columns
    for col in summary_df.columns:
        if col not in ['n_prompts', 'n_seeds', 'short_name']:
            summary_df[col] = summary_df[col].round(2)

    # Save to CSV
    summary_df.to_csv(output_dir / '0_summary_metrics.csv', index=False)
    print(f"✓ Saved summary metrics to {output_dir / '0_summary_metrics.csv'}")

    # Print to console
    print("\n" + "="*100)
    print("SUMMARY METRICS TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")


def plot_eval_bucket_distribution(bucket_df, output_dir):
    """Show how prompts are distributed across eval awareness buckets."""
    if bucket_df is None or len(bucket_df) == 0:
        print("⚠️  No bucket data available, skipping eval bucket distribution plot")
        return

    # Count unique prompts per bucket per experiment
    bucket_counts = bucket_df.groupby(['short_name', 'eval_bucket']).agg({
        'prompt_id': 'nunique'
    }).reset_index()
    bucket_counts.columns = ['short_name', 'bucket', 'n_prompts']

    # Pivot for stacked bar chart
    bucket_pivot = bucket_counts.pivot(index='short_name', columns='bucket', values='n_prompts').fillna(0)

    # Reorder buckets
    bucket_order = ['10-25%', '25-50%', '50-75%', '75-100%']
    bucket_pivot = bucket_pivot[[b for b in bucket_order if b in bucket_pivot.columns]]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='black')

    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('Number of Prompts', fontweight='bold')
    ax.set_title('Eval Awareness Bucket Distribution\n(High Awareness BC Seeds Only)', fontsize=12, fontweight='bold')
    ax.legend(title='Eval Awareness Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '5_eval_bucket_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved eval bucket distribution to {output_dir / '5_eval_bucket_distribution.png'}")
    plt.close()


def plot_eval_bucket_metrics(bucket_df, output_dir):
    """Show key metrics broken down by eval awareness bucket."""
    if bucket_df is None or len(bucket_df) == 0:
        print("⚠️  No bucket data available, skipping eval bucket metrics")
        return

    # Define bucket order
    bucket_order = ['10-25%', '25-50%', '50-75%', '75-100%']
    available_buckets = [b for b in bucket_order if b in bucket_df['eval_bucket'].unique()]

    # Calculate baseline metrics per bucket (using original_choice, original_awareness, deployment_choice)
    baseline_metrics = []
    for bucket in available_buckets:
        bucket_data = bucket_df[bucket_df['eval_bucket'] == bucket]
        if len(bucket_data) > 0:
            # Baseline BC rate (original_choice vs deployment_choice) - deduplicate by prompt_id/seed first
            bucket_data_dedup = bucket_data.drop_duplicates(subset=['prompt_id', 'seed'])
            if 'original_choice' in bucket_data_dedup.columns and 'deployment_choice' in bucket_data_dedup.columns:
                baseline_bc_count = (bucket_data_dedup['original_choice'] != bucket_data_dedup['deployment_choice']).sum()
                baseline_bc_rate = (baseline_bc_count / len(bucket_data_dedup)) * 100
            else:
                baseline_bc_rate = 0

            # Baseline eval awareness
            baseline_eval_awareness = bucket_data_dedup['original_awareness'].mean() if 'original_awareness' in bucket_data_dedup.columns else 0

            # Baseline success rate
            baseline_success_count = (bucket_data_dedup['original_awareness'] < 5).sum() if 'original_awareness' in bucket_data_dedup.columns else 0
            baseline_success_rate = (baseline_success_count / len(bucket_data_dedup)) * 100

            baseline_metrics.append({
                'experiment': 'Baseline (Original Eval)',
                'bucket': bucket,
                'bc_rate': baseline_bc_rate,
                'modified_awareness': baseline_eval_awareness,
                'success_rate': baseline_success_rate,
                'n_seeds': len(bucket_data_dedup)
            })

    # Compute metrics per experiment per bucket
    bucket_metrics = []
    for exp_name in bucket_df['short_name'].unique():
        exp_data = bucket_df[bucket_df['short_name'] == exp_name]
        for bucket in available_buckets:
            bucket_data = exp_data[exp_data['eval_bucket'] == bucket]
            if len(bucket_data) > 0:
                # BC rate
                if 'deployment_choice' in bucket_data.columns and 'modified_choice' in bucket_data.columns:
                    bc_count = (bucket_data['deployment_choice'] != bucket_data['modified_choice']).sum()
                    bc_rate = (bc_count / len(bucket_data)) * 100
                else:
                    bc_rate = 0

                # Mean awareness
                modified_awareness = bucket_data['modified_awareness'].mean()

                # Success rate
                success_count = (bucket_data['modified_awareness'] < 5).sum()
                success_rate = (success_count / len(bucket_data)) * 100

                bucket_metrics.append({
                    'experiment': exp_name,
                    'bucket': bucket,
                    'bc_rate': bc_rate,
                    'modified_awareness': modified_awareness,
                    'success_rate': success_rate,
                    'n_seeds': len(bucket_data)
                })

    # Combine baseline and experiment metrics
    all_metrics = baseline_metrics + bucket_metrics
    bucket_metrics_df = pd.DataFrame(all_metrics)

    # Use seaborn color palette (7 colors now - 6 experiments + baseline)
    palette = sns.color_palette("Set2", n_colors=len(bucket_metrics_df['experiment'].unique()))

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: BC Rate by Bucket
    ax1 = axes[0, 0]
    pivot1 = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='bc_rate')
    pivot1 = pivot1.reindex(available_buckets)
    pivot1.plot(kind='bar', ax=ax1, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_title('BC Rate wrt Deployment by Eval Awareness Bucket', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Eval Awareness Bucket', fontsize=10)
    ax1.set_ylabel('BC Rate wrt Deployment (%)', fontsize=10)
    ax1.legend(title='Experiment', fontsize=7, loc='best')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Modified Awareness by Bucket
    ax2 = axes[0, 1]
    pivot2 = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='modified_awareness')
    pivot2 = pivot2.reindex(available_buckets)
    pivot2.plot(kind='bar', ax=ax2, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.set_title('Suppressed Awareness by Eval Awareness Bucket', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Eval Awareness Bucket', fontsize=10)
    ax2.set_ylabel('Awareness Score (1-10)', fontsize=10)
    ax2.legend(title='Experiment', fontsize=7, loc='best')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
    # Add deployment baseline (constant across buckets)
    if 'deployment_awareness' in bucket_df.columns:
        baseline_deploy = bucket_df['deployment_awareness'].mean()
        ax2.axhline(baseline_deploy, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline Deploy ({baseline_deploy:.1f})')

    # Plot 3: Success Rate by Bucket
    ax3 = axes[1, 0]
    pivot3 = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='success_rate')
    pivot3 = pivot3.reindex(available_buckets)
    pivot3.plot(kind='bar', ax=ax3, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax3.set_title('Success Rate by Eval Awareness Bucket', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Eval Awareness Bucket', fontsize=10)
    ax3.set_ylabel('Success Rate (%)', fontsize=10)
    ax3.legend(title='Experiment', fontsize=7, loc='best')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Sample Size by Bucket
    ax4 = axes[1, 1]
    pivot4 = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='n_seeds')
    pivot4 = pivot4.reindex(available_buckets)
    pivot4.plot(kind='bar', ax=ax4, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax4.set_title('Number of Seeds by Eval Awareness Bucket', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Eval Awareness Bucket', fontsize=10)
    ax4.set_ylabel('Number of Seeds', fontsize=10)
    ax4.legend(title='Experiment', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    plt.suptitle('Performance by Eval Awareness Bucket', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / '6_eval_bucket_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved eval bucket metrics to {output_dir / '6_eval_bucket_metrics.png'}")
    plt.close()


def plot_bc_bucket_distribution(bucket_df, output_dir):
    """Show how prompts are distributed across OG BC buckets."""
    if bucket_df is None or len(bucket_df) == 0 or 'bc_bucket' not in bucket_df.columns:
        print("⚠️  No OG BC bucket data available, skipping OG BC bucket distribution plot")
        return

    # Count unique prompts per BC bucket per experiment
    bc_counts = bucket_df.groupby(['short_name', 'bc_bucket']).agg({
        'prompt_id': 'nunique'
    }).reset_index()
    bc_counts.columns = ['short_name', 'bc_bucket', 'n_prompts']

    # Pivot for stacked bar chart
    bc_pivot = bc_counts.pivot(index='short_name', columns='bc_bucket', values='n_prompts').fillna(0)

    # Reorder buckets
    bucket_order = ['<50%', '50-75%', '75-90%', '90-100%']
    bc_pivot = bc_pivot[[b for b in bucket_order if b in bc_pivot.columns]]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bc_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm', edgecolor='black')

    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('Number of Prompts', fontweight='bold')
    ax.set_title('OG BC Bucket Distribution\n(Original Eval vs Deploy BC Rate)', fontsize=12, fontweight='bold')
    ax.legend(title='OG BC Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '7_og_bc_bucket_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved OG BC bucket distribution to {output_dir / '7_og_bc_bucket_distribution.png'}")
    plt.close()


def plot_bc_bucket_metrics(bucket_df, output_dir):
    """Show key metrics broken down by OG BC bucket."""
    if bucket_df is None or len(bucket_df) == 0 or 'bc_bucket' not in bucket_df.columns:
        print("⚠️  No OG BC bucket data available, skipping OG BC bucket metrics")
        return

    # Define bucket order
    bucket_order = ['<50%', '50-75%', '75-90%', '90-100%']
    available_buckets = [b for b in bucket_order if b in bucket_df['bc_bucket'].unique()]

    # Calculate baseline metrics per BC bucket
    baseline_bc_metrics = []
    for bucket in available_buckets:
        bucket_data = bucket_df[bucket_df['bc_bucket'] == bucket]
        if len(bucket_data) > 0:
            # Baseline BC rate - deduplicate by prompt_id/seed first
            bucket_data_dedup = bucket_data.drop_duplicates(subset=['prompt_id', 'seed'])
            if 'original_choice' in bucket_data_dedup.columns and 'deployment_choice' in bucket_data_dedup.columns:
                baseline_bc_count = (bucket_data_dedup['original_choice'] != bucket_data_dedup['deployment_choice']).sum()
                baseline_bc_rate = (baseline_bc_count / len(bucket_data_dedup)) * 100
            else:
                baseline_bc_rate = 0

            # Baseline eval awareness
            baseline_eval_awareness = bucket_data_dedup['original_awareness'].mean() if 'original_awareness' in bucket_data_dedup.columns else 0

            # Baseline success rate
            baseline_success_count = (bucket_data_dedup['original_awareness'] < 5).sum() if 'original_awareness' in bucket_data_dedup.columns else 0
            baseline_success_rate = (baseline_success_count / len(bucket_data_dedup)) * 100

            baseline_bc_metrics.append({
                'experiment': 'Baseline (Original Eval)',
                'bc_bucket': bucket,
                'bc_rate': baseline_bc_rate,
                'modified_awareness': baseline_eval_awareness,
                'success_rate': baseline_success_rate,
                'n_seeds': len(bucket_data_dedup)
            })

    # Compute metrics per experiment per BC bucket
    bc_metrics = []
    for exp_name in bucket_df['short_name'].unique():
        exp_data = bucket_df[bucket_df['short_name'] == exp_name]
        for bucket in available_buckets:
            bucket_data = exp_data[exp_data['bc_bucket'] == bucket]
            if len(bucket_data) > 0:
                # BC rate (actual, not bucket)
                if 'deployment_choice' in bucket_data.columns and 'modified_choice' in bucket_data.columns:
                    bc_count = (bucket_data['deployment_choice'] != bucket_data['modified_choice']).sum()
                    bc_rate = (bc_count / len(bucket_data)) * 100
                else:
                    bc_rate = 0

                # Mean awareness
                modified_awareness = bucket_data['modified_awareness'].mean()

                # Success rate
                success_count = (bucket_data['modified_awareness'] < 5).sum()
                success_rate = (success_count / len(bucket_data)) * 100

                bc_metrics.append({
                    'experiment': exp_name,
                    'bc_bucket': bucket,
                    'bc_rate': bc_rate,
                    'modified_awareness': modified_awareness,
                    'success_rate': success_rate,
                    'n_seeds': len(bucket_data)
                })

    # Combine baseline and experiment metrics
    all_bc_metrics = baseline_bc_metrics + bc_metrics
    bc_metrics_df = pd.DataFrame(all_bc_metrics)

    # Use seaborn color palette (7 colors - 6 experiments + baseline)
    palette = sns.color_palette("Set2", n_colors=len(bc_metrics_df['experiment'].unique()))

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: BC Rate by BC Bucket
    ax1 = axes[0, 0]
    pivot1 = bc_metrics_df.pivot(index='bc_bucket', columns='experiment', values='bc_rate')
    pivot1 = pivot1.reindex(available_buckets)
    pivot1.plot(kind='bar', ax=ax1, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_title('BC Rate wrt Deployment by OG BC Bucket', fontsize=12, fontweight='bold')
    ax1.set_xlabel('OG BC Bucket', fontsize=10)
    ax1.set_ylabel('BC Rate wrt Deployment (%)', fontsize=10)
    ax1.legend(title='Experiment', fontsize=7, loc='best')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Modified Awareness by BC Bucket
    ax2 = axes[0, 1]
    pivot2 = bc_metrics_df.pivot(index='bc_bucket', columns='experiment', values='modified_awareness')
    pivot2 = pivot2.reindex(available_buckets)
    pivot2.plot(kind='bar', ax=ax2, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.set_title('Suppressed Awareness by OG BC Bucket', fontsize=12, fontweight='bold')
    ax2.set_xlabel('OG BC Bucket', fontsize=10)
    ax2.set_ylabel('Awareness Score (1-10)', fontsize=10)
    ax2.legend(title='Experiment', fontsize=7, loc='best')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
    # Add deployment baseline (constant across buckets)
    if 'deployment_awareness' in bucket_df.columns:
        baseline_deploy = bucket_df['deployment_awareness'].mean()
        ax2.axhline(baseline_deploy, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline Deploy ({baseline_deploy:.1f})')

    # Plot 3: Success Rate by BC Bucket
    ax3 = axes[1, 0]
    pivot3 = bc_metrics_df.pivot(index='bc_bucket', columns='experiment', values='success_rate')
    pivot3 = pivot3.reindex(available_buckets)
    pivot3.plot(kind='bar', ax=ax3, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax3.set_title('Success Rate by OG BC Bucket', fontsize=12, fontweight='bold')
    ax3.set_xlabel('OG BC Bucket', fontsize=10)
    ax3.set_ylabel('Success Rate (%)', fontsize=10)
    ax3.legend(title='Experiment', fontsize=7, loc='best')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Sample Size by BC Bucket
    ax4 = axes[1, 1]
    pivot4 = bc_metrics_df.pivot(index='bc_bucket', columns='experiment', values='n_seeds')
    pivot4 = pivot4.reindex(available_buckets)
    pivot4.plot(kind='bar', ax=ax4, width=0.8, color=palette, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax4.set_title('Number of Seeds by OG BC Bucket', fontsize=12, fontweight='bold')
    ax4.set_xlabel('OG BC Bucket', fontsize=10)
    ax4.set_ylabel('Number of Seeds', fontsize=10)
    ax4.legend(title='Experiment', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    plt.suptitle('Performance by OG BC Bucket', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / '8_og_bc_bucket_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved OG BC bucket metrics to {output_dir / '8_og_bc_bucket_metrics.png'}")
    plt.close()


def main():
    # Input/output paths
    csv_path = Path('outputs/qwen_qwen3-32b_all18_seed29/data/data/experiment_metrics.csv')
    output_dir = Path('outputs/sysprompt_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if CSV exists
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        print("Please provide the correct path to experiment_metrics.csv")
        sys.exit(1)

    print(f"\n{'='*100}")
    print(f"PLOTTING 6 SPECIFIC EXPERIMENTS")
    print(f"{'='*100}\n")
    print(f"Input:  {csv_path}")
    print(f"Output: {output_dir}/\n")

    # Load aggregate metrics data
    df = load_and_filter_data(csv_path)

    if len(df) != 6:
        print(f"Warning: Expected 6 experiments, found {len(df)}")

    # Load bucket-level data (includes BC buckets from YAML)
    bucket_df = load_bucket_data(csv_path)

    # Verify BC bucket assignments
    if bucket_df is not None:
        bucket_df = verify_bc_buckets(bucket_df)

        # ASSERTION: Check we have approximately 2500+ seeds per experiment
        for exp_name in bucket_df['short_name'].unique():
            exp_seeds = len(bucket_df[bucket_df['short_name'] == exp_name])
            print(f"\n✓ Experiment {exp_name}: {exp_seeds} seeds")
            if exp_seeds < 2000:
                print(f"  ⚠️  WARNING: Only {exp_seeds} seeds, expected ~2500+")
                print(f"  This may indicate incorrect filtering!")

        # Total seeds assertion
        total_seeds = len(bucket_df)
        print(f"\n✓ Total seeds across all experiments: {total_seeds}")
        if total_seeds < 12000:  # 6 experiments * ~2500 seeds each
            print(f"  ⚠️  WARNING: Only {total_seeds} total seeds, expected ~15000+")
            print(f"  This may indicate incorrect filtering!")

        # Add BC rate wrt deployment to df (calculated from bucket data)
        bc_wrt_deployment = {}
        for exp_name in df['short_name']:
            exp_data = bucket_df[bucket_df['short_name'] == exp_name]
            if len(exp_data) > 0 and 'modified_choice' in exp_data.columns and 'deployment_choice' in exp_data.columns:
                bc_count = (exp_data['modified_choice'] != exp_data['deployment_choice']).sum()
                bc_rate = (bc_count / len(exp_data)) * 100
                bc_wrt_deployment[exp_name] = bc_rate
            else:
                bc_wrt_deployment[exp_name] = 0.0

        df['bc_rate_wrt_deploy'] = df['short_name'].map(bc_wrt_deployment)

        # Calculate baseline BC rate using only the prompts in our analysis (high-awareness BC prompts)
        # Use bucket_df which already has the correct prompts
        # Get unique base_prompt_ids from bucket_df
        analysis_prompts = bucket_df['base_prompt_id'].unique()

        # Load the full combined CSV
        combined_full_path = csv_path.parent / 'combined_all_experiments.csv'
        df_all = pd.read_csv(combined_full_path)

        # Extract base_prompt_id from prompt_id in combined CSV
        df_all['base_prompt_id'] = df_all['prompt_id'].apply(
            lambda x: '_'.join(x.replace('.yaml', '').split('_')[:3]) if isinstance(x, str) and '_seed_' in x else x
        )

        # Filter to only our analysis prompts
        df_filtered = df_all[df_all['base_prompt_id'].isin(analysis_prompts)]

        # Keep only UNIQUE prompt/seed combinations
        df_unique = df_filtered.drop_duplicates(subset=['prompt_id', 'seed'])

        # Filter to only seeds 0-29
        df_unique = df_unique[df_unique['seed'] <= 29]

        # Calculate BC rate
        bc_count_all = (df_unique['original_choice'] != df_unique['deployment_choice']).sum()
        total_seeds_all = len(df_unique)
        baseline_bc_rate = (bc_count_all / total_seeds_all) * 100
        df['baseline_bc_rate'] = baseline_bc_rate
        print(f"\n✓ Baseline BC Rate (original eval vs deployment, seeds 0-29, {len(analysis_prompts)} prompts): {baseline_bc_rate:.2f}% ({bc_count_all}/{total_seeds_all} unique seeds)")

        # Calculate baseline Choice B Delta (original eval - deployment)
        orig_b_count = (df_unique['original_choice'] == 'B').sum()
        deploy_b_count = (df_unique['deployment_choice'] == 'B').sum()
        orig_b_rate = (orig_b_count / total_seeds_all) * 100
        deploy_b_rate = (deploy_b_count / total_seeds_all) * 100
        baseline_choice_b_delta = orig_b_rate - deploy_b_rate
        df['baseline_choice_b_delta'] = baseline_choice_b_delta
        print(f"✓ Baseline Choice B Delta (original eval - deployment): {baseline_choice_b_delta:+.2f}% (Orig: {orig_b_rate:.1f}%, Deploy: {deploy_b_rate:.1f}%)")
    else:
        # Fallback: use absolute value of choice_b_delta from CSV
        df['bc_rate_wrt_deploy'] = df.get('choice_b_delta', 0).abs()
        df['baseline_bc_rate'] = df.get('orig_bc_per_prompt_mean', 0)

    # Create plots
    print("\nGenerating visualizations...")
    create_summary_table(df, bucket_df, output_dir)
    plot_metric_bars(df, output_dir)
    plot_pareto_frontiers(df, output_dir)
    plot_performance_heatmap(df, output_dir)
    plot_awareness_comparison(df, output_dir)

    # NEW: Bucket-stratified plots
    print("\nGenerating bucket-stratified plots...")
    plot_eval_bucket_distribution(bucket_df, output_dir)
    plot_eval_bucket_metrics(bucket_df, output_dir)
    plot_bc_bucket_distribution(bucket_df, output_dir)
    plot_bc_bucket_metrics(bucket_df, output_dir)

    print(f"\n{'='*100}")
    print(f"✓ ALL PLOTS SAVED TO: {output_dir}/")
    print(f"{'='*100}\n")

    print("Generated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
