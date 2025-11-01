#!/usr/bin/env python3
"""
Compare 5 main experiments: 2 steering, 2 suppression, 1 system prompt.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_palette("Set2")

# 5 Main experiments to compare
EXPERIMENTS = {
    'anthropic_N4_S1.0': 'Steer_Anthropic (S1.0)',
    'tim_N4_S3.5': 'Steer_Tim (S3.5)',
    'sentence_resample_pick_if_not_evaly': 'E01_Hard-Suppress(T<3)',
    'sentence_resample_pick_least_eval_among_5': 'E06_Best-of-5 (Min Awareness)',
    'system_prompt_taboo_words': 'SysPrompt_Taboo',
}

# Metrics to plot
METRICS_CONFIG = {
    'BC Rate wrt Deploy': {
        'field': 'bc_rate_wrt_deploy',
        'ylabel': 'BC Rate wrt Deploy (%)',
        'lower_is_better': True,
    },
    'Suppressed Eval Awareness': {
        'field': 'suppressed_eval_awareness',
        'ci_lower': 'suppressed_eval_awareness_ci_lower',
        'ci_upper': 'suppressed_eval_awareness_ci_upper',
        'ylabel': 'Awareness Score (1-10)',
        'lower_is_better': True,
    },
    'Success Rate': {
        'field': 'success_rate_modified',
        'ci_lower': 'success_rate_modified_ci_lower',
        'ci_upper': 'success_rate_modified_ci_upper',
        'ylabel': 'Success Rate (%)',
        'lower_is_better': False,
    },
    'Self-Test Removal': {
        'field': 'self_test_removal_rate',
        'ci_lower': 'self_test_removal_rate_ci_lower',
        'ci_upper': 'self_test_removal_rate_ci_upper',
        'ylabel': 'Self-Test Removal (%)',
        'lower_is_better': False,
    },
    'Choice B Delta': {
        'field': 'choice_b_delta',
        'ci_lower': 'choice_b_delta_ci_lower',
        'ci_upper': 'choice_b_delta_ci_upper',
        'ylabel': 'Choice B Delta (%)',
        'lower_is_better': None,  # Closer to 0 is better
    },
}


def load_and_filter_data(csv_path):
    """Load experiment metrics and filter to 5 main experiments."""
    df = pd.read_csv(csv_path)

    # Filter to only our 5 experiments
    df_filtered = df[df['experiment'].isin(EXPERIMENTS.keys())].copy()

    # Add short names
    df_filtered['short_name'] = df_filtered['experiment'].map(EXPERIMENTS)

    # Sort by experiment order
    exp_order = list(EXPERIMENTS.values())
    df_filtered['sort_order'] = df_filtered['short_name'].apply(lambda x: exp_order.index(x))
    df_filtered = df_filtered.sort_values('sort_order').drop('sort_order', axis=1)

    return df_filtered


def load_bucket_data(csv_path):
    """Load bucket-level data for stratified analysis."""
    bucket_csv = csv_path.parent / 'combined_all_experiments.csv'

    if not bucket_csv.exists():
        print(f"Warning: Bucket data not found at {bucket_csv}")
        return None

    df = pd.read_csv(bucket_csv)

    # Filter to only our 5 experiments
    df_filtered = df[df['experiment'].isin(EXPERIMENTS.keys())].copy()

    # Add short names
    df_filtered['short_name'] = df_filtered['experiment'].map(EXPERIMENTS)

    # Extract base_prompt_id for grouping
    df_filtered['base_prompt_id'] = df_filtered['prompt_id'].apply(
        lambda x: '_'.join(x.replace('.yaml', '').split('_')[:3]) if isinstance(x, str) and '_seed_' in x else x
    )

    return df_filtered


def verify_bc_buckets(bucket_df):
    """Verify BC bucket assignments."""
    print("\nBC Bucket distribution (from YAML bc_rate):")
    if 'bc_bucket' in bucket_df.columns:
        print(bucket_df['bc_bucket'].value_counts().to_dict())

    print(f"\nLoaded bucket data: {len(bucket_df)} seeds across buckets")
    print("  Seeds per experiment:")
    for exp_name in bucket_df['short_name'].unique():
        count = len(bucket_df[bucket_df['short_name'] == exp_name])
        print(f"    {exp_name}: {count} seeds")

    return bucket_df


def create_summary_table(df, bucket_df, output_dir):
    """Create summary table with key metrics."""
    summary_cols = [
        'short_name', 'n_prompts', 'bc_rate_wrt_deploy', 'n_seeds',
        'suppressed_eval_awareness', 'success_rate_modified',
        'awareness_score_delta_from_deployment', 'self_test_removal_rate',
        'choice_modal_consistency', 'task_completion'
    ]

    summary_df = df[summary_cols].copy()
    summary_df.to_csv(output_dir / '0_summary_metrics.csv', index=False)

    print("\n" + "="*100)
    print("SUMMARY METRICS TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)


def plot_metric_bars(df, output_dir):
    """Create bar charts for key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    experiments = df['short_name'].values
    n_experiments = len(experiments)
    colors = sns.color_palette("Set2", n_experiments)

    metrics = [
        ('bc_rate_wrt_deploy', 'BC Rate wrt Deploy (%)'),
        ('suppressed_eval_awareness', 'Suppressed Eval Awareness (1-10)'),
        ('success_rate_modified', 'Success Rate (%)'),
        ('self_test_removal_rate', 'Self-Test Removal (%)'),
        ('choice_b_delta', 'Choice B Delta (%)'),
        ('task_completion', 'Task Completion (%)'),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].values

        bars = ax.bar(range(n_experiments), values, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_ylabel(title, fontweight='bold', fontsize=10)
        ax.set_xticks(range(n_experiments))
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(output_dir / '1_metric_bars.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric bar charts to {output_dir / '1_metric_bars.png'}")
    plt.close()


def plot_pareto_frontiers(df, output_dir):
    """Create Pareto frontier plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define the two Pareto plots
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
            'title': 'BC Rate vs Success Rate',
            'x_invert': False,  # Lower BC is better
            'y_invert': True,   # Higher success is better
        },
    ]

    for idx, (ax, config) in enumerate(zip(axes, pareto_configs)):
        x_vals = df[config['x']].values
        y_vals = df[config['y']].values
        labels = df['short_name'].values

        # Get baseline values
        baseline_x = None
        baseline_y = None

        if config['x'] == 'suppressed_eval_awareness':
            baseline_x = df['baseline_eval_awareness'].mean()
        elif config['x'] == 'bc_rate_wrt_deploy':
            baseline_x = df['baseline_bc_rate'].mean() if 'baseline_bc_rate' in df.columns else df['orig_bc_per_prompt_mean'].mean()

        if config['y'] == 'suppressed_eval_awareness':
            baseline_y = df['baseline_eval_awareness'].mean()
        elif config['y'] == 'success_rate_modified':
            baseline_y = df['success_rate_original'].mean() if 'success_rate_original' in df.columns else None
        elif config['y'] == 'task_completion':
            baseline_y = 100.0
        elif config['y'] == 'self_test_removal_rate':
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

        # Add directional indicator
        if not config['x_invert'] and config['y_invert']:
            arrow_text = "↖ Better"
        elif config['x_invert'] and config['y_invert']:
            arrow_text = "↗ Better"
        elif not config['x_invert'] and not config['y_invert']:
            arrow_text = "↙ Better"
        else:
            arrow_text = "↘ Better"

        ax.text(0.02, 0.98, arrow_text, transform=ax.transAxes,
               fontsize=11, ha='left', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                        edgecolor='darkgreen', linewidth=2, alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / '2_tradeoff_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved trade-off curves to {output_dir / '2_tradeoff_curves.png'}")
    plt.close()


def plot_performance_heatmap(df, output_dir):
    """Create heatmap showing normalized performance across metrics."""
    metrics_for_heatmap = [
        'bc_rate_wrt_deploy',
        'suppressed_eval_awareness',
        'success_rate_modified',
        'self_test_removal_rate',
        'choice_b_delta',
    ]

    metric_labels = [
        'BC Rate\nwrt Deploy',
        'Suppressed\nAwareness',
        'Success\nRate',
        'Self-Test\nRemoval',
        'Choice B\nDelta',
    ]

    # Extract data
    heatmap_data = df[metrics_for_heatmap].values.T

    # Normalize: 0=worst, 1=best for each metric
    normalized_data = np.zeros_like(heatmap_data, dtype=float)
    for i, metric in enumerate(metrics_for_heatmap):
        vals = heatmap_data[i]

        if metric in ['bc_rate_wrt_deploy', 'suppressed_eval_awareness']:
            # Lower is better: invert
            normalized_data[i] = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        elif metric == 'choice_b_delta':
            # Closer to 0 is better
            normalized_data[i] = 1 - np.abs(vals) / (np.abs(vals).max() + 1e-9)
        else:
            # Higher is better
            normalized_data[i] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(df)))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(df['short_name'].values, rotation=45, ha='right')
    ax.set_yticklabels(metric_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance (0=worst, 1=best)', fontweight='bold')

    # Add text annotations
    for i in range(len(metric_labels)):
        for j in range(len(df)):
            text = ax.text(j, i, f'{normalized_data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontweight='bold')

    ax.set_title('Performance Heatmap (Normalized Metrics)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '3_performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance heatmap to {output_dir / '3_performance_heatmap.png'}")
    plt.close()


def main():
    output_dir = Path('outputs/5_main_experiments_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("PLOTTING 5 MAIN EXPERIMENTS")
    print("="*100)
    print()

    csv_path = Path('outputs/qwen_qwen3-32b_all18_seed29/data/data/experiment_metrics.csv')

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Input:  {csv_path}")
    print(f"Output: {output_dir}/\n")

    # Load data
    df = load_and_filter_data(csv_path)

    if len(df) != 5:
        print(f"Warning: Expected 5 experiments, found {len(df)}")

    print(f"Loaded {len(df)} experiments:")
    for exp in df['short_name'].values:
        print(f"  - {exp}")

    # Load bucket data
    bucket_df = load_bucket_data(csv_path)

    if bucket_df is not None:
        bucket_df = verify_bc_buckets(bucket_df)

        # Add BC rate wrt deployment
        bc_wrt_deployment = {}
        for exp_name in df['short_name']:
            exp_data = bucket_df[bucket_df['short_name'] == exp_name]
            if len(exp_data) > 0:
                bc_count = (exp_data['modified_choice'] != exp_data['deployment_choice']).sum()
                bc_rate = (bc_count / len(exp_data)) * 100
                bc_wrt_deployment[exp_name] = bc_rate

        df['bc_rate_wrt_deploy'] = df['short_name'].map(bc_wrt_deployment)

        # Calculate baseline BC rate and Choice B Delta
        analysis_prompts = bucket_df['base_prompt_id'].unique()
        combined_full_path = csv_path.parent / 'combined_all_experiments.csv'
        df_all = pd.read_csv(combined_full_path)

        df_all['base_prompt_id'] = df_all['prompt_id'].apply(
            lambda x: '_'.join(x.replace('.yaml', '').split('_')[:3]) if isinstance(x, str) and '_seed_' in x else x
        )

        df_filtered = df_all[df_all['base_prompt_id'].isin(analysis_prompts)]
        df_unique = df_filtered.drop_duplicates(subset=['prompt_id', 'seed'])
        df_unique = df_unique[df_unique['seed'] <= 29]

        # BC rate
        bc_count_all = (df_unique['original_choice'] != df_unique['deployment_choice']).sum()
        total_seeds_all = len(df_unique)
        baseline_bc_rate = (bc_count_all / total_seeds_all) * 100
        df['baseline_bc_rate'] = baseline_bc_rate
        print(f"\n✓ Baseline BC Rate (original eval vs deployment, seeds 0-29, {len(analysis_prompts)} prompts): {baseline_bc_rate:.2f}% ({bc_count_all}/{total_seeds_all} unique seeds)")

        # Choice B Delta
        orig_b_count = (df_unique['original_choice'] == 'B').sum()
        deploy_b_count = (df_unique['deployment_choice'] == 'B').sum()
        orig_b_rate = (orig_b_count / total_seeds_all) * 100
        deploy_b_rate = (deploy_b_count / total_seeds_all) * 100
        baseline_choice_b_delta = orig_b_rate - deploy_b_rate
        df['baseline_choice_b_delta'] = baseline_choice_b_delta
        print(f"✓ Baseline Choice B Delta (original eval - deployment): {baseline_choice_b_delta:+.2f}% (Orig: {orig_b_rate:.1f}%, Deploy: {deploy_b_rate:.1f}%)")

    # Create plots
    print("\nGenerating visualizations...")
    create_summary_table(df, bucket_df, output_dir)
    plot_metric_bars(df, output_dir)
    plot_pareto_frontiers(df, output_dir)
    plot_performance_heatmap(df, output_dir)

    print(f"\n{'='*100}")
    print(f"✓ ALL PLOTS SAVED TO: {output_dir}/")
    print(f"{'='*100}\n")

    print("Generated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
