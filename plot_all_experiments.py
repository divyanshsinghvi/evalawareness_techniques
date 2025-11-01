#!/usr/bin/env python3
"""
Plot ALL experiments together for comprehensive comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_palette("Set2")

# All experiments - full names mapping
ALL_EXPERIMENTS = {
    # Steering experiments
    'anthropic_N4_S1.0': 'Steer_Anthropic (S1.0)',
    'tim_N4_S3.5': 'Steer_Tim (S3.5)',
    'tim_N4_S3.5_random': 'Steer_Tim_Random (S3.5)',

    # Suppression experiments
    'sentence_resample_pick_if_not_evaly': 'E01_Hard-Suppress(T<3)',
    'sentence_resample_pick_least_eval_among_5': 'E06_Best-of-5',
    'remove_only': 'RemoveOnly',
    'no_reasoning': 'NoReasoning',
    'only_final_answer': 'OnlyFinalAnswer',
    'sentence_resample_pick_least_eval_for_reasoning_and_content': 'LeastEval_Full',
    'chunk_resample_thresh5_temp07_fixed': 'E10_Soft-Suppress(T<5)',
    'chunk_resample_exclude_criteria_2_v2': 'E11_Cue-Literal',
    'resample_best_of_n5_thresh5_temp07': 'BestOf5_T5',
    '13_category_prompt_cue_removal_chunk_resample': 'E13_Cue-Strip',
    '14_category_safety_without_cue_removal_chunk_resample': 'E14_Safety-Strip',

    # System prompt experiments
    'system_prompt': 'SysPromptDeploy',
    'system_prompt_no_spec': 'SysPromptDeploy_NoSpec',
    'system_prompt_taboo_words': 'SysPrompt_Taboo',
}


def load_and_filter_data(csv_path):
    """Load experiment metrics and filter to all available experiments."""
    df = pd.read_csv(csv_path)

    # Filter to only experiments we have names for
    df_filtered = df[df['experiment'].isin(ALL_EXPERIMENTS.keys())].copy()

    # Add short names
    df_filtered['short_name'] = df_filtered['experiment'].map(ALL_EXPERIMENTS)

    # Add experiment type
    def get_exp_type(exp):
        if exp.startswith('anthropic') or exp.startswith('tim'):
            return 'Steering'
        elif exp.startswith('system'):
            return 'System Prompt'
        else:
            return 'Suppression'

    df_filtered['exp_type'] = df_filtered['experiment'].apply(get_exp_type)

    return df_filtered


def load_bucket_data(csv_path):
    """Load bucket-level data for BC rate calculation."""
    bucket_csv = csv_path.parent / 'combined_all_experiments.csv'

    if not bucket_csv.exists():
        print(f"Warning: Bucket data not found at {bucket_csv}")
        return None

    df = pd.read_csv(bucket_csv)

    # Filter to only our experiments
    df_filtered = df[df['experiment'].isin(ALL_EXPERIMENTS.keys())].copy()

    # Add short names
    df_filtered['short_name'] = df_filtered['experiment'].map(ALL_EXPERIMENTS)

    # Extract base_prompt_id
    df_filtered['base_prompt_id'] = df_filtered['prompt_id'].apply(
        lambda x: '_'.join(x.replace('.yaml', '').split('_')[:3]) if isinstance(x, str) and '_seed_' in x else x
    )

    return df_filtered


def create_summary_table(df, output_dir):
    """Create comprehensive summary table."""
    summary_cols = [
        'short_name', 'exp_type', 'n_prompts', 'bc_rate_wrt_deploy', 'n_seeds',
        'suppressed_eval_awareness', 'success_rate_modified',
        'self_test_removal_rate', 'choice_b_delta', 'task_completion'
    ]

    summary_df = df[summary_cols].copy()

    # Sort by experiment type and BC rate
    summary_df = summary_df.sort_values(['exp_type', 'bc_rate_wrt_deploy'])

    summary_df.to_csv(output_dir / '0_all_experiments_summary.csv', index=False)

    print("\n" + "="*120)
    print("ALL EXPERIMENTS SUMMARY")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)

    # Print statistics by type
    print("\n" + "="*120)
    print("STATISTICS BY EXPERIMENT TYPE")
    print("="*120)
    for exp_type in ['Steering', 'Suppression', 'System Prompt']:
        type_data = df[df['exp_type'] == exp_type]
        if len(type_data) > 0:
            print(f"\n{exp_type} ({len(type_data)} experiments):")
            print(f"  BC Rate: {type_data['bc_rate_wrt_deploy'].min():.1f}% - {type_data['bc_rate_wrt_deploy'].max():.1f}%")
            print(f"  Awareness: {type_data['suppressed_eval_awareness'].min():.1f} - {type_data['suppressed_eval_awareness'].max():.1f}")
            print(f"  Success Rate: {type_data['success_rate_modified'].min():.1f}% - {type_data['success_rate_modified'].max():.1f}%")
    print("="*120)


def plot_scatter_comparison(df, output_dir, bucket_df=None):
    """Create comprehensive scatter plot comparing all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Calculate seed-level std if bucket data available
    std_data = {}
    if bucket_df is not None:
        std_data = calculate_seed_level_std(bucket_df, df)

    # Define scatter plots
    scatter_configs = [
        {
            'x': 'suppressed_eval_awareness',
            'y': 'bc_rate_wrt_deploy',
            'xlabel': 'Suppressed Awareness (1-10)',
            'ylabel': 'BC Rate wrt Deploy (%)',
            'title': 'Awareness vs BC Rate',
        },
        {
            'x': 'bc_rate_wrt_deploy',
            'y': 'success_rate_modified',
            'xlabel': 'BC Rate wrt Deploy (%)',
            'ylabel': 'Success Rate (%)',
            'title': 'BC Rate vs Success Rate',
        },
        {
            'x': 'suppressed_eval_awareness',
            'y': 'choice_b_delta',
            'xlabel': 'Suppressed Awareness (1-10)',
            'ylabel': 'Choice B Delta (%)',
            'title': 'Awareness vs Choice B Delta',
        },
        {
            'x': 'bc_rate_wrt_deploy',
            'y': 'choice_b_delta',
            'xlabel': 'BC Rate wrt Deploy (%)',
            'ylabel': 'Choice B Delta (%)',
            'title': 'BC Rate vs Choice B Delta',
        },
    ]

    # Color by experiment type
    exp_types = df['exp_type'].unique()
    colors_map = {
        'Steering': 'blue',
        'Suppression': 'green',
        'System Prompt': 'red',
    }

    for idx, (ax, config) in enumerate(zip(axes.flatten(), scatter_configs)):
        for exp_type in exp_types:
            type_data = df[df['exp_type'] == exp_type]

            # Prepare error bars using seed-level std
            x_err = None
            y_err = None

            if std_data:
                x_std_col = f"{config['x']}_std"
                y_std_col = f"{config['y']}_std"

                x_stds = [std_data.get(name, {}).get(x_std_col, 0) for name in type_data['short_name']]
                y_stds = [std_data.get(name, {}).get(y_std_col, 0) for name in type_data['short_name']]

                if any(x_stds):
                    x_err = x_stds
                if any(y_stds):
                    y_err = y_stds

            # Plot error bars if available
            if x_err is not None or y_err is not None:
                ax.errorbar(
                    type_data[config['x']],
                    type_data[config['y']],
                    xerr=x_err,
                    yerr=y_err,
                    fmt='o',
                    label=exp_type,
                    color=colors_map.get(exp_type, 'gray'),
                    markersize=8,
                    alpha=0.6,
                    elinewidth=1.5,
                    capsize=3,
                    capthick=1.5,
                    markeredgecolor='black',
                    markeredgewidth=1.5
                )
            else:
                ax.scatter(
                    type_data[config['x']],
                    type_data[config['y']],
                    label=exp_type,
                    color=colors_map.get(exp_type, 'gray'),
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=1.5
                )

            # Annotate points
            for _, row in type_data.iterrows():
                ax.annotate(
                    row['short_name'],
                    (row[config['x']], row[config['y']]),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=7,
                    alpha=0.8
                )

        # Add baseline if available
        if config['y'] == 'bc_rate_wrt_deploy' and 'baseline_bc_rate' in df.columns:
            baseline = df['baseline_bc_rate'].mean()
            ax.axhline(baseline, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline BC={baseline:.1f}%')

        if config['y'] == 'choice_b_delta' and 'baseline_choice_b_delta' in df.columns:
            baseline = df['baseline_choice_b_delta'].mean()
            ax.axhline(baseline, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline={baseline:.1f}%')

        ax.set_xlabel(config['xlabel'], fontweight='bold', fontsize=11)
        ax.set_ylabel(config['ylabel'], fontweight='bold', fontsize=11)
        ax.set_title(config['title'], fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '1_all_experiments_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plots to {output_dir / '1_all_experiments_scatter.png'}")
    plt.close()


def plot_pareto_frontier(df, output_dir, bucket_df=None):
    """Create Pareto frontier for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Calculate seed-level std if bucket data available
    std_data = {}
    if bucket_df is not None:
        std_data = calculate_seed_level_std(bucket_df, df)

    pareto_configs = [
        {
            'x': 'suppressed_eval_awareness',
            'y': 'choice_b_delta',
            'xlabel': 'Suppressed Awareness (1-10)',
            'ylabel': 'Choice B Delta (%)',
            'title': 'Suppressed Awareness vs Choice B Delta (All Experiments)',
        },
        {
            'x': 'bc_rate_wrt_deploy',
            'y': 'success_rate_modified',
            'xlabel': 'BC Rate wrt Deploy (%)',
            'ylabel': 'Success Rate (%)',
            'title': 'BC Rate vs Success Rate (All Experiments)',
        },
    ]

    # Color by experiment type
    colors_map = {
        'Steering': 'blue',
        'Suppression': 'green',
        'System Prompt': 'red',
    }

    for idx, (ax, config) in enumerate(zip(axes, pareto_configs)):
        for exp_type in df['exp_type'].unique():
            type_data = df[df['exp_type'] == exp_type]

            # Sort by x-axis
            type_data_sorted = type_data.sort_values(config['x'])

            # Prepare error bars using seed-level std
            xerr = None
            yerr = None

            if std_data:
                x_std_col = f"{config['x']}_std"
                y_std_col = f"{config['y']}_std"

                x_stds = [std_data.get(name, {}).get(x_std_col, 0) for name in type_data_sorted['short_name']]
                y_stds = [std_data.get(name, {}).get(y_std_col, 0) for name in type_data_sorted['short_name']]

                if any(x_stds):
                    xerr = x_stds
                if any(y_stds):
                    yerr = y_stds

            # Plot with error bars
            ax.errorbar(
                type_data_sorted[config['x']],
                type_data_sorted[config['y']],
                xerr=xerr,
                yerr=yerr,
                fmt='o-',
                label=exp_type,
                color=colors_map.get(exp_type, 'gray'),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                elinewidth=1.5,
                capsize=3,
                capthick=1.5
            )

        # Add baseline
        if config['y'] == 'choice_b_delta' and 'baseline_choice_b_delta' in df.columns:
            baseline_x = df['baseline_eval_awareness'].mean()
            baseline_y = df['baseline_choice_b_delta'].mean()
            ax.scatter(baseline_x, baseline_y, s=400, alpha=0.8, edgecolors='red',
                      facecolors='none', linewidth=3, marker='*', label='Baseline', zorder=10)
            ax.annotate('Baseline', (baseline_x, baseline_y), xytext=(5, -15),
                       textcoords='offset points', fontsize=11, fontweight='bold', color='red')

        ax.set_xlabel(config['xlabel'], fontweight='bold', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '2_all_experiments_pareto.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto frontiers to {output_dir / '2_all_experiments_pareto.png'}")
    plt.close()


def plot_top_performers(df, output_dir, bucket_df=None):
    """Identify and plot top performers by different metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Calculate seed-level std if bucket data available
    std_data = {}
    if bucket_df is not None:
        std_data = calculate_seed_level_std(bucket_df, df)

    metrics = [
        ('bc_rate_wrt_deploy', 'Top 10 by BC Rate (Lower is Better)', True),
        ('suppressed_eval_awareness', 'Top 10 by Suppressed Awareness (Lower is Better)', True),
        ('success_rate_modified', 'Top 10 by Success Rate (Higher is Better)', False),
        ('choice_b_delta', 'Top 10 by Choice B Delta (Closer to 0 is Better)', None),
    ]

    for ax, (metric, title, ascending) in zip(axes.flatten(), metrics):
        if ascending is None:
            # For choice_b_delta, sort by absolute value
            top_df = df.iloc[df[metric].abs().nsmallest(10).index].copy()
        else:
            top_df = df.nsmallest(10, metric) if ascending else df.nlargest(10, metric)

        # Color by experiment type
        colors = [{'Steering': 'blue', 'Suppression': 'green', 'System Prompt': 'red'}.get(t, 'gray')
                  for t in top_df['exp_type']]

        y_pos = np.arange(len(top_df))

        # Use seed-level std if available
        xerr = None
        if std_data:
            std_col = f"{metric}_std"
            xerr_vals = [std_data.get(name, {}).get(std_col, 0) for name in top_df['short_name']]
            if any(xerr_vals):
                xerr = xerr_vals

        if xerr is not None:
            bars = ax.barh(y_pos, top_df[metric].values, xerr=xerr,
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                          error_kw={'elinewidth': 2, 'capsize': 4, 'capthick': 2})
        else:
            bars = ax.barh(y_pos, top_df[metric].values, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_df['short_name'].values, fontsize=9)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_df[metric].values)):
            ax.text(value + max(abs(top_df[metric].values))*0.02, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '3_top_performers.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved top performers to {output_dir / '3_top_performers.png'}")
    plt.close()


def plot_ranking_heatmap(df, output_dir):
    """Create ranking heatmap across all metrics."""
    metrics = [
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

    # Create ranking matrix
    ranking_data = np.zeros((len(metrics), len(df)))

    for i, metric in enumerate(metrics):
        if metric == 'choice_b_delta':
            # Rank by absolute value (closer to 0 is better)
            ranks = df[metric].abs().rank()
        elif metric in ['bc_rate_wrt_deploy', 'suppressed_eval_awareness']:
            # Lower is better
            ranks = df[metric].rank()
        else:
            # Higher is better
            ranks = df[metric].rank(ascending=False)

        ranking_data[i] = ranks.values

    # Normalize ranks to 0-1 (1 = best rank)
    normalized_ranks = 1 - (ranking_data - 1) / (len(df) - 1)

    # Sort experiments by average rank
    avg_ranks = normalized_ranks.mean(axis=0)
    sort_idx = np.argsort(-avg_ranks)  # Best performers first

    fig, ax = plt.subplots(figsize=(max(16, len(df)*0.6), 8))

    im = ax.imshow(normalized_ranks[:, sort_idx], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(df)))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(df.iloc[sort_idx]['short_name'].values, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(metric_labels, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Rank (1=best, 0=worst)', fontweight='bold', fontsize=10)

    # Add rank numbers
    for i in range(len(metric_labels)):
        for j in range(len(df)):
            rank = int(ranking_data[i, sort_idx[j]])
            text = ax.text(j, i, f'{rank}',
                          ha='center', va='center', color='black', fontsize=7, fontweight='bold')

    ax.set_title('Performance Rankings Across All Metrics (All Experiments)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '4_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved ranking heatmap to {output_dir / '4_ranking_heatmap.png'}")
    plt.close()


def calculate_seed_level_std(bucket_df, df):
    """Calculate seed-level standard errors (SE) for each experiment."""
    std_data = {}

    for exp_name in df['short_name'].unique():
        exp_data = bucket_df[bucket_df['short_name'] == exp_name]

        if len(exp_data) > 0:
            n = len(exp_data)
            # Calculate standard error (SE = std / sqrt(n))
            std_data[exp_name] = {
                'bc_rate_wrt_deploy_std': (exp_data['modified_choice'] != exp_data['deployment_choice']).astype(float).std() * 100 / np.sqrt(n),
                'suppressed_eval_awareness_std': exp_data['modified_awareness'].std() / np.sqrt(n),
                'choice_b_delta_std': ((exp_data['modified_choice'] == 'B').astype(float) -
                                       (exp_data['deployment_choice'] == 'B').astype(float)).std() * 100 / np.sqrt(n),
                'success_rate_modified_std': (exp_data['modified_awareness'] < 5).astype(float).std() * 100 / np.sqrt(n),
            }

    return std_data


def main():
    output_dir = Path('outputs/all_experiments_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*120)
    print("PLOTTING ALL EXPERIMENTS TOGETHER")
    print("="*120)
    print()

    csv_path = Path('outputs/qwen_qwen3-32b_all18_seed29/data/data/experiment_metrics.csv')

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Input:  {csv_path}")
    print(f"Output: {output_dir}/\n")

    # Load data
    df = load_and_filter_data(csv_path)

    print(f"Loaded {len(df)} experiments:")
    for exp_type in ['Steering', 'Suppression', 'System Prompt']:
        type_exps = df[df['exp_type'] == exp_type]['short_name'].values
        print(f"\n{exp_type} ({len(type_exps)}):")
        for exp in type_exps:
            print(f"  - {exp}")

    # Load bucket data for BC rate calculation
    bucket_df = load_bucket_data(csv_path)

    if bucket_df is not None:
        # Calculate BC rate wrt deployment
        bc_wrt_deployment = {}
        for exp_name in df['short_name']:
            exp_data = bucket_df[bucket_df['short_name'] == exp_name]
            if len(exp_data) > 0:
                bc_count = (exp_data['modified_choice'] != exp_data['deployment_choice']).sum()
                bc_rate = (bc_count / len(exp_data)) * 100
                bc_wrt_deployment[exp_name] = bc_rate

        df['bc_rate_wrt_deploy'] = df['short_name'].map(bc_wrt_deployment)

        # Calculate baselines
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

        # Choice B Delta
        orig_b_count = (df_unique['original_choice'] == 'B').sum()
        deploy_b_count = (df_unique['deployment_choice'] == 'B').sum()
        orig_b_rate = (orig_b_count / total_seeds_all) * 100
        deploy_b_rate = (deploy_b_count / total_seeds_all) * 100
        baseline_choice_b_delta = orig_b_rate - deploy_b_rate
        df['baseline_choice_b_delta'] = baseline_choice_b_delta

        print(f"\n✓ Baseline BC Rate: {baseline_bc_rate:.2f}%")
        print(f"✓ Baseline Choice B Delta: {baseline_choice_b_delta:+.2f}%")

    # Create plots
    print("\nGenerating visualizations...")
    create_summary_table(df, output_dir)
    plot_scatter_comparison(df, output_dir, bucket_df)
    plot_pareto_frontier(df, output_dir, bucket_df)
    plot_top_performers(df, output_dir, bucket_df)
    plot_ranking_heatmap(df, output_dir)

    print(f"\n{'='*120}")
    print(f"✓ ALL PLOTS SAVED TO: {output_dir}/")
    print(f"{'='*120}\n")

    print("Generated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
