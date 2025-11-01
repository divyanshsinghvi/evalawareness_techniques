#!/usr/bin/env python3
"""
Per-model steering comparison plots.

Creates separate plots for each model showing BC Rate and Choice B Delta
for all steering techniques available for that model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_palette("Set2")

# Model configuration - only qwen3-32b has complete data
MODEL = 'qwen_qwen3-32b'

# Steering experiments to include
STEERING_EXPERIMENTS = {
    'anthropic_N4_S1.0': 'Anthropic (S1.0)',
    'tim_N4_S3.5': 'Tim (S3.5)',
    'tim_N4_S3.5_random': 'Tim Random (S3.5)',
    'tim_deploy_N4_S-3.5': 'Tim Deploy (S-3.5)',
}


def load_and_analyze_data(model_id):
    """Load combined experiment data and calculate metrics."""
    csv_path = Path(f'outputs/{model_id}_all18_seed29/data/data/combined_all_experiments.csv')

    if not csv_path.exists():
        print(f"❌ Data file not found: {csv_path}")
        print(f"   Please run compare_experiment_variants.py for {model_id} first.")
        return None

    df = pd.read_csv(csv_path)

    # Filter to only steering experiments
    df_steering = df[df['experiment'].isin(STEERING_EXPERIMENTS.keys())].copy()

    if len(df_steering) == 0:
        print(f"❌ No steering experiments found in data")
        return None

    # Calculate metrics per experiment
    results = []

    for exp_name, exp_display in STEERING_EXPERIMENTS.items():
        exp_data = df_steering[df_steering['experiment'] == exp_name]

        if len(exp_data) == 0:
            continue

        # BC Rate wrt Deploy
        bc_count = (exp_data['modified_choice'] != exp_data['deployment_choice']).sum()
        bc_rate = (bc_count / len(exp_data)) * 100

        # Choice B rates
        deploy_b_count = (exp_data['deployment_choice'] == 'B').sum()
        modified_b_count = (exp_data['modified_choice'] == 'B').sum()

        deploy_b_rate = (deploy_b_count / len(exp_data)) * 100
        modified_b_rate = (modified_b_count / len(exp_data)) * 100
        choice_b_delta = deploy_b_rate - modified_b_rate  # Positive = moving toward deployment

        results.append({
            'experiment': exp_name,
            'experiment_display': exp_display,
            'bc_rate': bc_rate,
            'bc_count': bc_count,
            'total_count': len(exp_data),
            'deploy_b_rate': deploy_b_rate,
            'modified_b_rate': modified_b_rate,
            'choice_b_delta': choice_b_delta,
        })

        print(f"  ✓ {exp_display}:")
        print(f"      BC Rate: {bc_rate:.2f}% ({bc_count}/{len(exp_data)})")
        print(f"      Choice B Delta: {choice_b_delta:+.2f}% (Deploy: {deploy_b_rate:.1f}%, Steered: {modified_b_rate:.1f}%)")

    return pd.DataFrame(results)


def create_per_model_plots(results_df, model_id, output_dir):
    """Create two plots: BC Rate and Choice B Delta for this model."""

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    experiments = results_df['experiment_display'].values
    n_experiments = len(experiments)

    # Color palette
    colors = sns.color_palette("Set2", n_experiments)

    # Plot 1: BC Rate
    ax = axes[0]
    bc_rates = results_df['bc_rate'].values
    bars = ax.bar(range(n_experiments), bc_rates, alpha=0.85, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, value in zip(bars, bc_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('BC Rate wrt Deploy (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'BC Rate - {model_id}', fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_experiments))
    ax.set_xticklabels(experiments, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bc_rates) * 1.15)

    # Plot 2: Choice B Delta
    ax = axes[1]
    choice_b_deltas = results_df['choice_b_delta'].values
    bars = ax.bar(range(n_experiments), choice_b_deltas, alpha=0.85, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, value in zip(bars, choice_b_deltas):
        offset = 0.5 if value > 0 else -0.5
        va = 'bottom' if value > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
               f'{value:+.1f}%', ha='center', va=va, fontweight='bold', fontsize=11)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_ylabel('Choice B Delta (%)\n(Deploy - Steered)', fontweight='bold', fontsize=12)
    ax.set_title(f'Choice B Delta - {model_id}\n(Positive = Moving Toward Deployment)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_experiments))
    ax.set_xticklabels(experiments, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-limits to be symmetric around 0
    max_abs = max(abs(choice_b_deltas.min()), abs(choice_b_deltas.max()))
    ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_id}_steering_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved per-model comparison plot: {model_id}_steering_comparison.png")
    plt.close()


def main():
    output_dir = Path('outputs/steering_per_model_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print(f"PER-MODEL STEERING COMPARISON: {MODEL}")
    print("="*100)
    print()

    results_df = load_and_analyze_data(MODEL)

    if results_df is None or len(results_df) == 0:
        return

    # Save to CSV
    results_df.to_csv(output_dir / f'{MODEL}_steering_metrics.csv', index=False)

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(results_df[['experiment_display', 'bc_rate', 'choice_b_delta']].to_string(index=False,
          formatters={'bc_rate': lambda x: f'{x:.2f}%', 'choice_b_delta': lambda x: f'{x:+.2f}%'}))
    print("="*100)

    # Create plots
    create_per_model_plots(results_df, MODEL, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - {MODEL}_steering_metrics.csv")
    print(f"  - {MODEL}_steering_comparison.png")
    print()


if __name__ == '__main__':
    main()
