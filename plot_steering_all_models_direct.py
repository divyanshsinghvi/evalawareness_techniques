#!/usr/bin/env python3
"""
Direct steering comparison across all models without requiring categorization.

Reads steered outputs and original rollouts directly to calculate BC Rate and Choice B Delta.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import re

# Set style
sns.set_palette("Set2")

# Model configurations
MODELS = {
    'nvidia_llama-3.3-nemotron-super-49b-v1.5': {
        'short_name': 'Nemotron-49B',
        'experiments': {
            'anthropic/eval/N4_S1.0': 'Anthropic (S1.0)',
            'tim/eval/N4_S3.5': 'Tim (S3.5)',
        }
    },
    'qwen_qwen3-30b-a3b-thinking-2507': {
        'short_name': 'Qwen3-30B',
        'experiments': {
            'anthropic/eval/N4_S1.0': 'Anthropic (S1.0)',
            'tim/eval/N4_S3.5': 'Tim (S3.5)',
        }
    },
    'qwen_qwen3-32b': {
        'short_name': 'Qwen3-32B',
        'experiments': {
            'anthropic/eval/N4_S1.0': 'Anthropic (S1.0)',
            'tim/eval/N4_S3.5': 'Tim (S3.5)',
            'tim/eval/N4_S3.5_random': 'Tim Random (S3.5)',
            'tim/deploy/N4_S-3.5': 'Tim Deploy (S-3.5)',
        }
    },
}


def extract_choice_from_response(response_text):
    """Extract boxed choice (A or B) from response text."""
    if not response_text:
        return None

    # Look for \boxed{A} or \boxed{B}
    match = re.search(r'\\boxed\{([AB])\}', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look for **A** or **B** at end
    match = re.search(r'\*\*([AB])\*\*\s*$', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def find_original_rollout(model_id, steered_file_path):
    """Find and read the original rollout file for a steered output."""
    # Try to get source_file from steered output
    try:
        with open(steered_file_path, 'r') as f:
            steered_data = yaml.safe_load(f)

        source_file = steered_data.get('source_file', '')
        if source_file:
            # Try multiple possible locations
            possible_paths = [
                Path('working/rollouts') / model_id / source_file,
                Path('extracted_prompts') / source_file,
            ]

            for path in possible_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        rollout_data = yaml.safe_load(f)

                    # Extract deployment choice
                    if 'deployment' in rollout_data:
                        deploy_response = rollout_data['deployment'].get('assistant_response', '')
                        return extract_choice_from_response(deploy_response)
    except Exception as e:
        pass

    return None


def analyze_steering_experiment(model_id, exp_path_suffix, exp_name):
    """Calculate BC rate and Choice B Delta for a steering experiment."""
    steered_dir = Path(f'working/steered-outs/{model_id}/high_awareness_bc/{exp_path_suffix}')

    if not steered_dir.exists():
        print(f"  ✗ Directory not found: {steered_dir}")
        return None

    steered_files = list(steered_dir.glob('*_steer_out.yaml'))

    if len(steered_files) == 0:
        print(f"  ✗ No steered output files found")
        return None

    bc_count = 0
    total_count = 0
    failed_extractions = 0
    deploy_b_count = 0
    steered_b_count = 0

    for steered_file in steered_files:
        try:
            with open(steered_file, 'r') as f:
                steered_data = yaml.safe_load(f)

            # Extract steered choice from full_response
            full_response = steered_data.get('full_response', '')
            steered_choice = extract_choice_from_response(full_response)

            if steered_choice is None:
                failed_extractions += 1
                continue

            # Get deployment choice from original rollout
            deployment_choice = find_original_rollout(model_id, steered_file)

            if deployment_choice is None:
                continue

            # Check if BC occurred
            if steered_choice != deployment_choice:
                bc_count += 1

            # Count B choices
            if deployment_choice == 'B':
                deploy_b_count += 1
            if steered_choice == 'B':
                steered_b_count += 1

            total_count += 1

        except Exception as e:
            continue

    if total_count == 0:
        print(f"  ✗ No valid comparisons found")
        return None

    bc_rate = (bc_count / total_count) * 100
    deploy_b_rate = (deploy_b_count / total_count) * 100
    steered_b_rate = (steered_b_count / total_count) * 100
    choice_b_delta = deploy_b_rate - steered_b_rate  # Positive = moving toward deployment

    print(f"  ✓ {exp_name}:")
    print(f"      BC Rate: {bc_rate:.2f}% ({bc_count}/{total_count})")
    print(f"      Choice B Delta: {choice_b_delta:+.2f}% (Deploy: {deploy_b_rate:.1f}%, Steered: {steered_b_rate:.1f}%)")
    if failed_extractions > 0:
        print(f"      Failed extractions: {failed_extractions}")

    return {
        'model_id': model_id,
        'experiment': exp_name,
        'bc_rate': bc_rate,
        'bc_count': bc_count,
        'total_count': total_count,
        'deploy_b_rate': deploy_b_rate,
        'steered_b_rate': steered_b_rate,
        'choice_b_delta': choice_b_delta,
        'failed_extractions': failed_extractions,
    }


def create_per_model_plots(model_results, output_dir):
    """Create separate plots for each model."""

    for model_id, model_config in MODELS.items():
        model_data = [r for r in model_results if r['model_id'] == model_id]

        if len(model_data) == 0:
            continue

        df = pd.DataFrame(model_data)

        # Create figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        experiments = df['experiment'].values
        n_experiments = len(experiments)

        # Color palette
        colors = sns.color_palette("Set2", n_experiments)

        # Plot 1: BC Rate
        ax = axes[0]
        bc_rates = df['bc_rate'].values
        bars = ax.bar(range(n_experiments), bc_rates, alpha=0.85, color=colors,
                       edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, value in zip(bars, bc_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('BC Rate wrt Deploy (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'BC Rate - {model_config["short_name"]}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(n_experiments))
        ax.set_xticklabels(experiments, rotation=15, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(bc_rates) * 1.15)

        # Plot 2: Choice B Delta
        ax = axes[1]
        choice_b_deltas = df['choice_b_delta'].values
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
        ax.set_title(f'Choice B Delta - {model_config["short_name"]}\n(Positive = Moving Toward Deployment)',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(range(n_experiments))
        ax.set_xticklabels(experiments, rotation=15, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Set y-limits to be symmetric around 0
        max_abs = max(abs(choice_b_deltas.min()), abs(choice_b_deltas.max())) if len(choice_b_deltas) > 0 else 20
        ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_id}_steering_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {model_id}_steering_comparison.png")
        plt.close()


def main():
    output_dir = Path('outputs/steering_per_model_direct')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("DIRECT STEERING COMPARISON - ALL MODELS")
    print("="*100)
    print()

    all_results = []

    for model_id, model_config in MODELS.items():
        print(f"Model: {model_config['short_name']} ({model_id})")
        print("-" * 100)

        for exp_path, exp_name in model_config['experiments'].items():
            result = analyze_steering_experiment(model_id, exp_path, exp_name)
            if result:
                result['model_name'] = model_config['short_name']
                all_results.append(result)

        print()

    if len(all_results) == 0:
        print("❌ No results found!")
        return

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV
    results_df.to_csv(output_dir / 'all_models_steering_metrics.csv', index=False)

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY - ALL MODELS")
    print("="*100)
    for model_id, model_config in MODELS.items():
        model_data = results_df[results_df['model_id'] == model_id]
        if len(model_data) > 0:
            print(f"\n{model_config['short_name']}:")
            print(model_data[['experiment', 'bc_rate', 'choice_b_delta']].to_string(index=False,
                  formatters={'bc_rate': lambda x: f'{x:.2f}%', 'choice_b_delta': lambda x: f'{x:+.2f}%'}))
    print("="*100)

    # Create per-model plots
    create_per_model_plots(all_results, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - all_models_steering_metrics.csv")
    print(f"  - {len(MODELS)} model comparison plots")
    print()


if __name__ == '__main__':
    main()
