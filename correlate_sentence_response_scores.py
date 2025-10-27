#!/usr/bin/env python3
"""
Correlate sentence-level scores with response-level awareness scores.
Separates analysis for other_test and self_test responses.
"""

import yaml
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_yaml(filepath):
    """Load YAML file safely."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def extract_file_identifier(filepath):
    """Extract identifier from filepath (scenario_timestamp_hash_seed)."""
    parts = Path(filepath).stem.split('_seed_')
    if len(parts) == 2:
        base = parts[0]
        seed = parts[1]
        return f"{base}_seed_{seed}"
    return Path(filepath).stem

def aggregate_sentence_scores(categorized_sentences):
    """Aggregate sentence-level scores into summary statistics."""
    if not categorized_sentences:
        return {
            'mean_score': 0,
            'max_score': 0,
            'min_score': 0,
            'median_score': 0,
            'high_score_count': 0,  # score >= 4
            'very_high_score_count': 0,  # score == 5
            'total_sentences': 0,
            'high_score_ratio': 0,
        }

    scores = [s.get('score', 0) for s in categorized_sentences if isinstance(s, dict)]

    return {
        'mean_score': np.mean(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
        'median_score': np.median(scores) if scores else 0,
        'high_score_count': sum(1 for s in scores if s >= 4),
        'very_high_score_count': sum(1 for s in scores if s == 5),
        'total_sentences': len(scores),
        'high_score_ratio': sum(1 for s in scores if s >= 4) / len(scores) if scores else 0,
    }

def collect_data(categorization_dir, response_categorization_dir, model_name):
    """Collect and match data from both directories."""
    cat_dir = Path(categorization_dir) / model_name
    resp_dir = Path(response_categorization_dir) / model_name

    # Collect sentence-level data
    sentence_data = {}
    for yaml_file in cat_dir.rglob('*.yaml'):
        if yaml_file.is_file():
            try:
                data = load_yaml(yaml_file)
                identifier = extract_file_identifier(yaml_file)

                # Get deployment categorized_sentences
                deployment_sentences = data.get('deployment', {}).get('categorized_sentences', [])
                evaluation_sentences = data.get('evaluation', {}).get('categorized_sentences', [])

                sentence_data[identifier] = {
                    'deployment': aggregate_sentence_scores(deployment_sentences),
                    'evaluation': aggregate_sentence_scores(evaluation_sentences),
                    'file': str(yaml_file)
                }
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

    # Collect response-level data and match
    matched_data = []
    for yaml_file in resp_dir.rglob('*.yaml'):
        if yaml_file.is_file():
            try:
                data = load_yaml(yaml_file)
                identifier = extract_file_identifier(yaml_file)

                # Get deployment and evaluation awareness
                deployment_awareness = data.get('deployment', {}).get('awareness', {})
                evaluation_awareness = data.get('evaluation', {}).get('awareness', {})

                # Match with sentence data
                if identifier in sentence_data:
                    matched_data.append({
                        'identifier': identifier,
                        'deployment_awareness_type': deployment_awareness.get('awareness_type', 'unknown'),
                        'deployment_awareness_intensity': deployment_awareness.get('awareness_intensity', 0),
                        'deployment_awareness_level': deployment_awareness.get('awareness_level', 'Unknown'),
                        'evaluation_awareness_type': evaluation_awareness.get('awareness_type', 'unknown'),
                        'evaluation_awareness_intensity': evaluation_awareness.get('awareness_intensity', 0),
                        'evaluation_awareness_level': evaluation_awareness.get('awareness_level', 'Unknown'),
                        'deployment_mean_score': sentence_data[identifier]['deployment']['mean_score'],
                        'deployment_max_score': sentence_data[identifier]['deployment']['max_score'],
                        'deployment_high_score_ratio': sentence_data[identifier]['deployment']['high_score_ratio'],
                        'deployment_high_score_count': sentence_data[identifier]['deployment']['high_score_count'],
                        'deployment_total_sentences': sentence_data[identifier]['deployment']['total_sentences'],
                        'evaluation_mean_score': sentence_data[identifier]['evaluation']['mean_score'],
                        'evaluation_max_score': sentence_data[identifier]['evaluation']['max_score'],
                        'evaluation_high_score_ratio': sentence_data[identifier]['evaluation']['high_score_ratio'],
                        'evaluation_high_score_count': sentence_data[identifier]['evaluation']['high_score_count'],
                        'evaluation_total_sentences': sentence_data[identifier]['evaluation']['total_sentences'],
                    })
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

    return pd.DataFrame(matched_data)

def compute_correlations(df, test_type, phase):
    """Compute correlations for a specific test type and phase (deployment/evaluation)."""
    if df.empty:
        return None

    awareness_col = f'{phase}_awareness_intensity'
    mean_col = f'{phase}_mean_score'
    max_col = f'{phase}_max_score'
    ratio_col = f'{phase}_high_score_ratio'
    count_col = f'{phase}_high_score_count'

    results = {
        'test_type': test_type,
        'phase': phase,
        'n_samples': len(df),
        'correlations': {}
    }

    # Compute correlations
    for score_col, label in [
        (mean_col, 'mean_score'),
        (max_col, 'max_score'),
        (ratio_col, 'high_score_ratio'),
        (count_col, 'high_score_count')
    ]:
        if df[score_col].nunique() > 1 and df[awareness_col].nunique() > 1:
            pearson_r, pearson_p = stats.pearsonr(df[score_col], df[awareness_col])
            spearman_r, spearman_p = stats.spearmanr(df[score_col], df[awareness_col])

            results['correlations'][label] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            }

    return results

def plot_correlations(df, test_type, phase, output_dir):
    """Create scatter plots for correlations."""
    awareness_col = f'{phase}_awareness_intensity'

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{test_type.replace("_", " ").title()} - {phase.title()} Phase', fontsize=16)

    plots = [
        (f'{phase}_mean_score', 'Mean Sentence Score', axes[0, 0]),
        (f'{phase}_max_score', 'Max Sentence Score', axes[0, 1]),
        (f'{phase}_high_score_ratio', 'High Score Ratio (>=4)', axes[1, 0]),
        (f'{phase}_high_score_count', 'High Score Count (>=4)', axes[1, 1]),
    ]

    for score_col, label, ax in plots:
        if df[score_col].nunique() > 1 and df[awareness_col].nunique() > 1:
            ax.scatter(df[score_col], df[awareness_col], alpha=0.6)

            # Add regression line
            z = np.polyfit(df[score_col], df[awareness_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[score_col].min(), df[score_col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Add correlation
            r, p_val = stats.pearsonr(df[score_col], df[awareness_col])
            ax.set_title(f'{label}\nPearson r={r:.3f}, p={p_val:.3e}')
            ax.set_xlabel(label)
            ax.set_ylabel('Awareness Intensity')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient variance',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)

    plt.tight_layout()
    output_path = Path(output_dir) / f'correlation_{test_type}_{phase}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def main():
    # Configuration
    categorization_dir = 'working/categorization'
    response_categorization_dir = 'working/response_categorization'
    output_dir = 'working/correlation_analysis'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all models
    models = [d.name for d in Path(categorization_dir).iterdir() if d.is_dir()]

    print(f"Found {len(models)} models: {models}")

    all_results = []

    for model in models:
        print(f"\nProcessing model: {model}")

        # Collect data
        df = collect_data(categorization_dir, response_categorization_dir, model)
        print(f"  Matched {len(df)} files")

        if df.empty:
            print(f"  No data found for {model}")
            continue

        # Separate by test type for evaluation phase
        for test_type in ['self_test', 'other_test']:
            df_test = df[df['evaluation_awareness_type'] == test_type]

            if df_test.empty:
                print(f"  No {test_type} data")
                continue

            print(f"\n  {test_type}: {len(df_test)} samples")

            # Compute correlations for evaluation phase
            results = compute_correlations(df_test, test_type, 'evaluation')
            if results:
                results['model'] = model
                all_results.append(results)

                # Print results
                print(f"    Correlations with awareness_intensity:")
                for metric, corr in results['correlations'].items():
                    print(f"      {metric}:")
                    print(f"        Pearson:  r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.3e}")
                    print(f"        Spearman: r={corr['spearman_r']:.3f}, p={corr['spearman_p']:.3e}")

                # Create plots
                plot_correlations(df_test, test_type, 'evaluation', output_dir)

        # Also analyze deployment phase (baseline)
        print(f"\n  Deployment phase (baseline): {len(df)} samples")
        results = compute_correlations(df, 'baseline', 'deployment')
        if results:
            results['model'] = model
            all_results.append(results)
            plot_correlations(df, 'baseline', 'deployment', output_dir)

        # Save full dataset for this model
        df.to_csv(Path(output_dir) / f'{model}_full_data.csv', index=False)
        print(f"  Saved full data to {model}_full_data.csv")

    # Save summary results
    summary_path = Path(output_dir) / 'correlation_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"\nSaved summary to {summary_path}")

    # Create summary table
    summary_rows = []
    for result in all_results:
        for metric, corr in result['correlations'].items():
            summary_rows.append({
                'model': result['model'],
                'test_type': result['test_type'],
                'phase': result['phase'],
                'n_samples': result['n_samples'],
                'metric': metric,
                'pearson_r': corr['pearson_r'],
                'pearson_p': corr['pearson_p'],
                'spearman_r': corr['spearman_r'],
                'spearman_p': corr['spearman_p'],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(output_dir) / 'correlation_summary.csv', index=False)
    print(f"Saved summary table to correlation_summary.csv")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
