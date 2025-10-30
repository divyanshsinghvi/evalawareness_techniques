#!/usr/bin/env python3
"""
Boss script to compare awareness mitigation techniques across models with Pareto frontier analysis.

Compares 3 techniques (Suppression, Steering, Off-Policy) across multiple models.
Generates:
1. Per-model comparison (3 techniques per model)
2. Per-technique comparison (N models per technique)
3. Combined view (all models × all techniques)

Usage:
    python compare_techniques.py \
      --suppression-dir working/suppression_experiments/01_chunk_resample/.../  \
      --steering-dir working/steered_categorization/  \
      --offpolicy-dir working/off-policy-intervention/  \
      --output-dir outputs/technique_comparison
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from typing import Dict, List, Tuple
import json

# Color palette for models
MODEL_COLORS = plotly.colors.qualitative.Plotly

# Marker shapes for techniques
TECHNIQUE_MARKERS = {
    'suppression': 'circle',
    'steering': 'square',
    'offpolicy': 'triangle-up'
}

TECHNIQUE_NAMES = {
    'suppression': 'Suppression',
    'steering': 'Steering',
    'offpolicy': 'Off-Policy'
}


def auto_detect_models(suppression_dir: Path, steering_dir: Path, offpolicy_dir: Path, mode: str) -> List[str]:
    """Auto-detect models present in all three technique directories."""
    print(f"\n{'='*80}")
    print(f"AUTO-DETECTING MODELS (mode: {mode})")
    print(f"{'='*80}")

    # Find subdirectories in each technique dir
    supp_models = {d.name for d in suppression_dir.iterdir() if d.is_dir() and not d.name.startswith('.')}
    steer_models = {d.name for d in steering_dir.iterdir() if d.is_dir() and not d.name.startswith('.')}
    off_models = {d.name for d in offpolicy_dir.iterdir() if d.is_dir() and not d.name.startswith('.')}

    print(f"Suppression models: {supp_models}")
    print(f"Steering models: {steer_models}")
    print(f"Off-Policy models: {off_models}")

    # Find intersection (models present in all 3)
    common_models = supp_models & steer_models & off_models

    if not common_models:
        print("\n⚠️  WARNING: No models found in all 3 technique directories!")
        print("Using union of all models (some combinations may be missing)")
        common_models = supp_models | steer_models | off_models

    common_models = sorted(common_models)
    print(f"\n✓ Found {len(common_models)} models: {common_models}")
    print(f"{'='*80}\n")

    return common_models


def run_analysis_script(script_name: str, args_dict: Dict, output_csv: Path):
    """Run an analysis script and export to CSV."""
    # Build command
    cmd = ['python', script_name]
    for key, value in args_dict.items():
        if value is True:
            # Boolean flag - just add the flag
            cmd.append(f'--{key}')
        else:
            # Regular argument with value
            cmd.append(f'--{key}')
            cmd.append(str(value))
    cmd.extend(['--output-csv', str(output_csv)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"✓ Success")


def collect_data_for_all_combinations(
    models: List[str],
    suppression_dir: Path,
    steering_dir: Path,
    offpolicy_dir: Path,
    output_data_dir: Path,
    recursive: bool,
    mode: str
) -> pd.DataFrame:
    """Run analysis scripts for all model × technique combinations and merge data."""
    print(f"\n{'='*80}")
    print(f"COLLECTING DATA FOR ALL COMBINATIONS")
    print(f"{'='*80}\n")

    output_data_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = []

    for model in models:
        print(f"\nProcessing model: {model}")
        print(f"{'='*40}")

        # Suppression
        supp_dir = suppression_dir / model
        csv_path = output_data_dir / f"{model}_suppression.csv"
        args = {'experiment-dir': supp_dir}
        if recursive:
            args['recursive'] = True
        run_analysis_script('analyze_suppression_results.py', args, csv_path)
        df = pd.read_csv(csv_path)
        df['model'] = model
        df['technique'] = 'suppression'
        all_csvs.append(df)

        # Steering
        steer_dir = steering_dir / model
        csv_path = output_data_dir / f"{model}_steering.csv"
        # Find response_categorization dir (usually at working/response_categorization/{mode})
        # Navigate up from steering_dir to find working dir
        working_dir = steering_dir
        while working_dir.name != 'working' and working_dir.parent != working_dir:
            working_dir = working_dir.parent
        # Use the mode parameter for version
        # Note: response_cat_dir should NOT include model subdirectory
        response_cat_dir = working_dir / 'response_categorization' / mode
        args = {
            'categorization-dir': steer_dir,
            'response-categorization-dir': response_cat_dir
        }
        if recursive:
            args['recursive'] = True
        run_analysis_script('analyze_steered_results.py', args, csv_path)
        df = pd.read_csv(csv_path)
        df['model'] = model
        df['technique'] = 'steering'
        all_csvs.append(df)

        # Off-Policy
        off_dir = offpolicy_dir / model
        csv_path = output_data_dir / f"{model}_offpolicy.csv"
        args = {'experiment-dir': off_dir}
        if recursive:
            args['recursive'] = True
        try:
            run_analysis_script('analyze_off_policy_results.py', args, csv_path)
            df = pd.read_csv(csv_path)
            df['model'] = model
            df['technique'] = 'offpolicy'
            all_csvs.append(df)
        except Exception as e:
            print(f"⚠️  Warning: Off-policy analysis failed for {model}: {e}")
            print(f"   Continuing with other techniques...")

    if not all_csvs:
        raise ValueError("No data collected! Check that analysis scripts ran successfully.")

    # Combine all DataFrames
    combined_df = pd.concat(all_csvs, ignore_index=True)

    # Save combined data
    combined_path = output_data_dir / 'combined_all.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✓ Combined data saved to: {combined_path}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Models: {combined_df['model'].nunique()}")
    print(f"  Techniques: {combined_df['technique'].nunique()}")
    print(f"{'='*80}\n")

    return combined_df


def normalize_columns(df: pd.DataFrame, technique: str) -> pd.DataFrame:
    """Normalize column names across different techniques."""
    df = df.copy()

    # Define column mappings for each technique
    if technique == 'suppression':
        col_map = {
            'eval_orig_awareness': 'original_awareness',
            'eval_supp_awareness': 'modified_awareness',
            'deploy_awareness': 'deployment_awareness',
            'eval_orig_choice': 'original_choice',
            'eval_supp_choice': 'modified_choice',
            'deploy_choice': 'deployment_choice',
            'eval_orig_type': 'original_type',
            'eval_supp_type': 'modified_type',
            'eval_flipped': 'bc'
        }
    elif technique == 'steering':
        col_map = {
            'original_intensity': 'original_awareness',
            'steered_intensity': 'modified_awareness',
            'deployment_intensity': 'deployment_awareness',
            'original_choice': 'original_choice',
            'steered_choice': 'modified_choice',
            'deployment_choice': 'deployment_choice',
            'original_type': 'original_type',
            'steered_type': 'modified_type',
            'steered_bc': 'bc'
        }
    elif technique == 'offpolicy':
        col_map = {
            'intervention_awareness': 'modified_awareness',
            'intervention_choice': 'modified_choice',
            'intervention_type_category': 'modified_type',
            'intervention_bc': 'bc'
        }
    else:
        col_map = {}

    # For columns that need renaming, first check if target column exists
    # If it does and it's different from source, use the source data
    for old_name, new_name in col_map.items():
        if old_name in df.columns:
            if new_name in df.columns and old_name != new_name:
                # Drop the existing target column, we'll replace it with renamed source
                df = df.drop(columns=[new_name])
            df = df.rename(columns={old_name: new_name})

    return df


def compute_aggregated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated metrics for each (model, technique) combination.

    Aggregates by prompt_id to get prompt-level statistics, then computes
    overall metrics with clustering corrections.
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING AGGREGATED METRICS")
    print(f"{'='*80}\n")

    metrics_list = []

    for (model, technique), group in df.groupby(['model', 'technique']):
        print(f"Computing metrics for: {model} + {technique}")

        # Normalize columns
        group = normalize_columns(group, technique)

        # Debug: Check bc column
        if 'bc' in group.columns:
            print(f"  bc column type: {type(group['bc'])}, dtype: {group['bc'].dtype}")

        # Group by prompt to get prompt-level means
        prompt_groups = group.groupby('prompt_id')

        # 1. Awareness Reduction
        orig_aware = group['original_awareness'].values
        mod_aware = group['modified_awareness'].values
        reduction = orig_aware - mod_aware
        reduction_mean = np.mean(reduction)
        reduction_std = np.std(reduction, ddof=1)

        # Prompt-level for clustering
        prompt_reductions = prompt_groups.apply(lambda x: np.mean(x['original_awareness'] - x['modified_awareness']))
        reduction_se = stats.sem(prompt_reductions)
        reduction_ci_lower = reduction_mean - 1.96 * reduction_se
        reduction_ci_upper = reduction_mean + 1.96 * reduction_se

        # 2. Deployment Convergence
        deploy_aware = group['deployment_awareness'].values
        convergence = np.abs(mod_aware - deploy_aware)
        convergence_mean = np.mean(convergence)
        prompt_convergence = prompt_groups.apply(lambda x: np.mean(np.abs(x['modified_awareness'] - x['deployment_awareness'])))
        convergence_se = stats.sem(prompt_convergence)
        convergence_ci_lower = convergence_mean - 1.96 * convergence_se
        convergence_ci_upper = convergence_mean + 1.96 * convergence_se

        # 3. BC Rate
        if 'bc' in group.columns:
            # Convert to numeric to handle boolean/non-numeric values
            bc_numeric = pd.to_numeric(group['bc'], errors='coerce').fillna(0)
            bc_count = bc_numeric.sum()
            bc_rate = (bc_count / len(group)) * 100
            # Compute prompt-level BC rates
            prompt_bc_rates = prompt_groups.apply(lambda x: pd.to_numeric(x['bc'], errors='coerce').fillna(0).mean())
            bc_se = stats.sem(prompt_bc_rates) * 100 if len(prompt_bc_rates) > 1 else 0
            bc_ci_lower = bc_rate - 1.96 * bc_se
            bc_ci_upper = bc_rate + 1.96 * bc_se
        else:
            bc_rate = 0
            bc_ci_lower = 0
            bc_ci_upper = 0

        # 4. Task Completion
        valid_choices = group['modified_choice'].notna().sum()
        task_completion = (valid_choices / len(group)) * 100

        # 5. Overshoot Rate
        overshoot_count = (mod_aware < deploy_aware).sum()
        overshoot_rate = (overshoot_count / len(group)) * 100

        # 6. Self-Test Removal
        self_test_orig = (group['original_type'] == 'self_test').sum() if 'original_type' in group.columns else 0
        self_test_removed = ((group['original_type'] == 'self_test') & (group['modified_type'] != 'self_test')).sum() if 'original_type' in group.columns and 'modified_type' in group.columns else 0
        self_test_removal_rate = (self_test_removed / self_test_orig * 100) if self_test_orig > 0 else 0

        # 7. Consistency (inverse of std)
        consistency = 1 / reduction_std if reduction_std > 0 else 0

        # 8. Final Awareness
        final_awareness = np.mean(mod_aware)
        prompt_final = prompt_groups['modified_awareness'].mean()
        final_se = stats.sem(prompt_final)
        final_ci_lower = final_awareness - 1.96 * final_se
        final_ci_upper = final_awareness + 1.96 * final_se

        # 9. Success Rate (< 5)
        success_count = (mod_aware < 5).sum()
        success_rate = (success_count / len(group)) * 100

        # Clustering stats
        n_prompts = group['prompt_id'].nunique()
        n_seeds = len(group)

        # ICC (simplified)
        prompt_means = prompt_groups['modified_awareness'].mean()
        var_between = np.var(prompt_means, ddof=1) if n_prompts > 1 else 0
        var_within = prompt_groups['modified_awareness'].var(ddof=1).mean()
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
        n_eff = n_seeds / (1 + ((n_seeds / n_prompts) - 1) * icc) if icc > 0 else n_seeds

        metrics = {
            'model': model,
            'technique': technique,
            'n_prompts': n_prompts,
            'n_seeds': n_seeds,
            'icc': icc,
            'n_eff': n_eff,
            # Metric 1
            'awareness_reduction': reduction_mean,
            'awareness_reduction_ci_lower': reduction_ci_lower,
            'awareness_reduction_ci_upper': reduction_ci_upper,
            # Metric 2
            'deployment_convergence': convergence_mean,
            'deployment_convergence_ci_lower': convergence_ci_lower,
            'deployment_convergence_ci_upper': convergence_ci_upper,
            # Metric 3
            'bc_rate': bc_rate,
            'bc_rate_ci_lower': bc_ci_lower,
            'bc_rate_ci_upper': bc_ci_upper,
            # Metric 4
            'task_completion': task_completion,
            # Metric 5
            'overshoot_rate': overshoot_rate,
            # Metric 6
            'self_test_removal_rate': self_test_removal_rate,
            # Metric 7
            'consistency': consistency,
            # Metric 8
            'final_awareness': final_awareness,
            'final_awareness_ci_lower': final_ci_lower,
            'final_awareness_ci_upper': final_ci_upper,
            # Metric 9
            'success_rate': success_rate,
        }

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    print(f"\n✓ Computed metrics for {len(metrics_df)} combinations")
    print(metrics_df[['model', 'technique', 'awareness_reduction', 'bc_rate', 'task_completion']])
    print(f"{'='*80}\n")

    return metrics_df


def compute_pairwise_comparisons(combined_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise statistical comparisons between techniques.

    For each model, compare techniques pairwise using prompt-level t-tests.
    Returns a dataframe with comparison results including t-statistic, p-value, effect size.
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING PAIRWISE COMPARISONS")
    print(f"{'='*80}\n")

    comparison_list = []

    # Get list of all techniques and models
    techniques = combined_df['technique'].unique()
    models = combined_df['model'].unique()

    # Metrics to compare
    metric_configs = [
        ('awareness_reduction', 'original_awareness', 'modified_awareness', 'subtraction'),
        ('deployment_convergence', 'modified_awareness', 'deployment_awareness', 'abs_diff'),
        ('bc_rate', 'bc', None, 'rate'),
        ('final_awareness', 'modified_awareness', None, 'mean'),
    ]

    for model in models:
        model_df = combined_df[combined_df['model'] == model]

        # Compare all pairs of techniques
        for i, tech1 in enumerate(techniques):
            for tech2 in techniques[i+1:]:
                print(f"Comparing {model}: {tech1} vs {tech2}")

                tech1_df = model_df[model_df['technique'] == tech1].copy()
                tech2_df = model_df[model_df['technique'] == tech2].copy()

                # Normalize columns for both
                tech1_df = normalize_columns(tech1_df, tech1)
                tech2_df = normalize_columns(tech2_df, tech2)

                for metric_name, col1, col2, calc_type in metric_configs:
                    # Compute prompt-level statistics for each technique
                    if calc_type == 'subtraction':
                        tech1_prompts = tech1_df.groupby('prompt_id').apply(lambda x: np.mean(x[col1] - x[col2]))
                        tech2_prompts = tech2_df.groupby('prompt_id').apply(lambda x: np.mean(x[col1] - x[col2]))
                    elif calc_type == 'abs_diff':
                        tech1_prompts = tech1_df.groupby('prompt_id').apply(lambda x: np.mean(np.abs(x[col1] - x[col2])))
                        tech2_prompts = tech2_df.groupby('prompt_id').apply(lambda x: np.mean(np.abs(x[col1] - x[col2])))
                    elif calc_type == 'rate':
                        tech1_prompts = tech1_df.groupby('prompt_id').apply(lambda x: pd.to_numeric(x[col1], errors='coerce').fillna(0).mean())
                        tech2_prompts = tech2_df.groupby('prompt_id').apply(lambda x: pd.to_numeric(x[col1], errors='coerce').fillna(0).mean())
                    elif calc_type == 'mean':
                        tech1_prompts = tech1_df.groupby('prompt_id')[col1].mean()
                        tech2_prompts = tech2_df.groupby('prompt_id')[col1].mean()

                    # Skip if either has NaN or insufficient data
                    if tech1_prompts.isna().any() or tech2_prompts.isna().any() or len(tech1_prompts) < 2 or len(tech2_prompts) < 2:
                        continue

                    # Perform independent samples t-test
                    t_stat, p_value = stats.ttest_ind(tech1_prompts, tech2_prompts)

                    # Cohen's d effect size
                    pooled_std = np.sqrt((tech1_prompts.var() + tech2_prompts.var()) / 2)
                    cohens_d = (tech1_prompts.mean() - tech2_prompts.mean()) / pooled_std if pooled_std > 0 else 0

                    comparison = {
                        'model': model,
                        'technique1': tech1,
                        'technique2': tech2,
                        'metric': metric_name,
                        'tech1_mean': tech1_prompts.mean(),
                        'tech2_mean': tech2_prompts.mean(),
                        'difference': tech1_prompts.mean() - tech2_prompts.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'n_prompts_tech1': len(tech1_prompts),
                        'n_prompts_tech2': len(tech2_prompts),
                    }

                    comparison_list.append(comparison)

    if not comparison_list:
        print("⚠️  Warning: No comparisons could be computed")
        return pd.DataFrame()

    comparisons_df = pd.DataFrame(comparison_list)
    print(f"\n✓ Computed {len(comparisons_df)} pairwise comparisons")
    print(f"  Significant comparisons (p<0.05): {comparisons_df['significant'].sum()}")
    print(f"{'='*80}\n")

    return comparisons_df


def generate_per_model_plots(metrics_df: pd.DataFrame, comparisons_df: pd.DataFrame, output_dir: Path):
    """
    Generate per-model comparison plots (View 1).

    For each model, create Pareto frontier plots comparing all techniques.
    Each plot shows 2 metrics with techniques as points.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING PER-MODEL PLOTS (View 1)")
    print(f"{'='*80}\n")

    plots_dir = output_dir / 'plots' / 'per_model'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define Pareto plot configurations (x_metric, y_metric, description)
    plot_configs = [
        ('awareness_reduction', 'bc_rate', 'Effectiveness vs Behavior Change'),
        ('awareness_reduction', 'task_completion', 'Effectiveness vs Task Quality'),
        ('bc_rate', 'task_completion', 'Behavior Change vs Task Quality'),
        ('final_awareness', 'bc_rate', 'Final Awareness vs Behavior Change'),
        ('deployment_convergence', 'bc_rate', 'Convergence vs Behavior Change'),
    ]

    for model in metrics_df['model'].unique():
        print(f"Creating plots for model: {model}")

        model_metrics = metrics_df[metrics_df['model'] == model]

        # Create subplots for all configurations
        n_plots = len(plot_configs)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[config[2] for config in plot_configs],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        for idx, (x_metric, y_metric, title) in enumerate(plot_configs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Plot each technique
            for tech_idx, (tech, tech_data) in enumerate(model_metrics.groupby('technique')):
                x_val = tech_data[x_metric].values[0]
                y_val = tech_data[y_metric].values[0]

                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue

                # Get error bars if available
                x_ci_lower = tech_data.get(f'{x_metric}_ci_lower', pd.Series([x_val])).values[0]
                x_ci_upper = tech_data.get(f'{x_metric}_ci_upper', pd.Series([x_val])).values[0]
                y_ci_lower = tech_data.get(f'{y_metric}_ci_lower', pd.Series([y_val])).values[0]
                y_ci_upper = tech_data.get(f'{y_metric}_ci_upper', pd.Series([y_val])).values[0]

                # Calculate error bar lengths
                x_error = [x_val - x_ci_lower, x_ci_upper - x_val] if not pd.isna(x_ci_lower) else None
                y_error = [y_val - y_ci_lower, y_ci_upper - y_val] if not pd.isna(y_ci_lower) else None

                marker_symbol = TECHNIQUE_MARKERS.get(tech, 'circle')
                tech_name = TECHNIQUE_NAMES.get(tech, tech)

                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers',
                        marker=dict(
                            size=12,
                            symbol=marker_symbol,
                            line=dict(width=2, color='white')
                        ),
                        name=tech_name,
                        error_x=dict(type='data', array=[x_error[1]] if x_error else None, arrayminus=[x_error[0]] if x_error else None) if x_error else None,
                        error_y=dict(type='data', array=[y_error[1]] if y_error else None, arrayminus=[y_error[0]] if y_error else None) if y_error else None,
                        showlegend=(idx == 0),  # Only show legend for first plot
                        legendgroup=tech,
                        hovertemplate=f"<b>{tech_name}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=row, col=col
                )

            # Update axes
            fig.update_xaxes(title_text=x_metric.replace('_', ' ').title(), row=row, col=col)
            fig.update_yaxes(title_text=y_metric.replace('_', ' ').title(), row=row, col=col)

        # Update layout
        fig.update_layout(
            title_text=f"Technique Comparison: {model}",
            height=300 * n_rows,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Save HTML
        output_file = plots_dir / f"{model}_comparison.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    print(f"\n✓ Generated per-model plots in: {plots_dir}")
    print(f"{'='*80}\n")


def generate_per_technique_plots(metrics_df: pd.DataFrame, comparisons_df: pd.DataFrame, output_dir: Path):
    """
    Generate per-technique comparison plots (View 2).

    For each technique, create plots comparing all models.
    Each plot shows 2 metrics with models as points.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING PER-TECHNIQUE PLOTS (View 2)")
    print(f"{'='*80}\n")

    plots_dir = output_dir / 'plots' / 'per_technique'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define plot configurations
    plot_configs = [
        ('awareness_reduction', 'bc_rate', 'Effectiveness vs Behavior Change'),
        ('awareness_reduction', 'task_completion', 'Effectiveness vs Task Quality'),
        ('bc_rate', 'task_completion', 'Behavior Change vs Task Quality'),
        ('final_awareness', 'bc_rate', 'Final Awareness vs Behavior Change'),
        ('deployment_convergence', 'bc_rate', 'Convergence vs Behavior Change'),
    ]

    for technique in metrics_df['technique'].unique():
        print(f"Creating plots for technique: {technique}")

        tech_metrics = metrics_df[metrics_df['technique'] == technique]

        # Create subplots
        n_plots = len(plot_configs)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[config[2] for config in plot_configs],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        for idx, (x_metric, y_metric, title) in enumerate(plot_configs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Plot each model
            for model_idx, (model, model_data) in enumerate(tech_metrics.groupby('model')):
                x_val = model_data[x_metric].values[0]
                y_val = model_data[y_metric].values[0]

                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue

                # Get error bars if available
                x_ci_lower = model_data.get(f'{x_metric}_ci_lower', pd.Series([x_val])).values[0]
                x_ci_upper = model_data.get(f'{x_metric}_ci_upper', pd.Series([x_val])).values[0]
                y_ci_lower = model_data.get(f'{y_metric}_ci_lower', pd.Series([y_val])).values[0]
                y_ci_upper = model_data.get(f'{y_metric}_ci_upper', pd.Series([y_val])).values[0]

                # Calculate error bar lengths
                x_error = [x_val - x_ci_lower, x_ci_upper - x_val] if not pd.isna(x_ci_lower) else None
                y_error = [y_val - y_ci_lower, y_ci_upper - y_val] if not pd.isna(y_ci_lower) else None

                # Use color from palette
                color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]

                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        text=[model.split('_')[-1]],  # Short model name
                        textposition="top center",
                        name=model,
                        error_x=dict(type='data', array=[x_error[1]] if x_error else None, arrayminus=[x_error[0]] if x_error else None) if x_error else None,
                        error_y=dict(type='data', array=[y_error[1]] if y_error else None, arrayminus=[y_error[0]] if y_error else None) if y_error else None,
                        showlegend=(idx == 0),
                        legendgroup=model,
                        hovertemplate=f"<b>{model}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=row, col=col
                )

            # Update axes
            fig.update_xaxes(title_text=x_metric.replace('_', ' ').title(), row=row, col=col)
            fig.update_yaxes(title_text=y_metric.replace('_', ' ').title(), row=row, col=col)

        # Update layout
        tech_name = TECHNIQUE_NAMES.get(technique, technique)
        fig.update_layout(
            title_text=f"Model Comparison: {tech_name}",
            height=300 * n_rows,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Save HTML
        output_file = plots_dir / f"{technique}_models.html"
        fig.write_html(str(output_file))
        print(f"  Saved: {output_file}")

    print(f"\n✓ Generated per-technique plots in: {plots_dir}")
    print(f"{'='*80}\n")


def generate_combined_plots(metrics_df: pd.DataFrame, comparisons_df: pd.DataFrame, output_dir: Path):
    """
    Generate combined plots (View 3).

    Create plots showing all models × all techniques together.
    Uses different colors for models and different shapes for techniques.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED PLOTS (View 3)")
    print(f"{'='*80}\n")

    plots_dir = output_dir / 'plots' / 'combined'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define plot configurations
    plot_configs = [
        ('awareness_reduction', 'bc_rate', 'Effectiveness vs Behavior Change'),
        ('awareness_reduction', 'task_completion', 'Effectiveness vs Task Quality'),
        ('bc_rate', 'task_completion', 'Behavior Change vs Task Quality'),
        ('final_awareness', 'bc_rate', 'Final Awareness vs Behavior Change'),
        ('deployment_convergence', 'bc_rate', 'Convergence vs Behavior Change'),
    ]

    # Create a single figure with all plot configurations
    n_plots = len(plot_configs)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[config[2] for config in plot_configs],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # Get unique models and techniques
    models = sorted(metrics_df['model'].unique())
    techniques = sorted(metrics_df['technique'].unique())

    for idx, (x_metric, y_metric, title) in enumerate(plot_configs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Plot each model × technique combination
        for model_idx, model in enumerate(models):
            for tech in techniques:
                combo_data = metrics_df[(metrics_df['model'] == model) & (metrics_df['technique'] == tech)]

                if len(combo_data) == 0:
                    continue

                x_val = combo_data[x_metric].values[0]
                y_val = combo_data[y_metric].values[0]

                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue

                # Get error bars
                x_ci_lower = combo_data.get(f'{x_metric}_ci_lower', pd.Series([x_val])).values[0]
                x_ci_upper = combo_data.get(f'{x_metric}_ci_upper', pd.Series([x_val])).values[0]
                y_ci_lower = combo_data.get(f'{y_metric}_ci_lower', pd.Series([y_val])).values[0]
                y_ci_upper = combo_data.get(f'{y_metric}_ci_upper', pd.Series([y_val])).values[0]

                x_error = [x_val - x_ci_lower, x_ci_upper - x_val] if not pd.isna(x_ci_lower) else None
                y_error = [y_val - y_ci_lower, y_ci_upper - y_val] if not pd.isna(y_ci_lower) else None

                # Color by model, shape by technique
                color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
                marker_symbol = TECHNIQUE_MARKERS.get(tech, 'circle')
                tech_name = TECHNIQUE_NAMES.get(tech, tech)

                # Create legend label combining model and technique
                legend_label = f"{model} - {tech_name}"

                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=color,
                            symbol=marker_symbol,
                            line=dict(width=2, color='white')
                        ),
                        name=legend_label,
                        error_x=dict(type='data', array=[x_error[1]] if x_error else None, arrayminus=[x_error[0]] if x_error else None) if x_error else None,
                        error_y=dict(type='data', array=[y_error[1]] if y_error else None, arrayminus=[y_error[0]] if y_error else None) if y_error else None,
                        showlegend=(idx == 0),
                        legendgroup=f"{model}_{tech}",
                        hovertemplate=f"<b>{model} - {tech_name}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=row, col=col
                )

        # Update axes
        fig.update_xaxes(title_text=x_metric.replace('_', ' ').title(), row=row, col=col)
        fig.update_yaxes(title_text=y_metric.replace('_', ' ').title(), row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="Combined: All Models × All Techniques",
        height=300 * n_rows,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    # Save HTML
    output_file = plots_dir / "all_combinations.html"
    fig.write_html(str(output_file))
    print(f"  Saved: {output_file}")

    print(f"\n✓ Generated combined plots in: {plots_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare awareness mitigation techniques across models')
    parser.add_argument('--suppression-dir', type=str, required=True,
                       help='Directory containing suppression results (with model subdirectories)')
    parser.add_argument('--steering-dir', type=str, required=True,
                       help='Directory containing steering results (with model subdirectories)')
    parser.add_argument('--offpolicy-dir', type=str, required=True,
                       help='Directory containing off-policy results (with model subdirectories)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for all results')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search subdirectories in analysis scripts')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific models to compare (default: auto-detect all)')
    parser.add_argument('--mode', type=str, required=True, choices=['v0', 'v1'],
                       help='Version mode: v0 or v1 (required)')

    args = parser.parse_args()

    # Convert to Path objects
    suppression_dir = Path(args.suppression_dir)
    steering_dir = Path(args.steering_dir)
    offpolicy_dir = Path(args.offpolicy_dir)
    output_dir = Path(args.output_dir)

    # Verify directories exist
    for dir_path, name in [(suppression_dir, 'Suppression'),
                            (steering_dir, 'Steering'),
                            (offpolicy_dir, 'Off-Policy')]:
        if not dir_path.exists():
            print(f"Error: {name} directory not found: {dir_path}")
            return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Auto-detect models
    if args.models:
        models = args.models
        print(f"Using specified models: {models}")
    else:
        models = auto_detect_models(suppression_dir, steering_dir, offpolicy_dir, args.mode)

    if not models:
        print("Error: No models found or specified!")
        return 1

    # Step 2-3: Collect data for all combinations
    output_data_dir = output_dir / 'data'
    combined_df = collect_data_for_all_combinations(
        models,
        suppression_dir,
        steering_dir,
        offpolicy_dir,
        output_data_dir,
        args.recursive,
        args.mode
    )

    # Step 4: Compute aggregated metrics
    metrics_df = compute_aggregated_metrics(combined_df)
    metrics_df.to_csv(output_data_dir / 'combined_metrics.csv', index=False)

    # Step 5: Compute pairwise statistical comparisons
    comparisons_df = compute_pairwise_comparisons(combined_df, metrics_df)
    if not comparisons_df.empty:
        comparisons_df.to_csv(output_data_dir / 'pairwise_comparisons.csv', index=False)

    # Step 6: Generate plots
    generate_per_model_plots(metrics_df, comparisons_df, output_dir)
    generate_per_technique_plots(metrics_df, comparisons_df, output_dir)
    generate_combined_plots(metrics_df, comparisons_df, output_dir)

    # TODO: Step 7 (summaries)

    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"  Data: {output_data_dir}")
    print(f"  Combined data: {len(combined_df)} rows")
    print(f"  Aggregated metrics: {len(metrics_df)} combinations")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
