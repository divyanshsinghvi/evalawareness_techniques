#!/usr/bin/env python3
"""
Compare multiple experiment variants (suppression + steering) for a single model.

Generates comprehensive comparison plots showing:
- Pareto frontiers for trade-off analysis
- Bar charts with statistical significance
- Per-awareness-type breakdown
- Performance heatmaps
- Statistical significance matrices

Usage:
    python compare_experiment_variants.py \
      --model qwen_qwen3-32b \
      --suppression-experiments \
        "chunk_resample:working/suppression_experiments/01_chunk_resample/..." \
        "remove_only:working/suppression_experiments/02_remove_only/..." \
      --steering-experiments \
        "tim_N4_S3.5:working/steered-outs/qwen_qwen3-32b/.../N4_S3.5" \
      --mode v1 \
      --output-dir outputs/experiment_comparison
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from typing import Dict, List, Tuple
import json
import yaml

# Color schemes
SUPPRESSION_COLORS = ['#1f77b4', '#4e9ad1', '#82bcde']  # Blue shades
STEERING_COLORS = ['#ff7f0e', '#ffb366', '#ffd699']     # Orange shades

def run_analysis_script(script_name: str, args_dict: Dict, output_csv: Path):
    """Run an analysis script with given arguments."""
    cmd = ['python', script_name]
    for key, value in args_dict.items():
        if value is True:
            cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    cmd.extend(['--output-csv', str(output_csv)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"✓ Success")


def extract_seed_from_filename(filename: str) -> int:
    """Extract seed number from filename (e.g., 'prompt_seed_10_supp_0.yaml' -> 10)."""
    import re
    match = re.search(r'seed_(\d+)', filename)
    return int(match.group(1)) if match else 0


def standardize_suppression_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize suppression columns to common names."""
    rename_map = {
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

    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Extract seed from filename
    df['seed'] = df['file'].apply(extract_seed_from_filename)

    # Keep only relevant columns
    keep_cols = ['file', 'prompt_id', 'seed', 'experiment', 'technique', 'experiment_name',
                 'original_awareness', 'modified_awareness', 'deployment_awareness',
                 'original_choice', 'modified_choice', 'deployment_choice',
                 'original_type', 'modified_type', 'bc',
                 'seed_removed_sentences', 'seed_removed_pct']

    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def standardize_steering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize steering columns to common names."""
    rename_map = {
        'original_intensity': 'original_awareness',
        'steered_intensity': 'modified_awareness',
        'deployment_intensity': 'deployment_awareness',
        'steered_choice': 'modified_choice',
        'steered_type': 'modified_type',
        'steered_bc': 'bc'
    }

    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Extract seed from filename
    df['seed'] = df['file'].apply(extract_seed_from_filename)

    # Keep only relevant columns
    keep_cols = ['file', 'prompt_id', 'seed', 'experiment', 'technique', 'experiment_name',
                 'original_awareness', 'modified_awareness', 'deployment_awareness',
                 'original_choice', 'modified_choice', 'deployment_choice',
                 'original_type', 'modified_type', 'bc',
                 'n_value', 's_value', 'awareness_reduction']

    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def process_single_experiment(
    exp_tuple: Tuple[str, str, Path, Path, str, str]
) -> Tuple[bool, pd.DataFrame, str]:
    """
    Process a single experiment (suppression or steering).

    Args:
        exp_tuple: (exp_name, technique, exp_dir, output_data_dir, model, mode)

    Returns:
        (success, df, exp_name)
    """
    exp_name, technique, exp_dir, output_data_dir, model, mode = exp_tuple

    try:
        if technique == 'suppression':
            csv_path = output_data_dir / f"supp_{exp_name}.csv"
            full_path = exp_dir / model if not str(exp_dir).endswith(model) else exp_dir
            args = {
                'experiment-dir': full_path,
                'recursive': True,
                'experiment-name': exp_name
            }
            run_analysis_script('analyze_suppression_results.py', args, csv_path)
            df = pd.read_csv(csv_path)
            df['technique'] = 'suppression'
            df['experiment'] = exp_name
            df = standardize_suppression_columns(df)
            return (True, df, exp_name)

        else:  # steering
            csv_path = output_data_dir / f"steer_{exp_name}.csv"

            # Find working dir and build paths
            working_dir = exp_dir
            while working_dir.name != 'working' and working_dir.parent != working_dir:
                working_dir = working_dir.parent

            # Get the relative path from working dir to exp_dir
            try:
                rel_path = exp_dir.relative_to(working_dir / 'steered-outs' / model)
            except ValueError:
                rel_path = Path(*exp_dir.parts[-4:])

            # Steered categorizations are in working/steered_categorization/v1/model/...
            steered_cat_dir = working_dir / 'steered_categorization' / mode / model / rel_path

            # Original response categorizations for comparison
            response_cat_dir = working_dir / 'response_categorization' / mode

            args = {
                'categorization-dir': steered_cat_dir,
                'response-categorization-dir': response_cat_dir,
                'recursive': True,
                'experiment-name': exp_name
            }

            run_analysis_script('analyze_steered_results.py', args, csv_path)
            df = pd.read_csv(csv_path)
            df['technique'] = 'steering'
            df['experiment'] = exp_name
            df = standardize_steering_columns(df)
            return (True, df, exp_name)

    except Exception as e:
        print(f"\n❌ ERROR processing {technique} experiment '{exp_name}':")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, None, exp_name)


def load_bucket_mapping(model: str, mode: str) -> Dict[str, str]:
    """Load prompt_id → bucket mapping from high_awareness_bc_seeds.yaml."""
    bucket_file = Path(f"working/response_categorization/{mode}/{model}/high_awareness_bc_seeds.yaml")

    if not bucket_file.exists():
        print(f"  ⚠️  Warning: Bucket file not found: {bucket_file}")
        return {}

    try:
        with open(bucket_file, 'r') as f:
            data = yaml.safe_load(f)

        prompt_to_bucket = {}
        buckets = data.get('eval_awareness_buckets', {})

        for bucket_name, prompts in buckets.items():
            for prompt_id in prompts.keys():
                prompt_to_bucket[prompt_id] = bucket_name

        print(f"  ✓ Loaded {len(prompt_to_bucket)} prompts from bucket file")
        print(f"    Bucket distribution: {dict(sorted([(b, list(prompt_to_bucket.values()).count(b)) for b in set(prompt_to_bucket.values())]))}")
        return prompt_to_bucket

    except Exception as e:
        print(f"  ⚠️  Warning: Failed to load bucket file: {e}")
        return {}


def collect_all_experiments(
    model: str,
    suppression_experiments: List[Tuple[str, Path]],
    steering_experiments: List[Tuple[str, Path]],
    output_data_dir: Path,
    mode: str,
    limit_seeds: int = None
) -> pd.DataFrame:
    """Collect data from all experiment variants using parallel processing."""
    print(f"\n{'='*80}")
    print(f"COLLECTING DATA FROM ALL EXPERIMENTS")
    print(f"{'='*80}\n")

    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Prepare experiment tuples for parallel processing
    experiment_tuples = []

    # Add suppression experiments
    for exp_name, exp_dir in suppression_experiments:
        experiment_tuples.append((exp_name, 'suppression', exp_dir, output_data_dir, model, mode))

    # Add steering experiments
    for exp_name, exp_dir in steering_experiments:
        experiment_tuples.append((exp_name, 'steering', exp_dir, output_data_dir, model, mode))

    # Process experiments in parallel
    all_dfs = []
    max_workers = min(os.cpu_count() or 4, len(experiment_tuples))

    print(f"Processing {len(experiment_tuples)} experiments in parallel (max workers: {max_workers})...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_exp = {executor.submit(process_single_experiment, exp_tuple): exp_tuple
                        for exp_tuple in experiment_tuples}

        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp_tuple = future_to_exp[future]
            exp_name = exp_tuple[0]
            technique = exp_tuple[1]

            try:
                success, df, exp_name = future.result()
                if success:
                    print(f"✓ Success: {technique} - {exp_name}")
                    all_dfs.append(df)
                else:
                    print(f"⚠️  Warning: Failed to process {technique} - {exp_name}")
            except Exception as e:
                print(f"⚠️  Warning: Exception processing {technique} - {exp_name}: {e}")

    if not all_dfs:
        raise ValueError("No experiment data collected!")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Apply seed limit if specified
    if limit_seeds is not None:
        print(f"\n⚙️  Applying seed limit: keeping seeds 0-{limit_seeds}")
        original_rows = len(combined_df)
        combined_df = combined_df[combined_df['seed'] <= limit_seeds].copy()
        print(f"  Filtered: {original_rows} -> {len(combined_df)} rows")

    # Load and apply bucket mapping
    print("\n⚙️  Loading bucket mapping...")
    bucket_mapping = load_bucket_mapping(model, mode)
    if bucket_mapping:
        combined_df['bucket'] = combined_df['prompt_id'].map(bucket_mapping)
        combined_df['bucket'] = combined_df['bucket'].fillna('unknown')
        print(f"  ✓ Added bucket column")
        bucket_counts = combined_df['bucket'].value_counts().to_dict()
        print(f"    Bucket distribution in data: {dict(sorted(bucket_counts.items()))}")
    else:
        combined_df['bucket'] = 'unknown'
        print(f"  ⚠️  No bucket mapping available, all marked as 'unknown'")

    combined_path = output_data_dir / 'combined_all_experiments.csv'
    combined_df.to_csv(combined_path, index=False)

    print(f"\n✓ Combined data saved: {combined_path}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Experiments: {combined_df['experiment'].nunique()}")
    if 'seed' in combined_df.columns:
        print(f"  Seed range: {combined_df['seed'].min()}-{combined_df['seed'].max()}")
    print(f"{'='*80}\n")

    return combined_df


def normalize_columns_for_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names between suppression and steering."""
    df = df.copy()

    technique = df['technique'].iloc[0] if len(df) > 0 else None

    if technique == 'suppression':
        # Only rename columns that exist in this dataframe
        rename_map = {}
        if 'eval_orig_awareness' in df.columns:
            rename_map['eval_orig_awareness'] = 'original_awareness'
        if 'eval_supp_awareness' in df.columns:
            rename_map['eval_supp_awareness'] = 'modified_awareness'
        if 'deploy_awareness' in df.columns:
            rename_map['deploy_awareness'] = 'deployment_awareness'
        if 'eval_orig_choice' in df.columns:
            rename_map['eval_orig_choice'] = 'original_choice'
        if 'eval_supp_choice' in df.columns:
            rename_map['eval_supp_choice'] = 'modified_choice'
        if 'deploy_choice' in df.columns:
            rename_map['deploy_choice'] = 'deployment_choice'
        if 'eval_orig_type' in df.columns:
            rename_map['eval_orig_type'] = 'original_type'
        if 'eval_supp_type' in df.columns:
            rename_map['eval_supp_type'] = 'modified_type'
        if 'eval_flipped' in df.columns:
            rename_map['eval_flipped'] = 'bc'

        df = df.rename(columns=rename_map)

        # Drop steering-specific columns that shouldn't be in suppression data
        cols_to_drop = ['steered_intensity', 'steered_type', 'steered_choice', 'steered_bc',
                        'original_intensity', 'deployment_intensity', 'steered_level_label',
                        'original_level_label', 'steered_behavioral_impact', 'original_bc']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    elif technique == 'steering':
        # Only rename columns that exist in this dataframe
        rename_map = {}
        if 'original_intensity' in df.columns:
            rename_map['original_intensity'] = 'original_awareness'
        if 'steered_intensity' in df.columns:
            rename_map['steered_intensity'] = 'modified_awareness'
        if 'deployment_intensity' in df.columns:
            rename_map['deployment_intensity'] = 'deployment_awareness'
        if 'steered_choice' in df.columns:
            rename_map['steered_choice'] = 'modified_choice'
        if 'steered_type' in df.columns:
            rename_map['steered_type'] = 'modified_type'
        if 'steered_bc' in df.columns:
            rename_map['steered_bc'] = 'bc'

        df = df.rename(columns=rename_map)

        # Drop suppression-specific columns that shouldn't be in steering data
        cols_to_drop = ['eval_orig_awareness', 'eval_supp_awareness', 'eval_reduction',
                        'eval_orig_choice', 'eval_supp_choice', 'eval_flipped',
                        'eval_orig_type', 'eval_supp_type', 'seed_removed_sentences',
                        'seed_removed_pct', 'deploy_awareness', 'deploy_choice']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def compute_experiment_metrics(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated metrics for each experiment with clustering corrections."""
    print(f"\n{'='*80}")
    print(f"COMPUTING EXPERIMENT METRICS")
    print(f"{'='*80}\n")

    metrics_list = []

    for (experiment, technique), group in combined_df.groupby(['experiment', 'technique']):
        print(f"Computing metrics for: {experiment} ({technique})")

        group = group.reset_index(drop=True)  # Fix duplicate index issues
        prompt_groups = group.groupby('prompt_id')

        # Core metrics with clustering
        orig_aware = group['original_awareness'].values
        mod_aware = group['modified_awareness'].values
        deploy_aware = group['deployment_awareness'].values

        # Awareness reduction - mean per prompt, then diff of means
        prompt_reductions = prompt_groups.apply(
            lambda x: x['original_awareness'].mean() - x['modified_awareness'].mean(),
            include_groups=False
        )
        reduction_mean = np.mean(prompt_reductions)
        reduction_se = stats.sem(prompt_reductions)
        reduction_ci_lower = reduction_mean - 1.96 * reduction_se
        reduction_ci_upper = reduction_mean + 1.96 * reduction_se

        # BC rate
        if 'bc' in group.columns:
            bc_numeric = pd.to_numeric(group['bc'], errors='coerce').fillna(0)
            bc_rate = (bc_numeric.sum() / len(group)) * 100
            prompt_bc_rates = prompt_groups.apply(lambda x: pd.to_numeric(x['bc'], errors='coerce').fillna(0).mean(), include_groups=False)
            bc_se = stats.sem(prompt_bc_rates) * 100
            bc_ci_lower = bc_rate - 1.96 * bc_se
            bc_ci_upper = bc_rate + 1.96 * bc_se

            # Per-prompt BC rates (modified/steered BC)
            bc_per_prompt_mean = prompt_bc_rates.mean() * 100
            bc_per_prompt_std = prompt_bc_rates.std() * 100
        else:
            bc_rate = bc_ci_lower = bc_ci_upper = 0
            bc_per_prompt_mean = bc_per_prompt_std = 0

        # Original BC rate per prompt (prompts where original_choice == 'B')
        if 'original_choice' in group.columns:
            prompt_orig_bc = prompt_groups.apply(
                lambda x: (x['original_choice'] == 'B').mean() if len(x) > 0 else 0,
                include_groups=False
            )
            orig_bc_per_prompt_mean = prompt_orig_bc.mean() * 100
            orig_bc_per_prompt_std = prompt_orig_bc.std() * 100
        else:
            orig_bc_per_prompt_mean = orig_bc_per_prompt_std = 0

        # Choice distribution stats (% choosing A vs B)
        deploy_b_rate = deploy_a_rate = deploy_b_ci_lower = deploy_b_ci_upper = None
        orig_b_rate = orig_a_rate = orig_b_ci_lower = orig_b_ci_upper = None
        mod_b_rate = mod_a_rate = mod_b_ci_lower = mod_b_ci_upper = None

        if 'deployment_choice' in group.columns:
            deploy_b = (group['deployment_choice'] == 'B').astype(int)
            deploy_b_rate = deploy_b.mean() * 100
            deploy_a_rate = 100 - deploy_b_rate
            deploy_b_se = np.sqrt(deploy_b.mean() * (1 - deploy_b.mean()) / len(deploy_b)) * 100 if len(deploy_b) > 0 else 0
            deploy_b_ci_lower = deploy_b_rate - 1.96 * deploy_b_se
            deploy_b_ci_upper = deploy_b_rate + 1.96 * deploy_b_se

        if 'original_choice' in group.columns:
            orig_b = (group['original_choice'] == 'B').astype(int)
            orig_b_rate = orig_b.mean() * 100
            orig_a_rate = 100 - orig_b_rate
            orig_b_se = np.sqrt(orig_b.mean() * (1 - orig_b.mean()) / len(orig_b)) * 100 if len(orig_b) > 0 else 0
            orig_b_ci_lower = orig_b_rate - 1.96 * orig_b_se
            orig_b_ci_upper = orig_b_rate + 1.96 * orig_b_se

        if 'modified_choice' in group.columns:
            mod_b = (group['modified_choice'] == 'B').astype(int)
            mod_b_rate = mod_b.mean() * 100
            mod_a_rate = 100 - mod_b_rate
            mod_b_se = np.sqrt(mod_b.mean() * (1 - mod_b.mean()) / len(mod_b)) * 100 if len(mod_b) > 0 else 0
            mod_b_ci_lower = mod_b_rate - 1.96 * mod_b_se
            mod_b_ci_upper = mod_b_rate + 1.96 * mod_b_se

        # Task completion
        task_completion = (group['modified_choice'].notna().sum() / len(group)) * 100

        # Original awareness (baseline)
        original_awareness = np.mean(orig_aware)

        # Final awareness
        final_awareness = np.mean(mod_aware)
        prompt_final = prompt_groups['modified_awareness'].mean()
        final_se = stats.sem(prompt_final)
        final_ci_lower = final_awareness - 1.96 * final_se
        final_ci_upper = final_awareness + 1.96 * final_se

        # Deployment convergence - Option B: mean per prompt, then diff of means
        prompt_convergence = prompt_groups.apply(
            lambda x: np.abs(x['modified_awareness'].mean() - x['deployment_awareness'].mean()),
            include_groups=False
        )
        convergence_mean = np.mean(prompt_convergence)
        convergence_se = stats.sem(prompt_convergence)
        convergence_ci_lower = convergence_mean - 1.96 * convergence_se
        convergence_ci_upper = convergence_mean + 1.96 * convergence_se

        # Awareness shift - overall means diff (not per-prompt averaged)
        awareness_shift = np.abs(np.mean(mod_aware) - np.mean(deploy_aware))

        # Success rate (< 5 awareness) - prompt-level
        prompt_success = prompt_groups['modified_awareness'].apply(lambda x: (x < 5).any())
        success_rate = prompt_success.mean() * 100
        success_se = stats.sem(prompt_success) * 100
        success_ci_lower = success_rate - 1.96 * success_se
        success_ci_upper = success_rate + 1.96 * success_se

        # Overshoot rate - compare prompt-level means
        prompt_overshoot = prompt_groups.apply(
            lambda x: (x['modified_awareness'].mean() < x['deployment_awareness'].mean()) if len(x) > 0 else False,
            include_groups=False
        )
        overshoot_rate = prompt_overshoot.mean() * 100
        overshoot_se = stats.sem(prompt_overshoot) * 100
        overshoot_ci_lower = overshoot_rate - 1.96 * overshoot_se
        overshoot_ci_upper = overshoot_rate + 1.96 * overshoot_se

        # Self-test and other-test removal rates with prompt-level CIs
        if 'original_type' in group.columns and 'modified_type' in group.columns:
            # Self-test removal - prompt-level
            prompt_self_removal = []
            for prompt_id, prompt_data in group.groupby('prompt_id'):
                is_self = prompt_data['original_type'] == 'self_test'
                if is_self.any():
                    is_removed = (is_self & (prompt_data['modified_type'] != 'self_test')).any()
                    prompt_self_removal.append(is_removed)

            if prompt_self_removal:
                self_test_removal_rate = np.mean(prompt_self_removal) * 100
                self_removal_se = stats.sem(prompt_self_removal) * 100
                self_removal_ci_lower = self_test_removal_rate - 1.96 * self_removal_se
                self_removal_ci_upper = self_test_removal_rate + 1.96 * self_removal_se
            else:
                self_test_removal_rate = self_removal_ci_lower = self_removal_ci_upper = 0

            # Other-test removal - prompt-level
            prompt_other_removal = []
            for prompt_id, prompt_data in group.groupby('prompt_id'):
                is_other = prompt_data['original_type'] == 'other_test'
                if is_other.any():
                    is_removed = (is_other & (prompt_data['modified_type'] != 'other_test')).any()
                    prompt_other_removal.append(is_removed)

            if prompt_other_removal:
                other_test_removal_rate = np.mean(prompt_other_removal) * 100
                other_removal_se = stats.sem(prompt_other_removal) * 100
                other_removal_ci_lower = other_test_removal_rate - 1.96 * other_removal_se
                other_removal_ci_upper = other_test_removal_rate + 1.96 * other_removal_se
            else:
                other_test_removal_rate = other_removal_ci_lower = other_removal_ci_upper = 0
        else:
            self_test_removal_rate = self_removal_ci_lower = self_removal_ci_upper = 0
            other_test_removal_rate = other_removal_ci_lower = other_removal_ci_upper = 0

        # Choice consistency metrics (prompt-level comparison)
        if 'modified_choice' in group.columns and 'deployment_choice' in group.columns:
            # Group by prompt to compare modal choices
            prompt_choice_consistency = []
            prompt_choice_shifts = []
            prompt_original_consistency = []  # For original→deployment baseline

            for prompt_id, prompt_data in group.groupby('prompt_id'):
                modified_choices = prompt_data['modified_choice'].dropna()
                deployment_choices = prompt_data['deployment_choice'].dropna()
                original_choices = prompt_data['original_choice'].dropna() if 'original_choice' in prompt_data.columns else pd.Series()

                if len(modified_choices) > 0 and len(deployment_choices) > 0:
                    # Metric 1: Modified→Deploy modal choice agreement
                    modified_modal = modified_choices.mode()[0] if len(modified_choices.mode()) > 0 else None
                    deployment_modal = deployment_choices.mode()[0] if len(deployment_choices.mode()) > 0 else None

                    if modified_modal and deployment_modal:
                        prompt_choice_consistency.append(modified_modal == deployment_modal)

                    # Metric 2: Choice distribution shift (using choice B proportion)
                    modified_B_prop = (modified_choices == 'B').sum() / len(modified_choices)
                    deployment_B_prop = (deployment_choices == 'B').sum() / len(deployment_choices)
                    prompt_choice_shifts.append(abs(deployment_B_prop - modified_B_prop))

                # Baseline: Original→Deploy modal choice agreement
                if len(original_choices) > 0 and len(deployment_choices) > 0:
                    original_modal = original_choices.mode()[0] if len(original_choices.mode()) > 0 else None
                    deployment_modal = deployment_choices.mode()[0] if len(deployment_choices.mode()) > 0 else None

                    if original_modal and deployment_modal:
                        prompt_original_consistency.append(original_modal == deployment_modal)

            # Choice modal consistency with CI
            if prompt_choice_consistency:
                choice_modal_consistency = np.mean(prompt_choice_consistency)
                choice_consistency_se = stats.sem(prompt_choice_consistency)
                choice_consistency_ci_lower = choice_modal_consistency - 1.96 * choice_consistency_se
                choice_consistency_ci_upper = choice_modal_consistency + 1.96 * choice_consistency_se
            else:
                choice_modal_consistency = choice_consistency_ci_lower = choice_consistency_ci_upper = 0

            # Choice distribution shift with CI
            if prompt_choice_shifts:
                choice_distribution_shift = np.mean(prompt_choice_shifts)
                choice_shift_se = stats.sem(prompt_choice_shifts)
                choice_shift_ci_lower = choice_distribution_shift - 1.96 * choice_shift_se
                choice_shift_ci_upper = choice_distribution_shift + 1.96 * choice_shift_se
            else:
                choice_distribution_shift = choice_shift_ci_lower = choice_shift_ci_upper = 0

            # Original→deployment baseline consistency
            if prompt_original_consistency:
                original_choice_consistency = np.mean(prompt_original_consistency)
            else:
                original_choice_consistency = 0

            # Overall choice proportion shift (not per-prompt averaged)
            all_modified_choices = group['modified_choice'].dropna()
            all_deployment_choices = group['deployment_choice'].dropna()
            if len(all_modified_choices) > 0 and len(all_deployment_choices) > 0:
                modified_B_prop_overall = (all_modified_choices == 'B').sum() / len(all_modified_choices)
                deployment_B_prop_overall = (all_deployment_choices == 'B').sum() / len(all_deployment_choices)
                choice_proportion_shift = abs(modified_B_prop_overall - deployment_B_prop_overall)
            else:
                choice_proportion_shift = 0
        else:
            choice_modal_consistency = choice_consistency_ci_lower = choice_consistency_ci_upper = 0
            choice_distribution_shift = choice_shift_ci_lower = choice_shift_ci_upper = 0
            original_choice_consistency = 0
            choice_proportion_shift = 0

        # ICC
        prompt_means = prompt_groups['modified_awareness'].mean()
        var_between = np.var(prompt_means, ddof=1) if len(prompt_means) > 1 else 0
        var_within = prompt_groups['modified_awareness'].var(ddof=1).mean()
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0

        metrics = {
            'experiment': experiment,
            'technique': technique,
            'n_prompts': group['prompt_id'].nunique(),
            'n_seeds': len(group),
            'icc': icc,
            'original_awareness': original_awareness,
            'awareness_reduction': reduction_mean,
            'awareness_reduction_ci_lower': reduction_ci_lower,
            'awareness_reduction_ci_upper': reduction_ci_upper,
            'bc_rate': bc_rate,
            'bc_rate_ci_lower': bc_ci_lower,
            'bc_rate_ci_upper': bc_ci_upper,
            'bc_per_prompt_mean': bc_per_prompt_mean,
            'bc_per_prompt_std': bc_per_prompt_std,
            'orig_bc_per_prompt_mean': orig_bc_per_prompt_mean,
            'orig_bc_per_prompt_std': orig_bc_per_prompt_std,
            'task_completion': task_completion,
            'final_awareness': final_awareness,
            'final_awareness_ci_lower': final_ci_lower,
            'final_awareness_ci_upper': final_ci_upper,
            'deployment_convergence': convergence_mean,
            'deployment_convergence_ci_lower': convergence_ci_lower,
            'deployment_convergence_ci_upper': convergence_ci_upper,
            'success_rate': success_rate,
            'success_rate_ci_lower': success_ci_lower,
            'success_rate_ci_upper': success_ci_upper,
            'overshoot_rate': overshoot_rate,
            'overshoot_rate_ci_lower': overshoot_ci_lower,
            'overshoot_rate_ci_upper': overshoot_ci_upper,
            'self_test_removal_rate': self_test_removal_rate,
            'self_test_removal_rate_ci_lower': self_removal_ci_lower,
            'self_test_removal_rate_ci_upper': self_removal_ci_upper,
            'other_test_removal_rate': other_test_removal_rate,
            'other_test_removal_rate_ci_lower': other_removal_ci_lower,
            'other_test_removal_rate_ci_upper': other_removal_ci_upper,
            'choice_modal_consistency': choice_modal_consistency,
            'choice_modal_consistency_ci_lower': choice_consistency_ci_lower,
            'choice_modal_consistency_ci_upper': choice_consistency_ci_upper,
            'choice_distribution_shift': choice_distribution_shift,
            'choice_distribution_shift_ci_lower': choice_shift_ci_lower,
            'choice_distribution_shift_ci_upper': choice_shift_ci_upper,
            'original_choice_consistency': original_choice_consistency,
            'awareness_shift': awareness_shift,
            'choice_proportion_shift': choice_proportion_shift,
            # Choice distribution (% choosing A vs B)
            'deploy_b_rate': deploy_b_rate,
            'deploy_a_rate': deploy_a_rate,
            'deploy_b_ci_lower': deploy_b_ci_lower,
            'deploy_b_ci_upper': deploy_b_ci_upper,
            'orig_b_rate': orig_b_rate,
            'orig_a_rate': orig_a_rate,
            'orig_b_ci_lower': orig_b_ci_lower,
            'orig_b_ci_upper': orig_b_ci_upper,
            'mod_b_rate': mod_b_rate,
            'mod_a_rate': mod_a_rate,
            'mod_b_ci_lower': mod_b_ci_lower,
            'mod_b_ci_upper': mod_b_ci_upper,
        }

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    print(f"\n✓ Computed metrics for {len(metrics_df)} experiments")
    print(metrics_df[['experiment', 'awareness_reduction', 'bc_rate', 'task_completion']])
    print(f"{'='*80}\n")

    return metrics_df


def compute_pairwise_stats(combined_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise statistical comparisons between experiments."""
    print(f"\n{'='*80}")
    print(f"COMPUTING PAIRWISE COMPARISONS")
    print(f"{'='*80}\n")

    comparison_list = []
    experiments = metrics_df['experiment'].unique()

    metric_configs = [
        ('awareness_reduction', 'original_awareness', 'modified_awareness', 'subtraction'),
        ('bc_rate', 'bc', None, 'rate'),
        ('final_awareness', 'modified_awareness', None, 'mean'),
    ]

    for i, exp1 in enumerate(experiments):
        for exp2 in experiments[i+1:]:
            print(f"Comparing: {exp1} vs {exp2}")

            exp1_df = combined_df[combined_df['experiment'] == exp1].copy()
            exp2_df = combined_df[combined_df['experiment'] == exp2].copy()

            exp1_df = normalize_columns_for_experiment(exp1_df)
            exp2_df = normalize_columns_for_experiment(exp2_df)

            for metric_name, col1, col2, calc_type in metric_configs:
                if calc_type == 'subtraction':
                    exp1_prompts = exp1_df.groupby('prompt_id').apply(lambda x: np.mean(x[col1] - x[col2]), include_groups=False)
                    exp2_prompts = exp2_df.groupby('prompt_id').apply(lambda x: np.mean(x[col1] - x[col2]), include_groups=False)
                elif calc_type == 'rate':
                    exp1_prompts = exp1_df.groupby('prompt_id').apply(lambda x: pd.to_numeric(x[col1], errors='coerce').fillna(0).mean(), include_groups=False)
                    exp2_prompts = exp2_df.groupby('prompt_id').apply(lambda x: pd.to_numeric(x[col1], errors='coerce').fillna(0).mean(), include_groups=False)
                elif calc_type == 'mean':
                    exp1_prompts = exp1_df.groupby('prompt_id')[col1].mean()
                    exp2_prompts = exp2_df.groupby('prompt_id')[col1].mean()

                if exp1_prompts.isna().any() or exp2_prompts.isna().any() or len(exp1_prompts) < 2 or len(exp2_prompts) < 2:
                    continue

                t_stat, p_value = stats.ttest_ind(exp1_prompts, exp2_prompts)
                pooled_std = np.sqrt((exp1_prompts.var() + exp2_prompts.var()) / 2)
                cohens_d = (exp1_prompts.mean() - exp2_prompts.mean()) / pooled_std if pooled_std > 0 else 0

                comparison_list.append({
                    'exp1': exp1,
                    'exp2': exp2,
                    'metric': metric_name,
                    'exp1_mean': exp1_prompts.mean(),
                    'exp2_mean': exp2_prompts.mean(),
                    'difference': exp1_prompts.mean() - exp2_prompts.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                })

    comparisons_df = pd.DataFrame(comparison_list)
    print(f"\n✓ Computed {len(comparisons_df)} pairwise comparisons")
    if len(comparisons_df) > 0:
        print(f"  Significant (p<0.05): {comparisons_df['significant'].sum()}")
    print(f"{'='*80}\n")

    return comparisons_df


def generate_visualizations(
    combined_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    output_plots_dir: Path,
    model: str
):
    """Generate all 5 visualization views."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    print(f"\n{'='*80}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    output_plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # View 1: Pareto Frontier Plots
    print("Creating View 1: Pareto frontier plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    pareto_pairs = [
        ('awareness_reduction', 'bc_rate', 'Awareness Reduction', 'BC Rate (%)', True, True),
        ('awareness_reduction', 'final_awareness', 'Awareness Reduction', 'Final Awareness', True, False),
        ('bc_rate', 'task_completion', 'BC Rate (%)', 'Task Completion (%)', True, True),
        ('success_rate', 'bc_rate', 'Success Rate (%)', 'BC Rate (%)', True, True),
        ('self_test_removal_rate', 'bc_rate', 'Self-Test Removal (%)', 'BC Rate (%)', True, True),
        ('other_test_removal_rate', 'bc_rate', 'Other-Test Removal (%)', 'BC Rate (%)', True, True),
    ]

    colors = sns.color_palette("husl", len(metrics_df))

    for idx, (metric_x, metric_y, label_x, label_y, higher_x, higher_y) in enumerate(pareto_pairs):
        ax = axes[idx]

        for i, row in metrics_df.iterrows():
            x_val = row[metric_x]
            y_val = row[metric_y]

            if pd.isna(x_val) or pd.isna(y_val):
                continue

            marker = 'o' if row['technique'] == 'suppression' else '^'
            ax.scatter(x_val, y_val, s=200, marker=marker, color=colors[i],
                      alpha=0.7, edgecolors='black', linewidth=2)
            ax.text(x_val, y_val, f"  {row['experiment']}", fontsize=9,
                   ha='left', va='center')

        # Add original baseline reference for final_awareness plots
        if metric_y == 'final_awareness' and 'original_awareness' in metrics_df.columns:
            original_mean = metrics_df['original_awareness'].mean()
            ax.axhline(y=original_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.6, label=f'Original Baseline ({original_mean:.1f})')
            ax.legend(fontsize=9, loc='best')

        ax.set_xlabel(label_x, fontsize=11, fontweight='bold')
        ax.set_ylabel(label_y, fontsize=11, fontweight='bold')
        ax.set_title(f'{label_x} vs {label_y}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add Pareto frontier direction arrow
        ax_arrow_text = f"↗ Better" if (higher_x and higher_y) else "Better →" if higher_x else "↑ Better"
        ax.text(0.95, 0.95, ax_arrow_text, transform=ax.transAxes,
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add shared legend on first subplot
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=10, label='Suppression', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                  markersize=10, label='Steering', markeredgecolor='black')
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.suptitle(f'Pareto Frontiers - {model}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(pad=2.0, h_pad=3.0)
    plt.savefig(output_plots_dir / 'view1_pareto_frontiers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: view1_pareto_frontiers.png")

    # View 2: Metric Bar Charts with Significance
    print("Creating View 2: Metric bar charts with significance...")

    # Group metrics by whether they have baselines
    metrics_with_baselines = ['final_awareness', 'choice_modal_consistency']
    metrics_without_baselines = [
        'awareness_reduction', 'bc_rate', 'deployment_convergence',
        'success_rate', 'overshoot_rate', 'self_test_removal_rate',
        'other_test_removal_rate', 'choice_distribution_shift'
    ]

    # All metrics for plotting (with baselines first, then without)
    metrics_to_plot = metrics_with_baselines + metrics_without_baselines

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        experiments = metrics_df['experiment'].values
        values = metrics_df[metric].values
        ci_lower = metrics_df[f'{metric}_ci_lower'].values if f'{metric}_ci_lower' in metrics_df.columns else values
        ci_upper = metrics_df[f'{metric}_ci_upper'].values if f'{metric}_ci_upper' in metrics_df.columns else values

        errors_lower = values - ci_lower
        errors_upper = ci_upper - values
        errors = [errors_lower, errors_upper]

        x_pos = np.arange(len(experiments))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5,
                     color=colors[:len(experiments)], alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Add original baseline reference line for final_awareness only
        if metric == 'final_awareness' and 'original_awareness' in metrics_df.columns:
            original_mean = metrics_df['original_awareness'].mean()
            ax.axhline(y=original_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Original Baseline ({original_mean:.1f})')
            ax.legend(fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')

        # Add indicator for metrics with baselines
        title_text = metric.replace('_', ' ').title()
        if metric in metrics_with_baselines:
            # For choice_modal_consistency, add baseline value to title instead of line
            if metric == 'choice_modal_consistency' and 'original_choice_consistency' in metrics_df.columns:
                baseline_consistency = metrics_df['original_choice_consistency'].mean() * 100
                title_text += f" (Orig→Deploy baseline: {baseline_consistency:.1f}%)"
            ax.set_title(title_text, fontsize=11, fontweight='bold', color='darkgreen')
        else:
            ax.set_title(title_text, fontsize=11, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')

        # Add significance stars
        if not comparisons_df.empty:
            metric_comps = comparisons_df[comparisons_df['metric'] == metric]
            y_max = max(ci_upper[~np.isnan(ci_upper)]) if len(ci_upper[~np.isnan(ci_upper)]) > 0 else 1

            for _, comp in metric_comps.iterrows():
                if comp['significant']:
                    exp1_idx = np.where(experiments == comp['exp1'])[0]
                    exp2_idx = np.where(experiments == comp['exp2'])[0]

                    if len(exp1_idx) > 0 and len(exp2_idx) > 0:
                        x1, x2 = exp1_idx[0], exp2_idx[0]
                        y_offset = y_max * 1.1
                        ax.plot([x1, x2], [y_offset, y_offset], 'k-', linewidth=1.5)

                        stars = '***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*'
                        ax.text((x1 + x2) / 2, y_offset, stars, ha='center', va='bottom', fontsize=12)

    # Hide unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Metrics Comparison - {model}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(pad=2.0, h_pad=3.0)
    plt.savefig(output_plots_dir / 'view2_metrics_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: view2_metrics_bars.png")

    # View 3: Per-Awareness-Type Breakdown
    print("Creating View 3: Per-awareness-type breakdown...")

    type_breakdown = combined_df.groupby(['experiment', 'original_type']).agg({
        'bc': lambda x: pd.to_numeric(x, errors='coerce').fillna(0).mean() * 100,
        'original_awareness': 'mean',
        'modified_awareness': 'mean',
        'file': 'count'
    }).reset_index()
    type_breakdown.columns = ['experiment', 'original_type', 'bc_rate', 'orig_aware', 'mod_aware', 'count']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # BC rate by type
    ax = axes[0, 0]
    type_pivot = type_breakdown.pivot(index='original_type', columns='experiment', values='bc_rate')
    type_pivot.plot(kind='bar', ax=ax, color=colors[:len(metrics_df)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('BC Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Original Awareness Type', fontsize=11, fontweight='bold')
    ax.set_title('BC Rate by Awareness Type', fontsize=12, fontweight='bold')
    ax.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Awareness reduction by type
    ax = axes[0, 1]
    type_breakdown['awareness_reduction'] = type_breakdown['orig_aware'] - type_breakdown['mod_aware']
    aware_pivot = type_breakdown.pivot(index='original_type', columns='experiment', values='awareness_reduction')
    aware_pivot.plot(kind='bar', ax=ax, color=colors[:len(metrics_df)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Awareness Reduction', fontsize=11, fontweight='bold')
    ax.set_xlabel('Original Awareness Type', fontsize=11, fontweight='bold')
    ax.set_title('Awareness Reduction by Type', fontsize=12, fontweight='bold')
    ax.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Sample counts by type
    ax = axes[1, 0]
    count_pivot = type_breakdown.pivot(index='original_type', columns='experiment', values='count')
    count_pivot.plot(kind='bar', ax=ax, color=colors[:len(metrics_df)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax.set_xlabel('Original Awareness Type', fontsize=11, fontweight='bold')
    ax.set_title('Sample Distribution by Type', fontsize=12, fontweight='bold')
    ax.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Type shift matrix for first experiment
    ax = axes[1, 1]
    first_exp = combined_df['experiment'].iloc[0]
    exp_data = combined_df[combined_df['experiment'] == first_exp]
    shift_matrix = pd.crosstab(exp_data['original_type'], exp_data['modified_type'], normalize='index') * 100
    sns.heatmap(shift_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '% of Original Type'})
    ax.set_title(f'Type Shift Matrix - {first_exp}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Modified Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Original Type', fontsize=11, fontweight='bold')

    plt.suptitle(f'Per-Awareness-Type Analysis - {model}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(pad=2.0, h_pad=3.0)
    plt.savefig(output_plots_dir / 'view3_type_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: view3_type_breakdown.png")

    # View 4: Performance Heatmap
    print("Creating View 4: Performance heatmap...")

    heatmap_metrics = ['awareness_reduction', 'bc_rate', 'success_rate',
                      'task_completion', 'choice_modal_consistency', 'deployment_convergence',
                      'original_awareness', 'final_awareness']

    heatmap_data = metrics_df.set_index('experiment')[heatmap_metrics].T

    # Normalize each metric to 0-100 scale for visualization
    heatmap_normalized = heatmap_data.copy()
    for metric in heatmap_metrics:
        row = heatmap_normalized.loc[metric]
        if metric in ['deployment_convergence', 'original_awareness', 'final_awareness']:
            # Lower is better, invert
            heatmap_normalized.loc[metric] = 100 * (1 - (row - row.min()) / (row.max() - row.min() + 1e-10))
        else:
            # Higher is better
            heatmap_normalized.loc[metric] = 100 * (row - row.min()) / (row.max() - row.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.2f', cmap='RdYlGn',
               ax=ax, cbar_kws={'label': 'Normalized Performance (0-100)'},
               linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Heatmap - {model}', fontsize=14, fontweight='bold')
    ax.set_yticklabels([m.replace('_', ' ').title() for m in heatmap_metrics], rotation=0)

    plt.tight_layout(pad=2.0, h_pad=3.0)
    plt.savefig(output_plots_dir / 'view4_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: view4_performance_heatmap.png")

    # View 5: Statistical Significance Matrix
    print("Creating View 5: Statistical significance matrix...")

    if not comparisons_df.empty:
        experiments = metrics_df['experiment'].unique()
        n_exp = len(experiments)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, metric in enumerate(['awareness_reduction', 'bc_rate', 'final_awareness']):
            ax = axes[idx]

            # Create matrix: rows = exp1, cols = exp2
            sig_matrix = np.zeros((n_exp, n_exp))
            pval_matrix = np.ones((n_exp, n_exp))

            for _, comp in comparisons_df[comparisons_df['metric'] == metric].iterrows():
                exp1_idx = np.where(experiments == comp['exp1'])[0][0]
                exp2_idx = np.where(experiments == comp['exp2'])[0][0]

                # Symmetric matrix
                sig_val = 3 if comp['p_value'] < 0.001 else 2 if comp['p_value'] < 0.01 else 1 if comp['p_value'] < 0.05 else 0
                sig_matrix[exp1_idx, exp2_idx] = sig_val
                sig_matrix[exp2_idx, exp1_idx] = sig_val
                pval_matrix[exp1_idx, exp2_idx] = comp['p_value']
                pval_matrix[exp2_idx, exp1_idx] = comp['p_value']

            # Create annotations
            annot = np.empty((n_exp, n_exp), dtype=object)
            for i in range(n_exp):
                for j in range(n_exp):
                    if i == j:
                        annot[i, j] = '-'
                    elif sig_matrix[i, j] == 3:
                        annot[i, j] = '***'
                    elif sig_matrix[i, j] == 2:
                        annot[i, j] = '**'
                    elif sig_matrix[i, j] == 1:
                        annot[i, j] = '*'
                    else:
                        annot[i, j] = 'ns'

            sns.heatmap(sig_matrix, annot=annot, fmt='', cmap='Reds', ax=ax,
                       xticklabels=experiments, yticklabels=experiments,
                       cbar_kws={'label': 'Significance Level',
                                'ticks': [0, 1, 2, 3],
                                'format': plt.matplotlib.ticker.FuncFormatter(lambda x, p: ['ns', 'p<0.05', 'p<0.01', 'p<0.001'][int(x)])},
                       vmin=0, vmax=3, linewidths=0.5, linecolor='gray')

            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Experiment', fontsize=10, fontweight='bold')
            ax.set_ylabel('Experiment', fontsize=10, fontweight='bold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.suptitle(f'Statistical Significance Matrix - {model}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(pad=2.0, h_pad=3.0)
        plt.savefig(output_plots_dir / 'view5_significance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: view5_significance_matrix.png")
    else:
        print(f"  ⚠️  Skipped: No pairwise comparisons available")

    # View 6: Deployment Robustness Analysis
    print("Creating View 6: Deployment robustness analysis...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.3)

    # Subplot 1: Distribution Shifts (Orig → Mod → Deploy) for each experiment
    ax1 = fig.add_subplot(gs[0, 0])

    # For each experiment, plot violin/box showing distribution at each stage
    exp_names = metrics_df['experiment'].values
    x_positions = []
    positions_map = {}

    for i, row in metrics_df.iterrows():
        exp_data = combined_df[combined_df['experiment'] == row['experiment']].copy()

        if 'original_awareness' in exp_data.columns and 'modified_awareness' in exp_data.columns and 'deployment_awareness' in exp_data.columns:
            orig = exp_data['original_awareness'].dropna().values
            mod = exp_data['modified_awareness'].dropna().values
            dep = exp_data['deployment_awareness'].dropna().values

            # Plot as violin plots side by side
            base_pos = i * 4  # Space between experiments
            pos_orig = base_pos
            pos_mod = base_pos + 1
            pos_deploy = base_pos + 2

            if len(orig) > 0:
                parts = ax1.violinplot([orig], positions=[pos_orig], widths=0.7,
                                      showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.3)

            if len(mod) > 0:
                parts = ax1.violinplot([mod], positions=[pos_mod], widths=0.7,
                                      showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.6)

            if len(dep) > 0:
                parts = ax1.violinplot([dep], positions=[pos_deploy], widths=0.7,
                                      showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.9)

            # Add experiment label
            ax1.text(pos_mod, -1, row['experiment'], rotation=45, ha='right', fontsize=8)

    ax1.set_ylabel('Awareness Distribution', fontsize=11, fontweight='bold')
    ax1.set_title('Awareness Shift: Original→Modified→Deployment', fontsize=12, fontweight='bold')
    ax1.set_ylim([-1.5, 10])
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, label='Original'),
        Patch(facecolor='gray', alpha=0.6, label='Modified'),
        Patch(facecolor='gray', alpha=0.9, label='Deployment')
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc='upper right')

    # Subplot 2: Deployment Convergence Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    conv_values = metrics_df['deployment_convergence'].values
    conv_lower = metrics_df['deployment_convergence_ci_lower'].values if 'deployment_convergence_ci_lower' in metrics_df.columns else conv_values
    conv_upper = metrics_df['deployment_convergence_ci_upper'].values if 'deployment_convergence_ci_upper' in metrics_df.columns else conv_values
    errors_lower = conv_values - conv_lower
    errors_upper = conv_upper - conv_values

    x_pos = np.arange(len(metrics_df))
    ax2.bar(x_pos, conv_values, yerr=[errors_lower, errors_upper], capsize=5,
           color=colors[:len(metrics_df)], alpha=0.7, edgecolor='black', linewidth=1.5,
           label='Modified→Deploy')

    # Add original→deployment convergence baseline for comparison
    baseline_convergences = []
    for _, row in metrics_df.iterrows():
        exp_data = combined_df[combined_df['experiment'] == row['experiment']]
        if 'original_awareness' in exp_data.columns and 'deployment_awareness' in exp_data.columns:
            orig_deploy_conv = np.mean(np.abs(
                exp_data['original_awareness'].dropna() - exp_data['deployment_awareness'].dropna()
            ))
            baseline_convergences.append(orig_deploy_conv)
        else:
            baseline_convergences.append(np.nan)

    if baseline_convergences:
        baseline_mean = np.nanmean(baseline_convergences)
        ax2.axhline(y=baseline_mean, color='orange', linestyle=':', linewidth=2.5,
                   alpha=0.7, label=f'Baseline (Orig→Deploy): {baseline_mean:.2f}')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_df['experiment'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Deployment Convergence', fontsize=10, fontweight='bold')
    ax2.set_title('Deployment Convergence (Lower = Better)', fontsize=12, fontweight='bold')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: <2.0')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Choice Modal Consistency
    ax3 = fig.add_subplot(gs[0, 2])
    consistency_values = metrics_df['choice_modal_consistency'].values * 100

    # Add error bars if CIs are available
    if 'choice_modal_consistency_ci_lower' in metrics_df.columns and 'choice_modal_consistency_ci_upper' in metrics_df.columns:
        ci_lower = metrics_df['choice_modal_consistency_ci_lower'].values * 100
        ci_upper = metrics_df['choice_modal_consistency_ci_upper'].values * 100
        errors_lower = consistency_values - ci_lower
        errors_upper = ci_upper - consistency_values
        ax3.bar(x_pos, consistency_values, yerr=[errors_lower, errors_upper], capsize=8,
               color=colors[:len(metrics_df)], alpha=0.7, edgecolor='black',
               linewidth=1.5, ecolor='black', label='Modified→Deploy')
    else:
        ax3.bar(x_pos, consistency_values, color=colors[:len(metrics_df)],
               alpha=0.7, edgecolor='black', linewidth=1.5, label='Modified→Deploy')
        print(f"  ⚠️  Warning: Choice modal consistency CIs not available in data")

    # Add original→deployment baseline for comparison
    if 'original_choice_consistency' in metrics_df.columns:
        baseline_consistency = metrics_df['original_choice_consistency'].mean() * 100
        ax3.axhline(y=baseline_consistency, color='orange', linestyle=':', linewidth=2.5,
                   alpha=0.7, label=f'Baseline (Orig→Deploy): {baseline_consistency:.1f}%')

    # Show legend if we have labels (from bar plot or baseline)
    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(fontsize=8, loc='upper right')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_df['experiment'], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Modal Choice Consistency (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Modified vs Deploy Modal Choice Agreement', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 100])

    # Subplot 4-6: Progression Plots (Original → Modified → Deployment) for top 3 experiments
    top_experiments = metrics_df.nlargest(3, 'n_prompts')

    for idx, (_, row) in enumerate(top_experiments.iterrows()):
        ax = fig.add_subplot(gs[1, idx])
        exp_data = combined_df[combined_df['experiment'] == row['experiment']].copy()

        if 'original_awareness' in exp_data.columns and 'modified_awareness' in exp_data.columns and 'deployment_awareness' in exp_data.columns:
            orig = exp_data['original_awareness'].dropna()
            mod = exp_data['modified_awareness'].dropna()
            dep = exp_data['deployment_awareness'].dropna()

            # Match indices across all three
            common_idx = orig.index.intersection(mod.index).intersection(dep.index)
            if len(common_idx) > 100:
                # Sample for clarity if too many points
                sample_idx = np.random.choice(common_idx, size=min(500, len(common_idx)), replace=False)
                common_idx = sample_idx

            if len(common_idx) > 0:
                # Draw lines showing progression: Original → Modified → Deployment
                for i in common_idx:
                    orig_val = orig.loc[i]
                    mod_val = mod.loc[i]
                    dep_val = dep.loc[i]

                    # Determine color based on final outcome
                    if dep_val < 3:
                        color = 'green'  # Good: low deployment awareness
                    elif dep_val < 5:
                        color = 'orange'  # Medium
                    else:
                        color = 'red'  # Bad: high deployment awareness

                    # Draw progression line: (original, original) → (modified, modified) → (modified, deployment)
                    ax.plot([orig_val, mod_val, mod_val], [orig_val, mod_val, dep_val],
                           color=color, alpha=0.1, linewidth=1)
                    ax.scatter([mod_val], [dep_val], s=10, color=color, alpha=0.3)

        # Add diagonal line
        max_val = 10
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect match')
        ax.set_xlabel('Modified Awareness (Eval)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Deployment Awareness', fontsize=10, fontweight='bold')
        ax.set_title(f"{row['experiment']}: Orig→Mod→Deploy\n(Convergence: {row['deployment_convergence']:.2f})",
                    fontsize=10, fontweight='bold')
        ax.set_xlim([0, max_val])
        ax.set_ylim([0, max_val])
        ax.grid(True, alpha=0.3)

        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Deploy<3 (Good)'),
            Patch(facecolor='orange', alpha=0.5, label='Deploy 3-5 (Med)'),
            Patch(facecolor='red', alpha=0.5, label='Deploy>5 (Bad)')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.suptitle(f'Deployment Robustness Analysis - {model}', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_plots_dir / 'view6_deployment_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: view6_deployment_robustness.png")

    # View 7: Bucket-Stratified Analysis
    print("Creating View 7: Bucket-stratified analysis...")

    # Check if bucket column exists
    if 'bucket' not in combined_df.columns or combined_df['bucket'].isna().all():
        print("  ⚠️  Warning: No bucket data available, skipping View 7")
    else:
        # Filter out unknown buckets
        bucket_df = combined_df[combined_df['bucket'] != 'unknown'].copy()

        if len(bucket_df) == 0:
            print("  ⚠️  Warning: No data with known buckets, skipping View 7")
        else:
            # Define bucket order
            bucket_order = ['10-25%', '25-50%', '50-75%', '75-100%']
            available_buckets = [b for b in bucket_order if b in bucket_df['bucket'].unique()]

            # Compute metrics per experiment per bucket
            bucket_metrics = []
            for exp_name in bucket_df['experiment'].unique():
                exp_data = bucket_df[bucket_df['experiment'] == exp_name]
                for bucket in available_buckets:
                    bucket_data = exp_data[exp_data['bucket'] == bucket]
                    if len(bucket_data) > 0:
                        # Compute metrics
                        orig_aware = bucket_data['original_awareness'].dropna()
                        mod_aware = bucket_data['modified_awareness'].dropna()
                        deploy_aware = bucket_data['deployment_awareness'].dropna()

                        # Awareness reduction (prompt-level means)
                        if len(orig_aware) > 0 and len(mod_aware) > 0:
                            prompt_groups = bucket_data.groupby('prompt_id')
                            prompt_reductions = prompt_groups.apply(
                                lambda x: x['original_awareness'].mean() - x['modified_awareness'].mean(),
                                include_groups=False
                            )
                            awareness_reduction = prompt_reductions.mean()
                        else:
                            awareness_reduction = 0

                        # BC rate
                        total = len(bucket_data)
                        bc_count = ((bucket_data['original_choice'] == 'B') &
                                   (bucket_data['modified_choice'] != 'B')).sum()
                        bc_rate = (bc_count / total * 100) if total > 0 else 0

                        # Success rate
                        success_count = (bucket_data['modified_choice'].notna() &
                                       (bucket_data['modified_choice'] != 'error')).sum()
                        success_rate = (success_count / total * 100) if total > 0 else 0

                        # Deployment convergence (prompt-level means)
                        if len(mod_aware) > 0 and len(deploy_aware) > 0:
                            prompt_convergence = prompt_groups.apply(
                                lambda x: np.abs(x['modified_awareness'].mean() - x['deployment_awareness'].mean()),
                                include_groups=False
                            )
                            deployment_convergence = prompt_convergence.mean()
                        else:
                            deployment_convergence = 0

                        bucket_metrics.append({
                            'experiment': exp_name,
                            'bucket': bucket,
                            'awareness_reduction': awareness_reduction,
                            'bc_rate': bc_rate,
                            'success_rate': success_rate,
                            'deployment_convergence': deployment_convergence
                        })

            bucket_metrics_df = pd.DataFrame(bucket_metrics)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Subplot 1: Awareness Reduction by Bucket
            ax1 = axes[0, 0]
            bucket_pivot = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='awareness_reduction')
            bucket_pivot = bucket_pivot.reindex(available_buckets)
            bucket_pivot.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Awareness Reduction by Bucket', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Eval Awareness Bucket', fontsize=10)
            ax1.set_ylabel('Awareness Reduction', fontsize=10)
            ax1.legend(title='Experiment', fontsize=8, title_fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

            # Subplot 2: BC Rate by Bucket
            ax2 = axes[0, 1]
            bucket_pivot = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='bc_rate')
            bucket_pivot = bucket_pivot.reindex(available_buckets)
            bucket_pivot.plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_title('BC Rate by Bucket', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Eval Awareness Bucket', fontsize=10)
            ax2.set_ylabel('BC Rate (%)', fontsize=10)
            ax2.legend(title='Experiment', fontsize=8, title_fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

            # Subplot 3: Success Rate by Bucket
            ax3 = axes[1, 0]
            bucket_pivot = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='success_rate')
            bucket_pivot = bucket_pivot.reindex(available_buckets)
            bucket_pivot.plot(kind='bar', ax=ax3, width=0.8)
            ax3.set_title('Success Rate by Bucket', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Eval Awareness Bucket', fontsize=10)
            ax3.set_ylabel('Success Rate (%)', fontsize=10)
            ax3.legend(title='Experiment', fontsize=8, title_fontsize=9)
            ax3.grid(axis='y', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)

            # Subplot 4: Deployment Convergence by Bucket
            ax4 = axes[1, 1]
            bucket_pivot = bucket_metrics_df.pivot(index='bucket', columns='experiment', values='deployment_convergence')
            bucket_pivot = bucket_pivot.reindex(available_buckets)
            bucket_pivot.plot(kind='bar', ax=ax4, width=0.8)
            ax4.set_title('Deployment Convergence by Bucket (Lower = Better)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Eval Awareness Bucket', fontsize=10)
            ax4.set_ylabel('Deployment Convergence', fontsize=10)
            ax4.legend(title='Experiment', fontsize=8, title_fontsize=9)
            ax4.grid(axis='y', alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)

            plt.suptitle(f'Bucket-Stratified Analysis - {model}', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout(pad=2.0, h_pad=3.0)
            plt.savefig(output_plots_dir / 'view7_bucket_stratified.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: view7_bucket_stratified.png")

            # View 8: Bucket-Specific Performance Rankings
            print("Creating View 8: Bucket-specific performance rankings...")

            # Compute composite score for each experiment-bucket combination
            # Higher awareness reduction, lower deployment convergence, higher success rate = better
            # Normalize metrics to 0-1 scale and compute weighted average
            for bucket in available_buckets:
                bucket_data = bucket_metrics_df[bucket_metrics_df['bucket'] == bucket].copy()
                if len(bucket_data) > 0:
                    # Normalize metrics (handle division by zero)
                    max_aware_red = bucket_data['awareness_reduction'].max()
                    min_aware_red = bucket_data['awareness_reduction'].min()
                    if max_aware_red > min_aware_red:
                        bucket_data['norm_aware_red'] = (bucket_data['awareness_reduction'] - min_aware_red) / (max_aware_red - min_aware_red)
                    else:
                        bucket_data['norm_aware_red'] = 1.0

                    max_conv = bucket_data['deployment_convergence'].max()
                    min_conv = bucket_data['deployment_convergence'].min()
                    if max_conv > min_conv:
                        bucket_data['norm_conv'] = 1 - (bucket_data['deployment_convergence'] - min_conv) / (max_conv - min_conv)
                    else:
                        bucket_data['norm_conv'] = 1.0

                    max_success = bucket_data['success_rate'].max()
                    min_success = bucket_data['success_rate'].min()
                    if max_success > min_success:
                        bucket_data['norm_success'] = (bucket_data['success_rate'] - min_success) / (max_success - min_success)
                    else:
                        bucket_data['norm_success'] = 1.0

                    # Composite score: weighted average
                    bucket_data['composite_score'] = (
                        0.4 * bucket_data['norm_aware_red'] +
                        0.4 * bucket_data['norm_conv'] +
                        0.2 * bucket_data['norm_success']
                    )

                    # Update in main dataframe
                    bucket_metrics_df.loc[bucket_metrics_df['bucket'] == bucket, 'composite_score'] = bucket_data['composite_score'].values

            # Create heatmap showing composite scores
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Subplot 1: Heatmap of composite scores
            score_pivot = bucket_metrics_df.pivot(index='experiment', columns='bucket', values='composite_score')
            score_pivot = score_pivot[available_buckets]  # Reorder columns

            im1 = ax1.imshow(score_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(available_buckets)))
            ax1.set_xticklabels(available_buckets, rotation=45, ha='right')
            ax1.set_yticks(range(len(score_pivot.index)))
            ax1.set_yticklabels(score_pivot.index, fontsize=9)
            ax1.set_xlabel('Eval Awareness Bucket', fontsize=10)
            ax1.set_ylabel('Experiment', fontsize=10)
            ax1.set_title('Composite Performance Score by Bucket\n(Green = Better)', fontsize=12, fontweight='bold')

            # Add text annotations
            for i in range(len(score_pivot.index)):
                for j in range(len(available_buckets)):
                    value = score_pivot.iloc[i, j]
                    if not pd.isna(value):
                        text = ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                                      color='black' if value > 0.5 else 'white', fontsize=8)

            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Composite Score', fontsize=9)

            # Subplot 2: Rankings table showing best experiment per bucket
            rankings = []
            for bucket in available_buckets:
                bucket_data = bucket_metrics_df[bucket_metrics_df['bucket'] == bucket]
                if len(bucket_data) > 0:
                    # Sort by composite score
                    sorted_data = bucket_data.sort_values('composite_score', ascending=False)
                    top_exp = sorted_data.iloc[0]
                    rankings.append({
                        'Bucket': bucket,
                        'Best Intervention': top_exp['experiment'],
                        'Score': f"{top_exp['composite_score']:.3f}",
                        'Awareness Red.': f"{top_exp['awareness_reduction']:.2f}",
                        'Deploy Conv.': f"{top_exp['deployment_convergence']:.2f}",
                        'Success Rate': f"{top_exp['success_rate']:.1f}%"
                    })

            rankings_df = pd.DataFrame(rankings)

            # Create table
            ax2.axis('off')
            table = ax2.table(cellText=rankings_df.values, colLabels=rankings_df.columns,
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header
            for i in range(len(rankings_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

            # Alternate row colors
            for i in range(1, len(rankings_df) + 1):
                for j in range(len(rankings_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')

            ax2.set_title('Best Intervention per Bucket\n(Based on Composite Score)', fontsize=12, fontweight='bold', pad=20)

            plt.suptitle(f'Bucket-Specific Performance Rankings - {model}', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(pad=2.0, h_pad=3.0)
            plt.savefig(output_plots_dir / 'view8_bucket_rankings.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: view8_bucket_rankings.png")

    print(f"\n✓ All visualizations saved to: {output_plots_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment variants for a single model')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., qwen_qwen3-32b)')
    parser.add_argument('--suppression-experiments', type=str, nargs='+', required=True,
                       help='Suppression experiments as "name:path" pairs')
    parser.add_argument('--steering-experiments', type=str, nargs='+', required=True,
                       help='Steering experiments as "name:path" pairs')
    parser.add_argument('--mode', type=str, required=True, choices=['v0', 'v1'],
                       help='Version mode for categorization')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--limit-seeds', type=int, default=None,
                       help='Limit to seeds 0-N (e.g., --limit-seeds 30 uses only seeds 0-30)')

    args = parser.parse_args()

    # Parse experiment specifications
    supp_exps = []
    for spec in args.suppression_experiments:
        name, path = spec.split(':', 1)
        supp_exps.append((name, Path(path)))

    steer_exps = []
    for spec in args.steering_experiments:
        name, path = spec.split(':', 1)
        steer_exps.append((name, Path(path)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    output_data_dir = output_dir / 'data'
    combined_df = collect_all_experiments(args.model, supp_exps, steer_exps, output_data_dir, args.mode, args.limit_seeds)

    # Compute metrics
    metrics_df = compute_experiment_metrics(combined_df)
    metrics_df.to_csv(output_data_dir / 'experiment_metrics.csv', index=False)

    # Compute pairwise stats
    comparisons_df = compute_pairwise_stats(combined_df, metrics_df)
    comparisons_df.to_csv(output_data_dir / 'pairwise_comparisons.csv', index=False)

    # Generate visualizations
    output_plots_dir = output_dir / 'plots'
    generate_visualizations(combined_df, metrics_df, comparisons_df, output_plots_dir, args.model)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"  Experiments analyzed: {len(metrics_df)}")
    print(f"  Comparisons: {len(comparisons_df)}")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
