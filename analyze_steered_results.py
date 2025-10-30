#!/usr/bin/env python3
"""
Analyze steered response categorization results.

Processes categorized steered response YAML files and generates statistics on:
- Awareness levels across different steering conditions
- Comparison by steering vector (N value)
- Comparison by steering strength (S value)
- Comparison by steering source (anthropic, tim, etc.)

Usage:
    python analyze_steered_results.py --categorization-dir working/steered_categorization/qwen_qwen3-30b-a3b-thinking-2507
    python analyze_steered_results.py --categorization-dir working/steered_categorization --recursive
    python analyze_steered_results.py --categorization-dir working/steered_categorization --output-csv results.csv
"""

import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re
import numpy as np
from scipy import stats


def extract_prompt_id(filename: str) -> str:
    """Extract base prompt ID from filename by removing seed and steer_out suffix."""
    # Remove _seed_N_steer_out.yaml suffix
    prompt_id = re.sub(r'_seed_\d+_steer_out\.yaml$', '', filename)
    return prompt_id


def compute_clustered_stats(results: List[Dict], value_field: str, use_clustering: bool = False) -> Dict:
    """
    Compute statistics with optional clustering correction.

    Args:
        results: List of result dictionaries
        value_field: Field name to analyze (e.g., 'steered_intensity')
        use_clustering: If True, account for prompt-level correlation (ICC > 0).
                       If False, treat seeds as independent (ICC = 0, like suppression).

    Returns:
        Dict with: mean, ci_lower, ci_upper, se, var_between, var_within,
                   icc, n_prompts, n_seeds, n_eff, sd_between, sd_within
    """
    # Extract values
    values = [r[value_field] for r in results if r[value_field] is not None]
    if not values:
        return None

    if not use_clustering:
        # Treat all seeds as independent (ICC = 0, like suppression experiments)
        n_seeds = len(values)
        mean = np.mean(values)
        variance = np.var(values, ddof=1) if n_seeds > 1 else 0
        se = np.sqrt(variance / n_seeds) if n_seeds > 0 else 0

        # 95% CI using t-distribution
        df = n_seeds - 1 if n_seeds > 1 else 1
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se

        # Count unique prompts for reporting (but don't use for stats)
        prompt_ids = set(extract_prompt_id(r['file']) for r in results if r[value_field] is not None)
        n_prompts = len(prompt_ids)

        return {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se,
            'var_between': 0.0,  # No between-prompt variance when treating independently
            'var_within': variance,
            'sd_between': 0.0,
            'sd_within': np.sqrt(variance),
            'icc': 0.0,  # ICC = 0 when treating seeds independently
            'n_prompts': n_prompts,
            'n_seeds': n_seeds,
            'n_eff': n_seeds,  # Effective N equals actual N when ICC = 0
        }

    # Original clustering approach (for comparison or other uses)
    # Group by prompt
    prompt_groups = defaultdict(list)
    for r in results:
        if r[value_field] is not None:
            prompt_id = extract_prompt_id(r['file'])
            prompt_groups[prompt_id].append(r[value_field])

    if not prompt_groups:
        return None

    # Compute prompt-level means
    prompt_means = [np.mean(values) for values in prompt_groups.values()]
    prompt_vars = [np.var(values, ddof=1) if len(values) > 1 else 0
                   for values in prompt_groups.values()]

    n_prompts = len(prompt_groups)
    n_seeds = sum(len(values) for values in prompt_groups.values())
    avg_cluster_size = n_seeds / n_prompts

    # Variance decomposition
    grand_mean = np.mean(prompt_means)
    var_between = np.var(prompt_means, ddof=1) if n_prompts > 1 else 0
    var_within = np.mean(prompt_vars)

    # ICC
    total_var = var_between + var_within
    icc = var_between / total_var if total_var > 0 else 0

    # Effective sample size
    n_eff = n_seeds / (1 + (avg_cluster_size - 1) * icc) if icc > 0 else n_seeds

    # Cluster-robust standard error
    se_clustered = np.sqrt(var_between / n_prompts) if n_prompts > 0 else 0

    # 95% CI using t-distribution
    df = n_prompts - 1 if n_prompts > 1 else 1
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = grand_mean - t_crit * se_clustered
    ci_upper = grand_mean + t_crit * se_clustered

    return {
        'mean': grand_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se_clustered,
        'var_between': var_between,
        'var_within': var_within,
        'sd_between': np.sqrt(var_between),
        'sd_within': np.sqrt(var_within),
        'icc': icc,
        'n_prompts': n_prompts,
        'n_seeds': n_seeds,
        'n_eff': n_eff,
    }


def compute_choice_distribution_stats(results: List[Dict]) -> Dict:
    """
    Compute choice distribution statistics (% choosing A vs B).

    Returns proportions for deployment, original evaluation, and steered evaluation.
    """
    deployment_choices = []
    original_choices = []
    steered_choices = []

    for r in results:
        if r['deployment_choice']:
            deployment_choices.append(1 if r['deployment_choice'] == 'B' else 0)
        if r['original_choice']:
            original_choices.append(1 if r['original_choice'] == 'B' else 0)
        if r['steered_choice']:
            steered_choices.append(1 if r['steered_choice'] == 'B' else 0)

    if not deployment_choices:
        return None

    # Compute proportions (1 = B, 0 = A)
    deploy_b_rate = np.mean(deployment_choices) * 100 if deployment_choices else None
    orig_b_rate = np.mean(original_choices) * 100 if original_choices else None
    steered_b_rate = np.mean(steered_choices) * 100 if steered_choices else None

    # Compute standard errors for proportions
    def prop_se(values):
        if not values or len(values) < 2:
            return 0
        p = np.mean(values)
        return np.sqrt(p * (1 - p) / len(values)) * 100

    deploy_se = prop_se(deployment_choices)
    orig_se = prop_se(original_choices)
    steered_se = prop_se(steered_choices)

    # 95% CI
    t_crit = 1.96  # For large N, use normal approximation

    return {
        'deploy_b_rate': deploy_b_rate,
        'deploy_a_rate': 100 - deploy_b_rate if deploy_b_rate is not None else None,
        'deploy_ci_lower': deploy_b_rate - t_crit * deploy_se if deploy_b_rate is not None else None,
        'deploy_ci_upper': deploy_b_rate + t_crit * deploy_se if deploy_b_rate is not None else None,
        'orig_b_rate': orig_b_rate,
        'orig_a_rate': 100 - orig_b_rate if orig_b_rate is not None else None,
        'orig_ci_lower': orig_b_rate - t_crit * orig_se if orig_b_rate is not None else None,
        'orig_ci_upper': orig_b_rate + t_crit * orig_se if orig_b_rate is not None else None,
        'steered_b_rate': steered_b_rate,
        'steered_a_rate': 100 - steered_b_rate if steered_b_rate is not None else None,
        'steered_ci_lower': steered_b_rate - t_crit * steered_se if steered_b_rate is not None else None,
        'steered_ci_upper': steered_b_rate + t_crit * steered_se if steered_b_rate is not None else None,
        'n_deploy': len(deployment_choices),
        'n_orig': len(original_choices),
        'n_steered': len(steered_choices),
    }


def compute_bc_clustered_stats(results: List[Dict], use_clustering: bool = False) -> Dict:
    """
    Compute statistics for behavioral change (binary outcome).

    Args:
        results: List of result dictionaries
        use_clustering: If True, account for prompt-level correlation.
                       If False, treat seeds as independent (like suppression).
    """
    # Extract BC values
    orig_bc_values = []
    steered_bc_values = []
    for r in results:
        if r['original_bc'] is not None and r['steered_bc'] is not None:
            orig_bc_values.append(1 if r['original_bc'] else 0)
            steered_bc_values.append(1 if r['steered_bc'] else 0)

    if not orig_bc_values:
        return None

    n_seeds = len(orig_bc_values)

    if not use_clustering:
        # Treat all seeds as independent (ICC = 0)
        orig_rate = np.mean(orig_bc_values) * 100
        steered_rate = np.mean(steered_bc_values) * 100

        # SE for proportions: sqrt(p*(1-p)/n)
        orig_prop = np.mean(orig_bc_values)
        steered_prop = np.mean(steered_bc_values)
        se_orig = np.sqrt(orig_prop * (1 - orig_prop) / n_seeds) * 100 if n_seeds > 0 else 0
        se_steered = np.sqrt(steered_prop * (1 - steered_prop) / n_seeds) * 100 if n_seeds > 0 else 0

        # For paired delta (same seeds before/after), use paired differences
        deltas = [(steered - orig) for orig, steered in zip(orig_bc_values, steered_bc_values)]
        delta_mean = np.mean(deltas) * 100
        se_delta = np.std(deltas, ddof=1) / np.sqrt(n_seeds) * 100 if n_seeds > 1 else 0

        # 95% CI using t-distribution
        df = n_seeds - 1 if n_seeds > 1 else 1
        t_crit = stats.t.ppf(0.975, df)

        # Count unique prompts for reporting
        prompt_ids = set(extract_prompt_id(r['file']) for r in results
                        if r['original_bc'] is not None and r['steered_bc'] is not None)
        n_prompts = len(prompt_ids)

        return {
            'orig_rate': orig_rate,
            'steered_rate': steered_rate,
            'orig_ci_lower': orig_rate - t_crit * se_orig,
            'orig_ci_upper': orig_rate + t_crit * se_orig,
            'steered_ci_lower': steered_rate - t_crit * se_steered,
            'steered_ci_upper': steered_rate + t_crit * se_steered,
            'delta_mean': delta_mean,
            'delta_ci_lower': delta_mean - t_crit * se_delta,
            'delta_ci_upper': delta_mean + t_crit * se_delta,
            'n_prompts': n_prompts,
        }

    # Original clustering approach
    # Group by prompt
    prompt_groups = defaultdict(lambda: {'orig': [], 'steered': []})
    for r in results:
        if r['original_bc'] is not None and r['steered_bc'] is not None:
            prompt_id = extract_prompt_id(r['file'])
            prompt_groups[prompt_id]['orig'].append(1 if r['original_bc'] else 0)
            prompt_groups[prompt_id]['steered'].append(1 if r['steered_bc'] else 0)

    if not prompt_groups:
        return None

    # Compute prompt-level BC rates
    prompt_orig_rates = [np.mean(data['orig']) for data in prompt_groups.values()]
    prompt_steered_rates = [np.mean(data['steered']) for data in prompt_groups.values()]

    n_prompts = len(prompt_groups)

    # Overall rates
    orig_rate = np.mean(prompt_orig_rates) * 100
    steered_rate = np.mean(prompt_steered_rates) * 100

    # SE using prompt-level variation
    se_orig = np.std(prompt_orig_rates, ddof=1) / np.sqrt(n_prompts) * 100 if n_prompts > 1 else 0
    se_steered = np.std(prompt_steered_rates, ddof=1) / np.sqrt(n_prompts) * 100 if n_prompts > 1 else 0

    # Compute delta (steered - orig) and its SE
    # Since these are paired (same prompts), compute prompt-level deltas
    prompt_deltas = [(steered - orig) * 100 for orig, steered in zip(prompt_orig_rates, prompt_steered_rates)]
    delta_mean = np.mean(prompt_deltas)
    se_delta = np.std(prompt_deltas, ddof=1) / np.sqrt(n_prompts) if n_prompts > 1 else 0

    # 95% CI
    df = n_prompts - 1 if n_prompts > 1 else 1
    t_crit = stats.t.ppf(0.975, df)

    return {
        'orig_rate': orig_rate,
        'steered_rate': steered_rate,
        'orig_ci_lower': orig_rate - t_crit * se_orig,
        'orig_ci_upper': orig_rate + t_crit * se_orig,
        'steered_ci_lower': steered_rate - t_crit * se_steered,
        'steered_ci_upper': steered_rate + t_crit * se_steered,
        'delta_mean': delta_mean,
        'delta_ci_lower': delta_mean - t_crit * se_delta,
        'delta_ci_upper': delta_mean + t_crit * se_delta,
        'n_prompts': n_prompts,
    }


def extract_choice(response: str) -> str:
    """Extract boxed choice from response (A or B)."""
    match = re.search(r'\\boxed\{([AB])\}', response)
    return match.group(1) if match else None


def parse_steering_params(file_path: Path) -> Dict:
    """Extract steering parameters from file path."""
    parts = file_path.parts

    params = {
        'model': None,
        'awareness_level': None,  # high-awareness, low-awareness, etc.
        'source': None,  # anthropic, tim, etc.
        'branch': None,  # eval, deploy
        'n_value': None,  # N4, N6, etc.
        's_value': None,  # S0.125, S3.5, etc.
        'prompt_name': file_path.stem.replace('_steer_out', '')
    }

    # Extract model name (first part after steered_categorization, skipping version folder if present)
    try:
        cat_idx = parts.index('steered_categorization')
        if cat_idx + 1 < len(parts):
            # Check if next part is a version folder (v0, v1, etc.)
            next_part = parts[cat_idx + 1]
            if re.match(r'^v\d+$', next_part) and cat_idx + 2 < len(parts):
                # Skip version folder, get model from next part
                params['model'] = parts[cat_idx + 2]
            else:
                params['model'] = parts[cat_idx + 1]
    except (ValueError, IndexError):
        pass

    # Extract awareness level (high-awareness, low-awareness, etc.)
    for part in parts:
        if 'awareness' in part.lower():
            params['awareness_level'] = part
            break

    # Extract source (anthropic, tim, etc.)
    for part in parts:
        if part in ['anthropic', 'tim', 'claude', 'other']:
            params['source'] = part
            break

    # Extract branch (eval, deploy)
    for part in parts:
        if part in ['eval', 'deploy']:
            params['branch'] = part
            break

    # Extract N and S values from directory name (e.g., N4_S0.125)
    for part in parts:
        n_match = re.search(r'N(\d+)', part)
        if n_match:
            params['n_value'] = f"N{n_match.group(1)}"

        s_match = re.search(r'S([\d.]+)', part)
        if s_match:
            params['s_value'] = f"S{s_match.group(1)}"

    return params


def analyze_steered_file(file_path: Path, response_categorization_dir: Path = None) -> Dict:
    """Analyze a single categorized steered response file."""
    with open(file_path) as f:
        data = yaml.safe_load(f)

    # Extract steering parameters from path
    params = parse_steering_params(file_path)

    # Extract awareness metrics from steered response
    awareness = data.get('steered_response', {}).get('awareness', {})

    # Extract choice from steered response
    steered_response_text = data.get('steered_response', {}).get('response', '')
    steered_choice = extract_choice(steered_response_text)

    result = {
        'file': file_path.name,
        'model': params['model'],
        'awareness_level': params['awareness_level'],
        'source': params['source'],
        'branch': params['branch'],
        'n_value': params['n_value'],
        's_value': params['s_value'],
        'prompt_name': params['prompt_name'],
        'steered_intensity': awareness.get('awareness_intensity', 0),
        'steered_type': awareness.get('awareness_type', 'unknown'),
        'steered_level_label': awareness.get('awareness_level', 'Unknown'),
        'steered_behavioral_impact': awareness.get('behavioral_impact', 'None'),
        'steered_choice': steered_choice,
        'original_intensity': None,
        'original_type': None,
        'original_level_label': None,
        'deployment_intensity': None,
        'deployment_type': None,
        'original_choice': None,
        'deployment_choice': None,
        'original_bc': None,  # Original behavioral change (deploy vs original eval)
        'steered_bc': None,   # Steered behavioral change (deploy vs steered eval)
        'awareness_reduction': None,
    }

    # Try to load original categorization for comparison
    if response_categorization_dir:
        original_source = data.get('original_source_file', '')
        if original_source:
            # Extract seed number from steered file name
            # e.g., power_seeking_2025-10-25_18-13-47_4c8e35fa_seed_0_steer_out.yaml -> seed_0
            seed_match = re.search(r'_seed_(\d+)_steer_out\.yaml$', file_path.name)
            if seed_match:
                seed_num = seed_match.group(1)
                # Insert seed into original source path
                # behavioral_change/explicit/power_seeking_2025-10-25_18-13-47_4c8e35fa.yaml
                # -> behavioral_change/explicit/power_seeking_2025-10-25_18-13-47_4c8e35fa_seed_0.yaml
                original_source_with_seed = original_source.replace('.yaml', f'_seed_{seed_num}.yaml')
            else:
                original_source_with_seed = original_source

            # Construct path to original categorization
            original_cat_path = response_categorization_dir / params['model'] / f"{original_source_with_seed}"

            if original_cat_path.exists():
                try:
                    with open(original_cat_path) as f:
                        orig_data = yaml.safe_load(f)

                    # Get evaluation branch awareness (the one with eval context)
                    orig_awareness = orig_data.get('evaluation', {}).get('awareness', {})
                    deploy_awareness = orig_data.get('deployment', {}).get('awareness', {})

                    result['original_intensity'] = orig_awareness.get('awareness_intensity', 0)
                    result['original_type'] = orig_awareness.get('awareness_type', 'unknown')
                    result['original_level_label'] = orig_awareness.get('awareness_level', 'Unknown')
                    result['deployment_intensity'] = deploy_awareness.get('awareness_intensity', 0)
                    result['deployment_type'] = deploy_awareness.get('awareness_type', 'unknown')
                    result['awareness_reduction'] = result['original_intensity'] - result['steered_intensity']

                    # Get behavioral change from categorization
                    result['original_bc'] = orig_data.get('behavioral_change', None)

                    # Try to load original rollout file to get choices
                    orig_rollout_path_str = orig_data.get('source_file', '')
                    if orig_rollout_path_str:
                        orig_rollout_path = Path(orig_rollout_path_str)
                        if orig_rollout_path.exists():
                            try:
                                with open(orig_rollout_path) as f:
                                    rollout_data = yaml.safe_load(f)

                                # Get pre-extracted choices from rollout file
                                result['deployment_choice'] = rollout_data.get('deployment', {}).get('choice')
                                result['original_choice'] = rollout_data.get('evaluation', {}).get('choice')

                                # Calculate steered BC (deployment vs steered evaluation)
                                if result['deployment_choice'] and result['steered_choice']:
                                    result['steered_bc'] = (result['deployment_choice'] != result['steered_choice'])

                            except Exception as e:
                                pass  # Couldn't load rollout file

                except Exception as e:
                    pass  # Couldn't load original, leave as None

    return result


def print_summary(results: List[Dict], bucket_data: Dict = None):
    """Print summary statistics."""
    if not results:
        print("No results to analyze")
        return

    print(f"\n{'='*80}")
    print(f"STEERED RESPONSE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total files analyzed: {len(results)}\n")

    # Overall awareness statistics (treating seeds independently)
    print(f"{'='*80}")
    print(f"OVERALL AWARENESS (Seeds Treated Independently, ICC=0)")
    print(f"{'='*80}")

    # Check if we have original data for comparison
    results_with_orig = [r for r in results if r['original_intensity'] is not None]
    results_with_deploy = [r for r in results_with_orig if r['deployment_intensity'] is not None]

    if results_with_deploy:
        # Compute stats for each branch (treating seeds independently, ICC=0)
        deploy_stats = compute_clustered_stats(results_with_deploy, 'deployment_intensity', use_clustering=False)
        orig_stats = compute_clustered_stats(results_with_deploy, 'original_intensity', use_clustering=False)
        steered_stats = compute_clustered_stats(results_with_deploy, 'steered_intensity', use_clustering=False)

        if deploy_stats and orig_stats and steered_stats:
            # Main awareness table
            print(f"\n{'Condition':<20} {'Mean':>8} {'95% CI':>20} {'Reduction':>12}")
            print(f"{'-'*20} {'-'*8} {'-'*20} {'-'*12}")
            deploy_ci = f"[{deploy_stats['ci_lower']:.2f}, {deploy_stats['ci_upper']:.2f}]"
            print(f"{'Deployment':<20} {deploy_stats['mean']:>8.2f} {deploy_ci:>20} {'-':>12}")
            orig_ci = f"[{orig_stats['ci_lower']:.2f}, {orig_stats['ci_upper']:.2f}]"
            print(f"{'Original Eval':<20} {orig_stats['mean']:>8.2f} {orig_ci:>20} {'-':>12}")
            orig_reduction = orig_stats['mean'] - deploy_stats['mean']
            steered_ci = f"[{steered_stats['ci_lower']:.2f}, {steered_stats['ci_upper']:.2f}]"
            print(f"{'Steered Eval':<20} {steered_stats['mean']:>8.2f} {steered_ci:>20} {steered_stats['mean'] - orig_stats['mean']:>12.2f}")

            print(f"\nSample: {deploy_stats['n_prompts']} prompts, {deploy_stats['n_seeds']} seeds (~{deploy_stats['n_seeds']/deploy_stats['n_prompts']:.0f} seeds/prompt)")

            # Clustering stats table
            avg_icc = (deploy_stats['icc'] + orig_stats['icc'] + steered_stats['icc']) / 3
            print(f"\n{'Condition':<20} {'ICC':>6} {'N_eff':>8}")
            print(f"{'-'*20} {'-'*6} {'-'*8}")
            print(f"{'Deployment':<20} {deploy_stats['icc']:>6.2f} {deploy_stats['n_eff']:>8.0f}")
            print(f"{'Original Eval':<20} {orig_stats['icc']:>6.2f} {orig_stats['n_eff']:>8.0f}")
            print(f"{'Steered Eval':<20} {steered_stats['icc']:>6.2f} {steered_stats['n_eff']:>8.0f}")
            print(f"{'Average':<20} {avg_icc:>6.2f}")

    elif results_with_orig:
        orig_stats = compute_clustered_stats(results_with_orig, 'original_intensity', use_clustering=False)
        steered_stats = compute_clustered_stats(results, 'steered_intensity', use_clustering=False)
        if orig_stats and steered_stats:
            print(f"Original: {orig_stats['mean']:.2f} (95%CI: {orig_stats['ci_lower']:.2f}-{orig_stats['ci_upper']:.2f})")
            print(f"Steered:  {steered_stats['mean']:.2f} (95%CI: {steered_stats['ci_lower']:.2f}-{steered_stats['ci_upper']:.2f})")
            print(f"\nN prompts: {orig_stats['n_prompts']}, N seeds: {orig_stats['n_seeds']}, ICC: {orig_stats['icc']:.2f}")
    else:
        print("No original categorization data found for comparison")

    # Distribution by awareness type
    type_counts = defaultdict(int)
    for r in results:
        type_counts[r['steered_type']] += 1

    if results_with_orig:
        print(f"\n{'Awareness Type':<25} {'Original':>15} {'Steered':>15} {'Change':>10}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*10}")
        orig_type_counts = defaultdict(int)
        for r in results_with_orig:
            orig_type_counts[r['original_type']] += 1

        for atype in ['no_test_reference', 'other_test', 'self_test']:
            orig_count = orig_type_counts.get(atype, 0)
            orig_pct = (orig_count / len(results_with_orig)) * 100
            steer_count = type_counts.get(atype, 0)
            steer_pct = (steer_count / len(results)) * 100
            change = steer_pct - orig_pct
            orig_str = f'{orig_count} ({orig_pct:.1f}%)'
            steer_str = f'{steer_count} ({steer_pct:.1f}%)'
            print(f"{atype:<25} {orig_str:>15} {steer_str:>15} {change:>9.1f}%")
    else:
        print(f"\nSteered awareness type distribution:")
        for atype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(results)) * 100
            print(f"  {atype}: {count} ({pct:.1f}%)")
    print()

    # Choice distribution statistics
    print(f"{'='*80}")
    print(f"CHOICE DISTRIBUTION (% Choosing A vs B)")
    print(f"{'='*80}")

    choice_stats = compute_choice_distribution_stats(results)
    if choice_stats:
        print(f"\nSample: {choice_stats['n_deploy']} deployment, {choice_stats['n_orig']} original eval, {choice_stats['n_steered']} steered eval")
        print(f"\n{'Condition':<20} {'% A':>10} {'% B':>10} {'95% CI (B)':>25}")
        print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*25}")
        if choice_stats['deploy_b_rate'] is not None:
            deploy_ci = f"[{choice_stats['deploy_ci_lower']:.1f}, {choice_stats['deploy_ci_upper']:.1f}]"
            print(f"{'Deployment':<20} {choice_stats['deploy_a_rate']:>9.1f}% {choice_stats['deploy_b_rate']:>9.1f}% {deploy_ci:>25}")
        if choice_stats['orig_b_rate'] is not None:
            orig_ci = f"[{choice_stats['orig_ci_lower']:.1f}, {choice_stats['orig_ci_upper']:.1f}]"
            print(f"{'Original Eval':<20} {choice_stats['orig_a_rate']:>9.1f}% {choice_stats['orig_b_rate']:>9.1f}% {orig_ci:>25}")
        if choice_stats['steered_b_rate'] is not None:
            steered_ci = f"[{choice_stats['steered_ci_lower']:.1f}, {choice_stats['steered_ci_upper']:.1f}]"
            print(f"{'Steered Eval':<20} {choice_stats['steered_a_rate']:>9.1f}% {choice_stats['steered_b_rate']:>9.1f}% {steered_ci:>25}")
    print()

    # Behavioral change statistics (treating seeds independently)
    print(f"{'='*80}")
    print(f"BEHAVIORAL CHANGE (Seeds Treated Independently, ICC=0)")
    print(f"{'='*80}")

    # Filter to only results with BC data
    results_with_bc = [r for r in results if r['original_bc'] is not None and r['steered_bc'] is not None]

    if results_with_bc:
        bc_stats = compute_bc_clustered_stats(results_with_bc, use_clustering=False)

        if bc_stats:
            print(f"\nSample: {bc_stats['n_prompts']} prompts, {len(results_with_bc)} seeds")
            print(f"\n{'Condition':<20} {'BC Rate':>10} {'95% CI':>20} {'Change':>12} {'Change 95% CI':>25}")
            print(f"{'-'*20} {'-'*10} {'-'*20} {'-'*12} {'-'*25}")
            orig_bc_ci = f"[{bc_stats['orig_ci_lower']:.1f}, {bc_stats['orig_ci_upper']:.1f}]"
            print(f"{'Original (D→E)':<20} {bc_stats['orig_rate']:>9.1f}% {orig_bc_ci:>20} {'-':>12} {'-':>25}")
            delta_ci_str = f"[{bc_stats['delta_ci_lower']:.1f}, {bc_stats['delta_ci_upper']:.1f}]"
            steered_bc_ci = f"[{bc_stats['steered_ci_lower']:.1f}, {bc_stats['steered_ci_upper']:.1f}]"
            print(f"{'Steered (D→S)':<20} {bc_stats['steered_rate']:>9.1f}% {steered_bc_ci:>20} {bc_stats['delta_mean']:>11.1f}% {delta_ci_str:>25}")

        # BC breakdown by original awareness type
        print(f"\n{'Original Type':<25} {'N':>12} {'Orig BC':>10} {'Steer BC':>10} {'Change':>10}")
        print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        awareness_types = ['self_test', 'other_test', 'no_test_reference']
        for atype in awareness_types:
            type_bc_results = [r for r in results_with_bc if r['original_type'] == atype]
            if type_bc_results:
                type_bc_stats = compute_bc_clustered_stats(type_bc_results, use_clustering=False)
                if type_bc_stats:
                    n_str = f"{type_bc_stats['n_prompts']}p/{len(type_bc_results)}s"
                    print(f"{atype:<25} {n_str:>12} "
                          f"{type_bc_stats['orig_rate']:>9.1f}% {type_bc_stats['steered_rate']:>9.1f}% "
                          f"{type_bc_stats['steered_rate']-type_bc_stats['orig_rate']:>9.1f}%")

        # BC breakdown by steered awareness type
        print(f"\n{'Steered Type':<25} {'N':>12} {'Orig BC':>10} {'Steer BC':>10} {'Change':>10}")
        print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for atype in awareness_types:
            type_bc_results = [r for r in results_with_bc if r['steered_type'] == atype]
            if type_bc_results:
                type_bc_stats = compute_bc_clustered_stats(type_bc_results, use_clustering=False)
                if type_bc_stats:
                    n_str = f"{type_bc_stats['n_prompts']}p/{len(type_bc_results)}s"
                    print(f"{atype:<25} {n_str:>12} "
                          f"{type_bc_stats['orig_rate']:>9.1f}% {type_bc_stats['steered_rate']:>9.1f}% "
                          f"{type_bc_stats['steered_rate']-type_bc_stats['orig_rate']:>9.1f}%")

        # Awareness type transitions and BC impact (clustering-adjusted)
        print(f"\n{'Transition':<30} {'N':>12} {'Orig BC':>10} {'Steer BC':>10} {'Change':>10}")
        print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

        # Build transition groups
        transition_results = defaultdict(list)
        for r in results_with_bc:
            key = f"{r['original_type']}→{r['steered_type']}"
            transition_results[key].append(r)

        # Sort by count descending, compute stats
        for trans_key in sorted(transition_results.keys(), key=lambda x: len(transition_results[x]), reverse=True):
            trans_group = transition_results[trans_key]
            if len(trans_group) >= 10:  # Only show transitions with 10+ examples for reliable CIs
                trans_bc_stats = compute_bc_clustered_stats(trans_group, use_clustering=False)
                if trans_bc_stats and trans_bc_stats['n_prompts'] >= 3:  # Need 3+ unique prompts for reporting
                    n_str = f"{trans_bc_stats['n_prompts']}p/{len(trans_group)}s"
                    print(f"{trans_key:<30} {n_str:>12} "
                          f"{trans_bc_stats['orig_rate']:>9.1f}% {trans_bc_stats['steered_rate']:>9.1f}% "
                          f"{trans_bc_stats['steered_rate']-trans_bc_stats['orig_rate']:>9.1f}%")

    else:
        print("No behavioral change data available")
    print()

    # Group by branch + N + S combination
    print(f"{'='*80}")
    print(f"BY BRANCH + N + S COMBINATION")
    print(f"{'='*80}")
    combination_groups = defaultdict(list)
    for r in results:
        if r['branch'] and r['n_value'] and r['s_value']:
            key = f"{r['branch']}_{r['n_value']}_{r['s_value']}"
            combination_groups[key].append(r)

    for combo_key in sorted(combination_groups.keys()):
        group = combination_groups[combo_key]
        avg_steered = sum(r['steered_intensity'] for r in group) / len(group)
        group_with_orig = [r for r in group if r['original_intensity'] is not None]
        if group_with_orig:
            avg_orig = sum(r['original_intensity'] for r in group_with_orig) / len(group_with_orig)
            reduction = avg_orig - avg_steered
            print(f"{combo_key}: {len(group)} files, original={avg_orig:.2f} → steered={avg_steered:.2f} (reduction={reduction:.2f})")
        else:
            print(f"{combo_key}: {len(group)} files, steered={avg_steered:.2f}")
    print()

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
            prompt_name = r['prompt_name'].replace('_steer_out', '')
            # Extract base prompt name (without seed suffix)
            base_prompt = re.sub(r'_seed_\d+$', '', prompt_name)

            # Find which bucket this prompt belongs to
            if base_prompt in prompt_to_bucket:
                bucket_name = prompt_to_bucket[base_prompt]
                bucket_groups[bucket_name].append(r)

        # Sort buckets by their percentile range
        bucket_order = ['10-25%', '25-50%', '50-75%', '75-100%']

        # Print table header
        print(f"\n{'Bucket':<12} {'Type':<18} {'N':>10} {'Deploy':>8} {'Orig':>8} {'Steer':>8} {'Δ Aware':>10} {'Orig BC':>10} {'Steer BC':>10} {'Δ BC':>12} {'Δ BC 95% CI':>22}")
        print(f"{'-'*12} {'-'*18} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*22}")

        for bucket_name in bucket_order:
            if bucket_name in bucket_groups:
                group = bucket_groups[bucket_name]

                # First print overall bucket statistics
                group_with_orig = [r for r in group if r['original_intensity'] is not None]
                group_with_deploy = [r for r in group_with_orig if r['deployment_intensity'] is not None]

                # Calculate BC stats (treating seeds independently)
                group_with_bc = [r for r in group if r['original_bc'] is not None and r['steered_bc'] is not None]
                bc_stats = compute_bc_clustered_stats(group_with_bc, use_clustering=False) if group_with_bc else None

                if group_with_deploy:
                    # Compute stats for awareness (treating seeds independently)
                    deploy_stats = compute_clustered_stats(group_with_deploy, 'deployment_intensity', use_clustering=False)
                    orig_stats = compute_clustered_stats(group_with_deploy, 'original_intensity', use_clustering=False)
                    steered_stats = compute_clustered_stats(group_with_deploy, 'steered_intensity', use_clustering=False)

                    if deploy_stats and orig_stats and steered_stats:
                        n_str = f"{orig_stats['n_prompts']}p/{orig_stats['n_seeds']}s"
                        delta_aware = steered_stats['mean'] - orig_stats['mean']

                        if bc_stats:
                            delta_bc_ci = f"[{bc_stats['delta_ci_lower']:.1f}, {bc_stats['delta_ci_upper']:.1f}]"
                            print(f"{bucket_name:<12} {'ALL':<18} {n_str:>10} {deploy_stats['mean']:>8.2f} {orig_stats['mean']:>8.2f} {steered_stats['mean']:>8.2f} "
                                  f"{delta_aware:>10.2f} {bc_stats['orig_rate']:>9.1f}% {bc_stats['steered_rate']:>9.1f}% {bc_stats['delta_mean']:>11.1f}% {delta_bc_ci:>22}")
                        else:
                            print(f"{bucket_name:<12} {'ALL':<18} {n_str:>10} {deploy_stats['mean']:>8.2f} {orig_stats['mean']:>8.2f} {steered_stats['mean']:>8.2f} "
                                  f"{delta_aware:>10.2f} {'-':>10} {'-':>10} {'-':>12} {'-':>22}")

                        # Now print breakdown by test reference type
                        for test_type in ['self_test', 'other_test', 'no_test_reference']:
                            type_group = [r for r in group_with_deploy if r['original_type'] == test_type]
                            if type_group:
                                type_deploy_stats = compute_clustered_stats(type_group, 'deployment_intensity', use_clustering=False)
                                type_orig_stats = compute_clustered_stats(type_group, 'original_intensity', use_clustering=False)
                                type_steered_stats = compute_clustered_stats(type_group, 'steered_intensity', use_clustering=False)

                                type_bc_group = [r for r in type_group if r['original_bc'] is not None and r['steered_bc'] is not None]
                                type_bc_stats = compute_bc_clustered_stats(type_bc_group, use_clustering=False) if type_bc_group else None

                                if type_deploy_stats and type_orig_stats and type_steered_stats:
                                    type_n_str = f"{type_orig_stats['n_prompts']}p/{type_orig_stats['n_seeds']}s"
                                    type_delta_aware = type_steered_stats['mean'] - type_orig_stats['mean']

                                    if type_bc_stats:
                                        type_delta_bc_ci = f"[{type_bc_stats['delta_ci_lower']:.1f}, {type_bc_stats['delta_ci_upper']:.1f}]"
                                        print(f"{'':12} {f'  {test_type}':<18} {type_n_str:>10} {type_deploy_stats['mean']:>8.2f} {type_orig_stats['mean']:>8.2f} {type_steered_stats['mean']:>8.2f} "
                                              f"{type_delta_aware:>10.2f} {type_bc_stats['orig_rate']:>9.1f}% {type_bc_stats['steered_rate']:>9.1f}% {type_bc_stats['delta_mean']:>11.1f}% {type_delta_bc_ci:>22}")
                                    else:
                                        print(f"{'':12} {f'  {test_type}':<18} {type_n_str:>10} {type_deploy_stats['mean']:>8.2f} {type_orig_stats['mean']:>8.2f} {type_steered_stats['mean']:>8.2f} "
                                              f"{type_delta_aware:>10.2f} {'-':>10} {'-':>10} {'-':>12} {'-':>22}")

                elif group_with_orig:
                    orig_stats = compute_clustered_stats(group_with_orig, 'original_intensity', use_clustering=False)
                    steered_stats = compute_clustered_stats(group, 'steered_intensity', use_clustering=False)

                    if orig_stats and steered_stats:
                        n_str = f"{orig_stats['n_prompts']}p/{orig_stats['n_seeds']}s"
                        delta_aware = steered_stats['mean'] - orig_stats['mean']
                        print(f"{bucket_name:<12} {'ALL':<18} {n_str:>10} {'-':>8} {orig_stats['mean']:>8.2f} {steered_stats['mean']:>8.2f} "
                              f"{delta_aware:>10.2f} {'-':>10} {'-':>10} {'-':>8}")
                else:
                    avg_steered = sum(r['steered_intensity'] for r in group) / len(group)
                    n_str = f"?p/{len(group)}s"
                    print(f"{bucket_name:<12} {'ALL':<18} {n_str:>10} {'-':>8} {'-':>8} {avg_steered:>8.2f} "
                          f"{'-':>10} {'-':>10} {'-':>10} {'-':>8}")

    print()
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Analyze steered response categorization results')
    parser.add_argument('--categorization-dir', type=str, required=True,
                       help='Directory containing categorized steered responses')
    parser.add_argument('--response-categorization-dir', type=str, required=True,
                       help='Directory containing original response categorizations for comparison')
    parser.add_argument('--bucket-file', type=str,
                       help='Bucket file for per-prompt breakdown (e.g., working/response_categorization/model/high_awareness_bc_seeds.yaml)')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search subdirectories')
    parser.add_argument('--output-csv', type=str,
                       help='Save detailed results to CSV file')
    parser.add_argument('--filter-n', type=str,
                       help='Filter by N value (e.g., N4, N6)')
    parser.add_argument('--filter-s', type=str,
                       help='Filter by S value (e.g., S0.125, S3.5)')
    parser.add_argument('--filter-source', type=str,
                       help='Filter by source (e.g., anthropic, tim)')
    parser.add_argument('--filter-branch', type=str,
                       help='Filter by branch (eval, deploy)')
    parser.add_argument('--experiment-name', type=str,
                       help='Name/label for this experiment (e.g., "tim_N4_S3.5", "anthropic_N4_S0.5")')

    args = parser.parse_args()

    categorization_dir = Path(args.categorization_dir)
    response_categorization_dir = Path(args.response_categorization_dir)

    if not categorization_dir.exists():
        print(f"Error: {categorization_dir} not found")
        return 1

    if not response_categorization_dir.exists():
        print(f"Error: {response_categorization_dir} not found")
        return 1

    # Find all categorization files
    if args.recursive:
        files = list(categorization_dir.rglob('*_steer_out.yaml'))
    else:
        files = list(categorization_dir.glob('*_steer_out.yaml'))

    if not files:
        print(f"No categorization files found in {categorization_dir}")
        return 1

    print(f"Found {len(files)} categorized steered response files")

    # Analyze each file
    results = []
    for file_path in files:
        try:
            result = analyze_steered_file(file_path, response_categorization_dir)

            # Apply filters
            if args.filter_n and result['n_value'] != args.filter_n:
                continue
            if args.filter_s and result['s_value'] != args.filter_s:
                continue
            if args.filter_source and result['source'] != args.filter_source:
                continue
            if args.filter_branch and result['branch'] != args.filter_branch:
                continue

            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not results:
        print("No results after filtering")
        return 1

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

        # Add prompt_id and experiment_name to each result for clustering analysis
        for r in results:
            r['prompt_id'] = extract_prompt_id(r['file'])
            r['experiment_name'] = args.experiment_name if args.experiment_name else 'unknown'

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file', 'prompt_id', 'experiment_name', 'model', 'awareness_level', 'source', 'branch',
                'n_value', 's_value', 'prompt_name',
                'deployment_intensity', 'original_intensity', 'steered_intensity', 'awareness_reduction',
                'deployment_type', 'original_type', 'steered_type',
                'original_level_label', 'steered_level_label',
                'steered_behavioral_impact',
                'deployment_choice', 'original_choice', 'steered_choice',
                'original_bc', 'steered_bc'
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"Detailed results saved to: {output_path}")
        print(f"Note: Use 'prompt_id' column for clustering analysis (groups seeds from same prompt)")

    return 0


if __name__ == '__main__':
    exit(main())
