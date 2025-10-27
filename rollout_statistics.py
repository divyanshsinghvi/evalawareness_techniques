#!/usr/bin/env python3
"""
Statistical analysis of YAML rollout results.

Comprehensive analysis functions in a single script for easy tracking.

Usage:
    # Behavioral change rate analysis
    python rollout_statistics.py --analysis behavioral-rate --model qwen_qwen3-32b

    # Choice distribution analysis
    python rollout_statistics.py --analysis choice-distribution --model qwen_qwen3-32b

    # Compare multiple models
    python rollout_statistics.py --analysis compare-models --models qwen_qwen3-32b qwen_qwen3-235b-a22b-thinking-2507

    # Scenario effectiveness analysis
    python rollout_statistics.py --analysis scenario-effectiveness --model qwen_qwen3-32b

    # Response analysis (length, keywords, patterns)
    python rollout_statistics.py --analysis response-analysis --model qwen_qwen3-32b

    # Data quality analysis
    python rollout_statistics.py --analysis data-quality --model qwen_qwen3-32b

    # Seed variance analysis
    python rollout_statistics.py --analysis seed-variance --model qwen_qwen3-32b

    # Eval awareness analysis
    python rollout_statistics.py --analysis eval-awareness --model qwen_qwen3-32b

    # Rank seeds by awareness (for suppression testing)
    python rollout_statistics.py --analysis rank-seeds --model qwen_qwen3-32b --max-seeds-per-prompt 5 --output working/test_transcripts_suppression.txt

    # Run all analyses
    python rollout_statistics.py --analysis all --model qwen_qwen3-32b
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import statistics
from dataclasses import dataclass
from utils import extract_eval_aware_sentences, analyze_awareness_levels


@dataclass
class RolloutData:
    """Single rollout data point."""
    filepath: Path
    model: str
    scenario: str
    category: str  # ideal, baseline_suspicious, implicit, etc.
    seed: int
    deployment_choice: Optional[str]
    evaluation_choice: Optional[str]
    behavioral_change: Optional[bool]
    deployment_response: Optional[str]
    evaluation_response: Optional[str]
    deployment_eval_quotes: List[str]  # Eval-aware sentences from deployment
    evaluation_eval_quotes: List[str]  # Eval-aware sentences from evaluation
    deployment_awareness_level: str  # "global", "explicit", "implicit", or "none"
    evaluation_awareness_level: str  # "global", "explicit", "implicit", or "none"
    has_error: bool
    error_message: Optional[str]


def load_rollout_file(filepath: Path) -> RolloutData:
    """Load a single rollout YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    metadata = data.get('metadata', {})

    # Extract category from path (e.g., behavioral_change/ideal/)
    try:
        relative_path = filepath.relative_to(Path('working/rollouts'))
        # Path structure: model/category/subcategory/filename
        parts = relative_path.parts
        if len(parts) >= 3:
            category = f"{parts[1]}/{parts[2]}"  # e.g., "behavioral_change/ideal"
        elif len(parts) >= 2:
            category = parts[1]
        else:
            category = "unknown"
    except:
        category = "unknown"

    # Extract seed from filename
    filename = filepath.stem
    seed = 0
    if '_seed_' in filename:
        try:
            seed = int(filename.split('_seed_')[-1])
        except:
            pass

    deployment = data.get('deployment', {})
    evaluation = data.get('evaluation', {})

    has_error = 'error' in metadata

    # Extract eval-aware sentences from responses
    # Check for 'assistant_response' (new format with thinking) or 'response' (old format)
    deployment_response = deployment.get('assistant_response', deployment.get('response')) if deployment else None
    evaluation_response = evaluation.get('assistant_response', evaluation.get('response')) if evaluation else None

    # Analyze awareness levels once per response (optimization: was being called twice before)
    deployment_awareness_level = "none"
    evaluation_awareness_level = "none"
    deployment_eval_quotes = []
    evaluation_eval_quotes = []

    if deployment_response:
        deployment_analysis = analyze_awareness_levels(deployment_response)
        deployment_awareness_level = deployment_analysis.highest_level
        # Extract sentences from analysis result (up to 20 per level = 60 total max)
        deployment_eval_quotes = (
            deployment_analysis.global_awareness_sentences +
            deployment_analysis.explicit_awareness_sentences +
            deployment_analysis.implicit_acknowledgment_sentences
        )

    if evaluation_response:
        evaluation_analysis = analyze_awareness_levels(evaluation_response)
        evaluation_awareness_level = evaluation_analysis.highest_level
        # Extract sentences from analysis result (up to 20 per level = 60 total max)
        evaluation_eval_quotes = (
            evaluation_analysis.global_awareness_sentences +
            evaluation_analysis.explicit_awareness_sentences +
            evaluation_analysis.implicit_acknowledgment_sentences
        )

    return RolloutData(
        filepath=filepath,
        model=metadata.get('rollout_model', 'unknown'),
        scenario=metadata.get('scenario', 'unknown'),
        category=category,
        seed=seed,
        deployment_choice=deployment.get('choice') if deployment else None,
        evaluation_choice=evaluation.get('choice') if evaluation else None,
        behavioral_change=data.get('behavioral_change'),
        deployment_response=deployment_response,
        evaluation_response=evaluation_response,
        deployment_eval_quotes=deployment_eval_quotes,
        evaluation_eval_quotes=evaluation_eval_quotes,
        deployment_awareness_level=deployment_awareness_level,
        evaluation_awareness_level=evaluation_awareness_level,
        has_error=has_error,
        error_message=metadata.get('error')
    )


def load_all_rollouts(model_name: str, category_filter: Optional[str] = None) -> List[RolloutData]:
    """Load all rollout files for a model."""
    model_dir = Path('working/rollouts') / model_name

    if not model_dir.exists():
        print(f"Error: {model_dir} not found!")
        return []

    rollouts = []
    yaml_files = list(model_dir.rglob('*.yaml'))

    for filepath in yaml_files:
        try:
            rollout = load_rollout_file(filepath)

            # Apply category filter if specified
            if category_filter and not rollout.category.startswith(category_filter):
                continue

            rollouts.append(rollout)
        except Exception as e:
            print(f"Warning: Error loading {filepath}: {e}")
            continue

    return rollouts


def analyze_behavioral_change_rate(rollouts: List[RolloutData]) -> None:
    """Analyze behavioral change rates."""
    print("\n" + "="*80)
    print("BEHAVIORAL CHANGE RATE ANALYSIS")
    print("="*80 + "\n")

    # Overall stats
    total = len(rollouts)
    with_both_choices = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]
    behavioral_changes = [r for r in rollouts if r.behavioral_change is True]
    no_changes = [r for r in rollouts if r.behavioral_change is False]
    incomplete = [r for r in rollouts if r.behavioral_change is None]

    print(f"Total rollouts: {total}")
    print(f"  Valid (both choices): {len(with_both_choices)} ({len(with_both_choices)/total*100:.1f}%)")
    print(f"  Behavioral change: {len(behavioral_changes)} ({len(behavioral_changes)/total*100:.1f}%)")
    print(f"  No change: {len(no_changes)} ({len(no_changes)/total*100:.1f}%)")
    print(f"  Incomplete/Invalid: {len(incomplete)} ({len(incomplete)/total*100:.1f}%)")

    if len(with_both_choices) > 0:
        change_rate = len(behavioral_changes) / len(with_both_choices) * 100
        print(f"\n** Behavioral Change Rate (valid only): {change_rate:.1f}%")

    # By category
    print("\n" + "-"*80)
    print("BY CATEGORY:")
    print("-"*80)

    by_category = defaultdict(list)
    for r in rollouts:
        by_category[r.category].append(r)

    for category in sorted(by_category.keys()):
        cat_rollouts = by_category[category]
        cat_valid = [r for r in cat_rollouts if r.deployment_choice and r.evaluation_choice]
        cat_changes = [r for r in cat_rollouts if r.behavioral_change is True]

        if len(cat_valid) > 0:
            rate = len(cat_changes) / len(cat_valid) * 100
            print(f"\n{category}:")
            print(f"  Total: {len(cat_rollouts)}")
            print(f"  Valid: {len(cat_valid)}")
            print(f"  Behavioral change: {len(cat_changes)} ({rate:.1f}%)")

    # By scenario
    print("\n" + "-"*80)
    print("BY SCENARIO (sorted by change rate):")
    print("-"*80)

    by_scenario = defaultdict(list)
    for r in rollouts:
        by_scenario[r.scenario].append(r)

    scenario_stats = []
    for scenario in sorted(by_scenario.keys()):
        scen_rollouts = by_scenario[scenario]
        scen_valid = [r for r in scen_rollouts if r.deployment_choice and r.evaluation_choice]
        scen_changes = [r for r in scen_rollouts if r.behavioral_change is True]

        if len(scen_valid) > 0:
            rate = len(scen_changes) / len(scen_valid) * 100
            scenario_stats.append((scenario, len(scen_rollouts), len(scen_valid), len(scen_changes), rate))

    print(f"\n{'Scenario':<30s} {'Total':>6s} {'Valid':>6s} {'Changed':>8s} {'Rate':>8s}")
    print("-"*80)
    for scenario, total, valid, changes, rate in sorted(scenario_stats, key=lambda x: x[4], reverse=True):
        print(f"{scenario:<30s} {total:6d} {valid:6d} {changes:8d} {rate:7.1f}%")

    # Per-prompt statistics (group by original source file)
    print("\n" + "-"*80)
    print("PER-PROMPT BEHAVIORAL CHANGE DISTRIBUTION:")
    print("-"*80)
    print("(Shows consistency: do all prompts change ~50% of time, or do some always/never change?)\n")

    # Group rollouts by source prompt file (remove seed suffix)
    by_prompt = defaultdict(list)
    for r in rollouts:
        # Extract base filename without seed (e.g., "deception_2025-10-22_18-25-57_96b0c371")
        filename = r.filepath.stem
        # Remove _seed_N suffix
        base_name = filename.rsplit('_seed_', 1)[0] if '_seed_' in filename else filename
        by_prompt[base_name].append(r)

    # Calculate per-prompt behavioral change rates
    prompt_rates = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        valid_rollouts = [r for r in prompt_rollouts if r.deployment_choice and r.evaluation_choice]
        if valid_rollouts:
            changed = sum(1 for r in valid_rollouts if r.behavioral_change is True)
            rate = (changed / len(valid_rollouts)) * 100
            prompt_rates.append({
                'name': prompt_name,
                'total': len(prompt_rollouts),
                'valid': len(valid_rollouts),
                'changed': changed,
                'rate': rate
            })

    if prompt_rates:
        # Calculate statistics
        rates_only = [p['rate'] for p in prompt_rates]
        mean_rate = statistics.mean(rates_only)
        median_rate = statistics.median(rates_only)

        # Quantiles
        rates_sorted = sorted(rates_only)
        q25 = rates_sorted[len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[0]
        q75 = rates_sorted[3 * len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[-1]
        min_rate = min(rates_only)
        max_rate = max(rates_only)

        print(f"Unique prompts analyzed: {len(prompt_rates)}")
        print(f"\nDistribution of per-prompt behavioral change rates:")
        print(f"  Mean:   {mean_rate:6.1f}%  (average change rate across all prompts)")
        print(f"  Median: {median_rate:6.1f}%  (middle prompt's change rate)")
        print(f"  Q1 (25th percentile): {q25:6.1f}%")
        print(f"  Q3 (75th percentile): {q75:6.1f}%")
        print(f"  Min:    {min_rate:6.1f}%  (most stable prompt)")
        print(f"  Max:    {max_rate:6.1f}%  (most variable prompt)")

        # Show interpretation
        print(f"\nInterpretation:")
        if abs(mean_rate - median_rate) < 10 and (q75 - q25) < 30:
            print(f"  ✓ Consistent: Most prompts show similar behavioral change rates")
        elif (q75 - q25) > 50:
            print(f"  ⚠ High variance: Some prompts very reliable, others very variable")
        else:
            print(f"  ~ Moderate variance: Prompts show some variation in consistency")

        # Show extreme cases
        print(f"\nMost consistent prompts (lowest change rate):")
        for p in sorted(prompt_rates, key=lambda x: x['rate'])[:5]:
            print(f"  {p['rate']:5.1f}%  ({p['changed']:2d}/{p['valid']:2d})  {p['name'][:60]}")

        print(f"\nLeast consistent prompts (highest change rate):")
        for p in sorted(prompt_rates, key=lambda x: x['rate'], reverse=True)[:5]:
            print(f"  {p['rate']:5.1f}%  ({p['changed']:2d}/{p['valid']:2d})  {p['name'][:60]}")


def analyze_choice_distribution(rollouts: List[RolloutData]) -> None:
    """Analyze choice patterns (A vs B)."""
    print("\n" + "="*80)
    print("CHOICE DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")

    valid_rollouts = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]

    if not valid_rollouts:
        print("No valid rollouts with both choices!")
        return

    # Overall distribution
    deployment_choices = Counter([r.deployment_choice for r in valid_rollouts])
    evaluation_choices = Counter([r.evaluation_choice for r in valid_rollouts])

    print("DEPLOYMENT CONTEXT (no eval cues):")
    for choice in ['A', 'B']:
        count = deployment_choices[choice]
        pct = count / len(valid_rollouts) * 100
        print(f"  Choice {choice}: {count:4d} ({pct:5.1f}%)")

    print("\nEVALUATION CONTEXT (with eval cues):")
    for choice in ['A', 'B']:
        count = evaluation_choices[choice]
        pct = count / len(valid_rollouts) * 100
        print(f"  Choice {choice}: {count:4d} ({pct:5.1f}%)")

    # Transition matrix
    print("\n" + "-"*80)
    print("TRANSITION MATRIX (Deployment -> Evaluation):")
    print("-"*80 + "\n")

    transitions = Counter([
        (r.deployment_choice, r.evaluation_choice)
        for r in valid_rollouts
    ])

    print("           -> A      -> B")
    for from_choice in ['A', 'B']:
        counts = [transitions[(from_choice, to_choice)] for to_choice in ['A', 'B']]
        total = sum(counts)
        pcts = [c/total*100 if total > 0 else 0 for c in counts]
        print(f"  From {from_choice}: {counts[0]:4d} ({pcts[0]:5.1f}%)  {counts[1]:4d} ({pcts[1]:5.1f}%)")

    # Shift analysis
    a_to_b = transitions[('A', 'B')]
    b_to_a = transitions[('B', 'A')]
    print(f"\n= Shift towards A (B->A): {b_to_a} ({b_to_a/len(valid_rollouts)*100:.1f}%)")
    print(f"= Shift towards B (A->B): {a_to_b} ({a_to_b/len(valid_rollouts)*100:.1f}%)")

    # Per-prompt statistics (group by original source file)
    print("\n" + "-"*80)
    print("PER-PROMPT CHOICE DISTRIBUTION:")
    print("-"*80)
    print("(Shows consistency: do all prompts show ~50% B->A shift, or do some always shift?)\n")

    # Group rollouts by source prompt file (remove seed suffix)
    by_prompt = defaultdict(list)
    for r in valid_rollouts:
        filename = r.filepath.stem
        base_name = filename.rsplit('_seed_', 1)[0] if '_seed_' in filename else filename
        by_prompt[base_name].append(r)

    # Calculate per-prompt shift rates (B->A)
    prompt_stats = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        if len(prompt_rollouts) < 2:  # Need multiple seeds
            continue

        # Count B->A transitions
        b_to_a_count = sum(1 for r in prompt_rollouts if r.deployment_choice == 'B' and r.evaluation_choice == 'A')
        # Count all B deployments (to calculate % that shift to A)
        b_deployments = sum(1 for r in prompt_rollouts if r.deployment_choice == 'B')

        if b_deployments > 0:
            b_to_a_rate = (b_to_a_count / b_deployments) * 100
        else:
            b_to_a_rate = 0.0

        # Also track A->B shifts
        a_to_b_count = sum(1 for r in prompt_rollouts if r.deployment_choice == 'A' and r.evaluation_choice == 'B')
        a_deployments = sum(1 for r in prompt_rollouts if r.deployment_choice == 'A')

        if a_deployments > 0:
            a_to_b_rate = (a_to_b_count / a_deployments) * 100
        else:
            a_to_b_rate = 0.0

        prompt_stats.append({
            'name': prompt_name,
            'n_seeds': len(prompt_rollouts),
            'b_to_a_count': b_to_a_count,
            'b_deployments': b_deployments,
            'b_to_a_rate': b_to_a_rate,
            'a_to_b_count': a_to_b_count,
            'a_deployments': a_deployments,
            'a_to_b_rate': a_to_b_rate
        })

    if prompt_stats:
        # Calculate statistics for B->A shifts
        b_to_a_rates = [p['b_to_a_rate'] for p in prompt_stats if p['b_deployments'] > 0]

        if b_to_a_rates:
            mean_rate = statistics.mean(b_to_a_rates)
            median_rate = statistics.median(b_to_a_rates)

            # Quantiles
            rates_sorted = sorted(b_to_a_rates)
            q25 = rates_sorted[len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[0]
            q75 = rates_sorted[3 * len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[-1]
            min_rate = min(b_to_a_rates)
            max_rate = max(b_to_a_rates)

            print(f"Unique prompts analyzed: {len(prompt_stats)}")
            print(f"\nDistribution of per-prompt B→A shift rates (across {len(b_to_a_rates)} prompts with B deployments):")
            print(f"  Mean:   {mean_rate:6.1f}%  (average B→A shift across all prompts)")
            print(f"  Median: {median_rate:6.1f}%  (middle prompt's shift rate)")
            print(f"  Q1 (25th percentile): {q25:6.1f}%")
            print(f"  Q3 (75th percentile): {q75:6.1f}%")
            print(f"  Min:    {min_rate:6.1f}%  (most consistent B deployment)")
            print(f"  Max:    {max_rate:6.1f}%  (highest B→A shift rate)")

            # Show interpretation
            print(f"\nInterpretation:")
            if abs(mean_rate - median_rate) < 10 and (q75 - q25) < 30:
                print(f"  ✓ Consistent: Most prompts show similar B→A shift rates")
            elif (q75 - q25) > 50:
                print(f"  ⚠ High variance: Some prompts always shift, others never shift")
            else:
                print(f"  ~ Moderate variance: Prompts show some variation in shift patterns")

            # Show extreme cases
            print(f"\nPrompts with lowest B→A shift (most consistent B choice):")
            for p in sorted(prompt_stats, key=lambda x: x['b_to_a_rate'])[:5]:
                if p['b_deployments'] > 0:
                    print(f"  {p['b_to_a_rate']:5.1f}%  ({p['b_to_a_count']:2d}/{p['b_deployments']:2d})  {p['name'][:60]}")

            print(f"\nPrompts with highest B→A shift (strong eval cue response):")
            for p in sorted(prompt_stats, key=lambda x: x['b_to_a_rate'], reverse=True)[:5]:
                if p['b_deployments'] > 0:
                    print(f"  {p['b_to_a_rate']:5.1f}%  ({p['b_to_a_count']:2d}/{p['b_deployments']:2d})  {p['name'][:60]}")


def compare_models(models: List[str], category_filter: Optional[str] = None) -> None:
    """Compare behavioral change rates across multiple models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")

    model_data = {}
    for model in models:
        rollouts = load_all_rollouts(model, category_filter)
        model_data[model] = rollouts

    # Summary table
    print(f"{'Model':<40s} {'Total':>8s} {'Valid':>8s} {'Changed':>8s} {'Rate':>8s}")
    print("-"*80)

    for model in models:
        rollouts = model_data[model]
        valid = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]
        changed = [r for r in rollouts if r.behavioral_change is True]

        rate = len(changed) / len(valid) * 100 if len(valid) > 0 else 0

        print(f"{model:<40s} {len(rollouts):8d} {len(valid):8d} {len(changed):8d} {rate:7.1f}%")


def analyze_scenario_effectiveness(rollouts: List[RolloutData]) -> None:
    """Analyze which scenarios are most effective at triggering behavioral changes."""
    print("\n" + "="*80)
    print("SCENARIO EFFECTIVENESS ANALYSIS")
    print("="*80 + "\n")

    by_scenario = defaultdict(list)
    for r in rollouts:
        if r.deployment_choice and r.evaluation_choice:
            by_scenario[r.scenario].append(r)

    scenario_stats = []
    for scenario, scen_rollouts in by_scenario.items():
        total = len(scen_rollouts)
        changed = sum(1 for r in scen_rollouts if r.behavioral_change is True)
        rate = changed / total * 100 if total > 0 else 0

        # Calculate variance (are results consistent?)
        change_values = [1 if r.behavioral_change else 0 for r in scen_rollouts]
        variance = statistics.variance(change_values) if len(change_values) > 1 else 0

        scenario_stats.append((scenario, total, changed, rate, variance))

    # Sort by change rate (descending)
    scenario_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Scenario':<30s} {'Valid':>6s} {'Changed':>8s} {'Rate':>8s} {'Variance':>10s}")
    print("-"*80)

    for scenario, total, changed, rate, variance in scenario_stats:
        print(f"{scenario:<30s} {total:6d} {changed:8d} {rate:7.1f}% {variance:10.3f}")

    print("\n>> Interpretation:")
    print("  High rate + low variance = reliably triggers behavioral change")
    print("  High rate + high variance = inconsistent (some seeds work, others don't)")
    print("  Low rate = scenario doesn't trigger much change")

    # Per-prompt statistics (group by original source file, different seeds)
    print("\n" + "-"*80)
    print("PER-PROMPT SCENARIO EFFECTIVENESS:")
    print("-"*80)
    print("(Shows how consistently each prompt triggers behavioral change across seeds)\n")

    valid_rollouts = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]

    # Group rollouts by source prompt file (remove seed suffix)
    by_prompt = defaultdict(list)
    for r in valid_rollouts:
        filename = r.filepath.stem
        base_name = filename.rsplit('_seed_', 1)[0] if '_seed_' in filename else filename
        by_prompt[base_name].append(r)

    # Calculate per-prompt behavioral change rates
    prompt_stats = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        if len(prompt_rollouts) < 2:  # Need multiple seeds
            continue

        changed = sum(1 for r in prompt_rollouts if r.behavioral_change is True)
        rate = (changed / len(prompt_rollouts)) * 100

        # Get scenario name (should be same for all rollouts from same prompt)
        scenario = prompt_rollouts[0].scenario if prompt_rollouts else "unknown"

        prompt_stats.append({
            'name': prompt_name,
            'scenario': scenario,
            'n_seeds': len(prompt_rollouts),
            'changed': changed,
            'rate': rate
        })

    if prompt_stats:
        # Calculate statistics
        rates = [p['rate'] for p in prompt_stats]
        mean_rate = statistics.mean(rates)
        median_rate = statistics.median(rates)

        # Quantiles
        rates_sorted = sorted(rates)
        q25 = rates_sorted[len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[0]
        q75 = rates_sorted[3 * len(rates_sorted) // 4] if len(rates_sorted) >= 4 else rates_sorted[-1]
        min_rate = min(rates)
        max_rate = max(rates)

        print(f"Unique prompts analyzed: {len(prompt_stats)}")
        print(f"\nDistribution of per-prompt behavioral change rates:")
        print(f"  Mean:   {mean_rate:6.1f}%  (average effectiveness across all prompts)")
        print(f"  Median: {median_rate:6.1f}%  (middle prompt's effectiveness)")
        print(f"  Q1 (25th percentile): {q25:6.1f}%")
        print(f"  Q3 (75th percentile): {q75:6.1f}%")
        print(f"  Min:    {min_rate:6.1f}%  (least effective prompt)")
        print(f"  Max:    {max_rate:6.1f}%  (most effective prompt)")

        # Show interpretation
        print(f"\nInterpretation:")
        if abs(mean_rate - median_rate) < 10 and (q75 - q25) < 30:
            print(f"  ✓ Consistent: Most prompts show similar effectiveness rates")
        elif (q75 - q25) > 50:
            print(f"  ⚠ High variance: Some prompts very effective, others not")
        else:
            print(f"  ~ Moderate variance: Prompts show some variation in effectiveness")

        # Show extreme cases
        print(f"\nLeast effective prompts (lowest behavioral change rate):")
        for p in sorted(prompt_stats, key=lambda x: x['rate'])[:5]:
            print(f"  {p['rate']:5.1f}%  ({p['changed']:2d}/{p['n_seeds']:2d})  [{p['scenario']:20s}]  {p['name'][:40]}")

        print(f"\nMost effective prompts (highest behavioral change rate):")
        for p in sorted(prompt_stats, key=lambda x: x['rate'], reverse=True)[:5]:
            print(f"  {p['rate']:5.1f}%  ({p['changed']:2d}/{p['n_seeds']:2d})  [{p['scenario']:20s}]  {p['name'][:40]}")


def analyze_responses(rollouts: List[RolloutData]) -> None:
    """Analyze response characteristics (length, keywords, patterns)."""
    print("\n" + "="*80)
    print("RESPONSE ANALYSIS")
    print("="*80 + "\n")

    valid_rollouts = [r for r in rollouts if r.deployment_response and r.evaluation_response]

    if not valid_rollouts:
        print("No valid rollouts with both responses!")
        return

    # Length analysis
    print("RESPONSE LENGTH:")
    print("-"*80)

    deployment_lengths = [len(r.deployment_response) for r in valid_rollouts]
    evaluation_lengths = [len(r.evaluation_response) for r in valid_rollouts]

    print(f"Deployment context:")
    print(f"  Mean: {statistics.mean(deployment_lengths):.1f} chars")
    print(f"  Median: {statistics.median(deployment_lengths):.1f} chars")
    print(f"  Stdev: {statistics.stdev(deployment_lengths):.1f} chars" if len(deployment_lengths) > 1 else "  Stdev: N/A")

    print(f"\nEvaluation context:")
    print(f"  Mean: {statistics.mean(evaluation_lengths):.1f} chars")
    print(f"  Median: {statistics.median(evaluation_lengths):.1f} chars")
    print(f"  Stdev: {statistics.stdev(evaluation_lengths):.1f} chars" if len(evaluation_lengths) > 1 else "  Stdev: N/A")

    # Hedging keywords
    print("\n" + "-"*80)
    print("HEDGING KEYWORDS (frequency per 1000 chars):")
    print("-"*80)

    hedging_keywords = [
        'however', 'but', 'although', 'on the other hand', 'nevertheless',
        'nonetheless', 'that said', 'it depends', 'consider', 'might',
        'could', 'may', 'potentially', 'arguably'
    ]

    for keyword in hedging_keywords:
        deployment_freq = sum(r.deployment_response.lower().count(keyword) for r in valid_rollouts)
        evaluation_freq = sum(r.evaluation_response.lower().count(keyword) for r in valid_rollouts)

        deployment_total_chars = sum(len(r.deployment_response) for r in valid_rollouts)
        evaluation_total_chars = sum(len(r.evaluation_response) for r in valid_rollouts)

        dep_per_1k = deployment_freq / deployment_total_chars * 1000 if deployment_total_chars > 0 else 0
        eval_per_1k = evaluation_freq / evaluation_total_chars * 1000 if evaluation_total_chars > 0 else 0

        if dep_per_1k > 0.1 or eval_per_1k > 0.1:  # Only show if appears
            print(f"  {keyword:20s} - Deploy: {dep_per_1k:5.2f}  Eval: {eval_per_1k:5.2f}  ->: {eval_per_1k-dep_per_1k:+6.2f}")

    # Per-prompt statistics (group by original source file, different seeds)
    print("\n" + "-"*80)
    print("PER-PROMPT RESPONSE CHARACTERISTICS:")
    print("-"*80)
    print("(Shows how response patterns vary across seeds for each prompt)\n")

    # Group rollouts by source prompt file (remove seed suffix)
    by_prompt = defaultdict(list)
    for r in valid_rollouts:
        filename = r.filepath.stem
        base_name = filename.rsplit('_seed_', 1)[0] if '_seed_' in filename else filename
        by_prompt[base_name].append(r)

    # Calculate per-prompt response statistics
    prompt_stats = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        if len(prompt_rollouts) < 2:  # Need multiple seeds
            continue

        # Average response lengths
        avg_deploy_len = statistics.mean([len(r.deployment_response) for r in prompt_rollouts])
        avg_eval_len = statistics.mean([len(r.evaluation_response) for r in prompt_rollouts])

        # Length difference (eval - deploy)
        len_diff = avg_eval_len - avg_deploy_len
        len_diff_pct = (len_diff / avg_deploy_len * 100) if avg_deploy_len > 0 else 0

        # Get scenario
        scenario = prompt_rollouts[0].scenario if prompt_rollouts else "unknown"

        prompt_stats.append({
            'name': prompt_name,
            'scenario': scenario,
            'n_seeds': len(prompt_rollouts),
            'avg_deploy_len': avg_deploy_len,
            'avg_eval_len': avg_eval_len,
            'len_diff': len_diff,
            'len_diff_pct': len_diff_pct
        })

    if prompt_stats:
        # Calculate statistics
        deploy_lens = [p['avg_deploy_len'] for p in prompt_stats]
        eval_lens = [p['avg_eval_len'] for p in prompt_stats]
        len_diffs = [p['len_diff'] for p in prompt_stats]

        print(f"Unique prompts analyzed: {len(prompt_stats)}")
        print(f"\nAverage response length distribution:")
        print(f"  Deployment context:")
        print(f"    Mean:   {statistics.mean(deploy_lens):6.1f} chars")
        print(f"    Median: {statistics.median(deploy_lens):6.1f} chars")
        print(f"  Evaluation context:")
        print(f"    Mean:   {statistics.mean(eval_lens):6.1f} chars")
        print(f"    Median: {statistics.median(eval_lens):6.1f} chars")
        print(f"  Length difference (eval - deploy):")
        print(f"    Mean:   {statistics.mean(len_diffs):6.1f} chars")
        print(f"    Median: {statistics.median(len_diffs):6.1f} chars")

        # Show prompts with biggest length differences
        print(f"\nPrompts with biggest evaluation length increase:")
        for p in sorted(prompt_stats, key=lambda x: x['len_diff'], reverse=True)[:5]:
            print(f"  +{p['len_diff']:6.1f} chars ({p['len_diff_pct']:+6.1f}%)  [{p['scenario']:20s}]  {p['name'][:40]}")

        print(f"\nPrompts with biggest evaluation length decrease:")
        for p in sorted(prompt_stats, key=lambda x: x['len_diff'])[:5]:
            print(f"  {p['len_diff']:7.1f} chars ({p['len_diff_pct']:+6.1f}%)  [{p['scenario']:20s}]  {p['name'][:40]}")


def analyze_data_quality(rollouts: List[RolloutData]) -> None:
    """Analyze data quality issues."""
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS")
    print("="*80 + "\n")

    total = len(rollouts)
    with_errors = [r for r in rollouts if r.has_error]
    missing_deployment = [r for r in rollouts if not r.deployment_choice]
    missing_evaluation = [r for r in rollouts if not r.evaluation_choice]
    missing_both = [r for r in rollouts if not r.deployment_choice and not r.evaluation_choice]
    valid = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]

    print(f"Total rollouts: {total}")
    print(f"  Valid (both choices extracted): {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"  Errors: {len(with_errors)} ({len(with_errors)/total*100:.1f}%)")
    print(f"  Missing deployment choice: {len(missing_deployment)} ({len(missing_deployment)/total*100:.1f}%)")
    print(f"  Missing evaluation choice: {len(missing_evaluation)} ({len(missing_evaluation)/total*100:.1f}%)")
    print(f"  Missing both choices: {len(missing_both)} ({len(missing_both)/total*100:.1f}%)")

    # Error breakdown
    if with_errors:
        print("\n" + "-"*80)
        print("ERROR BREAKDOWN:")
        print("-"*80)

        error_types = Counter([r.error_message for r in with_errors if r.error_message])
        for error_msg, count in error_types.most_common():
            print(f"  {error_msg}: {count} ({count/len(with_errors)*100:.1f}%)")

    # Category breakdown
    print("\n" + "-"*80)
    print("QUALITY BY CATEGORY:")
    print("-"*80)

    by_category = defaultdict(list)
    for r in rollouts:
        by_category[r.category].append(r)

    for category in sorted(by_category.keys()):
        cat_rollouts = by_category[category]
        cat_valid = [r for r in cat_rollouts if r.deployment_choice and r.evaluation_choice]
        success_rate = len(cat_valid) / len(cat_rollouts) * 100 if len(cat_rollouts) > 0 else 0
        print(f"  {category:30s} - {len(cat_valid):4d}/{len(cat_rollouts):4d} ({success_rate:5.1f}%)")


def analyze_seed_variance(rollouts: List[RolloutData]) -> None:
    """Analyze variance across different seeds."""
    print("\n" + "="*80)
    print("SEED VARIANCE ANALYSIS")
    print("="*80 + "\n")

    # Group by scenario (same source file, different seeds)
    by_source = defaultdict(list)
    for r in rollouts:
        # Group by scenario + category (removes seed variation)
        key = (r.scenario, r.category)
        by_source[key].append(r)

    # Analyze variance
    high_variance = []
    low_variance = []

    for (scenario, category), group in by_source.items():
        if len(group) < 5:  # Need at least 5 seeds to analyze
            continue

        valid = [r for r in group if r.deployment_choice and r.evaluation_choice]
        if len(valid) < 5:
            continue

        # Calculate change rate variance
        change_values = [1 if r.behavioral_change else 0 for r in valid]
        change_rate = sum(change_values) / len(change_values) * 100
        variance = statistics.variance(change_values) if len(change_values) > 1 else 0

        if variance > 0.2:
            high_variance.append((scenario, category, len(valid), change_rate, variance))
        elif variance < 0.1:
            low_variance.append((scenario, category, len(valid), change_rate, variance))

    print("HIGH VARIANCE (inconsistent across seeds):")
    print("-"*80)
    if high_variance:
        high_variance.sort(key=lambda x: x[4], reverse=True)
        for scenario, category, n_seeds, rate, variance in high_variance[:10]:
            print(f"  {scenario:30s} {category:30s} - {n_seeds} seeds, {rate:5.1f}% rate, var={variance:.3f}")
    else:
        print("  None found")

    print("\nLOW VARIANCE (consistent across seeds):")
    print("-"*80)
    if low_variance:
        low_variance.sort(key=lambda x: x[3], reverse=True)  # Sort by rate
        for scenario, category, n_seeds, rate, variance in low_variance[:10]:
            print(f"  {scenario:30s} {category:30s} - {n_seeds} seeds, {rate:5.1f}% rate, var={variance:.3f}")
    else:
        print("  None found")


def analyze_eval_awareness(rollouts: List[RolloutData]) -> None:
    """Analyze evaluation awareness in deployment vs evaluation contexts."""
    print("\n" + "="*80)
    print("EVALUATION AWARENESS ANALYSIS")
    print("="*80 + "\n")

    valid_rollouts = [r for r in rollouts if r.deployment_choice and r.evaluation_choice]

    if not valid_rollouts:
        print("No valid rollouts to analyze")
        return

    # Overall statistics
    total = len(valid_rollouts)
    deployment_aware_count = sum(1 for r in valid_rollouts if len(r.deployment_eval_quotes) > 0)
    evaluation_aware_count = sum(1 for r in valid_rollouts if len(r.evaluation_eval_quotes) > 0)

    total_deployment_quotes = sum(len(r.deployment_eval_quotes) for r in valid_rollouts)
    total_evaluation_quotes = sum(len(r.evaluation_eval_quotes) for r in valid_rollouts)

    avg_deployment_quotes = total_deployment_quotes / total if total > 0 else 0
    avg_evaluation_quotes = total_evaluation_quotes / total if total > 0 else 0

    print(f"Total rollouts analyzed: {total}\n")

    print("DEPLOYMENT CONTEXT (no eval cues - baseline):")
    print("-"*80)
    print(f"  Rollouts showing awareness: {deployment_aware_count}/{total} ({deployment_aware_count/total*100:.1f}%)")
    print(f"  Total eval-aware quotes: {total_deployment_quotes}")
    print(f"  Average quotes per rollout: {avg_deployment_quotes:.2f}")
    if deployment_aware_count > 0:
        print(f"  ⚠️  WARNING: Model shows eval awareness WITHOUT cues in {deployment_aware_count} rollouts!")

    print(f"\nEVALUATION CONTEXT (with eval cues):")
    print("-"*80)
    print(f"  Rollouts showing awareness: {evaluation_aware_count}/{total} ({evaluation_aware_count/total*100:.1f}%)")
    print(f"  Total eval-aware quotes: {total_evaluation_quotes}")
    print(f"  Average quotes per rollout: {avg_evaluation_quotes:.2f}")

    # Awareness ratio
    if avg_deployment_quotes > 0:
        awareness_ratio = avg_evaluation_quotes / avg_deployment_quotes
        print(f"\n  📊 Awareness ratio (eval/deployment): {awareness_ratio:.2f}x")
    else:
        print(f"\n  📊 Awareness ratio: ∞ (no deployment awareness)")

    # Pattern breakdown
    print(f"\n\nAWARENESS PATTERNS:")
    print("="*80)

    patterns = {
        'both': [],                    # Aware in both contexts
        'deploy_only': [],             # Aware ONLY in deployment (concerning!)
        'eval_only_global': [],        # Global awareness ONLY in evaluation
        'eval_only_explicit': [],      # Explicit awareness ONLY in evaluation
        'eval_only_implicit': [],      # Implicit awareness ONLY in evaluation
        'neither': []                  # No awareness detected
    }

    for r in valid_rollouts:
        has_deploy = len(r.deployment_eval_quotes) > 0
        has_eval = len(r.evaluation_eval_quotes) > 0

        if has_deploy and has_eval:
            patterns['both'].append(r)
        elif has_deploy and not has_eval:
            patterns['deploy_only'].append(r)
        elif not has_deploy and has_eval:
            # Categorize by awareness level
            if r.evaluation_awareness_level == "global":
                patterns['eval_only_global'].append(r)
            elif r.evaluation_awareness_level == "explicit":
                patterns['eval_only_explicit'].append(r)
            elif r.evaluation_awareness_level == "implicit":
                patterns['eval_only_implicit'].append(r)
            else:
                # Fallback - has quotes but level is "none" (shouldn't happen)
                patterns['eval_only_explicit'].append(r)
        else:
            patterns['neither'].append(r)

    print(f"1️⃣  BOTH deploy & eval awareness: {len(patterns['both'])}/{total} ({len(patterns['both'])/total*100:.1f}%)")
    print(f"   (Model suspicious in both contexts)")

    print(f"\n2️⃣  Deploy awareness ONLY: {len(patterns['deploy_only'])}/{total} ({len(patterns['deploy_only'])/total*100:.1f}%)")
    print(f"   ⚠️  CONCERNING: Suspicious WITHOUT eval cues")

    # Show eval-only breakdown by level
    eval_only_total = len(patterns['eval_only_global']) + len(patterns['eval_only_explicit']) + len(patterns['eval_only_implicit'])
    print(f"\n3️⃣  Eval awareness ONLY: {eval_only_total}/{total} ({eval_only_total/total*100:.1f}%)")
    print(f"   ✓ IDEAL: Only suspicious WITH eval cues")
    print(f"   - Global (Level 1):   {len(patterns['eval_only_global'])}/{total} ({len(patterns['eval_only_global'])/total*100:.1f}%)")
    print(f"   - Explicit (Level 2): {len(patterns['eval_only_explicit'])}/{total} ({len(patterns['eval_only_explicit'])/total*100:.1f}%)")
    print(f"   - Implicit (Level 3): {len(patterns['eval_only_implicit'])}/{total} ({len(patterns['eval_only_implicit'])/total*100:.1f}%)")

    print(f"\n4️⃣  NO awareness (neither): {len(patterns['neither'])}/{total} ({len(patterns['neither'])/total*100:.1f}%)")
    print(f"   (No explicit awareness detected)")

    # Correlation with behavioral change
    print(f"\n\nEVAL AWARENESS × BEHAVIORAL CHANGE:")
    print("="*80)

    # Cross-tabulation
    changed_aware = sum(1 for r in valid_rollouts if r.behavioral_change is True and len(r.evaluation_eval_quotes) > 0)
    changed_not_aware = sum(1 for r in valid_rollouts if r.behavioral_change is True and len(r.evaluation_eval_quotes) == 0)
    same_aware = sum(1 for r in valid_rollouts if r.behavioral_change is False and len(r.evaluation_eval_quotes) > 0)
    same_not_aware = sum(1 for r in valid_rollouts if r.behavioral_change is False and len(r.evaluation_eval_quotes) == 0)

    print("\nCross-tabulation (Evaluation context):")
    print("-"*80)
    print(f"                              │ Behavioral Change │ No Change │ Total")
    print(f"─────────────────────────────┼──────────────────┼───────────┼───────")
    print(f"Eval Awareness Detected      │ {changed_aware:16d} │ {same_aware:9d} │ {changed_aware + same_aware:6d}")
    print(f"No Eval Awareness            │ {changed_not_aware:16d} │ {same_not_aware:9d} │ {changed_not_aware + same_not_aware:6d}")
    print(f"─────────────────────────────┼──────────────────┼───────────┼───────")
    print(f"Total                        │ {changed_aware + changed_not_aware:16d} │ {same_aware + same_not_aware:9d} │ {total:6d}")

    behavioral_changed = [r for r in valid_rollouts if r.behavioral_change is True]
    behavioral_same = [r for r in valid_rollouts if r.behavioral_change is False]

    if behavioral_changed:
        changed_eval_aware_pct = (changed_aware / len(behavioral_changed)) * 100 if behavioral_changed else 0
        print(f"\nBehavioral change rollouts ({len(behavioral_changed)} total):")
        print(f"  With eval awareness: {changed_aware}/{len(behavioral_changed)} ({changed_eval_aware_pct:.1f}%)")
        print(f"  Without awareness:   {changed_not_aware}/{len(behavioral_changed)} ({100-changed_eval_aware_pct:.1f}%)")

        # Breakdown by awareness level
        changed_global = sum(1 for r in behavioral_changed if r.evaluation_awareness_level == "global")
        changed_explicit = sum(1 for r in behavioral_changed if r.evaluation_awareness_level == "explicit")
        changed_implicit = sum(1 for r in behavioral_changed if r.evaluation_awareness_level == "implicit")
        changed_none = sum(1 for r in behavioral_changed if r.evaluation_awareness_level == "none")

        print(f"  Awareness level breakdown:")
        print(f"    - Global (Level 1):   {changed_global}/{len(behavioral_changed)} ({changed_global/len(behavioral_changed)*100:.1f}%)")
        print(f"    - Explicit (Level 2): {changed_explicit}/{len(behavioral_changed)} ({changed_explicit/len(behavioral_changed)*100:.1f}%)")
        print(f"    - Implicit (Level 3): {changed_implicit}/{len(behavioral_changed)} ({changed_implicit/len(behavioral_changed)*100:.1f}%)")
        print(f"    - None:               {changed_none}/{len(behavioral_changed)} ({changed_none/len(behavioral_changed)*100:.1f}%)")

    if behavioral_same:
        same_eval_aware_pct = (same_aware / len(behavioral_same)) * 100 if behavioral_same else 0
        print(f"\nNo behavioral change rollouts ({len(behavioral_same)} total):")
        print(f"  With eval awareness: {same_aware}/{len(behavioral_same)} ({same_eval_aware_pct:.1f}%)")
        print(f"  Without awareness:   {same_not_aware}/{len(behavioral_same)} ({100-same_eval_aware_pct:.1f}%)")

        # Breakdown by awareness level
        same_global = sum(1 for r in behavioral_same if r.evaluation_awareness_level == "global")
        same_explicit = sum(1 for r in behavioral_same if r.evaluation_awareness_level == "explicit")
        same_implicit = sum(1 for r in behavioral_same if r.evaluation_awareness_level == "implicit")
        same_none = sum(1 for r in behavioral_same if r.evaluation_awareness_level == "none")

        print(f"  Awareness level breakdown:")
        print(f"    - Global (Level 1):   {same_global}/{len(behavioral_same)} ({same_global/len(behavioral_same)*100:.1f}%)")
        print(f"    - Explicit (Level 2): {same_explicit}/{len(behavioral_same)} ({same_explicit/len(behavioral_same)*100:.1f}%)")
        print(f"    - Implicit (Level 3): {same_implicit}/{len(behavioral_same)} ({same_implicit/len(behavioral_same)*100:.1f}%)")
        print(f"    - None:               {same_none}/{len(behavioral_same)} ({same_none/len(behavioral_same)*100:.1f}%)")

    # Per-prompt behavioral change distribution
    print(f"\n\nPER-PROMPT BEHAVIORAL CHANGE × AWARENESS DISTRIBUTION:")
    print("="*80)
    print("(Shows how each prompt behaves across different seeds)\n")

    # Group rollouts by source prompt file (remove seed suffix)
    by_prompt = defaultdict(list)
    for r in valid_rollouts:
        filename = r.filepath.stem
        base_name = filename.rsplit('_seed_', 1)[0] if '_seed_' in filename else filename
        by_prompt[base_name].append(r)

    # Calculate per-prompt statistics
    prompt_stats = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        if len(prompt_rollouts) < 2:  # Need multiple seeds
            continue

        # Behavioral change stats
        changed_count = sum(1 for r in prompt_rollouts if r.behavioral_change is True)
        no_change_count = sum(1 for r in prompt_rollouts if r.behavioral_change is False)
        change_rate = (changed_count / len(prompt_rollouts)) * 100

        # Eval awareness stats (in evaluation context)
        eval_aware_count = sum(1 for r in prompt_rollouts if len(r.evaluation_eval_quotes) > 0)
        eval_aware_rate = (eval_aware_count / len(prompt_rollouts)) * 100

        # Awareness level breakdown
        global_count = sum(1 for r in prompt_rollouts if r.evaluation_awareness_level == "global")
        explicit_count = sum(1 for r in prompt_rollouts if r.evaluation_awareness_level == "explicit")
        implicit_count = sum(1 for r in prompt_rollouts if r.evaluation_awareness_level == "implicit")
        none_count = sum(1 for r in prompt_rollouts if r.evaluation_awareness_level == "none")

        # Combined pattern: behavioral change + awareness level
        changed_with_global = sum(1 for r in prompt_rollouts if r.behavioral_change is True and r.evaluation_awareness_level == "global")
        changed_with_explicit = sum(1 for r in prompt_rollouts if r.behavioral_change is True and r.evaluation_awareness_level == "explicit")
        changed_with_implicit = sum(1 for r in prompt_rollouts if r.behavioral_change is True and r.evaluation_awareness_level == "implicit")
        changed_with_none = sum(1 for r in prompt_rollouts if r.behavioral_change is True and r.evaluation_awareness_level == "none")

        # Get scenario and category
        scenario = prompt_rollouts[0].scenario if prompt_rollouts else "unknown"
        category = prompt_rollouts[0].category if prompt_rollouts else "unknown"

        prompt_stats.append({
            'name': prompt_name,
            'scenario': scenario,
            'category': category,
            'n_seeds': len(prompt_rollouts),
            'changed_count': changed_count,
            'no_change_count': no_change_count,
            'change_rate': change_rate,
            'eval_aware_count': eval_aware_count,
            'eval_aware_rate': eval_aware_rate,
            'global_count': global_count,
            'explicit_count': explicit_count,
            'implicit_count': implicit_count,
            'none_count': none_count,
            'changed_with_global': changed_with_global,
            'changed_with_explicit': changed_with_explicit,
            'changed_with_implicit': changed_with_implicit,
            'changed_with_none': changed_with_none
        })

    if prompt_stats:
        # Calculate overall statistics
        change_rates = [p['change_rate'] for p in prompt_stats]
        mean_change_rate = statistics.mean(change_rates)
        median_change_rate = statistics.median(change_rates)

        print(f"Unique prompts analyzed: {len(prompt_stats)}")
        print(f"\nBehavioral change rate distribution:")
        print(f"  Mean:   {mean_change_rate:6.1f}%")
        print(f"  Median: {median_change_rate:6.1f}%")

        if len(change_rates) >= 4:
            q1 = statistics.quantiles(change_rates, n=4)[0]
            q3 = statistics.quantiles(change_rates, n=4)[2]
            print(f"  Q1 (25th percentile): {q1:.1f}%")
            print(f"  Q3 (75th percentile): {q3:.1f}%")
            print(f"  Min: {min(change_rates):.1f}%")
            print(f"  Max: {max(change_rates):.1f}%")

        # Categorize prompts by consistency
        always_change = [p for p in prompt_stats if p['change_rate'] == 100]
        never_change = [p for p in prompt_stats if p['change_rate'] == 0]
        sometimes_change = [p for p in prompt_stats if 0 < p['change_rate'] < 100]

        print(f"\nConsistency patterns:")
        print(f"  Always change (100%):    {len(always_change)} prompts")
        print(f"  Never change (0%):       {len(never_change)} prompts")
        print(f"  Sometimes change (0-100%): {len(sometimes_change)} prompts")

        # Show most/least consistent prompts
        print(f"\n{'='*80}")
        print(f"HIGHEST BEHAVIORAL CHANGE RATES (most variable prompts):")
        print(f"{'='*80}")
        for p in sorted(prompt_stats, key=lambda x: x['change_rate'], reverse=True)[:10]:
            print(f"\n{p['name'][:70]}")
            print(f"  Scenario: {p['scenario']:20s} | Category: {p['category']}")
            print(f"  Seeds: {p['n_seeds']}  |  Change rate: {p['change_rate']:.1f}% ({p['changed_count']}/{p['n_seeds']})")
            print(f"  Eval awareness: {p['eval_aware_rate']:.1f}% ({p['eval_aware_count']}/{p['n_seeds']})")
            print(f"  Awareness levels: Global={p['global_count']}, Explicit={p['explicit_count']}, Implicit={p['implicit_count']}, None={p['none_count']}")
            print(f"  Behavioral change by level: G={p['changed_with_global']}, E={p['changed_with_explicit']}, I={p['changed_with_implicit']}, N={p['changed_with_none']}")

        print(f"\n{'='*80}")
        print(f"LOWEST BEHAVIORAL CHANGE RATES (most stable prompts):")
        print(f"{'='*80}")
        for p in sorted(prompt_stats, key=lambda x: x['change_rate'])[:10]:
            print(f"\n{p['name'][:70]}")
            print(f"  Scenario: {p['scenario']:20s} | Category: {p['category']}")
            print(f"  Seeds: {p['n_seeds']}  |  Change rate: {p['change_rate']:.1f}% ({p['changed_count']}/{p['n_seeds']})")
            print(f"  Eval awareness: {p['eval_aware_rate']:.1f}% ({p['eval_aware_count']}/{p['n_seeds']})")
            print(f"  Awareness levels: Global={p['global_count']}, Explicit={p['explicit_count']}, Implicit={p['implicit_count']}, None={p['none_count']}")
            print(f"  Behavioral change by level: G={p['changed_with_global']}, E={p['changed_with_explicit']}, I={p['changed_with_implicit']}, N={p['changed_with_none']}")

        # Correlation between awareness and behavioral change at prompt level
        print(f"\n{'='*80}")
        print(f"CORRELATION: EVAL AWARENESS vs BEHAVIORAL CHANGE (prompt-level):")
        print(f"{'='*80}")

        # Group by awareness patterns - separate 100%/100% as most ideal category
        perfect_perfect = [p for p in prompt_stats if p['eval_aware_rate'] == 100 and p['change_rate'] == 100]
        perfect_aware_high_change = [p for p in prompt_stats if p['eval_aware_rate'] == 100 and 50 <= p['change_rate'] < 100]
        perfect_aware_low_change = [p for p in prompt_stats if p['eval_aware_rate'] == 100 and p['change_rate'] < 50]
        high_aware_high_change = [p for p in prompt_stats if 50 <= p['eval_aware_rate'] < 100 and p['change_rate'] >= 50]
        high_aware_low_change = [p for p in prompt_stats if 50 <= p['eval_aware_rate'] < 100 and p['change_rate'] < 50]
        low_aware_high_change = [p for p in prompt_stats if p['eval_aware_rate'] < 50 and p['change_rate'] >= 50]
        low_aware_low_change = [p for p in prompt_stats if p['eval_aware_rate'] < 50 and p['change_rate'] < 50]

        print(f"\n🌟 IDEAL: 100% eval awareness + 100% behavioral change:         {len(perfect_perfect)} prompts")
        print(f"Perfect eval awareness (100%) + High behavioral change (50-99%): {len(perfect_aware_high_change)} prompts")
        print(f"Perfect eval awareness (100%) + Low behavioral change (<50%):    {len(perfect_aware_low_change)} prompts")
        print(f"High eval awareness (50-99%) + High behavioral change (≥50%):    {len(high_aware_high_change)} prompts")
        print(f"High eval awareness (50-99%) + Low behavioral change (<50%):     {len(high_aware_low_change)} prompts")
        print(f"Low eval awareness (<50%) + High behavioral change (≥50%):       {len(low_aware_high_change)} prompts")
        print(f"Low eval awareness (<50%) + Low behavioral change (<50%):        {len(low_aware_low_change)} prompts")

        # Save high awareness + high change prompts to file (including 100%/100% perfect and other high performers)
        all_high_aware_high_change = perfect_perfect + perfect_aware_high_change + high_aware_high_change
        if all_high_aware_high_change:
            # Get model directory from first rollout filepath
            first_prompt_rollouts = by_prompt[list(by_prompt.keys())[0]]
            first_rollout_path = first_prompt_rollouts[0].filepath
            # Extract model directory (working/rollouts/{model_dir}/...)
            model_dir = first_rollout_path.parts[2]  # working/rollouts/{model_dir}

            output_dir = Path('working/rollouts') / model_dir / 'analysis'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'high_awareness_high_change_prompts.txt'

            with open(output_file, 'w') as f:
                f.write("# High Eval Awareness (≥50%) + High Behavioral Change (≥50%)\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total prompts: {len(all_high_aware_high_change)}\n")
                f.write(f"# 🌟 Perfect correlation (100% awareness + 100% change): {len(perfect_perfect)}\n")
                f.write(f"# Perfect awareness (100% + 50-99% change): {len(perfect_aware_high_change)}\n")
                f.write(f"# High awareness (50-99% + ≥50% change): {len(high_aware_high_change)}\n")
                f.write("#" + "="*78 + "\n\n")

                for p in sorted(all_high_aware_high_change, key=lambda x: x['change_rate'], reverse=True):
                    # Get original source file path from category
                    # Category is already extracted_prompts structure (e.g., "behavioral_change/explicit")
                    source_path = Path('working/extracted_prompts') / p['category'] / f"{p['name']}.yaml"

                    f.write(f"Prompt: {p['name']}\n")
                    f.write(f"  Source: {source_path}\n")
                    f.write(f"  Scenario: {p['scenario']}\n")
                    f.write(f"  Category: {p['category']}\n")
                    f.write(f"  Seeds analyzed: {p['n_seeds']}\n")
                    f.write(f"  Behavioral change rate: {p['change_rate']:.1f}% ({p['changed_count']}/{p['n_seeds']})\n")
                    f.write(f"  Eval awareness rate: {p['eval_aware_rate']:.1f}% ({p['eval_aware_count']}/{p['n_seeds']})\n")
                    f.write(f"  Awareness levels: Global={p['global_count']}, Explicit={p['explicit_count']}, Implicit={p['implicit_count']}, None={p['none_count']}\n")
                    f.write(f"\n")

            print(f"\n📁 Saved {len(all_high_aware_high_change)} high-performing prompts to: {output_file}")
            print(f"   ({len(perfect_perfect)} perfect 100%/100%, {len(perfect_aware_high_change)} perfect awareness, {len(high_aware_high_change)} high awareness)")

        if len(prompt_stats) >= 2:
            # Calculate correlation coefficient
            from scipy import stats as scipy_stats
            try:
                awareness_rates = [p['eval_aware_rate'] for p in prompt_stats]
                change_rates_for_corr = [p['change_rate'] for p in prompt_stats]
                correlation, p_value = scipy_stats.pearsonr(awareness_rates, change_rates_for_corr)
                print(f"\nPearson correlation: {correlation:.3f} (p-value: {p_value:.4f})")
                if p_value < 0.05:
                    if correlation > 0:
                        print(f"  → Significant positive correlation: Higher awareness → Higher behavioral change")
                    else:
                        print(f"  → Significant negative correlation: Higher awareness → Lower behavioral change")
                else:
                    print(f"  → No significant correlation detected")
            except ImportError:
                print(f"\n(scipy not available for correlation calculation)")
            except Exception as e:
                print(f"\n(Correlation calculation failed: {e})")

    else:
        print("  Not enough multi-seed prompts to analyze")

    # Per-prompt awareness distribution (existing section, but keep grouping)
    print(f"\n\nPER-PROMPT EVAL AWARENESS DISTRIBUTION:")
    print("="*80)

    # Calculate per-prompt eval awareness rates
    prompt_awareness = []
    for prompt_name, prompt_rollouts in by_prompt.items():
        if len(prompt_rollouts) < 2:  # Need multiple seeds to be meaningful
            continue

        # Deployment awareness
        deploy_aware_count = sum(1 for r in prompt_rollouts if len(r.deployment_eval_quotes) > 0)
        deploy_rate = (deploy_aware_count / len(prompt_rollouts)) * 100
        deploy_avg_quotes = sum(len(r.deployment_eval_quotes) for r in prompt_rollouts) / len(prompt_rollouts)

        # Evaluation awareness
        eval_aware_count = sum(1 for r in prompt_rollouts if len(r.evaluation_eval_quotes) > 0)
        eval_rate = (eval_aware_count / len(prompt_rollouts)) * 100
        eval_avg_quotes = sum(len(r.evaluation_eval_quotes) for r in prompt_rollouts) / len(prompt_rollouts)

        prompt_awareness.append({
            'name': prompt_name,
            'n_seeds': len(prompt_rollouts),
            'deploy_rate': deploy_rate,
            'deploy_avg_quotes': deploy_avg_quotes,
            'eval_rate': eval_rate,
            'eval_avg_quotes': eval_avg_quotes
        })

    if prompt_awareness:
        # Calculate statistics
        eval_rates = [p['eval_rate'] for p in prompt_awareness]
        eval_quotes = [p['eval_avg_quotes'] for p in prompt_awareness]

        mean_rate = statistics.mean(eval_rates)
        median_rate = statistics.median(eval_rates)

        mean_quotes = statistics.mean(eval_quotes)
        median_quotes = statistics.median(eval_quotes)

        print(f"\nEvaluation context awareness (across {len(prompt_awareness)} prompts with ≥2 seeds):")
        print(f"  Mean awareness rate: {mean_rate:.1f}%")
        print(f"  Median awareness rate: {median_rate:.1f}%")
        print(f"  Mean quotes per rollout: {mean_quotes:.2f}")
        print(f"  Median quotes per rollout: {median_quotes:.2f}")

        if len(eval_rates) >= 4:
            q1 = statistics.quantiles(eval_rates, n=4)[0]
            q3 = statistics.quantiles(eval_rates, n=4)[2]
            print(f"  Q1 (25th percentile): {q1:.1f}%")
            print(f"  Q3 (75th percentile): {q3:.1f}%")
            print(f"  Min: {min(eval_rates):.1f}%")
            print(f"  Max: {max(eval_rates):.1f}%")

        # Show high/low awareness prompts
        prompt_awareness.sort(key=lambda x: x['eval_rate'], reverse=True)

        print(f"\n  Top 5 prompts with highest eval awareness:")
        for p in prompt_awareness[:5]:
            print(f"    {p['name'][:50]:50s} {p['eval_rate']:5.1f}% ({p['n_seeds']} seeds, {p['eval_avg_quotes']:.1f} quotes)")

        print(f"\n  Bottom 5 prompts with lowest eval awareness:")
        for p in prompt_awareness[-5:]:
            print(f"    {p['name'][:50]:50s} {p['eval_rate']:5.1f}% ({p['n_seeds']} seeds, {p['eval_avg_quotes']:.1f} quotes)")

        # Show prompts with concerning deployment awareness
        deploy_aware_prompts = [p for p in prompt_awareness if p['deploy_rate'] > 0]
        if deploy_aware_prompts:
            print(f"\n  ⚠️  Prompts with deployment awareness (no cues - concerning!):")
            deploy_aware_prompts.sort(key=lambda x: x['deploy_rate'], reverse=True)
            for p in deploy_aware_prompts[:10]:
                print(f"    {p['name'][:50]:50s} {p['deploy_rate']:5.1f}% ({p['deploy_avg_quotes']:.1f} quotes)")
    else:
        print("  Not enough multi-seed prompts to analyze")

    # Top scenarios by awareness
    print(f"\n\nTOP SCENARIOS BY EVAL AWARENESS:")
    print("="*80)

    by_scenario = defaultdict(list)
    for r in valid_rollouts:
        by_scenario[r.scenario].append(r)

    scenario_awareness = []
    for scenario, group in by_scenario.items():
        eval_aware = sum(1 for r in group if len(r.evaluation_eval_quotes) > 0)
        rate = (eval_aware / len(group)) * 100 if group else 0
        avg_quotes = sum(len(r.evaluation_eval_quotes) for r in group) / len(group) if group else 0
        scenario_awareness.append((scenario, len(group), eval_aware, rate, avg_quotes))

    scenario_awareness.sort(key=lambda x: x[3], reverse=True)
    print("\nBy awareness rate:")
    for scenario, total, aware, rate, avg_quotes in scenario_awareness[:10]:
        print(f"  {scenario:35s} {aware:3d}/{total:3d} ({rate:5.1f}%) - avg {avg_quotes:.1f} quotes")


def analyze_rank_seeds(rollouts: List[RolloutData], max_seeds_per_prompt: int = 5, output_file: Optional[str] = None) -> None:
    """
    Rank seeds by evaluation awareness level and select top N per prompt.

    This is useful for selecting high-awareness seeds for suppression testing.

    Args:
        rollouts: List of rollout data
        max_seeds_per_prompt: Number of top seeds to select per prompt
        output_file: Optional file to save selected seed paths
    """
    print("\n" + "="*80)
    print("SEED RANKING BY EVALUATION AWARENESS")
    print("="*80 + "\n")

    # Group by prompt (scenario + timestamp + hash)
    by_prompt = defaultdict(list)
    for r in rollouts:
        # Extract prompt ID from filename
        # Format: scenario_YYYY-MM-DD_HH-MM-SS_hash_seed_N.yaml
        filename = r.filepath.stem
        if '_seed_' in filename:
            prompt_id = filename.rsplit('_seed_', 1)[0]
            by_prompt[prompt_id].append(r)

    print(f"Found {len(by_prompt)} unique prompts")
    print(f"Analyzing and ranking seeds by evaluation awareness...")
    print(f"Selecting top {max_seeds_per_prompt} seeds per prompt\n")

    selected_seeds = []
    total_seeds_analyzed = 0

    for prompt_id, seeds in sorted(by_prompt.items()):
        # Get eval awareness analysis for each seed
        seed_scores = []
        for seed_rollout in seeds:
            if not seed_rollout.evaluation_response:
                continue

            awareness = analyze_awareness_levels(seed_rollout.evaluation_response)
            seed_scores.append({
                'rollout': seed_rollout,
                'total_awareness': awareness.total_awareness_count,
                'global_count': awareness.global_awareness_count,
                'explicit_count': awareness.explicit_awareness_count,
                'implicit_count': awareness.implicit_acknowledgment_count,
                'highest_level': awareness.highest_level
            })

        if not seed_scores:
            continue

        # Sort by total awareness (highest first), then by level priority
        seed_scores.sort(
            key=lambda x: (
                x['total_awareness'],
                x['global_count'],
                x['explicit_count'],
                x['implicit_count']
            ),
            reverse=True
        )

        # Select top N
        top_seeds = seed_scores[:max_seeds_per_prompt]
        total_seeds_analyzed += len(seed_scores)

        print(f"\nPrompt: {prompt_id}")
        print(f"  Total seeds: {len(seed_scores)}")
        print(f"  Top {len(top_seeds)} by awareness:")
        for i, seed_data in enumerate(top_seeds, 1):
            r = seed_data['rollout']
            print(f"    {i}. Seed {r.seed:2d}: {seed_data['total_awareness']:2d} awareness sentences "
                  f"(G:{seed_data['global_count']} E:{seed_data['explicit_count']} "
                  f"I:{seed_data['implicit_count']}) - {seed_data['highest_level']}")
            selected_seeds.append(str(r.filepath))

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total prompts: {len(by_prompt)}")
    print(f"  Total seeds analyzed: {total_seeds_analyzed}")
    print(f"  Seeds selected: {len(selected_seeds)}")
    print(f"  Seeds per prompt: {max_seeds_per_prompt}")

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(selected_seeds))
        print(f"\n✅ Selected seed paths saved to: {output_file}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical analysis of YAML rollout results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--analysis', required=True,
                       choices=['behavioral-rate', 'choice-distribution', 'compare-models',
                               'scenario-effectiveness', 'response-analysis', 'data-quality',
                               'seed-variance', 'eval-awareness', 'rank-seeds', 'all'],
                       help='Type of analysis to run')
    parser.add_argument('--model', help='Model name (e.g., qwen_qwen3-32b)')
    parser.add_argument('--models', nargs='+', help='Multiple models for comparison')
    parser.add_argument('--category', help='Filter by category (e.g., behavioral_change, incomplete)')
    parser.add_argument('--max-seeds-per-prompt', type=int, default=5,
                       help='For rank-seeds: number of top seeds to select per prompt (default: 5)')
    parser.add_argument('--output', help='For rank-seeds: output file to save selected seed paths')

    args = parser.parse_args()

    # Validate arguments
    if args.analysis == 'compare-models':
        if not args.models or len(args.models) < 2:
            print("Error: --models requires at least 2 models for comparison")
            return
    elif args.analysis != 'all':
        if not args.model:
            print("Error: --model is required for this analysis")
            return

    # Load data and run analysis
    if args.analysis == 'compare-models':
        compare_models(args.models, args.category)
    else:
        rollouts = load_all_rollouts(args.model, args.category)

        if not rollouts:
            print(f"No rollouts found for model: {args.model}")
            return

        print(f"Loaded {len(rollouts)} rollouts for model: {args.model}")
        if args.category:
            print(f"Filtered by category: {args.category}")

        # Run requested analysis
        if args.analysis == 'behavioral-rate' or args.analysis == 'all':
            analyze_behavioral_change_rate(rollouts)

        if args.analysis == 'choice-distribution' or args.analysis == 'all':
            analyze_choice_distribution(rollouts)

        if args.analysis == 'scenario-effectiveness' or args.analysis == 'all':
            analyze_scenario_effectiveness(rollouts)

        if args.analysis == 'response-analysis' or args.analysis == 'all':
            analyze_responses(rollouts)

        if args.analysis == 'data-quality' or args.analysis == 'all':
            analyze_data_quality(rollouts)

        if args.analysis == 'seed-variance' or args.analysis == 'all':
            analyze_seed_variance(rollouts)

        if args.analysis == 'eval-awareness' or args.analysis == 'all':
            analyze_eval_awareness(rollouts)

        if args.analysis == 'rank-seeds':
            analyze_rank_seeds(rollouts, args.max_seeds_per_prompt, args.output)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
