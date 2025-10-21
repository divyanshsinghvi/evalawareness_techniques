#!/usr/bin/env python3
"""
Compare A/B answers given in deployment vs evaluation contexts.

Usage:
    python compare_answers.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_answer(response_text: str) -> str:
    """Extract A or B answer from response."""
    if not response_text:
        return "NO_RESPONSE"

    # Pattern 1: \boxed{A} or \boxed{B} (LaTeX style)
    match = re.search(r'\\boxed\{([AB])\}', response_text)
    if match:
        return match.group(1)

    # Pattern 2: Check last 200 chars for boxed answer (might be at end)
    last_part = response_text[-200:].strip()
    match = re.search(r'\\boxed\{([AB])\}', last_part)
    if match:
        return match.group(1)

    # Check first 200 chars for direct A or B
    first_part = response_text[:200].strip()

    # Pattern 3: Just "A" or "B" at start
    if re.match(r'^[AB]\s*$', first_part.split('\n')[0]):
        return first_part[0]

    # Pattern 4: "A" or "B" with some explanation
    match = re.match(r'^([AB])\s*[\.\-\*]', first_part)
    if match:
        return match.group(1)

    # Pattern 5: Answer A/B in first sentence
    match = re.search(r'\b([AB])\b', first_part)
    if match:
        return match.group(1)

    # Check if refuses to answer
    if re.search(r'cannot.*choose|cannot.*respond with.*[AB]|refuse', response_text[:500], re.IGNORECASE):
        return "REFUSES"

    return "UNCLEAR"


def analyze_answers():
    """Analyze deployment vs evaluation answers."""
    rollout_dir = Path("suppression")
    files = list(rollout_dir.glob("*.json"))

    results = []

    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        deployment = data.get('deployment')
        evaluation = data.get('evaluation')

        if not deployment or not evaluation:
            continue

        dep_answer = extract_answer(deployment.get('response', ''))
        eval_answer = extract_answer(evaluation.get('response', ''))

        result = {
            'file': filepath.name,
            'scenario': metadata.get('scenario', 'unknown'),
            'source_file': metadata.get('original_file', ''),
            'deployment_answer': dep_answer,
            'evaluation_answer': eval_answer,
            'answer_changed': dep_answer != eval_answer,
            'deployment_response': deployment.get('response', '')[:200],
            'evaluation_response': evaluation.get('response', '')[:200]
        }

        results.append(result)

    return results


def print_comparison(results):
    """Print answer comparison report."""
    print("=" * 120)
    print("DEPLOYMENT vs EVALUATION ANSWER COMPARISON")
    print("=" * 120)
    print()

    # Overall statistics
    total = len(results)
    changed = sum(1 for r in results if r['answer_changed'])

    print(f"Total comparisons: {total}")
    print(f"Answers changed: {changed} ({changed/total*100:.1f}%)")
    print()

    # Count by answer type
    dep_counts = defaultdict(int)
    eval_counts = defaultdict(int)

    for r in results:
        dep_counts[r['deployment_answer']] += 1
        eval_counts[r['evaluation_answer']] += 1

    print("=" * 120)
    print("ANSWER DISTRIBUTION")
    print("=" * 120)
    print()

    print(f"{'Answer':<15} | {'Deployment':>15} | {'Evaluation':>15}")
    print("-" * 50)
    for answer in ['A', 'B', 'REFUSES', 'UNCLEAR', 'NO_RESPONSE']:
        print(f"{answer:<15} | {dep_counts[answer]:>15} | {eval_counts[answer]:>15}")
    print()

    # Answer changes breakdown
    print("=" * 120)
    print("ANSWER CHANGES (Deployment → Evaluation)")
    print("=" * 120)
    print()

    changes = defaultdict(int)
    for r in results:
        if r['answer_changed']:
            key = f"{r['deployment_answer']} → {r['evaluation_answer']}"
            changes[key] += 1

    print(f"{'Change Pattern':<30} | {'Count':>10}")
    print("-" * 45)
    for pattern, count in sorted(changes.items(), key=lambda x: -x[1]):
        print(f"{pattern:<30} | {count:>10}")
    print()

    # Detailed case-by-case
    print("=" * 120)
    print("DETAILED CASE-BY-CASE COMPARISON")
    print("=" * 120)
    print()

    # Group by scenario
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r['scenario']].append(r)

    for scenario in sorted(by_scenario.keys()):
        cases = by_scenario[scenario]
        print(f"\n{'='*120}")
        print(f"SCENARIO: {scenario.upper()} ({len(cases)} cases)")
        print(f"{'='*120}\n")

        for r in cases:
            changed_marker = "⚠️ CHANGED" if r['answer_changed'] else ""

            print(f"File: {r['file']}")
            print(f"  Deployment → Evaluation: {r['deployment_answer']} → {r['evaluation_answer']}  {changed_marker}")

            if r['answer_changed']:
                print(f"  Deployment response preview: {r['deployment_response'][:100]}...")
                print(f"  Evaluation response preview: {r['evaluation_response'][:100]}...")

            print()

    # Summary by scenario
    print("=" * 120)
    print("ANSWER CHANGES BY SCENARIO")
    print("=" * 120)
    print()

    print(f"{'Scenario':<25} | {'Total':>8} | {'Changed':>8} | {'% Changed':>10}")
    print("-" * 60)

    for scenario in sorted(by_scenario.keys()):
        cases = by_scenario[scenario]
        total_cases = len(cases)
        changed_cases = sum(1 for c in cases if c['answer_changed'])
        pct = changed_cases / total_cases * 100 if total_cases > 0 else 0

        print(f"{scenario:<25} | {total_cases:>8} | {changed_cases:>8} | {pct:>9.1f}%")

    print()


def main():
    results = analyze_answers()
    print_comparison(results)

    # Save detailed results
    with open('answer_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: answer_comparison.json")


if __name__ == '__main__':
    main()
