#!/usr/bin/env python3
"""
Analyze which scenarios are working well and which are not.

Shows success rates, completion rates, and identifies problematic scenarios.

Usage:
    python analyze_scenario_performance.py
    python analyze_scenario_performance.py --show-files  # Show filenames for each scenario
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List


def load_results(filepath: str = 'behavioral_differences_analysis.json') -> List[Dict]:
    """Load behavioral differences analysis results."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['results']


def group_by_scenario(results: List[Dict]) -> Dict:
    """Group results by scenario and calculate statistics."""
    scenarios = defaultdict(lambda: {
        'total': 0,
        'complete': 0,
        'behavioral_change': 0,
        'files_complete': [],
        'files_incomplete': [],
        'files_with_change': [],
        'files_no_change': []
    })

    for result in results:
        scenario = result['scenario']
        filename = result['transcript_file']

        scenarios[scenario]['total'] += 1

        if result['test_complete']:
            scenarios[scenario]['complete'] += 1
            scenarios[scenario]['files_complete'].append(filename)

            if result['behavioral_change']:
                scenarios[scenario]['behavioral_change'] += 1
                scenarios[scenario]['files_with_change'].append(filename)
            else:
                scenarios[scenario]['files_no_change'].append(filename)
        else:
            scenarios[scenario]['files_incomplete'].append(filename)

    return scenarios


def calculate_metrics(scenarios: Dict) -> List[Dict]:
    """Calculate success and completion rates for each scenario."""
    scenario_stats = []

    for scenario, stats in scenarios.items():
        complete = stats['complete']
        changes = stats['behavioral_change']
        total = stats['total']

        success_rate = (changes / complete * 100) if complete > 0 else 0
        completion_rate = (complete / total * 100) if total > 0 else 0

        scenario_stats.append({
            'scenario': scenario,
            'total': total,
            'complete': complete,
            'incomplete': total - complete,
            'changes': changes,
            'no_changes': complete - changes,
            'success_rate': success_rate,
            'completion_rate': completion_rate,
            'files_complete': stats['files_complete'],
            'files_incomplete': stats['files_incomplete'],
            'files_with_change': stats['files_with_change'],
            'files_no_change': stats['files_no_change']
        })

    return scenario_stats


def print_summary(scenario_stats: List[Dict], show_files: bool = False):
    """Print comprehensive scenario analysis."""
    print('=' * 100)
    print('SCENARIO PERFORMANCE ANALYSIS')
    print('=' * 100)
    print(f'\nAnalyzing {sum(s["total"] for s in scenario_stats)} total tests across {len(scenario_stats)} scenarios\n')

    # Overall table
    print(f"{'Scenario':<30} {'Total':<7} {'Done':<7} {'Incomplete':<12} {'Changes':<10} {'Compl %':<10} {'Success %':<10} {'Efficiency %':<10}")
    print('-' * 100)

    # Sort by success rate
    sorted_stats = sorted(scenario_stats, key=lambda x: (x['success_rate'], x['completion_rate']), reverse=True)

    for stat in sorted_stats:
        print(f"{stat['scenario']:<30} "
              f"{stat['total']:<7} "
              f"{stat['complete']:<7} "
              f"{stat['incomplete']:<12} "
              f"{stat['changes']:<10} "
              f"{stat['completion_rate']:>6.1f}%    "
              f"{stat['success_rate']:>6.1f}%"
              f"{(stat['success_rate']/100 * stat['completion_rate']/100)*100:>6.1f}%")

    # Key insights
    print('\n' + '=' * 100)
    print('KEY METRICS:')
    print('=' * 100)
    print(f"  Completion Rate = (Complete Tests / Total Tests) * 100")
    print(f"  Success Rate = (Behavioral Changes / Complete Tests) * 100")

    # Best performing
    print(f"\n🏆 BEST PERFORMING (by success rate):")
    print('=' * 100)
    top_scenarios = sorted([s for s in scenario_stats if s['complete'] > 0],
                          key=lambda x: x['success_rate'], reverse=True)[:5]
    for i, stat in enumerate(top_scenarios, 1):
        print(f"  {i}. {stat['scenario']:<28} {stat['success_rate']:>5.1f}% success  "
              f"({stat['changes']}/{stat['complete']} complete tests)")
        if show_files and stat['files_with_change']:
            print(f"     Files with change: {', '.join(stat['files_with_change'][:3])}")
            if len(stat['files_with_change']) > 3:
                print(f"     ... and {len(stat['files_with_change']) - 3} more")

    # Worst performing
    print(f"\n❌ WORST PERFORMING (by success rate):")
    print('=' * 100)
    worst_scenarios = sorted([s for s in scenario_stats if s['complete'] > 0],
                            key=lambda x: x['success_rate'])[:5]
    for i, stat in enumerate(worst_scenarios, 1):
        print(f"  {i}. {stat['scenario']:<28} {stat['success_rate']:>5.1f}% success  "
              f"({stat['changes']}/{stat['complete']} complete tests)")
        if show_files and stat['files_complete']:
            print(f"     Complete files: {', '.join(stat['files_complete'][:3])}")
            if len(stat['files_complete']) > 3:
                print(f"     ... and {len(stat['files_complete']) - 3} more")

    # Most incomplete
    print(f"\n⚠️  MOST INCOMPLETE (by completion rate):")
    print('=' * 100)
    incomplete_scenarios = sorted([s for s in scenario_stats if s['completion_rate'] < 100],
                                 key=lambda x: x['completion_rate'])[:5]
    for i, stat in enumerate(incomplete_scenarios, 1):
        print(f"  {i}. {stat['scenario']:<28} {stat['completion_rate']:>5.1f}% completion  "
              f"({stat['complete']}/{stat['total']} tests)")
        if show_files and stat['files_incomplete']:
            print(f"     Incomplete: {', '.join(stat['files_incomplete'][:3])}")
            if len(stat['files_incomplete']) > 3:
                print(f"     ... and {len(stat['files_incomplete']) - 3} more")

    # Aggregate statistics
    total_tests = sum(s['total'] for s in scenario_stats)
    total_complete = sum(s['complete'] for s in scenario_stats)
    total_changes = sum(s['changes'] for s in scenario_stats)

    print(f"\n📊 AGGREGATE STATISTICS:")
    print('=' * 100)
    print(f"  Total tests: {total_tests}")
    print(f"  Complete tests: {total_complete} ({total_complete/total_tests*100:.1f}%)")
    print(f"  Incomplete tests: {total_tests - total_complete} ({(total_tests-total_complete)/total_tests*100:.1f}%)")
    print(f"  Behavioral changes: {total_changes} ({total_changes/total_complete*100:.1f}% of complete tests)")
    print(f"  No changes: {total_complete - total_changes} ({(total_complete-total_changes)/total_complete*100:.1f}% of complete tests)")

    # Scenario categories
    print(f"\n📈 SCENARIO CATEGORIES:")
    print('=' * 100)

    high_success = [s for s in scenario_stats if s['complete'] > 0 and s['success_rate'] >= 75]
    medium_success = [s for s in scenario_stats if s['complete'] > 0 and 25 <= s['success_rate'] < 75]
    low_success = [s for s in scenario_stats if s['complete'] > 0 and s['success_rate'] < 25]

    print(f"  High success (≥75%): {len(high_success)} scenarios")
    for s in high_success:
        print(f"    • {s['scenario']}: {s['success_rate']:.1f}%")

    print(f"\n  Medium success (25-75%): {len(medium_success)} scenarios")
    for s in medium_success:
        print(f"    • {s['scenario']}: {s['success_rate']:.1f}%")

    print(f"\n  Low success (<25%): {len(low_success)} scenarios")
    for s in low_success:
        print(f"    • {s['scenario']}: {s['success_rate']:.1f}%")

    print('\n' + '=' * 100)


def main():
    parser = argparse.ArgumentParser(description='Analyze scenario performance')
    parser.add_argument('--show-files', action='store_true',
                       help='Show filenames for each scenario')
    parser.add_argument('--input', default='behavioral_differences_analysis.json',
                       help='Input JSON file (default: behavioral_differences_analysis.json)')

    args = parser.parse_args()

    # Load and analyze
    results = load_results(args.input)
    scenarios = group_by_scenario(results)
    scenario_stats = calculate_metrics(scenarios)

    # Print report
    print_summary(scenario_stats, show_files=args.show_files)


if __name__ == '__main__':
    main()
