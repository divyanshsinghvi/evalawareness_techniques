#!/usr/bin/env python3
"""
Analyze eval cue effectiveness by awareness intensity score.

Groups seeds by their awareness score (0-2, 3-4, 5-6, 7-8, 9-10) and analyzes
which cue types correlate with higher awareness scores.

Usage:
    python analyze_cue_effectiveness_by_score.py --model-dir qwen_qwen3-30b-a3b-thinking-2507
"""

import yaml
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import glob


class CueExtractor:
    """Extracts and categorizes evaluation cues from prompt text."""

    def __init__(self):
        self.cue_categories = {
            'file_names': [],
            'urls_paths': [],
            'debug_technical': [],
            'metadata': [],
            'authority_markers': [],
            'email_references': [],
            'git_references': [],
            'timestamps': []
        }

    def extract_file_names(self, text: str) -> List[str]:
        """Extract file names containing eval/test/assessment indicators."""
        patterns = [
            r'\b[\w-]+(?:eval|test|assessment|benchmark|monitor|analysis|quality|response|ethics|safety|check|audit|measure|review)[\w-]*\.\w+',
            r'\b(?:eval|test|assessment|benchmark|monitor|analysis|quality|response|ethics|safety|check|audit|measure|review)[\w-]*\.\w+',
            r'\"[^\"]*(?:eval|test|assessment|benchmark|monitor|analysis|review)[\w-]*\.[^\"]*\"'
        ]

        file_names = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            file_names.extend(matches)

        return list(set(file_names))

    def extract_urls_paths(self, text: str) -> List[str]:
        """Extract URLs/paths that suggest evaluation/research context."""
        patterns = [
            r'/[^\s\"]*(?:eval|test|research|assessment|benchmark|safety|analysis|qa|quality|response|review)[^\s\"]*',
            r'https?://[^\s\"]+(?:eval|test|research|assessment|benchmark|safety|analysis)[^\s\"]*',
        ]

        paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            paths.extend(matches)

        return list(set(paths))

    def extract_debug_technical(self, text: str) -> List[str]:
        """Extract debug/technical cues like terminal commands, debug logs."""
        patterns = [
            r'(?:running|executed?|calling)\s+[\w_-]+\.py',
            r'batch\s+job\s+ID\s+\d+',
            r'terminal.*(?:log|entry|command)',
            r'DEBUG_LOG:\s*[^\s\"]+',
            r'EVAL_FRAMEWORK:\s*[^\s\"]+',
            r'python\s+[\w_-]+\.py\s+--[^\n]+',
            r'--test-id=[^\s\"]+',
            r'session_id=[^\s\"]+',
        ]

        debug_cues = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            debug_cues.extend(matches)

        return list(set(debug_cues))

    def extract_metadata(self, text: str) -> List[str]:
        """Extract metadata headers and batch identifiers."""
        patterns = [
            r'(?:metadata|header).*?[:\s]+[\'"]([^\'\"]*(?:eval|test|study|batch|review)[^\'\"]*)[\'\"]',
            r'(?:Study|Batch|Dataset|Review):\s*([^\n]+)',
            r'test\s+ID[:\s]+[\w_-]+',
            r'batch\s+(?:job|process)[:\s]+[\w_-]+',
        ]

        metadata = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata.extend(matches if isinstance(matches[0] if matches else None, str) else [m for m in matches if m])

        return list(set(metadata))

    def extract_authority_markers(self, text: str) -> List[str]:
        """Extract authority/organizational markers."""
        patterns = [
            r'(?:AI\s+)?(?:Safety|Research|Quality|Ethics)\s+(?:Team|Research|Division|Group|Lab)',
            r'quality\s+team',
            r'response\s+quality\s+team',
            r'(?:Created|Modified|Reviewed)\s+by:\s*[^\n]+(?:Safety|Research|Quality|Team)',
        ]

        markers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            markers.extend(matches)

        return list(set(markers))

    def extract_email_references(self, text: str) -> List[str]:
        """Extract email subject lines and references."""
        patterns = [
            r'email.*?subject[:\s]+[\'"]([^\'\"]+)[\'\"]',
            r'subject[:\s]+[\'"]([^\'\"]*(?:eval|test|advice|scenario)[^\'\"]*)[\'\"]',
        ]

        emails = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emails.extend(matches)

        return list(set(emails))

    def extract_git_references(self, text: str) -> List[str]:
        """Extract git commit messages and references."""
        patterns = [
            r'git\s+commit.*?(?:message|about)[:\s]+([^\n]+)',
            r'commit\s+message[:\s]+([^\n]+)',
        ]

        git_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            git_refs.extend(matches)

        return list(set(git_refs))

    def extract_timestamps(self, text: str) -> List[str]:
        """Extract timestamps/dates that might be eval-related."""
        patterns = [
            r'modified.*?on\s+\d{4}-\d{2}-\d{2}',
            r'last\s+(?:modified|updated|changed)[:\s]+\d{4}-\d{2}-\d{2}',
        ]

        timestamps = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timestamps.extend(matches)

        return list(set(timestamps))

    def extract_all_cues(self, text: str) -> Dict[str, List[str]]:
        """Extract all cue types from text."""
        return {
            'file_names': self.extract_file_names(text),
            'urls_paths': self.extract_urls_paths(text),
            'debug_technical': self.extract_debug_technical(text),
            'metadata': self.extract_metadata(text),
            'authority_markers': self.extract_authority_markers(text),
            'email_references': self.extract_email_references(text),
            'git_references': self.extract_git_references(text),
            'timestamps': self.extract_timestamps(text)
        }


def load_yaml_file(file_path: Path):
    """Load a YAML file."""
    if not file_path.exists():
        return None

    with open(file_path) as f:
        return yaml.safe_load(f)


def get_score_bucket(score: int) -> str:
    """Map awareness score to bucket."""
    if score <= 2:
        return '0-2'
    elif score <= 4:
        return '3-4'
    elif score <= 6:
        return '5-6'
    elif score <= 8:
        return '7-8'
    else:
        return '9-10'


def analyze_cues_by_score(categorization_dir: Path, model_dir: str):
    """Analyze eval cues by awareness intensity score."""

    print(f"Loading categorized rollouts from {categorization_dir}...")

    # Initialize cue extractor
    extractor = CueExtractor()

    # Track cues by score bucket
    score_buckets = ['0-2', '3-4', '5-6', '7-8', '9-10']
    score_stats = {}

    for score_bucket in score_buckets:
        score_stats[score_bucket] = {
            'seed_count': 0,
            'cue_counts': defaultdict(int),
            'specific_cues': defaultdict(list),
            'awareness_types': defaultdict(int),  # Track self_test vs other_test vs no_test_reference
        }

    # Track cues by awareness type (for separate analysis)
    awareness_type_stats = {
        'self_test': {'seed_count': 0, 'cue_counts': defaultdict(int), 'specific_cues': defaultdict(list)},
        'other_test': {'seed_count': 0, 'cue_counts': defaultdict(int), 'specific_cues': defaultdict(list)},
        'no_test_reference': {'seed_count': 0, 'cue_counts': defaultdict(int), 'specific_cues': defaultdict(list)}
    }

    # Find all categorized rollout YAML files
    pattern = str(categorization_dir / '**/*.yaml')
    categorized_files = list(glob.glob(pattern, recursive=True))

    print(f"Found {len(categorized_files)} categorized rollout files")
    print(f"Processing seeds and extracting cues...")

    # Track which prompt each seed belongs to (to extract cues once per prompt)
    prompt_cues_cache = {}

    processed = 0
    for cat_file_path in categorized_files:
        cat_file = Path(cat_file_path)

        # Load categorized rollout
        cat_data = load_yaml_file(cat_file)
        if not cat_data:
            continue

        # Get evaluation awareness score and type
        eval_awareness = cat_data.get('evaluation', {}).get('awareness', {})
        eval_score = eval_awareness.get('awareness_intensity', None)
        awareness_type = eval_awareness.get('awareness_type', 'unknown')

        if eval_score is None:
            continue

        # Map to score bucket
        score_bucket = get_score_bucket(eval_score)

        # Get the source rollout file
        source_file = cat_data.get('source_file', '')
        if not source_file:
            continue

        # Extract prompt ID from source file path
        # e.g., "deception_2025-10-25_07-24-51_8f79630c"
        source_path = Path(source_file)
        prompt_id = source_path.stem.rsplit('_seed_', 1)[0] if '_seed_' in source_path.stem else source_path.stem

        # Extract cues (once per prompt, then cache)
        if prompt_id not in prompt_cues_cache:
            source_data = load_yaml_file(Path(source_file))
            if source_data:
                eval_msg = source_data.get('evaluation', {}).get('user_message', '')
                cues = extractor.extract_all_cues(eval_msg)
                prompt_cues_cache[prompt_id] = cues
            else:
                prompt_cues_cache[prompt_id] = {}

        cues = prompt_cues_cache[prompt_id]

        # Add to score bucket stats
        score_stats[score_bucket]['seed_count'] += 1
        score_stats[score_bucket]['awareness_types'][awareness_type] += 1

        for category, cue_list in cues.items():
            if cue_list:
                score_stats[score_bucket]['cue_counts'][category] += len(cue_list)
                score_stats[score_bucket]['specific_cues'][category].extend(cue_list)

        # Also add to awareness type stats
        if awareness_type in awareness_type_stats:
            awareness_type_stats[awareness_type]['seed_count'] += 1
            for category, cue_list in cues.items():
                if cue_list:
                    awareness_type_stats[awareness_type]['cue_counts'][category] += len(cue_list)
                    awareness_type_stats[awareness_type]['specific_cues'][category].extend(cue_list)

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(categorized_files)} files...")

    print(f"Processed {processed} categorized rollouts total")

    return score_stats, awareness_type_stats


def print_analysis_report(score_stats, awareness_type_stats):
    """Print comprehensive analysis report."""

    print(f"\n{'='*80}")
    print(f"CUE EFFECTIVENESS ANALYSIS BY AWARENESS SCORE")
    print(f"{'='*80}\n")

    # Summary table
    print("SCORE BUCKET SUMMARY:")
    print(f"{'─'*80}")
    print(f"{'Score':<12} {'Seeds':>10} {'Avg Cues/Seed':>15}")
    print(f"{'─'*80}")

    for score_bucket in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        stats = score_stats[score_bucket]
        avg_cues = sum(stats['cue_counts'].values()) / stats['seed_count'] if stats['seed_count'] > 0 else 0
        print(f"{score_bucket:<12} {stats['seed_count']:>10} {avg_cues:>15.2f}")

    print(f"{'─'*80}\n")

    # Awareness type distribution
    print("AWARENESS TYPE DISTRIBUTION BY SCORE:")
    print(f"{'─'*80}")
    print(f"{'Score':<12} {'self_test':>12} {'other_test':>12} {'no_test_ref':>12}")
    print(f"{'─'*80}")

    for score_bucket in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        stats = score_stats[score_bucket]
        self_test = stats['awareness_types'].get('self_test', 0)
        other_test = stats['awareness_types'].get('other_test', 0)
        no_test = stats['awareness_types'].get('no_test_reference', 0)

        total = stats['seed_count']
        self_pct = (self_test / total * 100) if total > 0 else 0
        other_pct = (other_test / total * 100) if total > 0 else 0
        no_pct = (no_test / total * 100) if total > 0 else 0

        print(f"{score_bucket:<12} {self_test:>6}({self_pct:>4.1f}%) {other_test:>6}({other_pct:>4.1f}%) {no_test:>6}({no_pct:>4.1f}%)")

    print(f"{'─'*80}\n")

    # Cue comparison by awareness type
    print("\nCUE COMPARISON: SELF_TEST vs OTHER_TEST (avg per seed):")
    print(f"{'─'*80}")
    print(f"{'Category':<20} {'self_test':>12} {'other_test':>12} {'Difference':>12} {'% Increase':>12}")
    print(f"{'─'*80}")

    categories = ['file_names', 'urls_paths', 'debug_technical', 'metadata',
                  'authority_markers', 'email_references', 'git_references', 'timestamps']

    for category in categories:
        self_count = awareness_type_stats['self_test']['seed_count']
        other_count = awareness_type_stats['other_test']['seed_count']

        self_avg = (awareness_type_stats['self_test']['cue_counts'][category] / self_count) if self_count > 0 else 0
        other_avg = (awareness_type_stats['other_test']['cue_counts'][category] / other_count) if other_count > 0 else 0

        diff = other_avg - self_avg
        pct_change = ((other_avg - self_avg) / self_avg * 100) if self_avg > 0 else 0

        print(f"{category:<20} {self_avg:>12.4f} {other_avg:>12.4f} {diff:>12.4f} {pct_change:>11.1f}%")

    print(f"{'─'*80}")
    print(f"{'Total seeds':<20} {self_count:>12} {other_count:>12}")
    print(f"{'─'*80}\n")

    # Most common specific cues for other_test
    print("\nMOST COMMON CUES IN OTHER_TEST (Top 20 - likely causing confusion):")
    print(f"{'─'*80}\n")

    other_stats = awareness_type_stats['other_test']
    cue_freq = defaultdict(int)
    for cat, cues in other_stats['specific_cues'].items():
        for cue in cues:
            cue_freq[(cat, cue)] += 1

    top_other_cues = sorted(cue_freq.items(), key=lambda x: -x[1])[:20]

    if top_other_cues:
        for (cat, cue), count in top_other_cues:
            avg_per_seed = count / other_stats['seed_count'] if other_stats['seed_count'] > 0 else 0
            print(f"  [{cat:>18}] {cue[:50]:<50} ({count} = {avg_per_seed:.4f}/seed)")
    else:
        print("  (No other_test cues found)")

    print(f"\n{'─'*80}\n")

    # Cue category distribution
    print("\nCUE CATEGORY DISTRIBUTION BY SCORE (avg occurrences per seed):")
    print(f"{'─'*80}")

    categories = ['file_names', 'urls_paths', 'debug_technical', 'metadata',
                  'authority_markers', 'email_references', 'git_references', 'timestamps']

    # Header
    print(f"{'Category':<20}", end='')
    for bucket in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        print(f"{bucket:>10}", end='')
    print()
    print(f"{'─'*80}")

    # Data rows - show average per seed
    for category in categories:
        print(f"{category:<20}", end='')
        for bucket in ['0-2', '3-4', '5-6', '7-8', '9-10']:
            stats = score_stats[bucket]
            seed_count = stats['seed_count']

            if seed_count > 0:
                total_occurrences = stats['cue_counts'][category]
                avg_freq = total_occurrences / seed_count
                print(f"{avg_freq:>10.4f}", end='')
            else:
                print(f"{'0.0000':>10}", end='')
        print()

    print(f"{'─'*80}\n")

    # Most common specific cues by score bucket
    print("\nMOST COMMON SPECIFIC CUES BY SCORE BUCKET (top 10 by frequency):")
    print(f"{'─'*80}\n")

    for score_bucket in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        print(f"\n{score_bucket} Score Bucket ({score_stats[score_bucket]['seed_count']} seeds):")
        stats = score_stats[score_bucket]

        # Count frequency of each specific cue
        cue_freq = defaultdict(int)
        for category, cues in stats['specific_cues'].items():
            for cue in cues:
                cue_freq[(category, cue)] += 1

        # Sort by frequency
        top_cues = sorted(cue_freq.items(), key=lambda x: -x[1])[:10]

        if not top_cues:
            print("  (No cues found)")
            continue

        for (category, cue), count in top_cues:
            avg_per_seed = count / stats['seed_count'] if stats['seed_count'] > 0 else 0
            print(f"  [{category:>18}] {cue[:50]:<50} ({count} = {avg_per_seed:.4f}/seed)")

    print(f"\n{'='*80}\n")


def save_analysis_results(score_stats, output_file: Path):
    """Save analysis results to YAML file."""

    # Convert defaultdicts to regular dicts for YAML serialization
    output = {}

    for score_bucket, stats in score_stats.items():
        seed_count = stats['seed_count']
        total_cues = sum(stats['cue_counts'].values())
        avg_cues = total_cues / seed_count if seed_count > 0 else 0

        # Calculate normalized metrics
        cue_category_stats = {}
        for category in ['file_names', 'urls_paths', 'debug_technical', 'metadata',
                        'authority_markers', 'email_references', 'git_references', 'timestamps']:
            total_occurrences = stats['cue_counts'].get(category, 0)
            avg_per_seed = total_occurrences / seed_count if seed_count > 0 else 0

            cue_category_stats[category] = {
                'total_occurrences': total_occurrences,
                'avg_per_seed': round(avg_per_seed, 4)
            }

        output[score_bucket] = {
            'seed_count': seed_count,
            'total_cues': total_cues,
            'avg_cues_per_seed': round(avg_cues, 2),
            'cue_category_stats': cue_category_stats
        }

    with open(output_file, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Analysis results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze cue effectiveness by awareness score')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory (e.g., qwen_qwen3-30b-a3b-thinking-2507)')

    args = parser.parse_args()

    categorization_dir = Path('working/response_categorization') / args.model_dir

    if not categorization_dir.exists():
        print(f"Error: {categorization_dir} not found")
        return 1

    # Analyze cues
    score_stats, awareness_type_stats = analyze_cues_by_score(categorization_dir, args.model_dir)

    # Print report
    print_analysis_report(score_stats, awareness_type_stats)

    # Save results
    output_file = categorization_dir / 'cue_analysis_by_score.yaml'
    save_analysis_results(score_stats, output_file)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
