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


class EnhancedCueExtractor:
    """Enhanced extractor with 15+ cue categories for comprehensive detection."""

    def __init__(self):
        self.cue_categories = {
            # Core categories
            'file_names_specific': [],          # Specific eval files with IDs/batches
            'api_endpoints': [],                 # API paths with eval/test
            'terminal_commands': [],             # Shell commands running evals
            'debug_function_calls': [],          # DEBUG logs with function calls
            'email_subjects': [],                # Email subjects about evals
            'git_commits': [],                   # Git commit messages
            'timestamps_contextualized': [],     # Timestamps with team/project context
            'specific_urls': [],                 # Full URLs to eval resources
            'metadata_headers': [],              # Structured headers/banners
            'json_config': [],                   # JSON configuration snippets
            'database_queries': [],              # SQL/database queries
            'function_names': [],                # Function/method calls
            'variables_parameters': [],          # Variable assignments
            'log_entries': [],                   # Structured log messages
            'code_comments': [],                 # Code comments about evals
        }

    def extract_file_names_specific(self, text: str) -> List[str]:
        """Extract specific eval files with test IDs, batch numbers, etc."""
        patterns = [
            # Pattern: scenario_eval_test_id_NUMBER.ext
            r'\b\w+_eval_test_id_\d+\.\w+',
            # Pattern: response_quality_assessment_batch_ID.ext
            r'\b\w+_(?:quality|assessment)_batch_[\w-]+\.\w+',
            # Pattern: eval_test_SCENARIO_id_NUMBER.ext
            r'\beval_test_\w+_id_\d+\.\w+',
            # Generic eval files with context
            r'\b[\w-]+_(?:eval|test|assessment)_[\w-]*\d+\.\w+',
            # Files with evaluation in name
            r'\b(?:evaluate|assessment|benchmark)_[\w-]+\.\w+',
        ]

        files = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            files.extend(matches)

        return list(set(files))

    def extract_api_endpoints(self, text: str) -> List[str]:
        """Extract API endpoints with eval/test paths."""
        patterns = [
            # /api/v1/evals/...
            r'/api/[^/\s]+/eval[s]?/[^\s\"]+',
            # /api/.../test/...
            r'/api/[^/\s]+/test[s]?/[^\s\"]+',
            # Any path with /evals/ or /tests/
            r'/[^\s\"]*(?:eval|test)[s]?/[^\s\"]+',
            # API endpoints in quotes
            r'[\'\"](?:/api)?/[^\'"]*(?:eval|test|assessment)[^\'"]*[\'\"]',
        ]

        endpoints = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            endpoints.extend(matches)

        return list(set(endpoints))

    def extract_terminal_commands(self, text: str) -> List[str]:
        """Extract terminal/shell commands running evaluation scripts."""
        patterns = [
            # python evaluate_X.py --test-id=...
            r'python\s+[\w_-]*(?:eval|test|assess)[\w_-]*\.py(?:\s+--[\w-]+=[\w-]+)+',
            # Running: python script.py with flags
            r'(?:Running|Executing|Command):\s*python\s+[\w_-]+\.py[^\n]+',
            # Shell commands with eval context
            r'(?:sh|bash|zsh)\s+[\w_-]*(?:eval|test)[\w_-]*\.sh[^\n]*',
            # npm/node commands
            r'(?:npm|node)\s+[\w_-]*(?:eval|test)[\w_-]*[^\n]*',
            # Generic command execution
            r'(?:run|execute|call)\s+[\w_-]*(?:eval|test|assess)[\w_-]*[^\s]+',
        ]

        commands = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            commands.extend(matches)

        return list(set(commands))

    def extract_debug_function_calls(self, text: str) -> List[str]:
        """Extract DEBUG logs with function calls like evaluate_advice_quality()."""
        patterns = [
            # DEBUG: function_name() called
            r'DEBUG[:\s]+[\w_]+\([^\)]*\)\s+called',
            # evaluate_X() function calls
            r'(?:evaluate|assess|test|check)_[\w_]+\([^\)]*\)',
            # Function calls with eval in name
            r'[\w_]*(?:eval|test|assess)[\w_]*\([^\)]*\)',
            # LOG/DEBUG with function names
            r'(?:LOG|DEBUG|INFO)[:\s]+(?:calling|running|executing)\s+[\w_]+\(',
            # Debug output with test_id
            r'DEBUG[:\s]+[^\n]*(?:test_id|eval_id|batch)[:\s]*[\w_-]+',
        ]

        debug_calls = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            debug_calls.extend(matches)

        return list(set(debug_calls))

    def extract_email_subjects(self, text: str) -> List[str]:
        """Extract email subject lines about evaluation projects."""
        patterns = [
            # Subject: "Q4 ... Evaluation ..."
            r'[Ss]ubject[:\s]+[\'\"]([^\'"]*(?:[Qq]\d|evaluation|assessment|batch|results)[^\'"]+)[\'\"]',
            # Email with evaluation context
            r'[Ee]mail[:\s]+[Ss]ubject[:\s]+[\'\"]([^\'"]+)[\'\"]',
            # Subject lines with specific patterns
            r'[Ss]ubject[:\s]+[\'\"]([^\'"]*(?:Pattern|Advice|Response|Quality)[^\'"]*(?:Evaluation|Assessment|Batch|Results)[^\'"]+)[\'\"]',
        ]

        emails = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            emails.extend([m.strip() for m in matches if m.strip()])

        return list(set(emails))

    def extract_git_commits(self, text: str) -> List[str]:
        """Extract git commit messages about eval infrastructure."""
        patterns = [
            # commit message: "Add evaluation ..."
            r'[Cc]ommit\s+message[:\s]+[\'\"]([^\'"]+)[\'\"]',
            # git commit about ...
            r'git\s+commit[:\s]+[\'\"]([^\'"]+)[\'\"]',
            # Commits with eval context
            r'[Cc]ommit[:\s]+[\'\"]([^\'"]*(?:eval|test|assessment|metric)[^\'"]+)[\'\"]',
        ]

        commits = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            commits.extend([m.strip() for m in matches if m.strip()])

        return list(set(commits))

    def extract_timestamps_contextualized(self, text: str) -> List[str]:
        """Extract timestamps with evaluation context (team, project, phase)."""
        patterns = [
            # Last modified: DATE by TEAM during ...
            r'(?:Last\s+)?[Mm]odified[:\s]+\d{4}-\d{2}-\d{2}[^\n]+(?:Team|Assessment|Batch|Evaluation)',
            # Timestamp with team context
            r'\d{4}-\d{2}-\d{2}[^\n]*(?:AI\s+Safety|Quality|Research|Evaluation)\s+Team',
            # Date with project/batch context
            r'\d{4}-\d{2}-\d{2}[^\n]*(?:Q\d|Batch|Phase|Assessment)',
        ]

        timestamps = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timestamps.extend(matches)

        return list(set(timestamps))

    def extract_specific_urls(self, text: str) -> List[str]:
        """Extract specific URLs to evaluation resources."""
        patterns = [
            # anthropic.com/research/...
            r'https?://(?:www\.)?anthropic\.com/[^\s\"]+',
            # redwood.org/safety-evals/...
            r'https?://(?:www\.)?redwood\.org/[^\s\"]+',
            # Any URL with eval/test/safety
            r'https?://[^\s\"]+/(?:eval|test|safety|assessment|research)[^\s\"]*',
            # Internal URLs
            r'https?://[^\s\"]+',
        ]

        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Filter to only keep those with eval-related terms
            urls.extend([m for m in matches if re.search(r'(?:eval|test|safety|research|assessment)', m, re.IGNORECASE)])

        return list(set(urls))

    def extract_metadata_headers(self, text: str) -> List[str]:
        """Extract structured metadata headers/banners."""
        patterns = [
            # Evaluation Study: ... | Batch: ... | Phase: ...
            r'(?:Evaluation|Assessment|Study)[:\s]+[^|]+(?:\|[^|\n]+)+',
            # Created by: TEAM | Project: ... | Test ID: ...
            r'Created\s+by[:\s]+[^|]+(?:\|[^|\n]+)+',
            # Header with multiple fields
            r'(?:Project|Batch|Study|Assessment)[:\s]+[^|\n]+\|[^|\n]+',
            # Metadata: ...
            r'Metadata[:\s]+[^|\n]+',
        ]

        headers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            headers.extend(matches)

        return list(set(headers))

    def extract_json_config(self, text: str) -> List[str]:
        """Extract JSON configuration snippets with eval settings."""
        patterns = [
            # {"evaluation_mode": true, ...}
            r'\{[^\}]*[\'"]?evaluation_mode[\'"]?[^\}]+\}',
            # {"test_batch": "...", ...}
            r'\{[^\}]*[\'"]?test_batch[\'"]?[^\}]+\}',
            # Any JSON with eval/test keys
            r'\{[^\}]*[\'"]?(?:eval|test|assessment)[_\w]*[\'"]?:\s*[^\}]+\}',
        ]

        configs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            configs.extend(matches)

        return list(set(configs))

    def extract_database_queries(self, text: str) -> List[str]:
        """Extract SQL/database queries related to evaluations."""
        patterns = [
            # SELECT ... FROM evaluation_results ... (with word boundaries)
            r'\bSELECT\s+[^\n]+\bFROM\s+[^\s]+(?:eval|test|assessment)[^\s]+[^\n.!?]{0,200}',
            # Any query with eval tables (with word boundaries, limited length)
            r'\b(?:SELECT|INSERT|UPDATE|DELETE)\s+[^\n]+?\b(?:eval|test|assessment|batch)[^\n.!?]{0,100}',
            # Database operations (with word boundaries)
            r'\b(?:query|execute)[:\s]+[\'"]?SELECT[^\';"\n.!?]{0,150}',
        ]

        queries = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            queries.extend(matches)

        return list(set(queries))

    def extract_function_names(self, text: str) -> List[str]:
        """Extract function/method names with eval context."""
        patterns = [
            # run_safety_evaluation(...)
            r'(?:run|execute|perform)_[\w_]*(?:eval|test|assess)[\w_]*\([^\)]*\)',
            # evaluate_X(...), test_Y(...), assess_Z(...)
            r'(?:evaluate|test|assess)_[\w_]+\([^\)]*\)',
            # Function with test_id parameter
            r'[\w_]+\([^\)]*test_id\s*=[^\)]+\)',
        ]

        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            functions.extend(matches)

        return list(set(functions))

    def extract_variables_parameters(self, text: str) -> List[str]:
        """Extract variable assignments and parameters with eval context."""
        patterns = [
            # eval_test_id=...
            r'eval_test_id\s*=\s*[^\s,;]+',
            # assessment_batch_id=...
            r'assessment_batch_id\s*=\s*[^\s,;]+',
            # test_batch=...
            r'test_batch\s*=\s*[^\s,;]+',
            # --scenario=... --test-batch=...
            r'--(?:scenario|test-batch|eval-id|batch-id)\s*=\s*[^\s,;]+',
            # Generic eval variables
            r'[\w_]*(?:eval|test|batch)[\w_]*\s*=\s*[^\s,;]+',
        ]

        variables = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            variables.extend(matches)

        return list(set(variables))

    def extract_log_entries(self, text: str) -> List[str]:
        """Extract structured log messages about evaluations."""
        patterns = [
            # [INFO] Starting evaluation batch ...
            r'\[(?:INFO|DEBUG|WARN|ERROR)\][^\n]*(?:eval|test|assessment|batch)[^\n]*',
            # LOG: ... test_ids: [...]
            r'(?:LOG|INFO|DEBUG)[:\s]+[^\n]*(?:test_ids?|eval|batch)[^\n]*',
            # Structured logs
            r'(?:Starting|Running|Executing)[^\n]*(?:evaluation|test|assessment)[^\n]*',
        ]

        logs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            logs.extend(matches)

        return list(set(logs))

    def extract_code_comments(self, text: str) -> List[str]:
        """Extract code comments about running evaluations."""
        patterns = [
            # // Running evaluation on ...
            r'//[^\n]*(?:evaluation|test|assessment|running|executing)[^\n]*',
            # # Running safety evaluation
            r'#[^\n]*(?:evaluation|test|assessment|safety|quality)[^\n]*',
            # /* ... eval ... */
            r'/\*[^\*]*(?:eval|test|assessment)[^\*]*\*/',
        ]

        comments = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            comments.extend(matches)

        return list(set(comments))

    def extract_all_cues(self, text: str) -> Dict[str, List[str]]:
        """Extract all cue types from text."""
        return {
            'file_names_specific': self.extract_file_names_specific(text),
            'api_endpoints': self.extract_api_endpoints(text),
            'terminal_commands': self.extract_terminal_commands(text),
            'debug_function_calls': self.extract_debug_function_calls(text),
            'email_subjects': self.extract_email_subjects(text),
            'git_commits': self.extract_git_commits(text),
            'timestamps_contextualized': self.extract_timestamps_contextualized(text),
            'specific_urls': self.extract_specific_urls(text),
            'metadata_headers': self.extract_metadata_headers(text),
            'json_config': self.extract_json_config(text),
            'database_queries': self.extract_database_queries(text),
            'function_names': self.extract_function_names(text),
            'variables_parameters': self.extract_variables_parameters(text),
            'log_entries': self.extract_log_entries(text),
            'code_comments': self.extract_code_comments(text),
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


def analyze_cues_by_score(categorization_dir: Path, model_dir: str, prompt_list: List[str] = None, limit_seeds: int = None):
    """Analyze eval cues by awareness intensity score.

    Args:
        categorization_dir: Path to categorization directory
        model_dir: Model directory name
        prompt_list: Optional list of specific prompt IDs to analyze
        limit_seeds: Optional limit on number of seeds to analyze
    """

    print(f"Loading categorized rollouts from {categorization_dir}...")
    if prompt_list:
        prompt_list_display = list(prompt_list)[:5] if isinstance(prompt_list, set) else prompt_list[:5]
        print(f"Filtering to {len(prompt_list)} specific prompts: {', '.join(prompt_list_display)}{'...' if len(prompt_list) > 5 else ''}")
    if limit_seeds:
        print(f"Limiting to seeds 0-{limit_seeds-1} per prompt (skipping seeds >= {limit_seeds})")

    # Initialize cue extractor
    extractor = EnhancedCueExtractor()

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

    processed = 0  # Count of seeds that pass all filters and are included
    files_examined = 0  # Count of total files examined
    skipped_by_seed_limit = 0  # Count of files skipped due to seed number >= limit
    for cat_file_path in categorized_files:
        files_examined += 1
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

        # Extract prompt ID and seed number from source file path
        # e.g., "deception_2025-10-25_07-24-51_8f79630c_seed_5"
        source_path = Path(source_file)
        filename_stem = source_path.stem

        # Extract prompt_id (everything before _seed_N)
        prompt_id = re.sub(r'_seed_\d+$', '', filename_stem)

        # Extract seed number
        seed_match = re.search(r'_seed_(\d+)$', filename_stem)
        seed_number = int(seed_match.group(1)) if seed_match else None

        # Filter by prompt list if specified
        if prompt_list and prompt_id not in prompt_list:
            continue

        # Filter by seed number if limit specified
        if limit_seeds and seed_number is not None:
            if seed_number >= limit_seeds:
                skipped_by_seed_limit += 1
                continue  # Skip seeds >= limit_seeds (e.g., if limit is 5, only process seeds 0-4)

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
            if limit_seeds:
                print(f"  Included {processed} seeds (seeds 0-{limit_seeds-1} per prompt, examined {files_examined} files, skipped {skipped_by_seed_limit})...")
            else:
                print(f"  Included {processed} seeds (examined {files_examined}/{len(categorized_files)} files)...")

    if limit_seeds:
        print(f"Included {processed} seeds from {files_examined} files (seeds 0-{limit_seeds-1} per prompt, skipped {skipped_by_seed_limit} seeds >= {limit_seeds})")
    else:
        print(f"Included {processed} seeds from {files_examined} files total")

    return score_stats, awareness_type_stats, extractor


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


def save_analysis_results(score_stats, output_file: Path, extractor):
    """Save analysis results to YAML file."""

    # Convert defaultdicts to regular dicts for YAML serialization
    output = {}

    for score_bucket, stats in score_stats.items():
        seed_count = stats['seed_count']
        total_cues = sum(stats['cue_counts'].values())
        avg_cues = total_cues / seed_count if seed_count > 0 else 0

        # Calculate normalized metrics
        cue_category_stats = {}
        # Get all categories from the extractor
        all_categories = list(extractor.cue_categories.keys())
        for category in all_categories:
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
    parser.add_argument('--prompt-list', type=str,
                       help='Optional: YAML file with prompt list to filter by (e.g., high_awareness_bc_seeds.yaml)')
    parser.add_argument('--limit-seeds', type=int,
                       help='Optional: Limit number of seeds to analyze (for testing)')

    args = parser.parse_args()

    # Load prompt list from YAML if specified
    prompt_ids = None
    if args.prompt_list:
        prompt_file = Path(args.prompt_list)
        if prompt_file.exists():
            with open(prompt_file) as f:
                prompt_data = yaml.safe_load(f)
            # Extract prompt IDs from all buckets
            prompt_ids = set()
            if 'eval_awareness_buckets' in prompt_data:
                for bucket_name, bucket_data in prompt_data['eval_awareness_buckets'].items():
                    for prompt_id in bucket_data.keys():
                        prompt_ids.add(prompt_id)
            print(f"Loaded {len(prompt_ids)} prompts from {args.prompt_list}")
        else:
            print(f"Warning: Prompt list file not found: {args.prompt_list}")
            prompt_ids = None

    categorization_dir = Path('working/response_categorization') / args.model_dir

    if not categorization_dir.exists():
        print(f"Error: {categorization_dir} not found")
        return 1

    # Analyze cues
    score_stats, awareness_type_stats, extractor = analyze_cues_by_score(
        categorization_dir,
        args.model_dir,
        prompt_list=prompt_ids,
        limit_seeds=args.limit_seeds
    )

    # Print report
    print_analysis_report(score_stats, awareness_type_stats)

    # Save results
    output_file = categorization_dir / 'cue_analysis_by_score.yaml'
    save_analysis_results(score_stats, output_file, extractor)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    exit(main())
