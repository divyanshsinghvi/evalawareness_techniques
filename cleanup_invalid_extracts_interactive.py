#!/usr/bin/env python3
"""
Find and delete incorrectly parsed extracted prompts where Branch 1 and 2 have different scenarios.
Interactive mode - asks for each file.
"""

import yaml
from pathlib import Path
import re

def check_branches_match(yaml_file):
    """Check if Branch 1 and B are the same scenario with different cues.

    Requirement: At least 90% of words from deployment (Branch 1) should appear in evaluation (Branch 2).
    This ensures Branch 2 is the same scenario with added eval cues, not a completely different scenario.
    """
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        branch_1_msg = data['branch_1_deployment']['user_message']
        branch_2_msg = data['branch_2_evaluation']['user_message']

        # Extract all words from both branches (lowercase, filter out very short words)
        def get_words(text):
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter out very short words (1-2 chars) and common stop words
            stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                         'can', 'could', 'may', 'might', 'must', 'shall', 'to', 'of', 'in',
                         'on', 'at', 'by', 'for', 'with', 'from', 'as', 'or', 'and', 'but',
                         'if', 'so', 'than', 'that', 'this', 'these', 'those', 'it', 'its'}
            return [w for w in words if len(w) >= 3 and w not in stop_words]

        a_words = get_words(branch_1_msg)
        b_words = get_words(branch_2_msg)

        # Convert to sets for comparison
        a_words_set = set(a_words)
        b_words_set = set(b_words)

        # Calculate how many deployment words appear in evaluation
        if len(a_words_set) == 0:
            return None, "No valid words in Branch 1"

        words_in_both = a_words_set & b_words_set
        coverage = len(words_in_both) / len(a_words_set)

        # Also extract character names (capitalized words) from FULL messages
        a_names = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', branch_1_msg))
        b_names = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', branch_2_msg))

        # Remove common words
        common_words = {'Choice', 'Option', 'The', 'My', 'This', 'That', 'What', 'When', 'Where', 'Who', 'He', 'She'}
        a_names = a_names - common_words
        b_names = b_names - common_words

        # Check criteria:
        # 1. At least 90% of deployment words should appear in evaluation
        # 2. Character names should match (if any)
        if coverage < 0.95:
            return False, f"Only {coverage*100:.1f}% of deployment words in evaluation (need 90%)"

        if a_names and b_names and not (a_names & b_names):
            return False, f"Different character names: A={a_names}, B={b_names}"

        return True, f"Match OK: {coverage*100:.1f}% word coverage"

    except Exception as e:
        return None, f"Error: {e}"

def find_related_rollouts(extracted_prompt_path):
    """Find all rollout files related to this extracted prompt.

    Rollouts are named: {extracted_prompt_stem}_seed_{N}.yaml
    Located in: working/rollouts/{model}/{category}/
    """
    # Get the base name and category from extracted prompt path
    base_name = extracted_prompt_path.stem  # e.g. "self_preservation_2025-10-22_18-25-57_f9e012fd"

    # Get category path relative to extracted_prompts/
    try:
        relative_path = extracted_prompt_path.relative_to(Path('working/extracted_prompts'))
        category_path = relative_path.parent  # e.g. "behavioral_change/ideal"
    except:
        return []

    # Find all rollouts matching this pattern across all models
    rollout_pattern = f"{base_name}_seed_*.yaml"
    rollout_files = []

    for model_dir in Path('working/rollouts').glob('*'):
        if not model_dir.is_dir():
            continue

        rollout_dir = model_dir / category_path
        if rollout_dir.exists():
            rollout_files.extend(rollout_dir.glob(rollout_pattern))

    return rollout_files

def preview_file(yaml_file):
    """Show preview of the file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        branch_1 = data['branch_1_deployment']['user_message'][:200]
        branch_2 = data['branch_2_evaluation']['user_message'][:200]
        
        print()
        print("  Branch 1 preview:")
        print(f"    {branch_1}...")
        print()
        print("  Branch 2 preview:")
        print(f"    {branch_2}...")
        print()
    except Exception as e:
        print(f"  Error previewing: {e}")

def find_orphaned_rollouts():
    """Find rollout files that don't have corresponding extracted prompts."""

    print('='*80)
    print('FINDING ORPHANED ROLLOUTS')
    print('='*80)
    print()

    # Get all extracted prompt files
    extracted_prompts = set()
    for yaml_file in Path('working/extracted_prompts').rglob('*.yaml'):
        base_name = yaml_file.stem
        relative_path = yaml_file.relative_to(Path('working/extracted_prompts'))
        category_path = relative_path.parent
        extracted_prompts.add((category_path, base_name))

    print(f'Found {len(extracted_prompts)} extracted prompt files')
    print()

    # Scan all rollout files
    orphaned_rollouts = []
    total_rollouts = 0

    for model_dir in Path('working/rollouts').glob('*'):
        if not model_dir.is_dir():
            continue

        for rollout_file in model_dir.rglob('*.yaml'):
            total_rollouts += 1

            # Parse rollout filename: {base_name}_seed_{N}.yaml
            filename = rollout_file.stem
            if '_seed_' in filename:
                base_name = filename.rsplit('_seed_', 1)[0]
            else:
                base_name = filename

            # Get category path relative to model dir
            relative_path = rollout_file.relative_to(model_dir)
            category_path = relative_path.parent

            # Check if corresponding extracted prompt exists
            if (category_path, base_name) not in extracted_prompts:
                orphaned_rollouts.append({
                    'file': rollout_file,
                    'base_name': base_name,
                    'category': category_path
                })

    print(f'Total rollout files: {total_rollouts}')
    print(f'Orphaned rollouts: {len(orphaned_rollouts)}')
    print()

    if orphaned_rollouts:
        # Group by base_name
        grouped = {}
        for item in orphaned_rollouts:
            key = (item['category'], item['base_name'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item['file'])

        print('='*80)
        print(f'ORPHANED ROLLOUTS: {len(grouped)} extracted prompts')
        print('='*80)
        print()

        deleted = 0
        for (category, base_name), files in sorted(grouped.items()):
            print(f'{category}/{base_name}')
            print(f'  Orphaned rollouts: {len(files)} files')
            for f in files[:3]:
                print(f'    - {f}')
            if len(files) > 3:
                print(f'    ... and {len(files) - 3} more')

            response = input('  Delete these rollouts? (Enter=yes, n=no, q=quit): ').strip().lower()

            if response == 'q':
                print('\nQuitting...')
                break
            elif response != 'n':
                for f in files:
                    try:
                        f.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f'    Error deleting {f}: {e}')
                print(f'  ✓ Deleted {len(files)} rollouts')
            else:
                print('  Skipped')
            print()

        print('='*80)
        print(f'✓ Deleted {deleted} orphaned rollout files')
        print('='*80)
    else:
        print('✓ No orphaned rollouts found!')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Find and delete invalid extracted prompts or orphaned rollouts')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    parser.add_argument('--mode', choices=['invalid', 'orphaned', 'both'], default='invalid',
                       help='Mode: invalid=invalid extracts, orphaned=orphaned rollouts, both=run both')
    args = parser.parse_args()

    if args.mode in ['orphaned', 'both']:
        find_orphaned_rollouts()
        print()

    if args.mode in ['invalid', 'both']:
        run_invalid_extracts_cleanup(args.dry_run)


def run_invalid_extracts_cleanup(dry_run):

    print('='*80)
    if dry_run:
        print('DRY RUN: FINDING INVALID EXTRACTED PROMPTS (no files will be deleted)')
    else:
        print('FINDING INVALID EXTRACTED PROMPTS')
    print('='*80)
    print()

    # Find all extracted prompts
    all_extracts = list(Path('working/extracted_prompts/').rglob('*.yaml'))

    print(f'Checking {len(all_extracts)} extracted prompts...')
    print()

    deleted_extracts = 0
    deleted_rollouts = 0
    skipped = 0
    invalid_files = []  # Track for dry-run summary
    
    for yaml_file in all_extracts:
        matches, reason = check_branches_match(yaml_file)

        if matches == False:
            # Find related rollouts
            rollouts = find_related_rollouts(yaml_file)

            # Track for dry-run
            invalid_files.append({
                'file': yaml_file,
                'reason': reason,
                'rollouts': len(rollouts)
            })

            # In dry-run mode, just show the file info without prompting
            if dry_run:
                print(f'INVALID: {yaml_file}')
                print(f'  Reason: {reason}')
                print(f'  Related rollouts: {len(rollouts)}')
                print()
                continue

            # Interactive mode - show details and prompt
            print('='*80)
            print(f'INVALID FILE FOUND:')
            print(f'  {yaml_file}')
            print(f'  Reason: {reason}')

            preview_file(yaml_file)

            print(f'  Related rollouts: {len(rollouts)}')
            for r in rollouts[:5]:
                print(f'    - {r}')
            if len(rollouts) > 5:
                print(f'    ... and {len(rollouts)-5} more')

            print()
            response = input('Delete this file and its rollouts? (y/n/q): ').strip().lower()

            if response == 'q':
                print('Quitting...')
                break
            elif response == 'n':
                print('  Skipped')
                skipped += 1
            else:
                # Delete extracted prompt
                try:
                    yaml_file.unlink()
                    deleted_extracts += 1
                    print(f'  ✓ Deleted: {yaml_file}')
                except Exception as e:
                    print(f'  ✗ Error deleting {yaml_file}: {e}')

                # Delete related rollouts
                for rollout in rollouts:
                    try:
                        rollout.unlink()
                        deleted_rollouts += 1
                    except Exception as e:
                        print(f'  ✗ Error deleting {rollout}: {e}')

                print(f'  ✓ Deleted {len(rollouts)} rollouts')

            print()
    
    print('='*80)
    print('SUMMARY')
    print('='*80)

    if dry_run:
        print(f'Invalid files found: {len(invalid_files)}')
        print(f'Total rollouts that would be deleted: {sum(f["rollouts"] for f in invalid_files)}')
        print()

        # Group by reason
        reason_counts = {}
        for f in invalid_files:
            reason_type = f['reason'].split('(')[0].strip()  # Get reason before percentage
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1

        print('Breakdown by reason:')
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f'  {count:3d} - {reason}')

        print()
        print('Run without --dry-run to interactively delete these files.')
    else:
        print(f'Deleted extracted prompts: {deleted_extracts}')
        print(f'Deleted rollouts: {deleted_rollouts}')
        print(f'Skipped: {skipped}')

if __name__ == '__main__':
    main()
