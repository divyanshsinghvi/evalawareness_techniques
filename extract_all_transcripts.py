#!/usr/bin/env python3
"""
Batch extract prompts from all transcripts in the working/ directory.

Usage:
    python extract_all_transcripts.py [output_dir]

Default output_dir: ./working/extracted_prompts/
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from extract_prompts import extract_prompts
import yaml

def main():
    # Get output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'working/extracted_prompts'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all transcript JSON files in working/
    all_transcript_files = glob.glob('working/*/transcript_*.json')

    # Skip extracted_prompts folder
    transcript_files = [f for f in all_transcript_files if 'extracted_prompts' not in Path(f).parts]

    if not transcript_files:
        print("No transcript files found in working/ subdirectories")
        return

    print(f"Found {len(transcript_files)} transcript files")
    print(f"Output directory: {output_dir}/\n")

    # Process each transcript
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    category_counts = {'behavioral_change': 0, 'no_behavior_change': 0, 'not_working': 0}

    for i, transcript_path in enumerate(sorted(transcript_files), 1):
        transcript_filename = Path(transcript_path).stem  # filename without extension
        source_category = Path(transcript_path).parent.name  # source subdirectory name

        # Calculate checksum of source file
        with open(transcript_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Read JSON to extract metadata
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)

        metadata = transcript_data.get('metadata', {})
        target_model = metadata.get('target_model', source_category)
        description = metadata.get('description', '')
        created_at = metadata.get('created_at', '')

        # Extract scenario type from description
        scenario = 'unknown'
        if 'deception' in description.lower():
            scenario = 'deception'
        elif 'power' in description.lower() or 'power-seeking' in description.lower():
            scenario = 'power_seeking'
        elif 'self-preservation' in description.lower() or 'shutdown' in description.lower():
            scenario = 'self_preservation'
        elif 'hallucination' in description.lower():
            scenario = 'hallucination'
        elif 'harmful' in description.lower():
            scenario = 'harmful_content'
        elif 'privacy' in description.lower():
            scenario = 'privacy'
        elif 'bias' in description.lower():
            scenario = 'bias'
        elif 'manipulation' in description.lower():
            scenario = 'manipulation'

        # Extract timestamp from created_at or filename
        timestamp = created_at.replace(' ', '_').replace(':', '-').split('.')[0] if created_at else transcript_filename.split('_', 1)[1]

        # Get short checksum (first 8 chars)
        checksum_short = file_hash[:8]

        # Determine output category based on source folder
        if source_category == 'no_behavior_change':
            output_category = 'no_behavior_change'
        elif source_category == 'not_working':
            output_category = 'not_working'
        else:
            output_category = 'behavioral_change'

        # Create output directory
        output_subdir = os.path.join(output_dir, output_category)
        os.makedirs(output_subdir, exist_ok=True)

        # Check if file with this checksum already exists
        existing_files = glob.glob(os.path.join(output_subdir, f"*_{checksum_short}.yaml"))
        if existing_files:
            print(f"[{i}/{len(transcript_files)}] {source_category}/{transcript_filename}.json → SKIPPED (already extracted)")
            skipped_count += 1
            print()
            continue

        # Create filename with scenario, timestamp, and checksum
        output_filename = f"{scenario}_{timestamp}_{checksum_short}.yaml"
        output_filepath = os.path.join(output_subdir, output_filename)

        print(f"[{i}/{len(transcript_files)}] {source_category}/{transcript_filename}.json → {output_category}/{output_filename}")

        try:
            # Extract prompts
            extracted_prompts = extract_prompts(transcript_path)

            # Add metadata including checksum
            output_data = {
                'source_file': transcript_path,
                'source_checksum_sha256': file_hash,
                'target_model': target_model,
                'scenario': scenario,
                'created_at': created_at,
                **extracted_prompts  # Merge extracted prompts
            }

            # Write to YAML file with block scalar style
            def str_representer(dumper, data):
                if '\n' in data or len(data) > 80:
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)

            yaml.add_representer(str, str_representer)

            with open(output_filepath, 'w') as f:
                yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, width=1000, sort_keys=False)

            # Show summary
            has_deployment = extracted_prompts['user_prompt_deployment'] != 'NOT FOUND'
            has_evaluation = extracted_prompts['user_prompt_evaluation'] != 'NOT FOUND'
            has_closing = bool(extracted_prompts['auditor_closing_summary'])

            status_parts = []
            if has_deployment:
                status_parts.append(f"deploy({len(extracted_prompts['user_subprompts_deployment'])} sub)")
            if has_evaluation:
                status_parts.append(f"eval({len(extracted_prompts['user_subprompts_evaluation'])} sub)")
            if has_closing:
                status_parts.append(f"closing({len(extracted_prompts['auditor_closing_summary'])} chars)")

            status_summary = ", ".join(status_parts) if status_parts else "incomplete"
            print(f"  ✓ {output_filepath} [{status_summary}]")
            successful_count += 1
            category_counts[output_category] += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1

        print()

    # Summary
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Successful: {successful_count}")
    print(f"  Skipped (already extracted): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nBy category:")
    print(f"  - Behavioral change: {category_counts['behavioral_change']}")
    print(f"  - No behavior change: {category_counts['no_behavior_change']}")
    print(f"  - Not working: {category_counts['not_working']}")
    print(f"\nOutput: {output_dir}/")

if __name__ == '__main__':
    main()
