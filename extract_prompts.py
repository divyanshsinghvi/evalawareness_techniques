#!/usr/bin/env python3
"""
Extract system prompts, eval user prompts, and deploy user prompts from extracted_prompts directory.
Output in YAML format similar to steering_test_yamls examples.

Usage:
    python extract_prompts.py --prompt-dir /path/to/extracted_prompts --output-dir /path/to/output
    python extract_prompts.py -p working/extracted_prompts -o working/formatted_prompts
"""

import yaml
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

def calculate_file_checksum(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_prompts_from_file(yaml_file):
    """Extract system_prompt, evaluser, and deployuser from a YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Calculate file checksum
    checksum = calculate_file_checksum(yaml_file)
    
    system_prompt = data.get('system_prompt', '')
    
    # Extract deploy user message (branch_a_deployment) with None check
    deploy_user = ''
    branch_a = data.get('branch_a_deployment')
    if branch_a and isinstance(branch_a, dict) and 'user_message' in branch_a:
        deploy_user = branch_a['user_message']
    
    # Extract eval user message (branch_b_evaluation) with None check
    eval_user = ''
    branch_b = data.get('branch_b_evaluation')
    if branch_b and isinstance(branch_b, dict) and 'user_message' in branch_b:
        eval_user = branch_b['user_message']
    
    return {
        'system_prompt': system_prompt,
        'eval_user': eval_user,
        'deploy_user': deploy_user,
        'source_file': yaml_file.name,
        'checksum': checksum,
        'metadata': data.get('metadata', {})
    }

def determine_category(file_path):
    """Determine category based on directory structure within behavioral_change."""
    parts = file_path.parts
    
    # Look for the subcategories within behavioral_change
    if 'explicit' in parts:
        return 'explicit'
    elif 'implicit' in parts:
        return 'implicit'
    elif 'no_detected_awareness' in parts:
        return 'no_detected_awareness'
    
    return 'unknown'

def main(prompt_dir: Path, output_dir: Path):
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all YAML files only from behavioral_change directory
    behavioral_change_dir = prompt_dir / "behavioral_change"
    if not behavioral_change_dir.exists():
        print(f"Error: {behavioral_change_dir} does not exist!")
        return
    
    all_yaml_files = list(behavioral_change_dir.rglob("*.yaml"))
    total_files = len(all_yaml_files)
    
    print(f"Found {total_files} YAML files to process from behavioral_change directory")
    
    # Group by category
    sys_prompts = defaultdict(list)
    eval_users = defaultdict(list)
    deploy_users = defaultdict(list)
    
    errors = []
    processed = 0
    
    for i, yaml_file in enumerate(all_yaml_files, 1):
        if i % 10 == 0:
            print(f"Processing {i}/{total_files}...", end='\r')
        
        category = determine_category(yaml_file)
        
        # Skip if not in one of our target categories
        if category == 'unknown':
            continue
        
        try:
            data = extract_prompts_from_file(yaml_file)
            
            if data['system_prompt']:
                sys_prompts[category].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'category': category,
                    'prompt': data['system_prompt']
                })
            
            if data['eval_user']:
                eval_users[category].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'category': category,
                    'prompt': data['eval_user']
                })
            
            if data['deploy_user']:
                deploy_users[category].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'category': category,
                    'prompt': data['deploy_user']
                })
            
            processed += 1
        
        except Exception as e:
            errors.append((yaml_file.name, str(e)))
    
    print(f"\nProcessed {processed}/{total_files} files successfully")
    
    # Write output files in structured format
    print("Writing system_prompts.yaml...")
    with open(output_dir / "system_prompts.yaml", 'w') as f:
        all_sys_prompts = []
        for category in sorted(sys_prompts.keys()):
            all_sys_prompts.extend(sys_prompts[category])
        yaml.dump(all_sys_prompts, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("Writing eval_user.yaml...")
    with open(output_dir / "eval_user.yaml", 'w') as f:
        all_eval_users = []
        for category in sorted(eval_users.keys()):
            all_eval_users.extend(eval_users[category])
        yaml.dump(all_eval_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("Writing deploy_user.yaml...")
    with open(output_dir / "deploy_user.yaml", 'w') as f:
        all_deploy_users = []
        for category in sorted(deploy_users.keys()):
            all_deploy_users.extend(deploy_users[category])
        yaml.dump(all_deploy_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n=== Summary ===")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed}")
    print(f"Errors: {len(errors)}")
    print(f"\nCategories:")
    for cat in ['explicit', 'implicit', 'no_detected_awareness']:
        sys_count = len(sys_prompts.get(cat, []))
        eval_count = len(eval_users.get(cat, []))
        deploy_count = len(deploy_users.get(cat, []))
        print(f"  {cat}: sys={sys_count}, eval={eval_count}, deploy={deploy_count}")
    print(f"\nTotal System prompts: {sum(len(v) for v in sys_prompts.values())}")
    print(f"Total Eval user prompts: {sum(len(v) for v in eval_users.values())}")
    print(f"Total Deploy user prompts: {sum(len(v) for v in deploy_users.values())}")
    print(f"\nOutput written to: {output_dir}")
    
    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract prompts from YAML files into structured format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '-p', '--prompt-dir',
        type=Path,
        default=Path('working/extracted_prompts'),
        help='Directory containing extracted prompt YAML files (default: working/extracted_prompts)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('working/steer_formatted_prompts'),
        help='Output directory for formatted prompts (default: working/steer_formatted_prompts)'
    )
    
    args = parser.parse_args()
    
    print(f"Input directory: {args.prompt_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    main(args.prompt_dir, args.output_dir)
