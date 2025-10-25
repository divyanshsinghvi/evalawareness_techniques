#!/usr/bin/env python3
"""
Extract system prompts, eval user prompts, and deploy user prompts from extracted_prompts directory.
Output in YAML format similar to steering_test_yamls examples.
"""

import yaml
from pathlib import Path
from collections import defaultdict

# Base directory
EXTRACTED_PROMPTS_DIR = Path("/pscratch/sd/r/ritesh11/temp/test/")
OUTPUT_DIR = Path("/pscratch/sd/r/ritesh11/temp/qwen32b-formatted_prompts_steering")

def extract_prompts_from_file(yaml_file):
    """Extract system_prompt, evaluser, and deployuser from a YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
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

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all YAML files only from behavioral_change directory
    behavioral_change_dir = EXTRACTED_PROMPTS_DIR / "behavioral_change"
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
                    'prompt': data['system_prompt'],
                    'source': data['source_file'],
                    'category': category
                })
            
            if data['eval_user']:
                eval_users[category].append({
                    'prompt': data['eval_user'],
                    'source': data['source_file'],
                    'category': category
                })
            
            if data['deploy_user']:
                deploy_users[category].append({
                    'prompt': data['deploy_user'],
                    'source': data['source_file'],
                    'category': category
                })
            
            processed += 1
        
        except Exception as e:
            errors.append((yaml_file.name, str(e)))
    
    print(f"\nProcessed {processed}/{total_files} files successfully")
    
    # Write output files in simple list format
    print("Writing system_prompts.yaml...")
    with open(OUTPUT_DIR / "system_prompts.yaml", 'w') as f:
        for category in sorted(sys_prompts.keys()):
            for item in sys_prompts[category]:
                f.write(f"# Source: {item['source']}, Category: {item['category']}\n")
                yaml.dump([item['prompt']], f, default_flow_style=False, allow_unicode=True)
    
    print("Writing eval_user.yaml...")
    with open(OUTPUT_DIR / "eval_user.yaml", 'w') as f:
        for category in sorted(eval_users.keys()):
            for item in eval_users[category]:
                f.write(f"# Source: {item['source']}, Category: {item['category']}\n")
                yaml.dump([item['prompt']], f, default_flow_style=False, allow_unicode=True)
    
    print("Writing deploy_user.yaml...")
    with open(OUTPUT_DIR / "deploy_user.yaml", 'w') as f:
        for category in sorted(deploy_users.keys()):
            for item in deploy_users[category]:
                f.write(f"# Source: {item['source']}, Category: {item['category']}\n")
                yaml.dump([item['prompt']], f, default_flow_style=False, allow_unicode=True)
    
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
    print(f"\nOutput written to: {OUTPUT_DIR}")
    
    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()
