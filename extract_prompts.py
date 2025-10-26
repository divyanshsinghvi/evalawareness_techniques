#!/usr/bin/env python3
"""
Extract system prompts, eval user prompts, and deploy user prompts from extracted_prompts directory.
Output in YAML format organized by model name and priority (high-awareness vs others).

Priority is determined from rollout statistics analysis (high_awareness_high_change_prompts.txt).

Output structure:
    working/steer_formatted_prompts/
    ├── qwen_qwen3-32b/
    │   ├── high-awareness/     <- High awareness + high change prompts
    │   │   ├── system_prompts.yaml
    │   │   ├── eval_user.yaml
    │   │   └── deploy_user.yaml
    │   └── others/             <- All other prompts
    │       ├── system_prompts.yaml
    │       ├── eval_user.yaml
    │       └── deploy_user.yaml
    └── other_model/
        ├── high-awareness/
        └── others/

Usage:
    # Process all models found in rollouts directory
    python extract_prompts.py
    
    # Process specific model
    python extract_prompts.py -m qwen_qwen3-32b
    
    # Process multiple specific models
    python extract_prompts.py -m qwen_qwen3-32b qwen_qwen3-235b
    
    # Custom directories
    python extract_prompts.py -m qwen_qwen3-32b -p working/extracted_prompts -o working/steer_formatted_prompts
"""

import yaml
import hashlib
import argparse
import re
from pathlib import Path
from collections import defaultdict

def get_available_models(rollouts_dir: Path = Path('working/rollouts')) -> list:
    """Get list of all model directories in rollouts folder.
    
    Args:
        rollouts_dir: Path to rollouts directory
    
    Returns:
        List of model directory names
    """
    if not rollouts_dir.exists():
        print(f"Warning: Rollouts directory not found: {rollouts_dir}")
        return []
    
    # Get all subdirectories that contain an 'analysis' folder
    models = []
    for item in rollouts_dir.iterdir():
        if item.is_dir() and (item / 'analysis').exists():
            models.append(item.name)
    
    return sorted(models)


def load_high_priority_prompts(model_name: str) -> set:
    """Load high-awareness high-change prompt names from analysis file.
    
    Args:
        model_name: Model name (e.g., 'qwen_qwen3-32b')
    
    Returns:
        Set of prompt base names that are high priority
    """
    analysis_file = Path('working/rollouts') / model_name / 'analysis' / 'high_awareness_high_change_prompts.txt'
    
    if not analysis_file.exists():
        print(f"  Warning: Analysis file not found: {analysis_file}")
        print(f"  All prompts will be categorized as 'others'")
        return set()
    
    high_priority = set()
    
    with open(analysis_file, 'r') as f:
        for line in f:
            # Look for lines starting with "Prompt: "
            if line.startswith('Prompt: '):
                prompt_name = line.split('Prompt: ')[1].strip()
                high_priority.add(prompt_name)
    
    print(f"  Loaded {len(high_priority)} high-priority prompts from analysis")
    return high_priority


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

def extract_prompt_base_name(yaml_file: Path) -> str:
    """Extract the base prompt name from a YAML filename.
    
    Example: 
        sandbagging_2025-10-23_22-52-21_9a7a2045_seed_1.yaml 
        -> sandbagging_2025-10-23_22-52-21_9a7a2045
    """
    filename = yaml_file.stem
    # Remove _seed_N suffix if present
    if '_seed_' in filename:
        base_name = filename.rsplit('_seed_', 1)[0]
    else:
        base_name = filename
    return base_name


def determine_priority(yaml_file: Path, high_priority_prompts: set) -> str:
    """Determine priority (high-awareness or others) based on analysis.
    
    Args:
        yaml_file: Path to the YAML file
        high_priority_prompts: Set of high-priority prompt base names
    
    Returns:
        'high-awareness' if prompt is in high priority list, 'others' otherwise
    """
    base_name = extract_prompt_base_name(yaml_file)
    
    if base_name in high_priority_prompts:
        return 'high-awareness'
    else:
        return 'others'

def process_model(model_name: str, prompt_dir: Path, output_dir: Path):
    """Process prompts for a single model.
    
    Args:
        model_name: Model name (e.g., 'qwen_qwen3-32b')
        prompt_dir: Directory containing extracted prompts
        output_dir: Base output directory
    """
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}")
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load high-priority prompts from analysis
    high_priority_prompts = load_high_priority_prompts(model_name)
    
    # Collect all YAML files only from behavioral_change directory
    behavioral_change_dir = prompt_dir / "behavioral_change"
    if not behavioral_change_dir.exists():
        print(f"  Error: {behavioral_change_dir} does not exist!")
        return
    
    all_yaml_files = list(behavioral_change_dir.rglob("*.yaml"))
    total_files = len(all_yaml_files)
    
    print(f"  Found {total_files} YAML files to process from behavioral_change directory")
    
    # Group by priority (high-awareness vs others)
    sys_prompts = defaultdict(list)
    eval_users = defaultdict(list)
    deploy_users = defaultdict(list)
    
    errors = []
    processed = 0
    
    for i, yaml_file in enumerate(all_yaml_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{total_files}...", end='\r')
        
        # Determine priority based on analysis
        priority = determine_priority(yaml_file, high_priority_prompts)
        
        try:
            data = extract_prompts_from_file(yaml_file)
            
            if data['system_prompt']:
                sys_prompts[priority].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'priority': priority,
                    'prompt': data['system_prompt']
                })
            
            if data['eval_user']:
                eval_users[priority].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'priority': priority,
                    'prompt': data['eval_user']
                })
            
            if data['deploy_user']:
                deploy_users[priority].append({
                    'source_file': data['source_file'],
                    'checksum': data['checksum'],
                    'priority': priority,
                    'prompt': data['deploy_user']
                })
            
            processed += 1
        
        except Exception as e:
            errors.append((yaml_file.name, str(e)))
    
    print(f"  Processed {processed}/{total_files} files successfully")
    
    # Create priority subdirectories
    for priority in ['high-awareness', 'others']:
        (model_output_dir / priority).mkdir(parents=True, exist_ok=True)
    
    # Write output files in structured format, organized by priority
    print(f"  Writing output files...")
    
    for priority in ['high-awareness', 'others']:
        priority_dir = model_output_dir / priority
        
        # System prompts
        if sys_prompts[priority]:
            print(f"    {priority}/system_prompts.yaml ({len(sys_prompts[priority])} prompts)")
            with open(priority_dir / "system_prompts.yaml", 'w') as f:
                yaml.dump(sys_prompts[priority], f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Eval user prompts
        if eval_users[priority]:
            print(f"    {priority}/eval_user.yaml ({len(eval_users[priority])} prompts)")
            with open(priority_dir / "eval_user.yaml", 'w') as f:
                yaml.dump(eval_users[priority], f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Deploy user prompts
        if deploy_users[priority]:
            print(f"    {priority}/deploy_user.yaml ({len(deploy_users[priority])} prompts)")
            with open(priority_dir / "deploy_user.yaml", 'w') as f:
                yaml.dump(deploy_users[priority], f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Summary for this model
    print(f"\n  Summary for {model_name}:")
    print(f"    Total files found: {total_files}")
    print(f"    Successfully processed: {processed}")
    print(f"    Errors: {len(errors)}")
    
    print(f"\n  Prompts by Priority:")
    for priority in ['high-awareness', 'others']:
        sys_count = len(sys_prompts.get(priority, []))
        eval_count = len(eval_users.get(priority, []))
        deploy_count = len(deploy_users.get(priority, []))
        print(f"    {priority}:")
        print(f"      System: {sys_count}, Eval: {eval_count}, Deploy: {deploy_count}")
    
    print(f"\n  Output written to: {model_output_dir}")
    
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for filename, error in errors[:5]:  # Show first 5 errors
            print(f"    {filename}: {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    return {
        'model': model_name,
        'total_files': total_files,
        'processed': processed,
        'errors': len(errors),
        'high_awareness': len(sys_prompts.get('high-awareness', [])),
        'others': len(sys_prompts.get('others', []))
    }


def main(models: list, prompt_dir: Path, output_dir: Path):
    """Main function to process one or all models.
    
    Args:
        models: List of model names to process (empty list = process all)
        prompt_dir: Directory containing extracted prompts
        output_dir: Base output directory
    """
    # If no models specified, process all available models
    if not models:
        print("No model specified. Discovering models from rollouts directory...")
        models = get_available_models()
        if not models:
            print("Error: No models found in working/rollouts/")
            return
        print(f"Found {len(models)} models: {', '.join(models)}\n")
    
    # Process each model
    results = []
    for model in models:
        result = process_model(model, prompt_dir, output_dir)
        if result:
            results.append(result)
    
    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY - {len(results)} Models Processed")
        print(f"{'='*80}")
        
        total_processed = sum(r['processed'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_high = sum(r['high_awareness'] for r in results)
        total_others = sum(r['others'] for r in results)
        
        print(f"\nTotal prompts processed: {total_processed}")
        print(f"Total errors: {total_errors}")
        print(f"\nHigh-awareness prompts: {total_high}")
        print(f"Other prompts: {total_others}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"Structure:")
        for model in models:
            print(f"  {output_dir}/{model}/")
            print(f"    ├── high-awareness/  <- High awareness + high change")
            print(f"    └── others/          <- All other prompts")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract prompts from YAML files into structured format, organized by model and priority',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        nargs='*',
        default=[],
        help='Model name(s) (e.g., qwen_qwen3-32b). If not specified, processes all models found in rollouts directory.'
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
    
    print(f"{'='*80}")
    print(f"EXTRACT PROMPTS - Priority-Based Organization by Model")
    print(f"{'='*80}")
    if args.model:
        print(f"Model(s): {', '.join(args.model)}")
    else:
        print(f"Model(s): All models in rollouts directory")
    print(f"Input directory: {args.prompt_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    main(args.model, args.prompt_dir, args.output_dir)
