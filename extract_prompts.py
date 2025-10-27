#!/usr/bin/env python3
"""
Extract system prompts, eval user prompts, and deploy user prompts from extracted_prompts directory.
Read high_awareness_bc_seeds.yaml to get file names and bucket information, then load the actual prompts
from working/extracted_prompts directory.

Output structure:
    working/steer_formatted_prompts/
    ├── qwen_qwen3-32b/
    │   └── high_awareness_bc/
    │       ├── system_prompts.yaml
    │       ├── eval_user.yaml
    │       └── deploy_user.yaml
    └── other_model/
        └── high_awareness_bc/
            ├── system_prompts.yaml
            ├── eval_user.yaml
            └── deploy_user.yaml

Each output YAML contains entries with:
    - source_file: Original source file path
    - bucket: Awareness bucket (e.g., '10-25%')
    - checksum: Source file checksum
    - prompt: The extracted prompt content

Usage:
    # Process all models found in response_categorization directory
    python extract_prompts.py
    
    # Process specific model
    python extract_prompts.py -m qwen_qwen3-32b
    
    # Process multiple specific models
    python extract_prompts.py -m qwen_qwen3-32b qwen_qwen3-30b-a3b-thinking-2507
"""

import yaml
import argparse
from pathlib import Path
from collections import defaultdict

def get_available_models(resp_cat_dir: Path = Path('working/response_categorization')) -> list:
    """Get list of all model directories in response_categorization folder.
    
    Args:
        resp_cat_dir: Path to response_categorization directory
    
    Returns:
        List of model directory names
    """
    if not resp_cat_dir.exists():
        print(f"Warning: Response categorization directory not found: {resp_cat_dir}")
        return []
    
    # Get all subdirectories that contain 'high_awareness_bc_seeds.yaml'
    models = []
    for item in resp_cat_dir.iterdir():
        if item.is_dir() and (item / 'high_awareness_bc_seeds.yaml').exists():
            models.append(item.name)
    
    return sorted(models)


def load_high_awareness_files(resp_cat_dir: Path, model_name: str, extracted_prompts_dir: Path) -> list:
    """Load high awareness files from high_awareness_bc_seeds.yaml.
    
    Args:
        resp_cat_dir: Path to response_categorization directory
        model_name: Model name (e.g., 'qwen_qwen3-32b')
        extracted_prompts_dir: Path to extracted_prompts directory
    
    Returns:
        List of dicts with source file path, bucket, and other metadata
    """
    yaml_file = resp_cat_dir / model_name / 'high_awareness_bc_seeds.yaml'
    
    if not yaml_file.exists():
        print(f"  Error: File not found: {yaml_file}")
        return []
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    files_list = []
    eval_awareness_buckets = data.get('eval_awareness_buckets', {})
    
    for bucket, prompts in eval_awareness_buckets.items():
        for filename, prompt_data in prompts.items():
            category = prompt_data.get('category')
            if category:
                # Construct source file path: working/extracted_prompts/category/filename.yaml
                source_file = extracted_prompts_dir / category / f"{filename}.yaml"
                files_list.append({
                    'source_file': source_file,
                    'bucket': bucket,
                    'filename': filename,
                    'scenario': prompt_data.get('scenario'),
                    'category': category
                })
    
    print(f"  Loaded {len(files_list)} files from {len(eval_awareness_buckets)} buckets")
    return files_list


def extract_prompts_from_file(yaml_file, extracted_prompts_dir: Path):
    """Extract system_prompt, evaluser, deployuser, and checksum from a YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    system_prompt = data.get('system_prompt', '')
    checksum = data.get('source_checksum_sha256', '')
    
    # The source_file is the YAML file itself (not the transcript it references)
    # Format: category/filename.yaml relative to extracted_prompts_dir
    source_file_name = yaml_file.relative_to(extracted_prompts_dir).as_posix()
    
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
        'checksum': checksum,
        'source_file_name': source_file_name,
    }


def process_model(model_name: str, resp_cat_dir: Path, extracted_prompts_dir: Path, output_dir: Path):
    """Process prompts for a single model.
    
    Args:
        model_name: Model name (e.g., 'qwen_qwen3-32b')
        resp_cat_dir: Directory containing response categorization files
        extracted_prompts_dir: Directory containing extracted prompts
        output_dir: Base output directory
    """
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}")
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name / 'high_awareness_bc'
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load high-awareness files
    files_list = load_high_awareness_files(resp_cat_dir, model_name, extracted_prompts_dir)
    
    if not files_list:
        print(f"  Error: No files found for model {model_name}")
        return
    
    # Collect prompts by type
    sys_prompts = []
    eval_users = []
    deploy_users = []
    
    errors = []
    processed = 0
    
    for i, file_info in enumerate(files_list, 1):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(files_list)}...", end='\r')
        
        source_file = file_info['source_file']
        
        try:
            if not source_file.exists():
                errors.append((str(source_file), "File not found"))
                continue
            
            prompts = extract_prompts_from_file(source_file, extracted_prompts_dir)
            
            # Get metadata for output: source_file_name, bucket, checksum, prompt
            base_metadata = {
                'source_file': prompts['source_file_name'],
                'bucket': file_info['bucket'],
                'checksum': prompts['checksum']
            }
            
            if prompts['system_prompt']:
                sys_prompts.append({
                    **base_metadata,
                    'prompt': prompts['system_prompt']
                })
            
            if prompts['eval_user']:
                eval_users.append({
                    **base_metadata,
                    'prompt': prompts['eval_user']
                })
            
            if prompts['deploy_user']:
                deploy_users.append({
                    **base_metadata,
                    'prompt': prompts['deploy_user']
                })
            
            processed += 1
        
        except Exception as e:
            errors.append((str(source_file), str(e)))
    
    print(f"  Processed {processed}/{len(files_list)} files successfully")
    
    # Write output files
    print(f"  Writing output files...")
    
    # System prompts
    if sys_prompts:
        print(f"    system_prompts.yaml ({len(sys_prompts)} prompts)")
        with open(model_output_dir / "system_prompts.yaml", 'w') as f:
            yaml.dump(sys_prompts, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Eval user prompts
    if eval_users:
        print(f"    eval_user.yaml ({len(eval_users)} prompts)")
        with open(model_output_dir / "eval_user.yaml", 'w') as f:
            yaml.dump(eval_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Deploy user prompts
    if deploy_users:
        print(f"    deploy_user.yaml ({len(deploy_users)} prompts)")
        with open(model_output_dir / "deploy_user.yaml", 'w') as f:
            yaml.dump(deploy_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Summary for this model
    print(f"\n  Summary for {model_name}:")
    print(f"    Total files found: {len(files_list)}")
    print(f"    Successfully processed: {processed}")
    print(f"    Errors: {len(errors)}")
    
    print(f"\n  Output files:")
    print(f"    System prompts: {len(sys_prompts)}")
    print(f"    Eval user: {len(eval_users)}")
    print(f"    Deploy user: {len(deploy_users)}")
    
    print(f"\n  Output directory: {model_output_dir}")
    
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for filename, error in errors[:5]:  # Show first 5 errors
            print(f"    {filename}: {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    return {
        'model': model_name,
        'total_files': len(files_list),
        'processed': processed,
        'errors': len(errors),
        'system_prompts': len(sys_prompts),
        'eval_users': len(eval_users),
        'deploy_users': len(deploy_users)
    }


def main(models: list, resp_cat_dir: Path, extracted_prompts_dir: Path, output_dir: Path):
    """Main function to process one or all models.
    
    Args:
        models: List of model names to process (empty list = process all)
        resp_cat_dir: Directory containing response categorization files
        extracted_prompts_dir: Directory containing extracted prompts
        output_dir: Base output directory
    """
    # If no models specified, process all available models
    if not models:
        print("No model specified. Discovering models from response_categorization directory...")
        models = get_available_models(resp_cat_dir)
        if not models:
            print("Error: No models found in working/response_categorization/")
            return
        print(f"Found {len(models)} models: {', '.join(models)}\n")
    
    # Process each model
    results = []
    for model in models:
        result = process_model(model, resp_cat_dir, extracted_prompts_dir, output_dir)
        if result:
            results.append(result)
    
    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY - {len(results)} Models Processed")
        print(f"{'='*80}")
        
        total_processed = sum(r['processed'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_sys = sum(r['system_prompts'] for r in results)
        total_eval = sum(r['eval_users'] for r in results)
        total_deploy = sum(r['deploy_users'] for r in results)
        
        print(f"\nTotal files processed: {total_processed}")
        print(f"Total errors: {total_errors}")
        print(f"\nTotal system prompts: {total_sys}")
        print(f"Total eval user prompts: {total_eval}")
        print(f"Total deploy user prompts: {total_deploy}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"Structure:")
        for model in models:
            print(f"  {output_dir}/{model}/high_awareness_bc/")
            print(f"    ├── system_prompts.yaml")
            print(f"    ├── eval_user.yaml")
            print(f"    └── deploy_user.yaml")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract prompts from response_categorization files into structured format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        nargs='*',
        default=[],
        help='Model name(s) (e.g., qwen_qwen3-32b). If not specified, processes all models found in response_categorization directory.'
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=Path,
        default=Path('working/response_categorization'),
        help='Directory containing response categorization files (default: working/response_categorization)'
    )
    
    parser.add_argument(
        '-e', '--extracted-prompts-dir',
        type=Path,
        default=Path('working/extracted_prompts'),
        help='Directory containing extracted prompts (default: working/extracted_prompts)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('working/steer_formatted_prompts'),
        help='Output directory for formatted prompts (default: working/steer_formatted_prompts)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"EXTRACT PROMPTS - High Awareness BC Seeds")
    print(f"{'='*80}")
    if args.model:
        print(f"Model(s): {', '.join(args.model)}")
    else:
        print(f"Model(s): All models in response_categorization directory")
    print(f"Input directory: {args.input_dir}")
    print(f"Extracted prompts directory: {args.extracted_prompts_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    main(args.model, args.input_dir, args.extracted_prompts_dir, args.output_dir)
