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
    
Prompts from both v0 and v1 are combined into the same output files.
Duplicate prompts (based on checksum and prompt content) are automatically removed.

Usage:
    # Process all models found in response_categorization directory (both v0 and v1)
    python extract_prompts.py
    
    # Process specific model (both v0 and v1)
    python extract_prompts.py -m qwen_qwen3-32b
    
    # Process specific model with specific version
    python extract_prompts.py -m qwen_qwen3-32b -v v1
    
    # Process multiple specific models with both versions
    python extract_prompts.py -m qwen_qwen3-32b qwen_qwen3-30b-a3b-thinking-2507
"""

import yaml
import argparse
from pathlib import Path
from collections import defaultdict

def get_available_models(resp_cat_dir: Path = Path('working/response_categorization'), versions: list = None) -> list:
    """Get list of all model directories in response_categorization folder.
    
    Args:
        resp_cat_dir: Path to response_categorization directory
        versions: List of versions to search (e.g., ['v0', 'v1']). If None, searches both.
    
    Returns:
        List of tuples (version, model_name)
    """
    if not resp_cat_dir.exists():
        print(f"Warning: Response categorization directory not found: {resp_cat_dir}")
        return []
    
    if versions is None:
        versions = ['v0', 'v1']
    
    # Get all subdirectories that contain 'high_awareness_bc_seeds.yaml'
    models = []
    for version in versions:
        version_dir = resp_cat_dir / version
        if not version_dir.exists():
            print(f"  Warning: Version directory not found: {version_dir}")
            continue
            
        for item in version_dir.iterdir():
            if item.is_dir() and (item / 'high_awareness_bc_seeds.yaml').exists():
                models.append((version, item.name))
    
    return sorted(models)


def load_high_awareness_files(resp_cat_dir: Path, version: str, model_name: str, extracted_prompts_dir: Path) -> list:
    """Load high awareness files from high_awareness_bc_seeds.yaml.
    
    Args:
        resp_cat_dir: Path to response_categorization directory
        version: Version directory (e.g., 'v0' or 'v1')
        model_name: Model name (e.g., 'qwen_qwen3-32b')
        extracted_prompts_dir: Path to extracted_prompts directory
    
    Returns:
        List of dicts with source file path, bucket, and other metadata
    """
    yaml_file = resp_cat_dir / version / model_name / 'high_awareness_bc_seeds.yaml'
    
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


def process_model(version: str, model_name: str, resp_cat_dir: Path, extracted_prompts_dir: Path, output_dir: Path):
    """Process prompts for a single model version.
    
    Args:
        version: Version directory (e.g., 'v0' or 'v1')
        model_name: Model name (e.g., 'qwen_qwen3-32b')
        resp_cat_dir: Directory containing response categorization files
        extracted_prompts_dir: Directory containing extracted prompts
        output_dir: Base output directory
        
    Returns:
        Dict with prompts by type and metadata
    """
    print(f"  Processing {version}/{model_name}...")
    
    # Load high-awareness files
    files_list = load_high_awareness_files(resp_cat_dir, version, model_name, extracted_prompts_dir)
    
    if not files_list:
        print(f"    Warning: No files found for {version}/{model_name}")
        return None
    
    # Collect prompts by type
    sys_prompts = []
    eval_users = []
    deploy_users = []
    
    errors = []
    processed = 0
    
    for file_info in files_list:
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
    
    print(f"    Loaded {len(files_list)} files, processed {processed} successfully")
    
    if errors:
        print(f"    Errors: {len(errors)}")
    
    return {
        'version': version,
        'model': model_name,
        'total_files': len(files_list),
        'processed': processed,
        'errors': errors,
        'sys_prompts': sys_prompts,
        'eval_users': eval_users,
        'deploy_users': deploy_users
    }


def main(models: list, versions: list, resp_cat_dir: Path, extracted_prompts_dir: Path, output_dir: Path):
    """Main function to process one or all models.
    
    Args:
        models: List of model names to process (empty list = process all)
        versions: List of versions to process (e.g., ['v0', 'v1'])
        resp_cat_dir: Directory containing response categorization files
        extracted_prompts_dir: Directory containing extracted prompts
        output_dir: Base output directory
    """
    # If no models specified, process all available models
    if not models:
        print("No model specified. Discovering models from response_categorization directory...")
        all_models = get_available_models(resp_cat_dir, versions)
        if not all_models:
            print("Error: No models found in working/response_categorization/")
            return
        model_list = [f"{v}/{m}" for v, m in all_models]
        print(f"Found {len(all_models)} version/model combinations: {', '.join(model_list)}\n")
        
        # Group by model name
        models_to_process = {}
        for version, model in all_models:
            if model not in models_to_process:
                models_to_process[model] = []
            models_to_process[model].append(version)
    else:
        # If models specified, create dict for requested versions
        models_to_process = {}
        for model in models:
            for version in versions:
                # Check if this version/model combination exists
                if (resp_cat_dir / version / model / 'high_awareness_bc_seeds.yaml').exists():
                    if model not in models_to_process:
                        models_to_process[model] = []
                    models_to_process[model].append(version)
        
        if not models_to_process:
            print(f"Error: No valid model/version combinations found")
            return
    
    # Process each model (combining versions)
    overall_results = []
    
    for model_name, model_versions in models_to_process.items():
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"Versions: {', '.join(model_versions)}")
        print(f"{'='*80}")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_name / 'high_awareness_bc'
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect prompts from all versions
        all_sys_prompts = []
        all_eval_users = []
        all_deploy_users = []
        all_errors = []
        total_files = 0
        total_processed = 0
        
        for version in model_versions:
            result = process_model(version, model_name, resp_cat_dir, extracted_prompts_dir, output_dir)
            if result:
                all_sys_prompts.extend(result['sys_prompts'])
                all_eval_users.extend(result['eval_users'])
                all_deploy_users.extend(result['deploy_users'])
                all_errors.extend(result['errors'])
                total_files += result['total_files']
                total_processed += result['processed']
        
        # Deduplicate prompts based on checksum and prompt content
        def deduplicate_prompts(prompts_list):
            """Remove duplicate prompts, keeping the first occurrence."""
            seen = set()
            unique_prompts = []
            duplicates = 0
            
            for prompt_entry in prompts_list:
                # Create a unique key from checksum and prompt text
                key = (prompt_entry['checksum'], prompt_entry['prompt'])
                
                if key not in seen:
                    seen.add(key)
                    unique_prompts.append(prompt_entry)
                else:
                    duplicates += 1
            
            return unique_prompts, duplicates
        
        # Deduplicate each prompt type
        sys_duplicates = 0
        eval_duplicates = 0
        deploy_duplicates = 0
        
        if all_sys_prompts:
            all_sys_prompts, sys_duplicates = deduplicate_prompts(all_sys_prompts)
        if all_eval_users:
            all_eval_users, eval_duplicates = deduplicate_prompts(all_eval_users)
        if all_deploy_users:
            all_deploy_users, deploy_duplicates = deduplicate_prompts(all_deploy_users)
        
        total_duplicates = sys_duplicates + eval_duplicates + deploy_duplicates
        
        if total_duplicates > 0:
            print(f"\n  Removed {total_duplicates} duplicate entries:")
            if sys_duplicates > 0:
                print(f"    System prompts: {sys_duplicates} duplicates")
            if eval_duplicates > 0:
                print(f"    Eval user: {eval_duplicates} duplicates")
            if deploy_duplicates > 0:
                print(f"    Deploy user: {deploy_duplicates} duplicates")
        
        # Write combined output files
        print(f"\n  Writing combined output files...")
        
        # System prompts
        if all_sys_prompts:
            print(f"    system_prompts.yaml ({len(all_sys_prompts)} prompts)")
            with open(model_output_dir / "system_prompts.yaml", 'w') as f:
                yaml.dump(all_sys_prompts, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Eval user prompts
        if all_eval_users:
            print(f"    eval_user.yaml ({len(all_eval_users)} prompts)")
            with open(model_output_dir / "eval_user.yaml", 'w') as f:
                yaml.dump(all_eval_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Deploy user prompts
        if all_deploy_users:
            print(f"    deploy_user.yaml ({len(all_deploy_users)} prompts)")
            with open(model_output_dir / "deploy_user.yaml", 'w') as f:
                yaml.dump(all_deploy_users, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Summary for this model
        print(f"\n  Summary for {model_name}:")
        print(f"    Versions processed: {', '.join(model_versions)}")
        print(f"    Total files found: {total_files}")
        print(f"    Successfully processed: {total_processed}")
        print(f"    Errors: {len(all_errors)}")
        
        print(f"\n  Output files:")
        print(f"    System prompts: {len(all_sys_prompts)}")
        print(f"    Eval user: {len(all_eval_users)}")
        print(f"    Deploy user: {len(all_deploy_users)}")
        
        print(f"\n  Output directory: {model_output_dir}")
        
        if all_errors:
            print(f"\n  Errors ({len(all_errors)}):")
            for filename, error in all_errors[:5]:  # Show first 5 errors
                print(f"    {filename}: {error}")
            if len(all_errors) > 5:
                print(f"    ... and {len(all_errors) - 5} more errors")
        
        overall_results.append({
            'model': model_name,
            'versions': model_versions,
            'total_files': total_files,
            'processed': total_processed,
            'errors': len(all_errors),
            'system_prompts': len(all_sys_prompts),
            'eval_users': len(all_eval_users),
            'deploy_users': len(all_deploy_users)
        })
    
    # Overall summary
    if len(overall_results) > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY - {len(overall_results)} Models Processed")
        print(f"{'='*80}")
        
        total_processed = sum(r['processed'] for r in overall_results)
        total_errors = sum(r['errors'] for r in overall_results)
        total_sys = sum(r['system_prompts'] for r in overall_results)
        total_eval = sum(r['eval_users'] for r in overall_results)
        total_deploy = sum(r['deploy_users'] for r in overall_results)
        
        print(f"\nTotal files processed: {total_processed}")
        print(f"Total errors: {total_errors}")
        print(f"\nTotal system prompts: {total_sys}")
        print(f"Total eval user prompts: {total_eval}")
        print(f"Total deploy user prompts: {total_deploy}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"Structure:")
        for result in overall_results:
            model = result['model']
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
        '-v', '--version',
        type=str,
        nargs='*',
        default=['v0', 'v1'],
        help='Version(s) to process (e.g., v0, v1). Default: both v0 and v1'
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
    print(f"Version(s): {', '.join(args.version)}")
    print(f"Input directory: {args.input_dir}")
    print(f"Extracted prompts directory: {args.extracted_prompts_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    main(args.model, args.version, args.input_dir, args.extracted_prompts_dir, args.output_dir)
