"""
Script to plot token length distributions for prompts in extracted_prompts directory.

Usage:
    python plot_token_distributions.py --model_id <hf_model_id> [--mode <explicit|implicit>] [--include_responses]

Examples:
    python plot_token_distributions.py --model_id "qwen/qwen3-32b" --mode explicit
    python plot_token_distributions.py --model_id "qwen/qwen3-32b" --mode explicit --include_responses
"""

import argparse
import yaml
import os
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot token length distributions for extracted prompts"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'qwen/qwen3-32b')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="explicit",
        choices=["explicit", "implicit"],
        help="Prompt mode: 'explicit' or 'implicit' (default: explicit)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: plot_stats/plots/<model_name>_<mode>)"
    )
    parser.add_argument(
        "--include_responses",
        action="store_true",
        help="Include model response token distributions in the analysis"
    )
    return parser.parse_args()


def load_prompts_from_yaml(yaml_path: str, include_responses: bool = False) -> Dict:
    """Load prompts from a single YAML file."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Extract the relevant fields
        system_prompt = data.get('system_prompt', '')
        branch_a = data.get('branch_a_deployment', {})
        branch_b = data.get('branch_b_evaluation', {})
        
        user_deploy = branch_a.get('user_message', '') if branch_a else ''
        user_eval = branch_b.get('user_message', '') if branch_b else ''
        
        result = {
            'system_prompt': system_prompt,
            'user_deploy': user_deploy,
            'user_eval': user_eval,
            'file': os.path.basename(yaml_path)
        }
        
        # Optionally include assistant responses
        if include_responses:
            assistant_deploy = branch_a.get('assistant_response', '') if branch_a else ''
            assistant_eval = branch_b.get('assistant_response', '') if branch_b else ''
            result['assistant_deploy'] = assistant_deploy
            result['assistant_eval'] = assistant_eval
        
        return result
    except Exception as e:
        print(f"Error loading {yaml_path}: {e}")
        return None


def get_token_count(tokenizer, text: str) -> int:
    """Get token count for a given text."""
    if not text or not isinstance(text, str):
        return 0
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return 0


def collect_token_lengths(prompts_dir: Path, tokenizer, include_responses: bool = False) -> Dict[str, List[int]]:
    """Collect token lengths for all prompts in the directory."""
    yaml_files = list(prompts_dir.glob("*.yaml"))
    
    token_lengths = {
        'system': [],
        'user_eval': [],
        'user_deploy': [],
        'sys_user_eval': [],
        'sys_user_deploy': []
    }
    
    # Add response categories if requested
    if include_responses:
        token_lengths['assistant_deploy'] = []
        token_lengths['assistant_eval'] = []
    
    print(f"Processing {len(yaml_files)} YAML files...")
    
    for yaml_file in tqdm(yaml_files):
        prompt_data = load_prompts_from_yaml(str(yaml_file), include_responses=include_responses)
        
        if prompt_data is None:
            continue
        
        sys_prompt = prompt_data['system_prompt']
        user_deploy = prompt_data['user_deploy']
        user_eval = prompt_data['user_eval']
        
        # Calculate token lengths
        sys_len = get_token_count(tokenizer, sys_prompt)
        user_deploy_len = get_token_count(tokenizer, user_deploy)
        user_eval_len = get_token_count(tokenizer, user_eval)
        
        # Store individual lengths
        token_lengths['system'].append(sys_len)
        token_lengths['user_eval'].append(user_eval_len)
        token_lengths['user_deploy'].append(user_deploy_len)
        
        # Calculate combined lengths
        sys_user_eval_combined = sys_prompt + "\n\n" + user_eval
        sys_user_deploy_combined = sys_prompt + "\n\n" + user_deploy
        
        token_lengths['sys_user_eval'].append(
            get_token_count(tokenizer, sys_user_eval_combined)
        )
        token_lengths['sys_user_deploy'].append(
            get_token_count(tokenizer, sys_user_deploy_combined)
        )
        
        # Add response token lengths if requested
        if include_responses:
            assistant_deploy = prompt_data.get('assistant_deploy', '')
            assistant_eval = prompt_data.get('assistant_eval', '')
            
            token_lengths['assistant_deploy'].append(
                get_token_count(tokenizer, assistant_deploy)
            )
            token_lengths['assistant_eval'].append(
                get_token_count(tokenizer, assistant_eval)
            )
    
    return token_lengths


def plot_distributions(token_lengths: Dict[str, List[int]], output_dir: Path, 
                       model_name: str, mode: str, include_responses: bool = False):
    """Create and save distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create individual plots for each category
    categories = [
        ('system', 'System Prompt Token Length Distribution'),
        ('user_eval', 'User Evaluation Prompt Token Length Distribution'),
        ('user_deploy', 'User Deployment Prompt Token Length Distribution'),
        ('sys_user_eval', 'System + User Eval Token Length Distribution'),
        ('sys_user_deploy', 'System + User Deploy Token Length Distribution')
    ]
    
    # Add response categories if requested
    if include_responses:
        categories.extend([
            ('assistant_deploy', 'Assistant Deployment Response Token Length Distribution'),
            ('assistant_eval', 'Assistant Evaluation Response Token Length Distribution')
        ])
    
    for category, title in categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = token_lengths[category]
        
        if not data:
            print(f"Warning: No data for {category}")
            continue
        
        # Create histogram
        ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        stats_text = (
            f'Mean: {mean_val:.1f}\n'
            f'Median: {median_val:.1f}\n'
            f'Std: {std_val:.1f}\n'
            f'Min: {min(data)}\n'
            f'Max: {max(data)}\n'
            f'N: {len(data)}'
        )
        
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
        
        ax.set_xlabel('Token Count', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title}\nModel: {model_name} | Mode: {mode}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'{category}_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    # Create a combined plot with all distributions
    # Adjust grid size based on number of categories
    if include_responses:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))  # 7 plots + 2 empty
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 5 plots + 1 empty
    axes = axes.flatten()
    
    for idx, (category, title) in enumerate(categories):
        ax = axes[idx]
        data = token_lengths[category]
        
        if not data:
            continue
        
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label='Mean')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label='Median')
        
        ax.set_xlabel('Token Count', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title.replace(' Token Length Distribution', ''), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'μ={mean_val:.0f}, σ={np.std(data):.0f}, N={len(data)}'
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Token Length Distributions - Model: {model_name} | Mode: {mode.upper()}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    combined_output = output_dir / 'combined_distributions.png'
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Saved: {combined_output}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS - Model: {model_name} | Mode: {mode.upper()}")
    print("="*80)
    
    for category, _ in categories:
        data = token_lengths[category]
        if data:
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Count:  {len(data)}")
            print(f"  Mean:   {np.mean(data):.2f}")
            print(f"  Median: {np.median(data):.2f}")
            print(f"  Std:    {np.std(data):.2f}")
            print(f"  Min:    {min(data)}")
            print(f"  Max:    {max(data)}")
            print(f"  25th percentile: {np.percentile(data, 25):.2f}")
            print(f"  75th percentile: {np.percentile(data, 75):.2f}")


def main():
    """Main function."""
    args = parse_args()
    
    # Determine the prompts directory
    script_dir = Path(__file__).parent.parent.parent  # Go up to Sprint directory
    prompts_dir = script_dir / "working" / "extracted_prompts" / "behavioral_change" / args.mode
    
    if not prompts_dir.exists():
        print(f"Error: Directory not found: {prompts_dir}")
        return
    
    print(f"Loading tokenizer for model: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure the model ID is correct and you have internet connection.")
        return
    
    # Collect token lengths
    print(f"\nAnalyzing prompts from: {prompts_dir}")
    if args.include_responses:
        print("Including model response token distributions...")
    token_lengths = collect_token_lengths(prompts_dir, tokenizer, args.include_responses)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a clean model name for directory
        model_name = args.model_id.replace("/", "_")
        suffix = "_with_responses" if args.include_responses else ""
        output_dir = Path(__file__).parent / "plots" / f"{model_name}_{args.mode}{suffix}"
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_distributions(token_lengths, output_dir, args.model_id, args.mode, args.include_responses)
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

