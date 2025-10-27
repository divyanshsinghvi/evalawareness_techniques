#!/usr/bin/env python3
"""
Analyze response-level categorization statistics.

Aggregates awareness scores from working/response_categorization/ to understand
evaluation awareness patterns at the full-response level.

Usage:
    python analyze_response_categorization_stats.py --model-dir qwen_qwen3-32b
"""

import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import json


def extract_prompt_id(file_path: str) -> str:
    """Extract unique prompt identifier from file path (remove _seed_N.yaml)."""
    import re
    filename = Path(file_path).name
    # Remove _seed_N.yaml to get base prompt name
    match = re.match(r'(.+)_seed_(\d+)\.yaml', filename)
    if match:
        return match.group(1)
    return filename.replace('.yaml', '')


def load_response_categorization_data(categorization_dir: Path):
    """Load all response categorization YAML files."""
    data = []
    
    for yaml_file in categorization_dir.rglob('*.yaml'):
        with open(yaml_file) as f:
            entry = yaml.safe_load(f)
            if entry:
                entry['_file_path'] = str(yaml_file.relative_to(categorization_dir))
                data.append(entry)
    
    return data


def print_summary_dashboard(data):
    """Print high-level summary dashboard."""
    total_seeds = len(data)
    unique_prompts = len(set(extract_prompt_id(entry.get('source_file', '')) for entry in data))
    
    # Behavioral change breakdown
    behavioral_change_count = sum(1 for e in data if e.get('behavioral_change', False))
    no_change_count = total_seeds - behavioral_change_count
    
    # Collect awareness intensities
    deploy_intensities = []
    eval_intensities = []
    deploy_types = defaultdict(int)
    eval_types = defaultdict(int)
    deploy_levels = defaultdict(int)
    eval_levels = defaultdict(int)
    
    for entry in data:
        deploy = entry.get('deployment', {}).get('awareness', {})
        evalu = entry.get('evaluation', {}).get('awareness', {})
        
        if deploy:
            intensity = deploy.get('awareness_intensity', 0)
            if isinstance(intensity, int):
                deploy_intensities.append(intensity)
            deploy_types[deploy.get('awareness_type', 'unknown')] += 1
            deploy_levels[deploy.get('awareness_level', 'unknown')] += 1
        
        if evalu:
            intensity = evalu.get('awareness_intensity', 0)
            if isinstance(intensity, int):
                eval_intensities.append(intensity)
            eval_types[evalu.get('awareness_type', 'unknown')] += 1
            eval_levels[evalu.get('awareness_level', 'unknown')] += 1
    
    print(f"\n{'='*80}")
    print(f"RESPONSE CATEGORIZATION SUMMARY DASHBOARD")
    print(f"{'='*80}\n")
    
    print(f"{'Total Seeds:':<30} {total_seeds}")
    print(f"{'Unique Prompts:':<30} {unique_prompts}")
    print(f"{'Behavioral Change:':<30} {behavioral_change_count} ({100*behavioral_change_count/total_seeds:.1f}%)")
    print(f"{'No Behavioral Change:':<30} {no_change_count} ({100*no_change_count/total_seeds:.1f}%)")
    
    if deploy_intensities:
        print(f"\n{'Deployment Awareness Intensity:':<30}")
        print(f"  {'Mean:':<28} {sum(deploy_intensities)/len(deploy_intensities):.2f}")
        print(f"  {'Median:':<28} {sorted(deploy_intensities)[len(deploy_intensities)//2]}")
        print(f"  {'Max:':<28} {max(deploy_intensities)}")
    
    if eval_intensities:
        print(f"\n{'Evaluation Awareness Intensity:':<30}")
        print(f"  {'Mean:':<28} {sum(eval_intensities)/len(eval_intensities):.2f}")
        print(f"  {'Median:':<28} {sorted(eval_intensities)[len(eval_intensities)//2]}")
        print(f"  {'Max:':<28} {max(eval_intensities)}")
    
    # Awareness type breakdown
    print(f"\n{'Awareness Type Distribution:':<30}")
    print(f"\n{'Type':<30} {'Deployment':>15} {'Evaluation':>15}")
    print(f"{'-'*60}")
    all_types = set(deploy_types.keys()) | set(eval_types.keys())
    for atype in sorted(all_types):
        deploy_count = deploy_types.get(atype, 0)
        eval_count = eval_types.get(atype, 0)
        print(f"{atype:<30} {deploy_count:>15} {eval_count:>15}")
    
    # Awareness level breakdown
    print(f"\n{'Awareness Level Distribution:':<30}")
    print(f"\n{'Level':<30} {'Deployment':>15} {'Evaluation':>15}")
    print(f"{'-'*60}")
    all_levels = set(deploy_levels.keys()) | set(eval_levels.keys())
    for level in sorted(all_levels):
        deploy_count = deploy_levels.get(level, 0)
        eval_count = eval_levels.get(level, 0)
        print(f"{level:<30} {deploy_count:>15} {eval_count:>15}")


def analyze_intensity_buckets(data):
    """Analyze awareness intensity distribution across buckets."""
    buckets = {
        '<5': (0, 5),
        '5-6': (5, 7),
        '7-8': (7, 9),
        '9-10': (9, 11)
    }
    
    deploy_buckets = defaultdict(int)
    eval_buckets = defaultdict(int)
    
    for entry in data:
        deploy = entry.get('deployment', {}).get('awareness', {})
        evalu = entry.get('evaluation', {}).get('awareness', {})
        
        deploy_intensity = deploy.get('awareness_intensity', 0) if deploy else 0
        eval_intensity = evalu.get('awareness_intensity', 0) if evalu else 0
        
        for bucket_name, (low, high) in buckets.items():
            if low <= deploy_intensity < high:
                deploy_buckets[bucket_name] += 1
            if low <= eval_intensity < high:
                eval_buckets[bucket_name] += 1
    
    print(f"\n{'='*80}")
    print(f"AWARENESS INTENSITY BUCKETS")
    print(f"{'='*80}\n")
    
    print(f"{'Bucket':<15} {'Deployment':>15} {'Evaluation':>15} {'Total':>15}")
    print(f"{'-'*60}")
    
    for bucket in ['<5', '5-6', '7-8', '9-10']:
        deploy_count = deploy_buckets[bucket]
        eval_count = eval_buckets[bucket]
        total = deploy_count + eval_count
        print(f"{bucket:<15} {deploy_count:>15} {eval_count:>15} {total:>15}")


def analyze_behavioral_change_correlation(data):
    """Analyze correlation between awareness and behavioral change."""
    behavioral_change_data = [e for e in data if e.get('behavioral_change', False)]
    no_change_data = [e for e in data if not e.get('behavioral_change', False)]
    
    def get_intensities(dataset):
        deploy = []
        evalu = []
        for entry in dataset:
            d = entry.get('deployment', {}).get('awareness', {})
            e = entry.get('evaluation', {}).get('awareness', {})
            if d:
                deploy.append(d.get('awareness_intensity', 0))
            if e:
                evalu.append(e.get('awareness_intensity', 0))
        return deploy, evalu
    
    bc_deploy, bc_eval = get_intensities(behavioral_change_data)
    nc_deploy, nc_eval = get_intensities(no_change_data)
    
    print(f"\n{'='*80}")
    print(f"BEHAVIORAL CHANGE CORRELATION")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<40} {'Behavioral Change':>20} {'No Change':>20}")
    print(f"{'-'*80}")
    
    if bc_deploy:
        print(f"{'Deployment Awareness (mean)':<40} {sum(bc_deploy)/len(bc_deploy):>20.2f} {sum(nc_deploy)/len(nc_deploy) if nc_deploy else 0:>20.2f}")
    if bc_eval:
        print(f"{'Evaluation Awareness (mean)':<40} {sum(bc_eval)/len(bc_eval):>20.2f} {sum(nc_eval)/len(nc_eval) if nc_eval else 0:>20.2f}")
    
    # Delta analysis
    deltas_bc = []
    deltas_nc = []
    
    for entry in behavioral_change_data:
        d = entry.get('deployment', {}).get('awareness', {}).get('awareness_intensity', 0)
        e = entry.get('evaluation', {}).get('awareness', {}).get('awareness_intensity', 0)
        deltas_bc.append(e - d)
    
    for entry in no_change_data:
        d = entry.get('deployment', {}).get('awareness', {}).get('awareness_intensity', 0)
        e = entry.get('evaluation', {}).get('awareness', {}).get('awareness_intensity', 0)
        deltas_nc.append(e - d)
    
    if deltas_bc:
        print(f"{'Eval - Deploy Delta (mean)':<40} {sum(deltas_bc)/len(deltas_bc):>20.2f} {sum(deltas_nc)/len(deltas_nc) if deltas_nc else 0:>20.2f}")


def analyze_self_test_prompts(data):
    """Analyze unique prompts by self_test vs other_test vs no_test_reference."""
    prompts_deploy_self = set()
    prompts_deploy_other = set()
    prompts_deploy_no = set()
    prompts_eval_self = set()
    prompts_eval_other = set()
    prompts_eval_no = set()

    for entry in data:
        prompt_id = extract_prompt_id(entry.get('source_file', ''))

        deploy = entry.get('deployment', {}).get('awareness', {})
        evalu = entry.get('evaluation', {}).get('awareness', {})

        deploy_type = deploy.get('awareness_type', '') if deploy else ''
        eval_type = evalu.get('awareness_type', '') if evalu else ''

        if deploy_type == 'self_test':
            prompts_deploy_self.add(prompt_id)
        elif deploy_type == 'other_test':
            prompts_deploy_other.add(prompt_id)
        elif deploy_type == 'no_test_reference':
            prompts_deploy_no.add(prompt_id)

        if eval_type == 'self_test':
            prompts_eval_self.add(prompt_id)
        elif eval_type == 'other_test':
            prompts_eval_other.add(prompt_id)
        elif eval_type == 'no_test_reference':
            prompts_eval_no.add(prompt_id)

    print(f"\n{'='*80}")
    print(f"UNIQUE PROMPTS BY TEST REFERENCE TYPE")
    print(f"{'='*80}\n")

    print(f"{'Branch':<30} {'Self-Test':>15} {'Other-Test':>15} {'No Test':>15}")
    print(f"{'-'*75}")
    print(f"{'Deployment':<30} {len(prompts_deploy_self):>15} {len(prompts_deploy_other):>15} {len(prompts_deploy_no):>15}")
    print(f"{'Evaluation':<30} {len(prompts_eval_self):>15} {len(prompts_eval_other):>15} {len(prompts_eval_no):>15}")


def analyze_mean_per_prompt(data):
    """Calculate mean awareness scores per prompt (averaged across seeds)."""
    from collections import defaultdict

    # Group by prompt
    prompt_data = defaultdict(lambda: {
        'deploy_intensities': [],
        'eval_intensities': [],
        'behavioral_changes': []
    })

    for entry in data:
        prompt_id = extract_prompt_id(entry.get('source_file', ''))

        deploy = entry.get('deployment', {}).get('awareness', {})
        evalu = entry.get('evaluation', {}).get('awareness', {})

        deploy_intensity = deploy.get('awareness_intensity', 0) if deploy else 0
        eval_intensity = evalu.get('awareness_intensity', 0) if evalu else 0
        behavioral_change = entry.get('behavioral_change', False)

        prompt_data[prompt_id]['deploy_intensities'].append(deploy_intensity)
        prompt_data[prompt_id]['eval_intensities'].append(eval_intensity)
        prompt_data[prompt_id]['behavioral_changes'].append(behavioral_change)

    # Calculate means per prompt
    prompt_stats = []
    for prompt_id, stats in prompt_data.items():
        deploy_mean = sum(stats['deploy_intensities']) / len(stats['deploy_intensities'])
        eval_mean = sum(stats['eval_intensities']) / len(stats['eval_intensities'])
        bc_rate = sum(1 for bc in stats['behavioral_changes'] if bc) / len(stats['behavioral_changes'])
        num_seeds = len(stats['deploy_intensities'])

        prompt_stats.append({
            'prompt_id': prompt_id,
            'deploy_mean': deploy_mean,
            'eval_mean': eval_mean,
            'delta_mean': eval_mean - deploy_mean,
            'bc_rate': bc_rate,
            'num_seeds': num_seeds
        })

    # Overall statistics
    all_deploy_means = [p['deploy_mean'] for p in prompt_stats]
    all_eval_means = [p['eval_mean'] for p in prompt_stats]
    all_delta_means = [p['delta_mean'] for p in prompt_stats]
    all_bc_rates = [p['bc_rate'] for p in prompt_stats]

    print(f"\n{'='*80}")
    print(f"MEAN AWARENESS PER PROMPT (AVERAGED ACROSS SEEDS)")
    print(f"{'='*80}\n")

    print(f"{'Total Unique Prompts:':<40} {len(prompt_stats)}")
    print(f"{'Mean Seeds per Prompt:':<40} {sum(p['num_seeds'] for p in prompt_stats) / len(prompt_stats):.1f}")

    print(f"\n{'Deployment Awareness (prompt means):':<40}")
    print(f"  {'Overall Mean:':<38} {sum(all_deploy_means)/len(all_deploy_means):.3f}")
    print(f"  {'Median:':<38} {sorted(all_deploy_means)[len(all_deploy_means)//2]:.3f}")
    print(f"  {'Max:':<38} {max(all_deploy_means):.3f}")

    print(f"\n{'Evaluation Awareness (prompt means):':<40}")
    print(f"  {'Overall Mean:':<38} {sum(all_eval_means)/len(all_eval_means):.3f}")
    print(f"  {'Median:':<38} {sorted(all_eval_means)[len(all_eval_means)//2]:.3f}")
    print(f"  {'Max:':<38} {max(all_eval_means):.3f}")

    print(f"\n{'Eval - Deploy Delta (prompt means):':<40}")
    print(f"  {'Overall Mean:':<38} {sum(all_delta_means)/len(all_delta_means):.3f}")
    print(f"  {'Median:':<38} {sorted(all_delta_means)[len(all_delta_means)//2]:.3f}")
    print(f"  {'Max:':<38} {max(all_delta_means):.3f}")

    print(f"\n{'Behavioral Change Rate (prompt means):':<40}")
    print(f"  {'Overall Mean:':<38} {sum(all_bc_rates)/len(all_bc_rates):.1%}")
    print(f"  {'Median:':<38} {sorted(all_bc_rates)[len(all_bc_rates)//2]:.1%}")
    print(f"  {'Prompts with 100% BC Rate:':<38} {sum(1 for r in all_bc_rates if r == 1.0)}")
    print(f"  {'Prompts with 0% BC Rate:':<38} {sum(1 for r in all_bc_rates if r == 0.0)}")

    # Top prompts by eval awareness
    top_eval = sorted(prompt_stats, key=lambda x: x['eval_mean'], reverse=True)[:10]
    print(f"\n{'Top 10 Prompts by Evaluation Awareness Mean:'}")
    print(f"\n{'Prompt ID':<50} {'Deploy':>10} {'Eval':>10} {'Delta':>10} {'BC Rate':>10}")
    print(f"{'-'*90}")
    for p in top_eval:
        prompt_display = p['prompt_id'][:47] + '...' if len(p['prompt_id']) > 50 else p['prompt_id']
        print(f"{prompt_display:<50} {p['deploy_mean']:>10.2f} {p['eval_mean']:>10.2f} {p['delta_mean']:>10.2f} {p['bc_rate']:>10.1%}")


def main():
    parser = argparse.ArgumentParser(description='Analyze response categorization statistics')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory (e.g., qwen_qwen3-32b)')
    
    args = parser.parse_args()
    
    categorization_dir = Path('working/response_categorization') / args.model_dir
    
    if not categorization_dir.exists():
        print(f"Error: {categorization_dir} not found")
        return 1
    
    print(f"Loading response categorization data from {categorization_dir}...")
    data = load_response_categorization_data(categorization_dir)
    
    if not data:
        print("No data found!")
        return 1
    
    print(f"Loaded {len(data)} categorized responses\n")
    
    # Run analyses
    print_summary_dashboard(data)
    analyze_intensity_buckets(data)
    analyze_behavioral_change_correlation(data)
    analyze_self_test_prompts(data)
    analyze_mean_per_prompt(data)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
