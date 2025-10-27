"""
Generate steering vectors from model activations.

This script loads activation data for each layer from evaluation and deployment caches,
and computes the steering vector using one of two modes:
- tim: difference of means (deploy_mean - eval_mean)
- anthropic: difference of sums (deploy_sum - eval_sum)

Output structure: output_dir/model_name/category/L{layer_num}.npy

Example:
    python generate_steering_vectors.py \
        --model_name qwen_qwen3-32b \
        --eval_dir /path/to/eval \
        --deploy_dir /path/to/deploy \
        --output_dir steering_vectors \
        --category tim \
        --start_layer 15 \
        --end_layer 48
"""

import numpy as np
# Compatibility for older numpy code referencing np.bool
if not hasattr(np, "bool"):
    np.bool = bool

import os
import argparse
import torch
from crosscoder_learning.dictionary_learning.cache import ActivationCache


def get_acts(eval_dir: str, deploy_dir: str, layer_num: int):
    """Load activations for a given layer from eval and deploy caches."""
    eval_cache = ActivationCache(eval_dir, f"layer_{layer_num}_out")
    deploy_cache = ActivationCache(deploy_dir, f"layer_{layer_num}_out")

    eval_indices = eval_cache.sequence_ranges[1:] - 3
    eval_acts = [eval_cache[int(idx)] for idx in eval_indices]
    eval_acts = torch.stack(eval_acts, dim=0)

    deploy_indices = deploy_cache.sequence_ranges[1:] - 3
    deploy_acts = [deploy_cache[int(idx)] for idx in deploy_indices]
    deploy_acts = torch.stack(deploy_acts, dim=0)

    return eval_acts, deploy_acts


def main(args):
    # Construct output directory: output_dir/model_name/category
    final_output_dir = os.path.join(args.output_dir, args.model_name, args.category)
    os.makedirs(final_output_dir, exist_ok=True)
    
    print(f"Output directory: {final_output_dir}\n")

    for layer_num in range(args.start_layer, args.end_layer + 1):
        print(f"Processing layer {layer_num}...")
        eval_acts, deploy_acts = get_acts(args.eval_dir, args.deploy_dir, layer_num)

        if args.category == "tim":
            # Difference of means
            steering_vector = deploy_acts.mean(dim=0) - eval_acts.mean(dim=0)
        elif args.category == "anthropic":
            # Difference of sums
            steering_vector = deploy_acts.sum(dim=0) - eval_acts.sum(dim=0)
        else:
            raise ValueError(f"Unknown type: {args.category}")

        save_path = os.path.join(final_output_dir, f"L{layer_num}.npy")
        np.save(save_path, steering_vector.to(torch.float32).cpu().numpy())
        print(f"Saved steering vector: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate steering vectors from activation caches.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for organizing output (e.g., qwen_qwen3-32b).")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to evaluation activation cache directory.")
    parser.add_argument("--deploy_dir", type=str, required=True, help="Path to deployment activation cache directory.")
    parser.add_argument("--category", type=str, choices=["tim", "anthropic"], required=True,
                        help="Type of steering vector computation: 'tim' (mean diff) or 'anthropic' (sum diff).")
    parser.add_argument("--output_dir", type=str, default="steering_vectors", help="Base directory to save steering vectors.")
    parser.add_argument("--start_layer", type=int, default=45, help="Starting layer index.")
    parser.add_argument("--end_layer", type=int, default=96, help="Ending layer index (inclusive).")

    args = parser.parse_args()
    main(args)
