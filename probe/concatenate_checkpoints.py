"""Concatenate multiple checkpoint files into a single larger checkpoint."""

import sys
import pickle
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def concatenate_checkpoints(checkpoint_paths, output_path):
    """
    Concatenate multiple checkpoint files into one.

    Args:
        checkpoint_paths: List of checkpoint file paths (in order)
        output_path: Output path for combined checkpoint
    """
    print("="*80)
    print("CONCATENATING CHECKPOINTS")
    print("="*80)

    all_deploy_acts = []
    all_eval_acts = []
    all_pairs = []
    layers = None

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n[{i+1}/{len(checkpoint_paths)}] Loading: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        deploy_acts = data['deploy_activations']
        eval_acts = data['eval_activations']
        pairs = data['pairs']

        if layers is None:
            layers = data['layers']

        print(f"  Pairs: {len(pairs)}")
        print(f"  Deploy activations: {len(deploy_acts)}")
        print(f"  Eval activations: {len(eval_acts)}")

        # Check for overlap - skip if this checkpoint starts before our current end
        current_total = len(all_pairs)
        if len(pairs) <= current_total:
            print(f"  ⚠️  SKIPPING - Already have {current_total} pairs (this has {len(pairs)})")
            continue

        # Add new pairs only (skip overlapping ones)
        new_pairs = pairs[current_total:]
        new_deploy = deploy_acts[current_total:]
        new_eval = eval_acts[current_total:]

        if new_pairs:
            print(f"  ✓ Adding {len(new_pairs)} new pairs")
            all_pairs.extend(new_pairs)
            all_deploy_acts.extend(new_deploy)
            all_eval_acts.extend(new_eval)
        else:
            print(f"  ✓ No new pairs to add")

    # Save combined checkpoint
    print(f"\n" + "="*80)
    print(f"SAVING COMBINED CHECKPOINT")
    print(f"="*80)
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Total deploy activations: {len(all_deploy_acts)}")
    print(f"  Total eval activations: {len(all_eval_acts)}")
    print(f"  Layers: {layers}")
    print(f"  Output: {output_path}")

    combined_data = {
        'deploy_activations': all_deploy_acts,
        'eval_activations': all_eval_acts,
        'pairs': all_pairs,
        'layers': layers,
        'n_processed': len(all_pairs)
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"\n✓ Saved combined checkpoint with {len(all_pairs)} pairs")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Concatenate checkpoint files')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint files to concatenate (in order)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for combined checkpoint')
    args = parser.parse_args()

    # Verify all checkpoints exist
    for checkpoint in args.checkpoints:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    concatenate_checkpoints(args.checkpoints, args.output)

if __name__ == "__main__":
    main()
