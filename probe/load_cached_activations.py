"""
Load pre-collected activations and create features for probe training.

Reads from: working/model_activations/qwen3-32B/qwen_3_32B_model_activations/
Structure:
  - deploy/layer_X_out/shard_0.memmap  (token-level activations)
  - deploy/tokens.pt                   (token IDs)
  - deploy/sequence_ranges.pt          (which tokens belong to which sequence)
  - eval/...                           (same structure)
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from tqdm import tqdm


def load_memmap_shard(layer_dir: str, shard_idx: int = 0):
    """
    Load a memmap shard for a specific layer.

    Args:
        layer_dir: Path to layer directory (e.g., deploy/layer_10_out/)
        shard_idx: Shard index

    Returns:
        np.memmap array of shape (n_tokens, d_model)
    """
    config_path = os.path.join(layer_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    d_model = config['d_model']
    total_size = config['total_size']

    shard_path = os.path.join(layer_dir, f"shard_{shard_idx}.memmap")

    # Load memmap
    activations = np.memmap(
        shard_path,
        dtype=np.float16,  # Activations stored as float16
        mode='r',
        shape=(total_size, d_model)
    )

    return activations


def load_sequence_metadata(base_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load tokens and sequence ranges.

    Args:
        base_dir: Base directory (e.g., deploy/)

    Returns:
        (tokens, sequence_ranges)
        tokens: shape (n_tokens,) - token IDs
        sequence_ranges: shape (n_sequences, 2) - [start_idx, end_idx] for each sequence
    """
    tokens = torch.load(os.path.join(base_dir, "tokens.pt"))
    sequence_ranges = torch.load(os.path.join(base_dir, "sequence_ranges.pt"))

    return tokens, sequence_ranges


def find_thinking_token_range(
    tokens: torch.Tensor,
    start_idx: int,
    end_idx: int
) -> Tuple[int, int]:
    """
    Find the token range corresponding to <thinking>...</thinking> or <think>...</think> tags.

    Token IDs for Qwen tokenizer (multi-token sequences):
    - <thinking>: [13708, 15736, 29]
    - </thinking>: [522, 82260, 29]
    - <think>: [13708, 766, 29]
    - </think>: [522, 26865, 29]

    Args:
        tokens: All tokens (n_tokens,)
        start_idx: Sequence start
        end_idx: Sequence end

    Returns:
        (thinking_start, thinking_end) indices
    """
    seq_tokens = tokens[start_idx:end_idx]

    # Multi-token sequences for thinking tags (Qwen tokenizer)
    # Handle variants with different newline combinations
    # Note: > = 29, >\n = 397, >\n\n = 1339
    thinking_open_sequences = [
        [13708, 15736, 29],    # <thinking>
        [13708, 15736, 397],   # <thinking>\n
        [13708, 15736, 1339],  # <thinking>\n\n
        [13708, 766, 29],      # <think>
        [13708, 766, 397],     # <think>\n
        [13708, 766, 1339]     # <think>\n\n
    ]
    thinking_close_sequences = [
        [522, 82260, 29],      # </thinking>
        [522, 82260, 397],     # </thinking>\n
        [522, 82260, 1339],    # </thinking>\n\n
        [522, 26865, 29],      # </think>
        [522, 26865, 397],     # </think>\n
        [522, 26865, 1339]     # </think>\n\n
    ]

    # Search for opening tag (multi-token sequence)
    think_start_local = None
    for i in range(len(seq_tokens)):
        for open_seq in thinking_open_sequences:
            seq_len = len(open_seq)
            if i + seq_len <= len(seq_tokens):
                # Convert to list for comparison
                token_window = [seq_tokens[i + j].item() for j in range(seq_len)]
                if token_window == open_seq:
                    think_start_local = i + seq_len  # Start after the opening tag
                    break
        if think_start_local is not None:
            break

    # Search for closing tag (multi-token sequence, from the end backwards)
    think_end_local = None
    for close_seq in thinking_close_sequences:
        seq_len = len(close_seq)
        # Search from end backwards for the full sequence
        for i in range(len(seq_tokens) - seq_len, -1, -1):
            # Convert to list for comparison
            token_window = [seq_tokens[i + j].item() for j in range(seq_len)]
            if token_window == close_seq:
                think_end_local = i  # End before the closing tag
                break
        if think_end_local is not None:
            break

    # If we found both tags, use them
    if think_start_local is not None and think_end_local is not None and think_end_local > think_start_local:
        thinking_start = start_idx + think_start_local
        thinking_end = start_idx + think_end_local

        # Sanity check
        if thinking_end > thinking_start:
            return thinking_start, thinking_end

    # Fallback: use heuristic (middle 70% of tokens)
    # This assumes: [system prompt tokens][thinking tokens][answer tokens]
    seq_len = len(seq_tokens)
    thinking_start = start_idx + int(0.15 * seq_len)
    thinking_end = start_idx + int(0.85 * seq_len)

    return thinking_start, thinking_end


def aggregate_sequence_activations(
    activations: np.ndarray,
    tokens: torch.Tensor,
    start_idx: int,
    end_idx: int,
    aggregation: str = "mean",
    token_mode: str = "thinking"
) -> np.ndarray:
    """
    Aggregate activations across a sequence (token range).

    Args:
        activations: Full activation memmap (n_tokens, d_model)
        tokens: All tokens (for filtering)
        start_idx: Start token index
        end_idx: End token index (exclusive)
        aggregation: "mean", "max", or "last"
        token_mode: "all", "thinking", "assistant", "last_n"

    Returns:
        Aggregated activation (d_model,)
    """
    # Filter tokens based on mode
    if token_mode == "thinking":
        # Extract only thinking tokens
        think_start, think_end = find_thinking_token_range(tokens, start_idx, end_idx)
        seq_acts = activations[think_start:think_end]
    elif token_mode == "all":
        # All tokens in response
        seq_acts = activations[start_idx:end_idx]
    elif token_mode == "assistant":
        # Skip system prompt, take assistant response
        # Heuristic: last 70% of tokens
        seq_len = end_idx - start_idx
        assistant_start = start_idx + int(0.3 * seq_len)
        seq_acts = activations[assistant_start:end_idx]
    elif token_mode.startswith("last_"):
        # Last N tokens
        n = int(token_mode.split("_")[1])
        seq_acts = activations[max(start_idx, end_idx - n):end_idx]
    else:
        raise ValueError(f"Unknown token_mode: {token_mode}")

    if len(seq_acts) == 0:
        # Fallback to full sequence
        seq_acts = activations[start_idx:end_idx]

    # Aggregate
    if aggregation == "mean":
        return seq_acts.mean(axis=0)
    elif aggregation == "max":
        return seq_acts.max(axis=0)
    elif aggregation == "last":
        return seq_acts[-1]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def project_onto_steering_vector(
    activation: np.ndarray,
    steering_vector: np.ndarray
) -> float:
    """
    Project activation onto steering vector.

    Args:
        activation: shape (d_model,)
        steering_vector: shape (d_model,)

    Returns:
        Projection (scalar)
    """
    return float(np.dot(activation, steering_vector))


def extract_features_for_sequence_pair(
    deploy_seq_idx: int,
    eval_seq_idx: int,
    deploy_base: str,
    eval_base: str,
    layers: List[int],
    steering_vectors: Dict[int, np.ndarray],
    aggregation: str = "mean",
    token_mode: str = "thinking"
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features for a deployment-evaluation sequence pair.

    Args:
        deploy_seq_idx: Deployment sequence index
        eval_seq_idx: Evaluation sequence index
        deploy_base: Path to deployment activations (e.g., deploy/)
        eval_base: Path to evaluation activations (e.g., eval/)
        layers: List of layer indices
        steering_vectors: Dict mapping layer_idx -> steering vector
        aggregation: Aggregation method
        token_mode: Which tokens to use ("thinking", "all", "assistant", "last_N")

    Returns:
        (features, feature_names)
        features shape: (3 * n_layers,) - [deploy_projs, eval_projs, delta_projs]
    """
    # Load sequence ranges and tokens
    deploy_tokens, deploy_ranges = load_sequence_metadata(deploy_base)
    eval_tokens, eval_ranges = load_sequence_metadata(eval_base)

    # Ranges are cumulative: [0, 12, 27, 44, 59] means seq 0 is [0:12], seq 1 is [12:27], etc.
    deploy_start = int(deploy_ranges[deploy_seq_idx])
    deploy_end = int(deploy_ranges[deploy_seq_idx + 1]) if deploy_seq_idx + 1 < len(deploy_ranges) else len(deploy_tokens)

    eval_start = int(eval_ranges[eval_seq_idx])
    eval_end = int(eval_ranges[eval_seq_idx + 1]) if eval_seq_idx + 1 < len(eval_ranges) else len(eval_tokens)

    deploy_projections = []
    eval_projections = []

    for layer in layers:
        layer_name = f"layer_{layer}_out"

        # Load deployment activations for this layer
        deploy_layer_dir = os.path.join(deploy_base, layer_name)
        deploy_acts = load_memmap_shard(deploy_layer_dir, shard_idx=0)

        # Load evaluation activations for this layer
        eval_layer_dir = os.path.join(eval_base, layer_name)
        eval_acts = load_memmap_shard(eval_layer_dir, shard_idx=0)

        # Aggregate with token filtering
        deploy_agg = aggregate_sequence_activations(
            deploy_acts, deploy_tokens, int(deploy_start), int(deploy_end),
            aggregation, token_mode
        )
        eval_agg = aggregate_sequence_activations(
            eval_acts, eval_tokens, int(eval_start), int(eval_end),
            aggregation, token_mode
        )

        # Project
        if layer in steering_vectors:
            deploy_proj = project_onto_steering_vector(deploy_agg, steering_vectors[layer])
            eval_proj = project_onto_steering_vector(eval_agg, steering_vectors[layer])
        else:
            deploy_proj = 0.0
            eval_proj = 0.0

        deploy_projections.append(deploy_proj)
        eval_projections.append(eval_proj)

    # Compute deltas
    deploy_projections = np.array(deploy_projections)
    eval_projections = np.array(eval_projections)
    delta_projections = eval_projections - deploy_projections

    # Concatenate
    features = np.concatenate([deploy_projections, eval_projections, delta_projections])

    # Feature names
    feature_names = []
    for prefix in ["deploy", "eval", "delta"]:
        for layer in layers:
            feature_names.append(f"{prefix}_L{layer}")

    return features, feature_names


def extract_features_for_dataset(
    sequence_pairs: List[Tuple[int, int]],  # [(deploy_idx, eval_idx), ...]
    deploy_base: str,
    eval_base: str,
    layers: List[int],
    steering_vectors: Dict[int, np.ndarray],
    aggregation: str = "mean",
    token_mode: str = "thinking",
    output_path: Optional[str] = None,
    pairs_metadata: Optional[List] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features for all sequence pairs.

    Args:
        sequence_pairs: List of (deploy_seq_idx, eval_seq_idx) tuples
        deploy_base: Deployment activations directory
        eval_base: Evaluation activations directory
        layers: Layers to extract
        steering_vectors: Steering vectors by layer
        aggregation: Aggregation method
        token_mode: Which tokens to use ("thinking", "all", "assistant", "last_N")
        output_path: Optional path to save features
        pairs_metadata: Optional list of ResponsePair objects for metadata

    Returns:
        (X, feature_names)
        X shape: (n_pairs, n_features)
    """
    print(f"\nExtracting features for {len(sequence_pairs)} sequence pairs...")
    print(f"Layers: {layers}")
    print(f"Aggregation: {aggregation}")
    print(f"Token mode: {token_mode}")

    all_features = []
    feature_names = None

    for i, (deploy_idx, eval_idx) in enumerate(tqdm(sequence_pairs, desc="Processing pairs")):
        try:
            features, names = extract_features_for_sequence_pair(
                deploy_idx,
                eval_idx,
                deploy_base,
                eval_base,
                layers,
                steering_vectors,
                aggregation,
                token_mode
            )

            all_features.append(features)

            if feature_names is None:
                feature_names = names

        except Exception as e:
            print(f"\nError processing pair {i} (deploy={deploy_idx}, eval={eval_idx}): {e}")
            # Add zeros as placeholder
            if feature_names is not None:
                all_features.append(np.zeros(len(feature_names)))
            continue

    # Convert to array
    X = np.array(all_features)

    print(f"\nFeature extraction complete:")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {len(feature_names)}")

    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_data = {
            'features': X,
            'feature_names': feature_names,
            'sequence_pairs': sequence_pairs,
            'layers': layers,
            'aggregation': aggregation,
            'pairs_metadata': pairs_metadata
        }

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\nFeatures saved to: {output_path}")

    return X, feature_names


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from probe.response_features import load_steering_vectors

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=str,
                       default="working/model_activations/qwen3-32B/qwen_3_32B_model_activations")
    parser.add_argument("--steering-dir", type=str, default="working/steering-vecs")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--aggregation", type=str, default="mean",
                       choices=["mean", "max", "last"])
    parser.add_argument("--token-mode", type=str, default="thinking",
                       help="Which tokens to use: thinking, all, assistant, last_N")
    args = parser.parse_args()

    # Load steering vectors
    print("Loading steering vectors...")
    steering_vecs = load_steering_vectors(
        steering_dir=args.steering_dir,
        method="anthropic",
        model_name="qwen_qwen3-32b"
    )

    layers = steering_vecs.get_layers()
    steering_dict = steering_vecs.vectors

    deploy_base = os.path.join(args.activation_dir, "deploy")
    eval_base = os.path.join(args.activation_dir, "eval")

    # Filter to only layers that exist in activation cache
    available_layers = []
    for layer in layers:
        layer_dir = os.path.join(deploy_base, f"layer_{layer}_out")
        if os.path.exists(layer_dir):
            available_layers.append(layer)

    if len(available_layers) < len(layers):
        print(f"\nWarning: Only {len(available_layers)}/{len(layers)} layers found in activation cache")
        print(f"  Available: {available_layers}")
        layers = available_layers

    # Load sequence metadata to see how many we have
    _, deploy_ranges = load_sequence_metadata(deploy_base)
    _, eval_ranges = load_sequence_metadata(eval_base)

    n_deploy = len(deploy_ranges) - 1  # -1 because cumulative indices
    n_eval = len(eval_ranges) - 1

    print(f"\nFound {n_deploy} deployment sequences")
    print(f"Found {n_eval} evaluation sequences")

    # For now, assume 1:1 correspondence
    n_pairs = min(n_deploy, n_eval)
    sequence_pairs = [(i, i) for i in range(n_pairs)]

    print(f"\nCreating {n_pairs} sequence pairs (1:1 mapping)")

    # Extract features
    X, feature_names = extract_features_for_dataset(
        sequence_pairs,
        deploy_base,
        eval_base,
        layers,
        steering_dict,
        aggregation=args.aggregation,
        token_mode=args.token_mode,
        output_path=args.output_path
    )

    print(f"\nDone! Features shape: {X.shape}")
