"""
Response-level feature extraction using steering vector projections.

This module computes features for response pairs by projecting response text
onto pre-trained steering vectors at each layer.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import re


@dataclass
class SteeringVectors:
    """Container for steering vectors across layers."""

    method: str  # "anthropic" or "tim"
    model_name: str
    vectors: Dict[int, np.ndarray]  # layer_idx -> vector (d_model,)
    d_model: int

    def get_layers(self) -> List[int]:
        """Get list of available layers."""
        return sorted(self.vectors.keys())

    def get_vector(self, layer: int) -> Optional[np.ndarray]:
        """Get steering vector for a specific layer."""
        return self.vectors.get(layer)


def load_steering_vectors(
    steering_dir: str,
    method: str = "anthropic",
    model_name: str = "qwen_qwen3-32b"
) -> SteeringVectors:
    """
    Load steering vectors from disk.

    Args:
        steering_dir: Base directory containing steering vectors
        method: "anthropic" or "tim"
        model_name: Model identifier

    Returns:
        SteeringVectors object
    """
    method_dir = os.path.join(steering_dir, model_name, method)

    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"Steering vector directory not found: {method_dir}")

    vectors = {}
    d_model = None

    # Load all .npy files
    for filename in os.listdir(method_dir):
        if not filename.endswith('.npy'):
            continue

        # Extract layer number from filename (e.g., "L5.npy" -> 5)
        match = re.match(r'L(\d+)\.npy', filename)
        if not match:
            continue

        layer_idx = int(match.group(1))
        filepath = os.path.join(method_dir, filename)

        # Load vector (allow_pickle for compatibility)
        vector = np.load(filepath, allow_pickle=True)
        vectors[layer_idx] = vector

        # Infer d_model from first vector
        if d_model is None:
            d_model = vector.shape[0]

    if not vectors:
        raise ValueError(f"No steering vectors found in {method_dir}")

    print(f"Loaded {len(vectors)} steering vectors ({method}) for {model_name}")
    print(f"  Layers: {sorted(vectors.keys())}")
    print(f"  d_model: {d_model}")

    return SteeringVectors(
        method=method,
        model_name=model_name,
        vectors=vectors,
        d_model=d_model
    )


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer for text.

    This is a rough approximation - ideally we'd use the actual model tokenizer,
    but for projection onto steering vectors, whitespace tokenization gives
    reasonable results.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Split on whitespace and punctuation boundaries
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens


def compute_text_projection(
    text: str,
    steering_vector: np.ndarray,
    aggregation: str = "mean"
) -> float:
    """
    Compute projection of text onto steering vector.

    This is a simplified approach that doesn't require actual model activations.
    We create pseudo-embeddings based on character n-grams and project.

    Args:
        text: Input text
        steering_vector: Steering vector (d_model,)
        aggregation: How to aggregate token projections ("mean", "max", "std")

    Returns:
        Aggregated projection score
    """
    if not text.strip():
        return 0.0

    d_model = steering_vector.shape[0]

    # Create simple character-based embeddings for each token
    tokens = simple_tokenize(text)
    if not tokens:
        return 0.0

    # For each token, create a pseudo-embedding using character n-grams
    token_projections = []
    for token in tokens:
        # Simple hash-based embedding
        token_embedding = np.zeros(d_model)
        for i, char in enumerate(token.lower()):
            # Spread character influence across embedding dimensions
            hash_val = hash(char + str(i)) % d_model
            token_embedding[hash_val] += 1.0

        # Normalize
        norm = np.linalg.norm(token_embedding)
        if norm > 0:
            token_embedding /= norm

        # Project onto steering vector
        projection = np.dot(token_embedding, steering_vector)
        token_projections.append(projection)

    # Aggregate
    if aggregation == "mean":
        return float(np.mean(token_projections))
    elif aggregation == "max":
        return float(np.max(np.abs(token_projections)))
    elif aggregation == "std":
        return float(np.std(token_projections))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def extract_response_features(
    text: str,
    steering_vectors: SteeringVectors,
    aggregations: List[str] = ["mean", "max", "std"]
) -> np.ndarray:
    """
    Extract features from a single response using steering vector projections.

    Args:
        text: Response text
        steering_vectors: Pre-loaded steering vectors
        aggregations: List of aggregation methods to use

    Returns:
        Feature vector (n_layers * n_aggregations,)
    """
    features = []
    layers = steering_vectors.get_layers()

    for layer in layers:
        vector = steering_vectors.get_vector(layer)

        for agg in aggregations:
            projection = compute_text_projection(text, vector, aggregation=agg)
            features.append(projection)

    return np.array(features)


def extract_pair_features(
    deployment_text: str,
    evaluation_text: str,
    steering_vectors: SteeringVectors,
    include_deployment: bool = True,
    include_evaluation: bool = True,
    include_delta: bool = True,
    aggregations: List[str] = ["mean", "max", "std"]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from a deployment-evaluation response pair.

    Args:
        deployment_text: Deployment context response
        evaluation_text: Evaluation context response
        steering_vectors: Pre-loaded steering vectors
        include_deployment: Include deployment features
        include_evaluation: Include evaluation features
        include_delta: Include delta features (eval - deploy)
        aggregations: List of aggregation methods

    Returns:
        (features, feature_names)
    """
    deploy_feats = extract_response_features(deployment_text, steering_vectors, aggregations)
    eval_feats = extract_response_features(evaluation_text, steering_vectors, aggregations)

    features = []
    feature_names = []

    layers = steering_vectors.get_layers()

    # Deployment features
    if include_deployment:
        features.append(deploy_feats)
        for layer in layers:
            for agg in aggregations:
                feature_names.append(f"deploy_L{layer}_{agg}")

    # Evaluation features
    if include_evaluation:
        features.append(eval_feats)
        for layer in layers:
            for agg in aggregations:
                feature_names.append(f"eval_L{layer}_{agg}")

    # Delta features
    if include_delta:
        delta_feats = eval_feats - deploy_feats
        features.append(delta_feats)
        for layer in layers:
            for agg in aggregations:
                feature_names.append(f"delta_L{layer}_{agg}")

    # Concatenate all features
    final_features = np.concatenate(features) if features else np.array([])

    return final_features, feature_names


def extract_dataset_features(
    response_pairs: List,  # List[ResponsePair]
    steering_vectors: SteeringVectors,
    include_deployment: bool = True,
    include_evaluation: bool = True,
    include_delta: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features for all response pairs in a dataset.

    Args:
        response_pairs: List of ResponsePair objects
        steering_vectors: Pre-loaded steering vectors
        include_deployment: Include deployment features
        include_evaluation: Include evaluation features
        include_delta: Include delta features
        verbose: Print progress

    Returns:
        (X, feature_names) where X is (n_pairs, n_features)
    """
    all_features = []
    feature_names = None

    for i, pair in enumerate(response_pairs):
        feats, names = extract_pair_features(
            pair.deployment_text,
            pair.evaluation_text,
            steering_vectors,
            include_deployment=include_deployment,
            include_evaluation=include_evaluation,
            include_delta=include_delta
        )

        all_features.append(feats)

        if feature_names is None:
            feature_names = names

        if verbose and (i + 1) % 50 == 0:
            print(f"Extracted features for {i + 1}/{len(response_pairs)} pairs")

    X = np.vstack(all_features)

    if verbose:
        print(f"\nFeature extraction complete:")
        print(f"  Shape: {X.shape}")
        print(f"  n_features: {len(feature_names)}")

    return X, feature_names


if __name__ == "__main__":
    # Example usage
    from response_data_prep import load_all_response_pairs, split_by_prompt

    print("Loading steering vectors...")
    steering_vecs = load_steering_vectors(
        steering_dir="working/steering-vecs",
        method="anthropic",
        model_name="qwen_qwen3-32b"
    )

    print("\nLoading response pairs...")
    pairs = load_all_response_pairs(
        "working/response_categorization/v1/qwen_qwen3-32b",
        limit=20,
        verbose=False
    )

    print(f"Loaded {len(pairs)} pairs")

    print("\nExtracting features...")
    X, feature_names = extract_dataset_features(
        pairs,
        steering_vecs,
        include_deployment=True,
        include_evaluation=True,
        include_delta=True,
        verbose=True
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Sample features (first 5): {feature_names[:5]}")
    print(f"Sample values: {X[0, :5]}")
