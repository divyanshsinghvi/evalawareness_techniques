"""
Response-level data preparation for evaluation awareness probe.

Loads response pairs (deployment + evaluation) from categorization YAMLs
and creates labels for binary, regression, and behavioral change prediction.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict


@dataclass
class ResponsePair:
    """Container for a deployment-evaluation response pair."""

    # Identifiers
    prompt_id: str  # Unique prompt identifier (for train/test split)
    seed: int  # Rollout seed
    source_file: str  # Path to rollout YAML

    # Response texts (assistant response only, for backward compat)
    deployment_text: str
    evaluation_text: str

    # Full conversation texts (system + user + assistant)
    deployment_full: Optional[str] = None
    evaluation_full: Optional[str] = None

    # Deployment labels
    deployment_awareness_intensity: float = 1.0  # 1-10
    deployment_awareness_binary: bool = False  # >= 5
    deployment_choice: Optional[str] = None  # A or B

    # Evaluation labels
    evaluation_awareness_intensity: float = 1.0  # 1-10
    evaluation_awareness_binary: bool = False  # >= 5
    evaluation_choice: Optional[str] = None  # A or B

    # Behavioral change
    behavioral_change: bool = False  # deployment_choice != evaluation_choice

    # Metadata
    category: str = ""  # e.g., "behavioral_change"
    subcategory: str = ""  # e.g., "explicit"
    awareness_level_deploy: str = ""  # e.g., "No Awareness"
    awareness_level_eval: str = ""  # e.g., "Explicit Meta-Recognition"


def extract_prompt_id(source_file: str) -> str:
    """
    Extract unique prompt identifier from source file path.

    Example:
        working/rollouts/.../hallucination_2025-10-28_07-22-41_76c9f8d4_seed_6.yaml
        -> hallucination_2025-10-28_07-22-41_76c9f8d4
    """
    filename = Path(source_file).stem  # Remove .yaml
    # Remove _seed_N suffix
    if '_seed_' in filename:
        prompt_id = filename.rsplit('_seed_', 1)[0]
    else:
        prompt_id = filename
    return prompt_id


def extract_seed(source_file: str) -> int:
    """
    Extract seed number from source file path.

    Example:
        hallucination_2025-10-28_07-22-41_76c9f8d4_seed_6.yaml -> 6
    """
    filename = Path(source_file).stem
    if '_seed_' in filename:
        seed_str = filename.rsplit('_seed_', 1)[1]
        try:
            return int(seed_str)
        except ValueError:
            return 0
    return 0


def load_rollout_file(rollout_path: str) -> Optional[Dict]:
    """Load rollout YAML file containing response texts."""
    try:
        with open(rollout_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"Error loading rollout {rollout_path}: {e}")
        return None


def load_categorization_file(cat_path: str, verbose: bool = False) -> Optional[Dict]:
    """Load response categorization YAML file."""
    try:
        with open(cat_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        if verbose:
            print(f"Error loading categorization {cat_path}: {e}")
        return None


def extract_response_pair(cat_data: Dict, base_dir: str = ".", tokenizer=None, verbose: bool = False) -> Optional[ResponsePair]:
    """
    Extract response pair from categorization data.

    Args:
        cat_data: Loaded categorization YAML dict
        base_dir: Base directory for resolving relative paths (default: current dir)
        tokenizer: Tokenizer for chat template formatting (required for full context)
        verbose: Print debug information

    Returns:
        ResponsePair object or None if data is incomplete
    """
    # Get source file path
    source_file = cat_data.get('source_file', '')
    if not source_file:
        if verbose:
            print("No source_file in cat_data")
        return None

    # Resolve absolute path
    # Source files in categorization are already relative to project root
    if not os.path.isabs(source_file):
        source_file = source_file  # Already correct relative path

    # Load rollout to get response texts
    rollout_data = load_rollout_file(source_file)
    if not rollout_data:
        return None

    # Extract system prompt
    system_prompt = rollout_data.get('system_prompt', '')

    # Extract deployment response
    deployment = rollout_data.get('deployment', {})
    deployment_text = deployment.get('assistant_response', '')
    deployment_choice = deployment.get('choice')  # May be None
    deployment_user = deployment.get('user_message', '')

    # Extract evaluation response
    evaluation = rollout_data.get('evaluation', {})
    evaluation_text = evaluation.get('assistant_response', '')
    evaluation_choice = evaluation.get('choice')  # May be None
    evaluation_user = evaluation.get('user_message', '')

    if not deployment_text or not evaluation_text:
        return None

    # Build full conversation context using tokenizer's chat template
    deployment_full = None
    evaluation_full = None

    if tokenizer is not None:
        try:
            deployment_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": deployment_user},
                {"role": "assistant", "content": deployment_text}
            ]
            evaluation_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_user},
                {"role": "assistant", "content": evaluation_text}
            ]

            deployment_full = tokenizer.apply_chat_template(
                deployment_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            evaluation_full = tokenizer.apply_chat_template(
                evaluation_messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            if verbose:
                print(f"Error applying chat template for {source_file}: {e}")
            return None

    # Extract labels from categorization
    deploy_awareness = cat_data.get('deployment', {}).get('awareness', {})
    eval_awareness = cat_data.get('evaluation', {}).get('awareness', {})

    deploy_intensity = deploy_awareness.get('awareness_intensity', 1)
    eval_intensity = eval_awareness.get('awareness_intensity', 1)

    # Binary labels (threshold >= 5)
    deploy_binary = deploy_intensity >= 5
    eval_binary = eval_intensity >= 5

    # Behavioral change
    behavioral_change = cat_data.get('behavioral_change', False)

    # Extract category/subcategory from path
    # Example: working/response_categorization/qwen_qwen3-32b/behavioral_change/explicit/file.yaml
    path_parts = Path(cat_data.get('source_file', '')).parts
    category = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    subcategory = path_parts[-2] if len(path_parts) >= 2 else "unknown"

    # Prompt ID and seed
    prompt_id = extract_prompt_id(source_file)
    seed = extract_seed(source_file)

    return ResponsePair(
        prompt_id=prompt_id,
        seed=seed,
        source_file=source_file,
        deployment_text=deployment_text,
        evaluation_text=evaluation_text,
        deployment_full=deployment_full,
        evaluation_full=evaluation_full,
        deployment_awareness_intensity=deploy_intensity,
        deployment_awareness_binary=deploy_binary,
        deployment_choice=deployment_choice,
        evaluation_awareness_intensity=eval_intensity,
        evaluation_awareness_binary=eval_binary,
        evaluation_choice=evaluation_choice,
        behavioral_change=behavioral_change,
        category=category,
        subcategory=subcategory,
        awareness_level_deploy=deploy_awareness.get('awareness_level', 'Unknown'),
        awareness_level_eval=eval_awareness.get('awareness_level', 'Unknown'),
    )


def load_all_response_pairs(
    categorization_dir: str,
    limit: Optional[int] = None,
    verbose: bool = True,
    tokenizer=None
) -> List[ResponsePair]:
    """
    Load all response pairs from categorization directory.

    Args:
        categorization_dir: Path to response_categorization directory
        limit: Optional limit on number of files to process
        verbose: Print progress messages
        tokenizer: Tokenizer for chat template formatting (required for full context)

    Returns:
        List of ResponsePair objects
    """
    pairs = []
    file_count = 0

    # Walk directory tree
    for root, dirs, files in os.walk(categorization_dir):
        for filename in files:
            if not filename.endswith('.yaml'):
                continue

            file_count += 1
            if limit and file_count > limit:
                break

            cat_path = os.path.join(root, filename)
            cat_data = load_categorization_file(cat_path, verbose=verbose)

            if cat_data:
                pair = extract_response_pair(cat_data, tokenizer=tokenizer, verbose=verbose)
                if pair:
                    pairs.append(pair)

            if verbose and file_count % 100 == 0:
                print(f"Processed {file_count} files, loaded {len(pairs)} pairs")

        if limit and file_count >= limit:
            break

    if verbose:
        print(f"\nTotal: Processed {file_count} files, loaded {len(pairs)} response pairs")

    return pairs


def split_by_prompt(
    pairs: List[ResponsePair],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[ResponsePair], List[ResponsePair], List[ResponsePair]]:
    """
    Split response pairs by prompt ID to prevent data leakage.

    All seeds of the same prompt go into the same split.

    Args:
        pairs: List of ResponsePair objects
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Group pairs by prompt_id
    prompt_groups = defaultdict(list)
    for pair in pairs:
        prompt_groups[pair.prompt_id].append(pair)

    # Get unique prompt IDs and shuffle
    prompt_ids = list(prompt_groups.keys())
    rng = np.random.RandomState(random_seed)
    rng.shuffle(prompt_ids)

    # Calculate split indices
    n_prompts = len(prompt_ids)
    train_end = int(n_prompts * train_ratio)
    val_end = train_end + int(n_prompts * val_ratio)

    # Split prompt IDs
    train_prompts = prompt_ids[:train_end]
    val_prompts = prompt_ids[train_end:val_end]
    test_prompts = prompt_ids[val_end:]

    # Collect all pairs for each split
    train_pairs = [pair for pid in train_prompts for pair in prompt_groups[pid]]
    val_pairs = [pair for pid in val_prompts for pair in prompt_groups[pid]]
    test_pairs = [pair for pid in test_prompts for pair in prompt_groups[pid]]

    print(f"\nSplit by prompt:")
    print(f"  Train: {len(train_prompts)} prompts, {len(train_pairs)} pairs")
    print(f"  Val:   {len(val_prompts)} prompts, {len(val_pairs)} pairs")
    print(f"  Test:  {len(test_prompts)} prompts, {len(test_pairs)} pairs")

    return train_pairs, val_pairs, test_pairs


def get_label_statistics(pairs: List[ResponsePair]) -> Dict:
    """
    Compute label distribution statistics.

    Args:
        pairs: List of ResponsePair objects

    Returns:
        Dictionary with label counts and percentages
    """
    n_total = len(pairs)

    # Binary awareness counts
    n_deploy_aware = sum(p.deployment_awareness_binary for p in pairs)
    n_eval_aware = sum(p.evaluation_awareness_binary for p in pairs)
    n_behavioral_change = sum(p.behavioral_change for p in pairs)

    # Intensity statistics
    deploy_intensities = [p.deployment_awareness_intensity for p in pairs]
    eval_intensities = [p.evaluation_awareness_intensity for p in pairs]

    stats = {
        'total_pairs': n_total,
        'deployment_aware': {
            'count': n_deploy_aware,
            'percentage': 100 * n_deploy_aware / n_total if n_total > 0 else 0
        },
        'evaluation_aware': {
            'count': n_eval_aware,
            'percentage': 100 * n_eval_aware / n_total if n_total > 0 else 0
        },
        'behavioral_change': {
            'count': n_behavioral_change,
            'percentage': 100 * n_behavioral_change / n_total if n_total > 0 else 0
        },
        'deployment_intensity': {
            'mean': np.mean(deploy_intensities) if deploy_intensities else 0.0,
            'std': np.std(deploy_intensities) if deploy_intensities else 0.0,
            'min': np.min(deploy_intensities) if deploy_intensities else 0.0,
            'max': np.max(deploy_intensities) if deploy_intensities else 0.0,
        },
        'evaluation_intensity': {
            'mean': np.mean(eval_intensities) if eval_intensities else 0.0,
            'std': np.std(eval_intensities) if eval_intensities else 0.0,
            'min': np.min(eval_intensities) if eval_intensities else 0.0,
            'max': np.max(eval_intensities) if eval_intensities else 0.0,
        }
    }

    return stats


def print_label_statistics(pairs: List[ResponsePair], split_name: str = "Dataset"):
    """Print formatted label statistics."""
    stats = get_label_statistics(pairs)

    print(f"\n{split_name} Label Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"\n  Deployment aware: {stats['deployment_aware']['count']} ({stats['deployment_aware']['percentage']:.1f}%)")
    print(f"  Evaluation aware: {stats['evaluation_aware']['count']} ({stats['evaluation_aware']['percentage']:.1f}%)")
    print(f"  Behavioral change: {stats['behavioral_change']['count']} ({stats['behavioral_change']['percentage']:.1f}%)")
    print(f"\n  Deployment intensity: {stats['deployment_intensity']['mean']:.2f} ± {stats['deployment_intensity']['std']:.2f}")
    print(f"  Evaluation intensity: {stats['evaluation_intensity']['mean']:.2f} ± {stats['evaluation_intensity']['std']:.2f}")


if __name__ == "__main__":
    # Example usage
    categorization_dir = "working/response_categorization/v1/qwen_qwen3-32b"

    print("Loading response pairs...")
    pairs = load_all_response_pairs(categorization_dir, limit=100, verbose=True)

    print_label_statistics(pairs, "Full Dataset")

    print("\nSplitting data...")
    train, val, test = split_by_prompt(pairs)

    print_label_statistics(train, "Training Set")
    print_label_statistics(val, "Validation Set")
    print_label_statistics(test, "Test Set")

    print("\nExample response pair:")
    if pairs:
        p = pairs[0]
        print(f"  Prompt ID: {p.prompt_id}")
        print(f"  Seed: {p.seed}")
        print(f"  Category: {p.category}/{p.subcategory}")
        print(f"  Deployment awareness: {p.deployment_awareness_intensity} ({'aware' if p.deployment_awareness_binary else 'not aware'})")
        print(f"  Evaluation awareness: {p.evaluation_awareness_intensity} ({'aware' if p.evaluation_awareness_binary else 'not aware'})")
        print(f"  Behavioral change: {p.behavioral_change}")
        print(f"  Deployment text (first 100 chars): {p.deployment_text[:100]}...")
        print(f"  Evaluation text (first 100 chars): {p.evaluation_text[:100]}...")
