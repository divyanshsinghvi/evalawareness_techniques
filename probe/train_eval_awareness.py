"""Train probe to predict evaluation awareness from evaluation activations only."""

import sys
import pickle
import numpy as np
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Parse arguments
parser = argparse.ArgumentParser(description='Train awareness probe with configurable regions')
parser.add_argument('--activation-file', type=str,
                   default='probe/activations/qwen32b_adaptive_full_checkpoint_599.pkl',
                   help='Path to activation checkpoint file')
parser.add_argument('--region', type=str, default='combined',
                   choices=['first_10', 'thinking', 'thinking_open_tag', 'thinking_close_tag', 'thinking_first_n', 'post_thinking', 'combined', 'all'],
                   help='Which activation region(s) to use for features')
parser.add_argument('--awareness-threshold', type=int, default=5,
                   help='Awareness intensity threshold (default: 5)')
parser.add_argument('--no-projection', action='store_true',
                   help='Use raw activations without steering vector projection')
parser.add_argument('--combine-features', action='store_true',
                   help='Use BOTH raw activations AND steering vector projections (concatenated)')
parser.add_argument('--elementwise-projection', action='store_true',
                   help='Use element-wise multiplication with steering vector (5120-dim weighted activations)')
args = parser.parse_args()

print("="*80)
print("PREDICT EVALUATION AWARENESS FROM EVALUATION ACTIVATIONS")
print("="*80)
print(f"Region mode: {args.region}")
print(f"Awareness threshold: >= {args.awareness_threshold}")
if args.combine_features:
    print(f"Projection mode: Combined (Raw + Projected)")
elif args.elementwise_projection:
    print(f"Projection mode: Element-wise multiplication with steering vector")
elif args.no_projection:
    print(f"Projection mode: Raw activations (no projection)")
else:
    print(f"Projection mode: Steering vector projection (dot product)")
print("="*80)

# Load activations
print("\n[1/5] Loading activations...")
activation_file = args.activation_file
print(f"  Loading from: {activation_file}")
with open(activation_file, 'rb') as f:
    data = pickle.load(f)

deploy_acts = data['deploy_activations']
eval_acts = data['eval_activations']
pairs = data['pairs']
layers = data['layers']

print(f"  Loaded {len(pairs)} pairs")
print(f"  Layers: {layers}")

# Extract features from both deployment and evaluation activations
print("\n[2/5] Extracting features...")

if args.no_projection and not args.combine_features and not args.elementwise_projection:
    # Raw activation mode - no steering vectors needed
    print("  Using raw activations (no steering vectors)")
    steering_vectors = None
else:
    # Projection, combined, or elementwise mode - load steering vectors
    from probe.response_features import load_steering_vectors

    steering_vectors_obj = load_steering_vectors(
        steering_dir='working/steering-vecs',
        method='anthropic',
        model_name='qwen_qwen3-32b'
    )
    steering_vectors = steering_vectors_obj.vectors

    print(f"  Loaded {len(steering_vectors)} steering vectors")
    if args.combine_features:
        print("  Will concatenate raw activations + projections")
    elif args.elementwise_projection:
        print("  Will use element-wise multiplication (activation * steering_vec)")

# Determine which regions to use
if args.region == 'all':
    regions_to_use = ['first_10', 'thinking', 'thinking_open_tag', 'thinking_close_tag', 'thinking_first_n', 'post_thinking', 'combined']
else:
    regions_to_use = [args.region]

print(f"  Using regions: {regions_to_use}")

# Project activations
n_pairs = len(pairs)  # Use pairs length, not activations (can have mismatch)
n_layers = len(layers)
n_regions = len(regions_to_use)
aggregations = ['mean', 'max', 'std']
n_aggs = len(aggregations)

if args.combine_features:
    # Combined mode: raw (5120-dim) + projected (3 scalars)
    print("  Note: Combining raw 'mean' (5120 dims) + projections (mean/max/std)")
    aggregations_to_use = ['mean']  # For raw component
    n_features_raw = n_layers * n_regions * 5120
    n_features_proj = n_layers * n_regions * n_aggs
    n_features = n_features_raw + n_features_proj
elif args.elementwise_projection:
    # Element-wise multiplication: weighted 5120-dim vectors
    print("  Note: Using element-wise weighted 'mean' (5120 dims per layer)")
    aggregations_to_use = ['mean']
    n_features = n_layers * n_regions * 5120  # Weighted 5120-dim vectors
elif args.no_projection:
    # Raw activations: each aggregation is 5120-dimensional
    # For now, just use mean aggregation to keep dimensionality manageable
    print("  Note: Using only 'mean' aggregation for raw activations (5120 dims per layer)")
    aggregations_to_use = ['mean']
    n_features = n_layers * n_regions * 5120  # Full 5120-dim vectors
else:
    # Projected: scalar per aggregation
    aggregations_to_use = aggregations
    n_features = n_layers * n_regions * n_aggs

X_deploy = np.zeros((n_pairs, n_features))
X_eval = np.zeros((n_pairs, n_features))
feature_names_deploy = []
feature_names_eval = []

# Generate feature names
if args.combine_features:
    # Combined: raw features first, then projected features
    for region in regions_to_use:
        for layer in layers:
            for dim in range(5120):
                feature_names_deploy.append(f"deploy_{region}_L{layer}_mean_d{dim}")
                feature_names_eval.append(f"eval_{region}_L{layer}_mean_d{dim}")
    for region in regions_to_use:
        for layer in layers:
            for agg in aggregations:
                feature_names_deploy.append(f"deploy_{region}_L{layer}_{agg}_proj")
                feature_names_eval.append(f"eval_{region}_L{layer}_{agg}_proj")
elif args.elementwise_projection:
    # Element-wise: weighted mean features
    for region in regions_to_use:
        for layer in layers:
            for dim in range(5120):
                feature_names_deploy.append(f"deploy_{region}_L{layer}_mean_weighted_d{dim}")
                feature_names_eval.append(f"eval_{region}_L{layer}_mean_weighted_d{dim}")
elif args.no_projection:
    for region in regions_to_use:
        for layer in layers:
            for dim in range(5120):
                feature_names_deploy.append(f"deploy_{region}_L{layer}_mean_d{dim}")
                feature_names_eval.append(f"eval_{region}_L{layer}_mean_d{dim}")
else:
    for region in regions_to_use:
        for layer in layers:
            for agg in aggregations:
                feature_names_deploy.append(f"deploy_{region}_L{layer}_{agg}")
                feature_names_eval.append(f"eval_{region}_L{layer}_{agg}")

# Compute projections
for i in range(n_pairs):
    deploy_act = deploy_acts[i]
    eval_act = eval_acts[i]

    feature_idx = 0

    if args.elementwise_projection:
        # Element-wise mode: multiply activations by steering vector element-wise
        for region in regions_to_use:
            for layer in layers:
                deploy_layer = deploy_act.get(layer, {})
                eval_layer = eval_act.get(layer, {})
                deploy_agg_acts = deploy_layer.get(region, {})
                eval_agg_acts = eval_layer.get(region, {})

                # Get steering vector for this layer
                if layer in steering_vectors:
                    steering_vec = steering_vectors[layer]
                else:
                    steering_vec = np.ones(5120)  # Fallback to no weighting

                # Extract mean and multiply element-wise with steering vector
                if deploy_agg_acts and 'mean' in deploy_agg_acts:
                    deploy_vec = deploy_agg_acts['mean'] * steering_vec
                else:
                    deploy_vec = np.zeros(5120)

                if eval_agg_acts and 'mean' in eval_agg_acts:
                    eval_vec = eval_agg_acts['mean'] * steering_vec
                else:
                    eval_vec = np.zeros(5120)

                X_deploy[i, feature_idx:feature_idx+5120] = deploy_vec
                X_eval[i, feature_idx:feature_idx+5120] = eval_vec
                feature_idx += 5120
    elif args.combine_features:
        # PHASE 1: Extract raw features (5120-dim mean per layer per region)
        for region in regions_to_use:
            for layer in layers:
                deploy_layer = deploy_act.get(layer, {})
                eval_layer = eval_act.get(layer, {})
                deploy_agg_acts = deploy_layer.get(region, {})
                eval_agg_acts = eval_layer.get(region, {})

                if deploy_agg_acts and 'mean' in deploy_agg_acts:
                    deploy_vec = deploy_agg_acts['mean']
                else:
                    deploy_vec = np.zeros(5120)

                if eval_agg_acts and 'mean' in eval_agg_acts:
                    eval_vec = eval_agg_acts['mean']
                else:
                    eval_vec = np.zeros(5120)

                X_deploy[i, feature_idx:feature_idx+5120] = deploy_vec
                X_eval[i, feature_idx:feature_idx+5120] = eval_vec
                feature_idx += 5120

        # PHASE 2: Add projected features (3 scalars per layer per region)
        for region in regions_to_use:
            for layer in layers:
                deploy_layer = deploy_act.get(layer, {})
                eval_layer = eval_act.get(layer, {})
                deploy_agg_acts = deploy_layer.get(region, {})
                eval_agg_acts = eval_layer.get(region, {})

                if layer not in steering_vectors:
                    feature_idx += n_aggs
                    continue

                steering_vec = steering_vectors[layer]

                for agg in aggregations:
                    if deploy_agg_acts and agg in deploy_agg_acts:
                        deploy_vec = deploy_agg_acts[agg]
                        deploy_proj = np.dot(deploy_vec, steering_vec)
                    else:
                        deploy_proj = 0.0
                    X_deploy[i, feature_idx] = deploy_proj

                    if eval_agg_acts and agg in eval_agg_acts:
                        eval_vec = eval_agg_acts[agg]
                        eval_proj = np.dot(eval_vec, steering_vec)
                    else:
                        eval_proj = 0.0
                    X_eval[i, feature_idx] = eval_proj

                    feature_idx += 1
    else:
        # Original logic for no_projection or projection only
        for region in regions_to_use:
            for layer in layers:
                # Get layer data (regional format)
                deploy_layer = deploy_act.get(layer, {})
                eval_layer = eval_act.get(layer, {})

                # Extract features from this region
                deploy_agg_acts = deploy_layer.get(region, {})
                eval_agg_acts = eval_layer.get(region, {})

                if args.no_projection:
                    # Raw activation mode - use full 5120-dim vector
                    if deploy_agg_acts and 'mean' in deploy_agg_acts:
                        deploy_vec = deploy_agg_acts['mean']
                    else:
                        deploy_vec = np.zeros(5120)

                    if eval_agg_acts and 'mean' in eval_agg_acts:
                        eval_vec = eval_agg_acts['mean']
                    else:
                        eval_vec = np.zeros(5120)

                    # Store full 5120-dim vector
                    X_deploy[i, feature_idx:feature_idx+5120] = deploy_vec
                    X_eval[i, feature_idx:feature_idx+5120] = eval_vec
                    feature_idx += 5120
                else:
                    # Projection mode - dot product with steering vectors
                    if layer not in steering_vectors:
                        # Skip this layer but advance feature index
                        feature_idx += n_aggs
                        continue

                    steering_vec = steering_vectors[layer]

                    for agg in aggregations:
                        # Deployment
                        if deploy_agg_acts and agg in deploy_agg_acts:
                            deploy_vec = deploy_agg_acts[agg]
                            deploy_proj = np.dot(deploy_vec, steering_vec)
                        else:
                            deploy_proj = 0.0
                        X_deploy[i, feature_idx] = deploy_proj

                        # Evaluation
                        if eval_agg_acts and agg in eval_agg_acts:
                            eval_vec = eval_agg_acts[agg]
                            eval_proj = np.dot(eval_vec, steering_vec)
                        else:
                            eval_proj = 0.0
                        X_eval[i, feature_idx] = eval_proj

                        feature_idx += 1

print(f"  Deployment feature matrix shape: {X_deploy.shape}")
print(f"  Evaluation feature matrix shape: {X_eval.shape}")
print(f"  Number of features per context: {len(feature_names_eval)}")

# Extract multiple targets
print("\n[3/5] Extracting labels...")

# Extract labels for both deployment and evaluation
deploy_aware_binary = []
eval_aware_binary = []
deploy_choices = []
eval_choices = []
behavioral_changes = []

for pair in pairs:
    # Deployment awareness
    if hasattr(pair, 'deployment_awareness_intensity'):
        deploy_intensity = pair.deployment_awareness_intensity
    else:
        deploy_intensity = 1

    deploy_aware_binary.append(1 if deploy_intensity >= args.awareness_threshold else 0)

    # Evaluation awareness
    if hasattr(pair, 'evaluation_awareness_intensity'):
        eval_intensity = pair.evaluation_awareness_intensity
    else:
        eval_intensity = 1

    eval_aware_binary.append(1 if eval_intensity >= args.awareness_threshold else 0)

    # Deployment choice
    if hasattr(pair, 'deployment_choice'):
        deploy_choices.append(1 if pair.deployment_choice == 'B' else 0)
    else:
        deploy_choices.append(None)

    # Evaluation choice
    if hasattr(pair, 'evaluation_choice'):
        eval_choices.append(1 if pair.evaluation_choice == 'B' else 0)
    else:
        eval_choices.append(None)

    behavioral_changes.append(pair.behavioral_change)

y_deploy_aware = np.array(deploy_aware_binary)
y_eval_aware = np.array(eval_aware_binary)
y_bc = np.array(behavioral_changes)

print(f"\nLabel distributions:")
print(f"  Deployment aware (intensity >= {args.awareness_threshold}): {y_deploy_aware.sum()} / {len(y_deploy_aware)} ({100*y_deploy_aware.mean():.1f}%)")
print(f"  Evaluation aware (intensity >= {args.awareness_threshold}): {y_eval_aware.sum()} / {len(y_eval_aware)} ({100*y_eval_aware.mean():.1f}%)")
print(f"  Behavioral change: {y_bc.sum()} / {len(y_bc)} ({100*y_bc.mean():.1f}%)")

has_deploy_choices = None not in deploy_choices
has_eval_choices = None not in eval_choices

if has_deploy_choices:
    y_deploy_choice = np.array(deploy_choices)
    print(f"  Deployment choice B: {y_deploy_choice.sum()} / {len(y_deploy_choice)} ({100*y_deploy_choice.mean():.1f}%)")
else:
    print(f"  Deployment choices: Not available")

if has_eval_choices:
    y_eval_choice = np.array(eval_choices)
    print(f"  Evaluation choice B: {y_eval_choice.sum()} / {len(y_eval_choice)} ({100*y_eval_choice.mean():.1f}%)")
else:
    print(f"  Evaluation choices: Not available")

# Split by prompt ID (50/50)
print("\n[4/5] Splitting by prompt ID (50/50)...")
prompt_groups = defaultdict(list)
for i, pair in enumerate(pairs):
    prompt_groups[pair.prompt_id].append(i)

prompt_ids = list(prompt_groups.keys())
n_prompts = len(prompt_ids)

print(f"  Total prompts: {n_prompts}")

np.random.seed(42)
np.random.shuffle(prompt_ids)

train_end = int(n_prompts * 0.5)

train_prompts = prompt_ids[:train_end]
test_prompts = prompt_ids[train_end:]

train_idx = [i for pid in train_prompts for i in prompt_groups[pid]]
test_idx = [i for pid in test_prompts for i in prompt_groups[pid]]

X_deploy_train, X_deploy_test = X_deploy[train_idx], X_deploy[test_idx]
X_eval_train, X_eval_test = X_eval[train_idx], X_eval[test_idx]

y_deploy_aware_train, y_deploy_aware_test = y_deploy_aware[train_idx], y_deploy_aware[test_idx]
y_eval_aware_train, y_eval_aware_test = y_eval_aware[train_idx], y_eval_aware[test_idx]
y_bc_train, y_bc_test = y_bc[train_idx], y_bc[test_idx]

print(f"\nSplit sizes:")
print(f"  Train: {len(train_prompts)} prompts, {len(train_idx)} pairs")
print(f"  Test:  {len(test_prompts)} prompts, {len(test_idx)} pairs")

# Train probes
print("\n[5/5] Training probes...")

results = {}
C = 0.01
MAX_ITER = 10000  # Balanced between convergence and speed

# Probe 1: Predict evaluation awareness
print("\n" + "="*80)
print(f"PROBE 1: EVALUATION AWARENESS (intensity >= {args.awareness_threshold})")
print("="*80)
probe_awareness = LogisticRegression(C=C, class_weight='balanced', max_iter=MAX_ITER, random_state=42)
probe_awareness.fit(X_eval_train, y_eval_aware_train)

for split_name, X_split, y_split in [("Train", X_eval_train, y_eval_aware_train), ("Test", X_eval_test, y_eval_aware_test)]:
    if len(np.unique(y_split)) > 1:
        y_pred = probe_awareness.predict(X_split)
        y_proba = probe_awareness.predict_proba(X_split)[:, 1]

        acc = accuracy_score(y_split, y_pred)
        auroc = roc_auc_score(y_split, y_proba)
        precision = precision_score(y_split, y_pred, zero_division=0)
        recall = recall_score(y_split, y_pred, zero_division=0)
        f1 = f1_score(y_split, y_pred, zero_division=0)

        print(f"\n{split_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  AUROC:     {auroc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Base rate: {y_split.mean():.1%}")

        if split_name == "Test":
            results['awareness_auroc'] = auroc
            results['awareness_acc'] = acc

# Probe 2: Predict behavioral change (for comparison)
print("\n" + "="*80)
print("PROBE 2: BEHAVIORAL CHANGE (from eval activations)")
print("="*80)
probe_bc = LogisticRegression(C=C, class_weight='balanced', max_iter=MAX_ITER, random_state=42)
probe_bc.fit(X_eval_train, y_bc_train)

for split_name, X_split, y_split in [("Train", X_eval_train, y_bc_train), ("Test", X_eval_test, y_bc_test)]:
    if len(np.unique(y_split)) > 1:
        y_pred = probe_bc.predict(X_split)
        y_proba = probe_bc.predict_proba(X_split)[:, 1]

        acc = accuracy_score(y_split, y_pred)
        auroc = roc_auc_score(y_split, y_proba)
        f1 = f1_score(y_split, y_pred, zero_division=0)

        print(f"\n{split_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  AUROC:     {auroc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Base rate: {y_split.mean():.1%}")

        if split_name == "Test":
            results['bc_auroc'] = auroc

# Probe 3: Predict evaluation choice (if available)
if has_eval_choices:
    print("\n" + "="*80)
    print("PROBE 3: EVALUATION CHOICE (A vs B)")
    print("="*80)
    y_eval_choice_train, y_eval_choice_test = y_eval_choice[train_idx], y_eval_choice[test_idx]

    probe_choice = LogisticRegression(C=C, class_weight='balanced', max_iter=MAX_ITER, random_state=42)
    probe_choice.fit(X_eval_train, y_eval_choice_train)

    for split_name, X_split, y_split in [("Train", X_eval_train, y_eval_choice_train), ("Test", X_eval_test, y_eval_choice_test)]:
        if len(np.unique(y_split)) > 1:
            y_pred = probe_choice.predict(X_split)
            y_proba = probe_choice.predict_proba(X_split)[:, 1]

            acc = accuracy_score(y_split, y_pred)
            precision = precision_score(y_split, y_pred, zero_division=0)
            recall = recall_score(y_split, y_pred, zero_division=0)
            f1 = f1_score(y_split, y_pred, zero_division=0)
            auroc = roc_auc_score(y_split, y_proba)

            print(f"\n{split_name}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  AUROC:     {auroc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1:        {f1:.4f}")
            print(f"  Base rate (choice B): {y_split.mean():.1%}")

            if split_name == "Test":
                results['choice_auroc'] = auroc
                results['choice_acc'] = acc

# Feature importance for awareness probe
print("\n" + "="*80)
print("TOP 20 FEATURES FOR EVALUATION AWARENESS")
print("="*80)

coeffs = probe_awareness.coef_[0]
top_idx = np.argsort(np.abs(coeffs))[-20:][::-1]

for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2d}. {feature_names_eval[idx]:30s}: {coeffs[idx]:+.4f}")

# Summary with baselines
print("\n" + "="*80)
print("SUMMARY: PROBE PERFORMANCE vs RANDOM BASELINE")
print("="*80)

# Random baseline AUROC = 0.5 (random guessing)
# Random baseline accuracy depends on class balance
test_aware_rate = y_eval_aware_test.mean()
test_bc_rate = y_bc_test.mean()
test_choice_rate = y_eval_choice_test.mean() if has_eval_choices else 0

print("\n" + "-"*80)
print("PROBE 1: EVALUATION AWARENESS")
print("-"*80)
test_aware_auroc = results.get('awareness_auroc', 0)
test_aware_acc = results.get('awareness_acc', 0)
random_acc_aware = max(test_aware_rate, 1 - test_aware_rate)  # Always predict majority class
print(f"  Probe Test AUROC:     {test_aware_auroc:.4f}")
print(f"  Random Baseline AUROC: 0.5000 (random guessing)")
print(f"  Improvement:          {test_aware_auroc - 0.5:+.4f} ({100*(test_aware_auroc - 0.5)/0.5:.1f}%)")
print(f"\n  Probe Test Accuracy:  {test_aware_acc:.4f}")
print(f"  Random Baseline Acc:   {random_acc_aware:.4f} (always predict majority)")
print(f"  Improvement:          {test_aware_acc - random_acc_aware:+.4f}")

print("\n" + "-"*80)
print("PROBE 2: BEHAVIORAL CHANGE")
print("-"*80)
test_bc_auroc = results.get('bc_auroc', 0)
random_acc_bc = max(test_bc_rate, 1 - test_bc_rate)
print(f"  Probe Test AUROC:     {test_bc_auroc:.4f}")
print(f"  Random Baseline AUROC: 0.5000")
print(f"  Improvement:          {test_bc_auroc - 0.5:+.4f} ({100*(test_bc_auroc - 0.5)/0.5:.1f}%)")
print(f"\n  Baseline (pseudo-embeddings): 0.6920")
print(f"  Delta from baseline:  {test_bc_auroc - 0.6920:+.4f}")

if has_eval_choices:
    print("\n" + "-"*80)
    print("PROBE 3: EVALUATION CHOICE (A vs B)")
    print("-"*80)
    test_choice_auroc = results.get('choice_auroc', 0)
    test_choice_acc = results.get('choice_acc', 0)
    random_acc_choice = max(test_choice_rate, 1 - test_choice_rate)
    print(f"  Probe Test AUROC:     {test_choice_auroc:.4f}")
    print(f"  Random Baseline AUROC: 0.5000")
    print(f"  Improvement:          {test_choice_auroc - 0.5:+.4f} ({100*(test_choice_auroc - 0.5)/0.5:.1f}%)")
    print(f"\n  Probe Test Accuracy:  {test_choice_acc:.4f}")
    print(f"  Random Baseline Acc:   {random_acc_choice:.4f} (always predict majority)")
    print(f"  Improvement:          {test_choice_acc - random_acc_choice:+.4f}")

print("\n" + "="*80)
