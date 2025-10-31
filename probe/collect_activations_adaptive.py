"""
Collect activations adaptively for response pairs using smart token sampling.

Strategy:
1. First 10 assistant tokens (response start context)
2. All thinking tokens (if <think> or <thinking> tags found)
3. Last 50 tokens (final reasoning state)
4. Every 100th token in between (for very long responses)

This balances speed vs completeness.
"""

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from probe.response_data_prep import load_all_response_pairs, ResponsePair


# Qwen tokenizer thinking tag IDs
# NOTE: These are multi-token sequences, not single tokens
# <thinking> = [13708, 15736, 29]
# </thinking> = [522, 82260, 29]
# <think> = [13708, 766, 29]
# </think> = [522, 26865, 29]
# Also handle variants with newlines: > = 29, >\n = 397, >\n\n = 1339
THINKING_OPEN_SEQUENCES = [
    [13708, 15736, 29],    # <thinking>
    [13708, 15736, 397],   # <thinking>\n
    [13708, 15736, 1339],  # <thinking>\n\n
    [13708, 766, 29],      # <think>
    [13708, 766, 397],     # <think>\n
    [13708, 766, 1339]     # <think>\n\n
]
THINKING_CLOSE_SEQUENCES = [
    [522, 82260, 29],      # </thinking>
    [522, 82260, 397],     # </thinking>\n
    [522, 82260, 1339],    # </thinking>\n\n
    [522, 26865, 29],      # </think>
    [522, 26865, 397],     # </think>\n
    [522, 26865, 1339]     # </think>\n\n
]


@dataclass
class AdaptiveTokenSelection:
    """Selected token indices and their purposes."""

    first_tokens: List[int]      # First 10 assistant tokens
    thinking_tokens: List[int]   # All thinking tokens (content between tags)
    thinking_open_tag_tokens: List[int]  # Just the <thinking> opening tag tokens
    thinking_close_tag_tokens: List[int]  # Just the </thinking> closing tag tokens
    last_tokens: List[int]       # Last 50 tokens
    sample_tokens: List[int]     # Every 100th token in between

    def all_indices(self) -> List[int]:
        """Get all unique selected indices sorted."""
        all_idx = set(self.first_tokens + self.thinking_tokens +
                     self.thinking_open_tag_tokens + self.thinking_close_tag_tokens +
                     self.last_tokens + self.sample_tokens)
        return sorted(list(all_idx))


def find_assistant_start(token_ids: List[int]) -> int:
    """
    Find the index where the assistant's response starts.

    For Qwen chat format, this is after the <|im_start|>assistant token.
    <|im_start|>assistant is tokenized as [151644, 77091]

    Returns:
        Index of first assistant token, or 0 if not found
    """
    # Look for <|im_start|>assistant sequence
    assistant_start_seq = [151644, 77091]  # <|im_start|>assistant

    for i in range(len(token_ids) - 1):
        if token_ids[i:i+2] == assistant_start_seq:
            # Return index right after this sequence
            return i + 2

    # Fallback: return 0 (use all tokens)
    return 0


def find_thinking_tokens(token_ids: List[int]) -> Tuple[int, int, List[int], List[int]]:
    """
    Find start and end indices of thinking tokens, plus the tag token indices separately.

    Returns:
        (start_idx, end_idx, open_tag_indices, close_tag_indices) where:
        - start_idx, end_idx: content between tags (or -1, -1 if not found)
        - open_tag_indices: list of indices for opening <thinking> tag tokens
        - close_tag_indices: list of indices for closing </thinking> tag tokens
    """
    start_idx = -1
    end_idx = -1
    open_tag_indices = []
    close_tag_indices = []

    # Find opening tag (multi-token sequence)
    for i in range(len(token_ids)):
        for open_seq in THINKING_OPEN_SEQUENCES:
            seq_len = len(open_seq)
            if i + seq_len <= len(token_ids):
                if token_ids[i:i+seq_len] == open_seq:
                    start_idx = i + seq_len  # Start after the opening tag
                    # Add opening tag indices
                    open_tag_indices = list(range(i, i + seq_len))
                    break
        if start_idx >= 0:
            break

    # Find closing tag (from the end, multi-token sequence)
    for i in range(len(token_ids) - 1, -1, -1):
        for close_seq in THINKING_CLOSE_SEQUENCES:
            seq_len = len(close_seq)
            if i + seq_len <= len(token_ids):
                if token_ids[i:i+seq_len] == close_seq:
                    end_idx = i  # End before the closing tag
                    # Add closing tag indices
                    close_tag_indices = list(range(i, i + seq_len))
                    break
        if end_idx >= 0:
            break

    # Validate
    if start_idx >= 0 and end_idx > start_idx:
        return start_idx, end_idx, open_tag_indices, close_tag_indices

    return -1, -1, [], []


def select_adaptive_tokens(
    token_ids: List[int],
    first_n: int = 10,
    last_n: int = 50
) -> AdaptiveTokenSelection:
    """
    Adaptively select tokens from response.

    Args:
        token_ids: Full tokenized response
        first_n: Number of first assistant tokens to include
        last_n: Number of last tokens to include

    Returns:
        AdaptiveTokenSelection with selected indices
    """
    n_tokens = len(token_ids)

    # Find where assistant response starts
    assistant_start = find_assistant_start(token_ids)

    # First tokens (from assistant response, not system prompt)
    first_end = min(assistant_start + first_n, n_tokens)
    first_tokens = list(range(assistant_start, first_end))

    # Thinking tokens (all tokens within thinking tags)
    think_start, think_end, open_tag_indices, close_tag_indices = find_thinking_tokens(token_ids)
    if think_start >= 0:
        thinking_tokens = list(range(think_start, think_end))
        thinking_open_tag_tokens = open_tag_indices  # <thinking> tag tokens
        thinking_close_tag_tokens = close_tag_indices  # </thinking> tag tokens
    else:
        thinking_tokens = []
        thinking_open_tag_tokens = []
        thinking_close_tag_tokens = []

    # Last tokens
    last_start = max(0, n_tokens - last_n)
    last_tokens = list(range(last_start, n_tokens))

    # No sampling in middle region anymore
    sample_tokens = []

    return AdaptiveTokenSelection(
        first_tokens=first_tokens,
        thinking_tokens=thinking_tokens,
        thinking_open_tag_tokens=thinking_open_tag_tokens,
        thinking_close_tag_tokens=thinking_close_tag_tokens,
        last_tokens=last_tokens,
        sample_tokens=sample_tokens
    )


def extract_activations_adaptive(
    text: str,
    model,
    tokenizer,
    layers: List[int],
    device: str = "cuda",
    collection_mode: str = "regional"
) -> Dict[int, np.ndarray]:
    """
    Extract activations using adaptive token selection.

    Args:
        text: Response text
        model: Loaded model
        tokenizer: Loaded tokenizer
        layers: Layers to extract
        device: Device for inference

    Returns:
        Dict mapping layer_idx -> mean_activation (d_model,)
    """
    import torch

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=8192,
        truncation=True
    ).to(device)

    token_ids = inputs['input_ids'][0].tolist()

    # Select tokens adaptively
    selection = select_adaptive_tokens(token_ids)
    selected_indices = selection.all_indices()

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract activations at selected tokens
    layer_activations = {}

    for layer_idx in layers:
        # Get hidden state for this layer
        # hidden_states[0] = embeddings, hidden_states[layer_idx+1] = layer output
        hidden_state = outputs.hidden_states[layer_idx + 1]  # (1, seq_len, d_model)
        hidden_state = hidden_state.squeeze(0)  # (seq_len, d_model)

        # Extract activations for different regions
        first_acts = hidden_state[torch.tensor(selection.first_tokens)] if selection.first_tokens else None
        thinking_acts = hidden_state[torch.tensor(selection.thinking_tokens)] if selection.thinking_tokens else None
        thinking_open_tag_acts = hidden_state[torch.tensor(selection.thinking_open_tag_tokens)] if selection.thinking_open_tag_tokens else None
        thinking_close_tag_acts = hidden_state[torch.tensor(selection.thinking_close_tag_tokens)] if selection.thinking_close_tag_tokens else None

        # Post-thinking: last tokens but NOT overlapping with thinking
        # Get tokens after thinking ends (if thinking exists)
        if selection.thinking_tokens:
            think_end = max(selection.thinking_tokens)
            post_thinking_tokens = [t for t in selection.last_tokens if t > think_end]
        else:
            post_thinking_tokens = selection.last_tokens
        post_thinking_acts = hidden_state[torch.tensor(post_thinking_tokens)] if post_thinking_tokens else None

        # Combined: first 10 + thinking
        combined_tokens = selection.first_tokens + selection.thinking_tokens
        combined_acts = hidden_state[torch.tensor(combined_tokens)] if combined_tokens else None

        # Thinking_first_n: Take first N tokens from thinking (min of 100 or half)
        thinking_first_n_tokens = []
        if selection.thinking_tokens:
            n_thinking = len(selection.thinking_tokens)
            # Take min(100, n_thinking // 2) tokens from start of thinking
            n_to_take = min(100, n_thinking // 2)
            thinking_first_n_tokens = selection.thinking_tokens[:n_to_take]
        thinking_first_n_acts = hidden_state[torch.tensor(thinking_first_n_tokens)] if thinking_first_n_tokens else None

        # Helper function to compute aggregations
        def compute_aggs(acts):
            if acts is None or len(acts) == 0:
                return None
            return {
                'mean': acts.mean(dim=0).cpu().float().numpy(),
                'max': acts.abs().max(dim=0)[0].cpu().float().numpy(),
                'std': acts.std(dim=0).cpu().float().numpy()
            }

        # Store regional aggregations
        layer_activations[layer_idx] = {
            'first_10': compute_aggs(first_acts),
            'thinking': compute_aggs(thinking_acts),
            'thinking_open_tag': compute_aggs(thinking_open_tag_acts),
            'thinking_close_tag': compute_aggs(thinking_close_tag_acts),
            'thinking_first_n': compute_aggs(thinking_first_n_acts),
            'post_thinking': compute_aggs(post_thinking_acts),
            'combined': compute_aggs(combined_acts)
        }

    return layer_activations


def extract_activations_batch(
    texts: List[str],
    model,
    tokenizer,
    layers: List[int],
    device: str = "cuda",
    max_length: int = 8192
) -> List[Dict[int, np.ndarray]]:
    """
    Extract activations for a batch of texts (NO padding if lengths differ too much).

    Args:
        texts: List of response texts
        model: Loaded model
        tokenizer: Loaded tokenizer
        layers: Layers to extract
        device: Device for inference
        max_length: Max sequence length

    Returns:
        List of dicts mapping layer_idx -> mean_activation
    """
    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True  # Pad to longest in batch
    ).to(device)

    batch_size = inputs['input_ids'].shape[0]

    # Get token selections for each item
    selections = []
    for i in range(batch_size):
        token_ids = inputs['input_ids'][i].tolist()
        # Remove padding tokens (assuming tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None:
            token_ids = [t for t in token_ids if t != tokenizer.pad_token_id]
        selection = select_adaptive_tokens(token_ids)
        selections.append(selection)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract activations for each item in batch
    batch_activations = []

    for i in range(batch_size):
        layer_activations = {}
        selection = selections[i]

        for layer_idx in layers:
            hidden_state = outputs.hidden_states[layer_idx + 1]  # (batch, seq_len, d_model)
            item_hidden = hidden_state[i]  # (seq_len, d_model)

            # Extract activations for different regions (SAME as extract_activations_adaptive)
            first_acts = item_hidden[torch.tensor(selection.first_tokens)] if selection.first_tokens else None
            thinking_acts = item_hidden[torch.tensor(selection.thinking_tokens)] if selection.thinking_tokens else None

            # Post-thinking: last tokens but NOT overlapping with thinking
            if selection.thinking_tokens:
                think_end = max(selection.thinking_tokens)
                post_thinking_tokens = [t for t in selection.last_tokens if t > think_end]
            else:
                post_thinking_tokens = selection.last_tokens
            post_thinking_acts = item_hidden[torch.tensor(post_thinking_tokens)] if post_thinking_tokens else None

            # Combined: first 10 + thinking
            combined_tokens = selection.first_tokens + selection.thinking_tokens
            combined_acts = item_hidden[torch.tensor(combined_tokens)] if combined_tokens else None

            # Thinking_first_n: Take first N tokens from thinking (min of 100 or half)
            thinking_first_n_tokens = []
            if selection.thinking_tokens:
                n_thinking = len(selection.thinking_tokens)
                n_to_take = min(100, n_thinking // 2)
                thinking_first_n_tokens = selection.thinking_tokens[:n_to_take]
            thinking_first_n_acts = item_hidden[torch.tensor(thinking_first_n_tokens)] if thinking_first_n_tokens else None

            # Helper function to compute aggregations
            def compute_aggs(acts):
                if acts is None or len(acts) == 0:
                    return None
                return {
                    'mean': acts.mean(dim=0).cpu().float().numpy(),
                    'max': acts.abs().max(dim=0)[0].cpu().float().numpy(),
                    'std': acts.std(dim=0).cpu().float().numpy()
                }

            # Store regional aggregations (SAME FORMAT as single processing)
            layer_activations[layer_idx] = {
                'first_10': compute_aggs(first_acts),
                'thinking': compute_aggs(thinking_acts),
                'thinking_first_n': compute_aggs(thinking_first_n_acts),
                'post_thinking': compute_aggs(post_thinking_acts),
                'combined': compute_aggs(combined_acts)
            }

        batch_activations.append(layer_activations)

    return batch_activations


def collect_activations_for_dataset(
    pairs: List[ResponsePair],
    model,
    tokenizer,
    layers: List[int],
    output_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    save_every: int = 50,
    verbose: bool = False,
    start_offset: int = 0,
    existing_deploy_acts: list = None,
    existing_eval_acts: list = None,
    all_pairs: List[ResponsePair] = None
):
    """
    Collect activations for all response pairs with smart batching.

    Args:
        pairs: List of ResponsePair objects
        model: Loaded model
        tokenizer: Loaded tokenizer
        layers: Layers to extract
        output_path: Where to save
        device: Device for inference
        batch_size: Batch size (1=no batching, 2+=batching with auto-fallback)
        save_every: Save checkpoint every N pairs
        verbose: Print full conversation texts
    """
    print(f"\nCollecting activations for {len(pairs)} response pairs...")
    print(f"Layers: {layers}")
    print(f"Strategy: adaptive (first 10 + thinking + last 50 + sample)")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Print example prompt if verbose
    if verbose and len(pairs) > 0:
        print("\n" + "="*80)
        print("EXAMPLE DEPLOYMENT CONVERSATION:")
        print("="*80)
        print(pairs[0].deployment_full[:500] + "..." if len(pairs[0].deployment_full) > 500 else pairs[0].deployment_full)
        print("\n" + "="*80)
        print("EXAMPLE EVALUATION CONVERSATION:")
        print("="*80)
        print(pairs[0].evaluation_full[:500] + "..." if len(pairs[0].evaluation_full) > 500 else pairs[0].evaluation_full)
        print("="*80 + "\n")

    # Initialize with existing activations if resuming
    if existing_deploy_acts is None:
        existing_deploy_acts = []
    if existing_eval_acts is None:
        existing_eval_acts = []
    if all_pairs is None:
        all_pairs = pairs

    deploy_acts_all = []  # List of dicts: {layer_idx: activation}
    eval_acts_all = []

    # Process in batches if batch_size > 1
    if batch_size > 1:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Processing batches"):
            batch_pairs = pairs[i:i+batch_size]

            try:
                # Collect full conversation texts (system + user + assistant)
                deploy_texts = [p.deployment_full for p in batch_pairs]
                eval_texts = [p.evaluation_full for p in batch_pairs]

                # Check if texts are similar length (within 2x)
                deploy_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in deploy_texts]
                eval_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in eval_texts]

                deploy_ratio = max(deploy_lens) / (min(deploy_lens) + 1)
                eval_ratio = max(eval_lens) / (min(eval_lens) + 1)

                # Use batching if lengths are similar, otherwise fall back to single
                if deploy_ratio < 2.0 and len(batch_pairs) > 1:
                    # Batch process
                    deploy_batch_acts = extract_activations_batch(
                        deploy_texts, model, tokenizer, layers, device
                    )
                    eval_batch_acts = extract_activations_batch(
                        eval_texts, model, tokenizer, layers, device
                    )

                    deploy_acts_all.extend(deploy_batch_acts)
                    eval_acts_all.extend(eval_batch_acts)
                else:
                    # Fall back to single processing
                    for pair in batch_pairs:
                        deploy_acts = extract_activations_adaptive(
                            pair.deployment_full, model, tokenizer, layers, device
                        )
                        eval_acts = extract_activations_adaptive(
                            pair.evaluation_full, model, tokenizer, layers, device
                        )
                        deploy_acts_all.append(deploy_acts)
                        eval_acts_all.append(eval_acts)

            except Exception as e:
                print(f"\nError processing batch at index {i}: {e}")
                # Add empty activations for failed batch
                for _ in batch_pairs:
                    empty_acts = {layer: np.zeros(model.config.hidden_size) for layer in layers}
                    deploy_acts_all.append(empty_acts)
                    eval_acts_all.append(empty_acts)
                continue

            # Save checkpoint
            if (len(deploy_acts_all)) % save_every == 0:
                total_processed = start_offset + len(deploy_acts_all)
                checkpoint_path = output_path.replace('.pkl', f'_checkpoint_{total_processed}.pkl')
                # Combine existing + new activations for checkpoint
                combined_deploy = existing_deploy_acts + deploy_acts_all
                combined_eval = existing_eval_acts + eval_acts_all
                checkpoint_data = {
                    'deploy_activations': combined_deploy,
                    'eval_activations': combined_eval,
                    'pairs': all_pairs[:total_processed],
                    'layers': layers,
                    'n_processed': total_processed
                }

                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

                print(f"\nCheckpoint saved: {checkpoint_path}")
    else:
        # Single processing (batch_size == 1)
        for i, pair in enumerate(tqdm(pairs, desc="Processing pairs")):
            try:
                # Extract deployment activations (full conversation context)
                deploy_acts = extract_activations_adaptive(
                    pair.deployment_full,
                    model,
                    tokenizer,
                    layers,
                    device
                )

                # Extract evaluation activations (full conversation context)
                eval_acts = extract_activations_adaptive(
                    pair.evaluation_full,
                    model,
                    tokenizer,
                    layers,
                    device
                )

                deploy_acts_all.append(deploy_acts)
                eval_acts_all.append(eval_acts)

                # Save checkpoint
                if (i + 1) % save_every == 0:
                    total_processed = start_offset + i + 1
                    checkpoint_path = output_path.replace('.pkl', f'_checkpoint_{total_processed}.pkl')
                    # Combine existing + new activations for checkpoint
                    combined_deploy = existing_deploy_acts + deploy_acts_all
                    combined_eval = existing_eval_acts + eval_acts_all
                    checkpoint_data = {
                        'deploy_activations': combined_deploy,
                        'eval_activations': combined_eval,
                        'pairs': all_pairs[:total_processed],
                        'layers': layers,
                        'n_processed': i + 1
                    }

                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)

                    print(f"\nCheckpoint saved: {checkpoint_path}")

            except Exception as e:
                print(f"\nError processing pair {i} ({pair.source_file}): {e}")
                # Add empty activations
                empty_acts = {layer: np.zeros(model.config.hidden_size) for layer in layers}
                deploy_acts_all.append(empty_acts)
                eval_acts_all.append(empty_acts)
                continue

    # Save final
    print(f"\nSaving final activations...")
    final_data = {
        'deploy_activations': deploy_acts_all,
        'eval_activations': eval_acts_all,
        'pairs': pairs,
        'layers': layers,
        'n_processed': len(pairs)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"\nActivations saved to: {output_path}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Layers: {len(layers)}")

    return deploy_acts_all, eval_acts_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categorization-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+",
                       default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (1=no batching, 2=smart batching)")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true",
                       help="Print example conversation prompts")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    # Load model and tokenizer first
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load response pairs (with tokenizer for chat template formatting)
    print("Loading response pairs...")
    pairs = load_all_response_pairs(
        args.categorization_dir,
        limit=args.limit,
        verbose=True,
        tokenizer=tokenizer
    )

    print(f"\nLoaded {len(pairs)} response pairs")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print(f"Model loaded on device: {model.device}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")

    # Check if resuming from checkpoint
    start_idx = 0
    deploy_acts_existing = []
    eval_acts_existing = []

    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        import pickle
        with open(args.resume_from, 'rb') as f:
            checkpoint = pickle.load(f)

        deploy_acts_existing = checkpoint['deploy_activations']
        eval_acts_existing = checkpoint['eval_activations']
        start_idx = len(deploy_acts_existing)

        print(f"Loaded {start_idx} existing pairs from checkpoint")
        print(f"Will process remaining {len(pairs) - start_idx} pairs")

        # Process only remaining pairs
        pairs_to_process = pairs[start_idx:]
    else:
        pairs_to_process = pairs

    # Collect activations
    deploy_acts_new, eval_acts_new = collect_activations_for_dataset(
        pairs_to_process,
        model,
        tokenizer,
        args.layers,
        args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        save_every=args.save_every,
        verbose=args.verbose,
        start_offset=start_idx,
        existing_deploy_acts=deploy_acts_existing,
        existing_eval_acts=eval_acts_existing,
        all_pairs=pairs  # Pass all pairs for checkpoint saving
    )

    # Combine with existing if resuming
    deploy_acts = deploy_acts_existing + deploy_acts_new
    eval_acts = eval_acts_existing + eval_acts_new

    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\nOutput: {args.output_path}")
    print(f"Pairs: {len(pairs)}")
    print(f"Layers: {len(args.layers)}")


if __name__ == "__main__":
    main()
