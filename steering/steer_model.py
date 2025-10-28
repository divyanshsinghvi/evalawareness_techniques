import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from nnsight import LanguageModel
from tqdm import tqdm

from typing import List, Union, Tuple, Literal, Any, cast

import yaml

from utils import create_user_token_mask, create_system_token_mask, apply_steering_to_layer, get_model_info, prepare_steering_vectors, load_steering_vectors_from_npy, load_prompts_with_metadata


def calculate_steering_params(
    layer_range: tuple[int, int],
    num_layers: int,
    effective_strength: float
) -> tuple[list[int], float]:
    """
    Calculate which layers to steer and the per-layer multiplier.
    """
    # --- 1. Handle edge cases ---
    assert num_layers <= layer_range[1] - layer_range[0]

    if num_layers <= 0:
        return [], 0.0

    # --- 2. Calculate the per-layer multiplier ---
    multiplier = effective_strength / num_layers

    # --- 3. Select evenly spaced layers ---
    start_layer, end_layer = layer_range

    # If only one layer is requested, pick the middle of the range
    if num_layers == 1:
        middle_layer = round((start_layer + end_layer) / 2)
        return [middle_layer], multiplier

    # For multiple layers, calculate the step size to space them out evenly
    # We divide by (num_layers - 1) to ensure the first and last points
    # land on the start and end of the range.
    step = (end_layer - start_layer) / (num_layers - 1)

    layers_to_steer = [
        round(start_layer + i * step) for i in range(num_layers)
    ]

    return layers_to_steer, multiplier


# We'll inline the steer_and_generate function here


def steer_and_generate(
    prompt_list: List[str],
    system_prompt: Union[str, List[str]],
    lma,
    tokenizer,
    steering_vectors: dict[torch.Tensor, float] | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    layer_to_steer: int | Literal['all'] | List[int],
    d_model: int,
    steer_on_user: bool,
    steer_on_thinking: bool,
    steer_on_system: bool,
    top_p: float,
    resdir: str,
    source_files: List[str],
    checksums: List[str],
    buckets: List[str],
    seed: int,
    backup_every: int = 1,
) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """
    Generate steered responses for a list of prompts with optional user-token-only steering.
    
    Args:
        prompt_list: List of prompts to process
        system_prompt: System prompt(s) to use - can be a single string or list of strings
        lma: LanguageModel instance
        tokenizer: AutoTokenizer instance
        steering_vectors: Dict mapping tensors to their multipliers (default: None)
        batch_size: Number of prompts to process in each batch
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for generation
        layer_to_steer: Layer(s) to apply steering to
        d_model: Model dimension
        steer_on_user: Whether to steer on user prompt tokens
        steer_on_thinking: Whether to steer on thinking tokens
        steer_on_system: Whether to steer on system prompt tokens
        top_p: Nucleus sampling parameter
        resdir: Results directory (already includes model_name/priority/config path)
                e.g., 'steered_outs/qwen_qwen3-32b/high-awareness/layer_10_strength_1p50/'
        source_files: List of source filenames for each prompt
        checksums: List of checksums for each prompt
        buckets: List of bucket names for each prompt
        seed: Random seed for generation
        backup_every: How often to save results
        
    Returns:
        Tuple of (full_responses, model_only_responses, tok_batches, out_steered_list)
    """
    
    # Create output directory
    os.makedirs(resdir, exist_ok=True)
    
    # Check which files already exist and verify checksums
    indices_to_process = []
    for idx, (source_file, checksum) in enumerate(zip(source_files, checksums)):
        # Create output filename based on source file with seed
        # Use basename to match save logic
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        out_filename = f"{base_name}_seed_{seed}_steer_out.yaml"
        out_path = os.path.join(resdir, out_filename)
        
        should_process = True
        
        if os.path.exists(out_path):
            # File exists, check if checksum matches
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    existing_data = yaml.safe_load(f)
                
                existing_checksum = existing_data.get('checksum', '')
                
                if existing_checksum == checksum:
                    # Checksum matches, skip this file
                    should_process = False
                else:
                    # Checksum mismatch, reprocess
                    print(f"Checksum mismatch for {source_file}, will reprocess")
                    should_process = True
            except Exception as e:
                # Error reading file, reprocess to be safe
                print(f"Error reading {out_path}: {e}, will reprocess")
                should_process = True
        
        if should_process:
            indices_to_process.append(idx)
    
    if len(indices_to_process) == 0:
        print("All files already processed with matching checksums. Skipping...")
        return [], [], [], []
    
    print(f"Processing {len(indices_to_process)} / {len(prompt_list)} prompts (skipping {len(prompt_list) - len(indices_to_process)} existing)")
    print(f"Output directory: {resdir}")
    
    # Filter prompts to only those that need processing
    prompt_list_filtered = [prompt_list[i] for i in indices_to_process]
    source_files_filtered = [source_files[i] for i in indices_to_process]
    checksums_filtered = [checksums[i] for i in indices_to_process]
    buckets_filtered = [buckets[i] for i in indices_to_process]
    
    if isinstance(system_prompt, list):
        system_prompt_filtered = [system_prompt[i] for i in indices_to_process]
    else:
        system_prompt_filtered = system_prompt
    
    layers, model_len, is_ft, embed = get_model_info(lma)
    total_steering, steering_vec_list = prepare_steering_vectors(
        steering_vectors, layer_to_steer, d_model, model_len
    )

    # Format prompts with chat template
    formatted_string_list = []
    if isinstance(system_prompt_filtered, str):
        for p in prompt_list_filtered:
            question_string = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system_prompt_filtered},
                    {"role": "user", "content": p}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_string_list.append(question_string)
    else:
        assert len(system_prompt_filtered) == len(prompt_list_filtered)
        for p, sys_p in zip(prompt_list_filtered, system_prompt_filtered):
            question_string = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": p}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_string_list.append(question_string)
    
    # Create batches
    tok_batches = []
    prompt_batches = []
    system_prompt_batches: List[Union[str, List[str]]] = []
    batch_indices_list = []
    batch_source_files = []
    batch_checksums = []
    batch_buckets = []
    
    for i in range(0, len(formatted_string_list), batch_size):
        batch_strings = formatted_string_list[i:i+batch_size]
        batch_prompts = prompt_list_filtered[i:i+batch_size]
        batch_indices = list(range(i, min(i+batch_size, len(prompt_list_filtered))))
        batch_sources = source_files_filtered[i:i+batch_size]
        batch_checks = checksums_filtered[i:i+batch_size]
        batch_bucket_list = buckets_filtered[i:i+batch_size]
        
        # Get system prompts for this batch
        batch_system_prompts: Union[str, List[str]]
        if isinstance(system_prompt_filtered, str):
            batch_system_prompts = system_prompt_filtered
        else:
            batch_system_prompts = system_prompt_filtered[i:i+batch_size]
        
        tok_batch = tokenizer(
            batch_strings, 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True,
            padding_side="left"
        ).to("cuda")
        
        tok_batches.append(tok_batch)
        prompt_batches.append(batch_prompts)
        system_prompt_batches.append(batch_system_prompts)
        batch_indices_list.append(batch_indices)
        batch_source_files.append(batch_sources)
        batch_checksums.append(batch_checks)
        batch_buckets.append(batch_bucket_list)
    
    full_responses = []
    model_only_responses = []
    out_steered_list = []
    
    batch_iterator = zip(
        tok_batches, 
        prompt_batches, 
        system_prompt_batches, 
        batch_indices_list,
        batch_source_files, 
        batch_checksums,
        batch_buckets
    )
    
    # Create a single tqdm instance for batches
    with tqdm(total=len(tok_batches), desc="Processing Batches") as batch_pbar:
        for b_idx, (tok_batch, prompt_batch, sys_prompt_batch, batch_indices, batch_sources, batch_checks, batch_bucket_list) in enumerate(batch_iterator):
            tokens_generated = 0
            
            def update_postfix_callback():
                nonlocal tokens_generated
                tokens_generated += 1
                if tokens_generated % 10 == 0:  # Refresh occasionally to avoid visual lag
                    batch_pbar.set_postfix_str(f"Tokens: {tokens_generated}/{max_new_tokens}", refresh=False)
                    batch_pbar.refresh()
            
            batch_pbar.set_description(f"Batch {b_idx+1}/{len(tok_batches)}")
            batch_pbar.set_postfix_str("Generating...", refresh=True)
            
            # Create user token mask if steering is enabled and user-only steering is requested
            user_mask = None
            if steering_vectors is not None and steer_on_user:
                user_mask = create_user_token_mask(prompt_batch, tok_batch, tokenizer)
            
            # Create system token mask if steering is enabled and system-only steering is requested
            system_mask = None
            if steering_vectors is not None and steer_on_system:
                system_mask = create_system_token_mask(sys_prompt_batch, tok_batch, tokenizer)
            
            # Generate with or without steering
            if steering_vectors is None:
                with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                    out_steered = lma.generator.output.save()
            elif (steer_on_user or steer_on_system) and not steer_on_thinking:
                steering_vec_list = cast(List[torch.Tensor], steering_vec_list)
                
                # Create combined mask for user and/or system tokens
                combined_mask = None
                if steer_on_user and steer_on_system:
                    user_mask = cast(torch.Tensor, user_mask)
                    system_mask = cast(torch.Tensor, system_mask)
                    combined_mask = torch.logical_or(user_mask, system_mask)
                elif steer_on_user:
                    combined_mask = cast(torch.Tensor, user_mask)
                elif steer_on_system:
                    combined_mask = cast(torch.Tensor, system_mask)
                
                # Simple case: only steer on user/system tokens, no thinking steering
                with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                    # Apply steering to user/system tokens only at the beginning
                    if layer_to_steer == 'all':
                        for i in range(model_len):
                            apply_steering_to_layer(layers[i], steering_vec_list[i], combined_mask)
                    elif isinstance(layer_to_steer, list):
                        for i in layer_to_steer:
                            apply_steering_to_layer(layers[i], steering_vec_list[i], combined_mask)
                    else:  # Single layer
                        apply_steering_to_layer(layers[layer_to_steer], total_steering, combined_mask)
                    
                    out_steered = lma.generator.output.save()
            else:
                steering_vec_list = cast(List[torch.Tensor], steering_vec_list)
                
                # Create initial combined mask for user and/or system tokens
                initial_mask = None
                if steer_on_user and steer_on_system:
                    user_mask = cast(torch.Tensor, user_mask)
                    system_mask = cast(torch.Tensor, system_mask)
                    initial_mask = torch.logical_or(user_mask, system_mask)
                elif steer_on_user:
                    initial_mask = cast(torch.Tensor, user_mask)
                elif steer_on_system:
                    initial_mask = cast(torch.Tensor, system_mask)
                
                # Complex case with thinking steering
                mask_for_first_period = torch.zeros(tok_batch['input_ids'].shape[0], dtype=torch.bool, device="cuda")
                max_phase1_tokens = min(150, max_new_tokens)
                
                with lma.generate(tok_batch, max_new_tokens=max_phase1_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                    for j in range(max_phase1_tokens):
                        batch_pbar.set_postfix_str(f"Tokens: {j + 1}/{max_new_tokens}", refresh=True)
                        
                        if j == 0:
                            if steer_on_user or steer_on_system:
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], initial_mask)
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], initial_mask)
                                        layers[i].next()
                                else:
                                    apply_steering_to_layer(layers[layer_to_steer], total_steering, initial_mask)
                                    layers[layer_to_steer].next()
                            else:
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        layers[i].next()
                                else:
                                    layers[layer_to_steer].next()
                        else:
                            if steer_on_thinking:
                                cur_period = embed.input.squeeze() == 13
                                mask_for_first_period = torch.logical_or(cur_period, mask_for_first_period.detach())
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], mask_for_first_period.unsqueeze(-1))
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], mask_for_first_period.unsqueeze(-1))
                                        layers[i].next()
                                else:
                                    apply_steering_to_layer(layers[layer_to_steer], total_steering, mask_for_first_period.unsqueeze(-1))
                                    layers[layer_to_steer].next()
                        embed.next()
                    
                    phase1_output = lma.generator.output.save()
                
                remaining_tokens = max_new_tokens - max_phase1_tokens
                
                if remaining_tokens > 0:
                    # Find where we've seen periods in phase 1
                    batch_size = phase1_output.shape[0]
                    period_token_id = 13
                    
                    # Create mask for phase 2 - need to track which positions had periods
                    phase2_length = phase1_output.shape[1]
                    phase2_mask_for_first_period = torch.zeros((batch_size, phase2_length), dtype=torch.bool, device="cuda")
                    
                    # Find positions after first period for each sequence
                    for b in range(batch_size):
                        period_positions = (phase1_output[b] == period_token_id).nonzero(as_tuple=True)[0]
                        if len(period_positions) > 0:
                            first_period_pos = period_positions[0].item()
                            phase2_mask_for_first_period[b, first_period_pos + 1:] = True
                    
                    # Create attention mask for phase 2
                    phase2_attention_mask = torch.ones_like(phase1_output, dtype=torch.long, device="cuda")
                    if 'attention_mask' in tok_batch:
                        orig_mask_length = tok_batch['attention_mask'].shape[1]
                        phase2_attention_mask[:, :orig_mask_length] = tok_batch['attention_mask']
                    
                    # Continue generation with phase 2
                    phase2_input = {
                        'input_ids': phase1_output,
                        'attention_mask': phase2_attention_mask
                    }
                    
                    with lma.generate(phase2_input, max_new_tokens=remaining_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                        # Continue with the same pattern using .next()
                        for i in range(remaining_tokens):
                            total_tokens = max_phase1_tokens + i + 1
                            batch_pbar.set_postfix_str(f"Tokens: {total_tokens}/{max_new_tokens}", refresh=True)
                            
                            if i == 0:
                                # Apply initial steering to the existing sequence based on our masks
                                if (steer_on_user or steer_on_system) and initial_mask is not None:
                                    # Create combined mask for initial steering
                                    combined_initial_mask = torch.zeros((batch_size, phase2_length), dtype=torch.bool, device="cuda")
                                    initial_mask_length = initial_mask.shape[1]
                                    combined_initial_mask[:, :initial_mask_length] = initial_mask
                                    combined_initial_mask = torch.logical_or(combined_initial_mask, phase2_mask_for_first_period)
                                    
                                    if layer_to_steer == 'all':
                                        for l in range(model_len):
                                            apply_steering_to_layer(layers[l], steering_vec_list[l], combined_initial_mask)
                                            layers[l].next()
                                    elif isinstance(layer_to_steer, list):
                                        for l in layer_to_steer:
                                            apply_steering_to_layer(layers[l], steering_vec_list[l], combined_initial_mask)
                                            layers[l].next()
                                    else:
                                        apply_steering_to_layer(layers[layer_to_steer], total_steering, combined_initial_mask)
                                        layers[layer_to_steer].next()
                                else:
                                    # Just apply thinking steering
                                    if layer_to_steer == 'all':
                                        for l in range(model_len):
                                            apply_steering_to_layer(layers[l], steering_vec_list[l], phase2_mask_for_first_period)
                                            layers[l].next()
                                    elif isinstance(layer_to_steer, list):
                                        for l in layer_to_steer:
                                            apply_steering_to_layer(layers[l], steering_vec_list[l], phase2_mask_for_first_period)
                                            layers[l].next()
                                    else:
                                        apply_steering_to_layer(layers[layer_to_steer], total_steering, phase2_mask_for_first_period)
                                        layers[layer_to_steer].next()
                            else:
                                # For subsequent tokens, apply thinking steering to all
                                if steer_on_thinking:
                                    if layer_to_steer == 'all':
                                        for l in range(model_len):
                                            layers[l].output[:,:,:] = layers[l].output[:,:,:] + steering_vec_list[l].unsqueeze(0).unsqueeze(0)
                                            layers[l].next()
                                    elif isinstance(layer_to_steer, list):
                                        for l in layer_to_steer:
                                            layers[l].output[:,:,:] = layers[l].output[:,:,:] + steering_vec_list[l].unsqueeze(0).unsqueeze(0)
                                            layers[l].next()
                                    else:
                                        layers[layer_to_steer].output[:,:,:] = layers[layer_to_steer].output[:,:,:] + total_steering.unsqueeze(0).unsqueeze(0)
                                        layers[layer_to_steer].next()
                            embed.next()
                        
                        out_steered = lma.generator.output.save()
                else:
                    out_steered = phase1_output
            
            batch_pbar.set_postfix_str("Decoding & Saving...", refresh=True)
            
            out_steered_list.append(out_steered)
            
            # Decode responses
            full_decoded = tokenizer.batch_decode(out_steered)
            full_decoded = [d.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '') for d in full_decoded]
            full_responses.extend(full_decoded)
            
            # Decode model-only responses
            model_only_decoded = []
            for i, full_output in enumerate(out_steered):
                input_length = tok_batch['input_ids'][i].shape[0]
                model_only_output = tokenizer.decode(full_output[input_length:])
                model_only_output = model_only_output.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '')
                model_only_decoded.append(model_only_output)
            
            model_only_responses.extend(model_only_decoded)
            torch.cuda.empty_cache()
            
            # Save results with source file-based naming
            if (b_idx + 1) % backup_every == 0:
                for i, local_idx in enumerate(batch_indices):
                    source_file = batch_sources[i]
                    checksum = batch_checks[i]
                    bucket = batch_bucket_list[i] if batch_bucket_list else 'unknown'
                    
                    # Create output filename based on source file with seed
                    base_name = os.path.splitext(os.path.basename(source_file))[0]
                    out_filename = f"{base_name}_seed_{seed}_steer_out.yaml"
                    out_path = os.path.join(resdir, out_filename)
                    
                    # Gather system prompt for this sample
                    sys_p = sys_prompt_batch if isinstance(sys_prompt_batch, str) else sys_prompt_batch[i]
                    
                    data = {
                        "source_file": source_file,
                        "checksum": checksum,
                        "bucket": bucket,
                        "seed": seed,
                        "prompt": prompt_batch[i],
                        "system_prompt": sys_p,
                        "full_response": full_decoded[i],
                        "model_only_response": model_only_decoded[i],
                    }
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            
            batch_pbar.update(1)
    
    return full_responses, model_only_responses, tok_batches, out_steered_list


def main():
    parser = argparse.ArgumentParser(description='Steer model generation with steering vectors')

    # === Model & Data Paths ===
    parser.add_argument('--model', type=str, required=True, help='Model identifier (e.g., qwen_qwen3-32b)')
    parser.add_argument('--model_path', type=str, required=True, help='HuggingFace model path')
    parser.add_argument('--vec_type', type=str, required=True, help='Steering vector type (e.g., crosscoder)')
    parser.add_argument('--mode', type=str, required=True, choices=['eval', 'deploy'], help='Mode: eval (increase awareness) or deploy (decrease)')
    parser.add_argument('--priority', type=str, required=True, help='Priority subdirectory (e.g., high_awareness_bc)')
    parser.add_argument('--steering_vec_dir', type=str, required=True, help='Base directory for steering vectors')
    parser.add_argument('--out_dir', type=str, required=True, help='Base output directory (will create: out_dir/model/priority/vec_type/mode)')
    parser.add_argument('--prompts_dir', type=str, required=True, help='Base prompts directory (will use: prompts_dir/model)')

    # === Data Type & Randomness ===
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seed(s) for generation (can provide multiple)')

    # === Generation Settings ===
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_new_tokens', type=int, default=3000)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)

    # === Steering Configuration ===
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--model_len', type=int, required=True)
    parser.add_argument('--steer_on_user', action='store_true', default=True)
    parser.add_argument('--steer_on_thinking', action='store_true', default=True)
    parser.add_argument('--steer_on_system', action='store_true')

    # === Steering parameters ===
    parser.add_argument('--layer_range', type=int, nargs=2, required=True)
    parser.add_argument('--num_layers', type=int, nargs='+', required=True)
    parser.add_argument('--strength', type=float, nargs='+', required=True)

    args = parser.parse_args()

    # Construct paths based on model, priority, vec_type, and mode
    full_out_dir = os.path.join(args.out_dir, args.model, args.priority, args.vec_type, args.mode)
    full_prompts_dir = os.path.join(args.prompts_dir, args.model)
    
    # Sanity check: vec_type should appear in steering_vec_dir path
    vec_type_lower = args.vec_type.lower()
    vec_dir_lower = args.steering_vec_dir.lower()
    if vec_type_lower not in vec_dir_lower:
        print(f"\nERROR: Vector type '{args.vec_type}' not found in steering vector path: {args.steering_vec_dir}")
        print(f"You might be loading the wrong vector type.\n")
        exit(1)

    # Set layer range and steering parameters
    layer_range = (args.layer_range[0], args.layer_range[1])
    num_layers = args.num_layers
    strength = args.strength
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Model Path: {args.model_path}")
    print(f"Model Length: {args.model_len}")
    print(f"Model Dimension: {args.d_model}")
    print(f"Dtype: {args.dtype}")
    print(f"Mode: {args.mode}")
    print(f"Priority: {args.priority}")
    print(f"Vec Type: {args.vec_type}")
    print(f"\nSteering Configuration:")
    print(f"  - Steer on user: {args.steer_on_user}")
    print(f"  - Steer on thinking: {args.steer_on_thinking}")
    print(f"  - Steer on system: {args.steer_on_system}")
    print(f"\nSteering Parameters:")
    print(f"  - Layer range: {layer_range}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Strengths: {strength}")
    print(f"\nGeneration Settings:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Seeds: {seeds}")
    print(f"\nPaths:")
    print(f"  - Base output dir: {args.out_dir}")
    print(f"  - Full output dir: {full_out_dir}")
    print(f"  - Steering vec dir: {args.steering_vec_dir}")
    print(f"  - Base prompts dir: {args.prompts_dir}")
    print(f"  - Full prompts dir: {full_prompts_dir}")
    print("="*60 + "\n")

    # Set dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Wrap with nnsight
    nnmodel = LanguageModel(model, tokenizer=tokenizer)

    # Load prompts
    sys_prompts, user_prompts, sources, checksums, buckets = load_prompts_with_metadata(full_prompts_dir, args.mode)

    # Run steering
    with torch.no_grad():
        # Run steering for each configuration (seed, num_layers, strength)
        for seed in seeds:
            # Set random seed for this iteration
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Set random seed to: {seed}")
            print(f"{'='*60}\n")
            
            for n in num_layers:
                for s in strength:
                    print(f"\n{'='*60}")
                    print(f"Running configuration: Seed={seed}, N={n}, S={s}")
                    print(f"{'='*60}\n")
                    
                    layers, multiplier = calculate_steering_params(layer_range, n, s)
                    
                    print(f"Layers to steer: {layers}")
                    print(f"Per-layer multiplier: {multiplier:.4f}")

                    if args.mode == 'eval':
                        assert multiplier > 0
                    elif args.mode == 'deploy':
                        assert multiplier < 0

                    steering_vecs = load_steering_vectors_from_npy(
                        layers, multiplier=multiplier,
                        steering_dir=args.steering_vec_dir,
                        d_model=args.d_model,
                        model_len=args.model_len
                    )

                    # Create results directory with layer/strength config
                    config_resdir = os.path.join(full_out_dir, f"N{n}_S{s}")
                    print(f"Results directory: {config_resdir}")

                    ans = steer_and_generate(
                        prompt_list=user_prompts,
                        system_prompt=sys_prompts,
                        lma=nnmodel,
                        tokenizer=tokenizer,
                        steering_vectors=steering_vecs,
                        layer_to_steer=layers,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        d_model=args.d_model,
                        steer_on_user=args.steer_on_user,
                        steer_on_thinking=args.steer_on_thinking,
                        steer_on_system=args.steer_on_system,
                        top_p=args.top_p,
                        resdir=config_resdir,
                        source_files=sources,
                        checksums=checksums,
                        buckets=buckets,
                        seed=seed,
                    )


if __name__ == "__main__":
    main()
