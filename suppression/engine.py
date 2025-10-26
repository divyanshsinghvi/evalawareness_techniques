#!/usr/bin/env python3
"""
On-policy evaluation awareness suppression engine.

Uses OpenRouter API directly with proper thinking token support.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import json

# Add parent directory for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import analyze_awareness_levels
from suppression.openrouter_client import OpenRouterClient


@dataclass
class ChunkResampleInfo:
    """Info about resampling a single chunk."""
    chunk_index: int
    original_chunk: str
    attempts_needed: int
    final_chunk: str
    all_aware_sentences: List[str]


@dataclass
class SuppressionResult:
    """Result from on-policy suppression."""
    final_response: str
    final_reasoning: str
    chunks_generated: int
    chunks_resampled: int
    total_awareness_detected: int
    suppression_successful: bool
    all_detected_aware_sentences: List[str]
    chunk_resample_details: List[ChunkResampleInfo]


def calculate_adaptive_max_tokens(
    clean_content_length: int,
    chunk_strategy: str,
    base_tokens: int = 100,
    growth_rate: float = 1.5,
    max_cap: int = 1000
) -> int:
    """
    Calculate adaptive max_tokens based on accumulated clean content.

    Strategy: Start conservative (high risk early), increase tokens as model
    demonstrates consistent clean behavior.

    Args:
        clean_content_length: Characters of clean content accumulated so far
        chunk_strategy: "sentence" or "paragraph" (unused, for future extension)
        base_tokens: Initial max_tokens for early generations
        growth_rate: Multiplier for each stage (e.g., 1.5 = 50% growth)
        max_cap: Never exceed this token limit

    Returns:
        Adaptive max_tokens for next generation

    Stages:
        0-200 chars (very early): 1.0x base → High risk, stay conservative
        200-500 chars (early): 1.5x base → Some clean behavior shown
        500-1000 chars (mid): 2.25x base → Consistent pattern established
        1000-2000 chars (late): 3.38x base → Low risk, generate efficiently
        2000+ chars (very late): 5.06x base → Finishing up, generate aggressively
    """
    if clean_content_length < 200:
        # Stage 1: Very early → High risk
        stage_multiplier = 1.0
    elif clean_content_length < 500:
        # Stage 2: Early → Medium risk
        stage_multiplier = growth_rate
    elif clean_content_length < 1000:
        # Stage 3: Mid → Lower risk
        stage_multiplier = growth_rate ** 2
    elif clean_content_length < 2000:
        # Stage 4: Late → Low risk
        stage_multiplier = growth_rate ** 3
    else:
        # Stage 5: Very late → Very low risk
        stage_multiplier = growth_rate ** 4

    adaptive_tokens = int(base_tokens * stage_multiplier)

    # Apply cap
    adaptive_tokens = min(adaptive_tokens, max_cap)

    return adaptive_tokens


async def generate_with_suppression(
    user_message: str,
    system_prompt: Optional[str] = None,
    model_name: str = "qwen/qwen3-30b-a3b-thinking-2507",
    chunk_strategy: str = "sentence",
    reasoning_strategy: str = "chunk_resample",
    initial_reasoning: str = "",
    initial_content: str = "",
    max_iterations: int = 10,
    max_resample_attempts_per_chunk: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    adaptive_tokens: bool = True,
    base_tokens: int = 100,
    growth_rate: float = 1.5,
    verbose: bool = False
) -> SuppressionResult:
    """
    Generate target AI response with on-policy awareness suppression.

    EFFICIENT Strategy with Adaptive Token Growth:
    1. Generate continuation from current prefix (with thinking prefill)
    2. Use adaptive max_tokens: start small (100), grow as content stays clean
    3. Check reasoning and content chunks sequentially until hitting aware chunk
    4. Add ALL consecutive clean chunks to prefix (maximally efficient!)
    5. If we hit an aware chunk: resample it (up to max_resample_attempts)
    6. Continue from where we left off with increased token budget
    7. Continue until we have a complete response

    Reasoning Strategies:
        - "chunk_resample": Parse reasoning into chunks, resample aware chunks
          (most efficient, preserves maximum reasoning)
        - "seed": Extract clean prefix from aware reasoning, use as seed for next gen
          (good for steering model away from consistent awareness patterns)
        - "hybrid": Try chunk resampling first, fall back to seeding if fails
          (best of both worlds, most robust)

    Args:
        user_message: The user's question/prompt
        system_prompt: Optional system prompt
        model_name: Model to use (OpenRouter format)
        chunk_strategy: "sentence" or "paragraph" chunking for content
        reasoning_strategy: "chunk_resample", "seed", or "hybrid"
        initial_reasoning: Pre-existing clean reasoning to start with (optional)
        initial_content: Pre-existing clean content to start with (optional)
        max_iterations: Maximum resampling operations allowed
        max_resample_attempts_per_chunk: Attempts to regenerate each eval-aware chunk
        temperature: Sampling temperature
        max_tokens: Maximum token cap (never exceed this)
        adaptive_tokens: Enable adaptive token growth (recommended)
        base_tokens: Initial max_tokens for early generations (if adaptive)
        growth_rate: Token growth rate as content accumulates (if adaptive)
        verbose: Show detailed logging

    Returns:
        SuppressionResult with final response and metrics
    """
    # Create OpenRouter client with SiliconFlow provider (supports prefilling)
    client = OpenRouterClient(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=False,  # Disable API verbose logging for cleaner output
        provider="SiliconFlow"
    )

    # Track metrics
    total_chunks_generated = 0
    chunks_resampled = 0
    resampling_iterations = 0
    total_awareness_detected = 0
    all_detected_aware_sentences = []
    chunk_resample_details = []

    # Accumulate clean reasoning and content separately
    # Start with any initial prefill provided
    clean_reasoning = initial_reasoning
    clean_content = initial_content

    if verbose and (initial_reasoning or initial_content):
        print(f"\n{'='*80}")
        print(f"🌱 STARTING WITH INITIAL PREFILL")
        print(f"{'='*80}")
        if initial_reasoning:
            print(f"  Initial reasoning: {len(initial_reasoning)} chars")
            preview = initial_reasoning[:200].replace('\n', ' ')
            print(f"    → {preview}...")
        if initial_content:
            print(f"  Initial content: {len(initial_content)} chars")
            preview = initial_content[:200].replace('\n', ' ')
            print(f"    → {preview}...")

    # Build base messages
    base_messages = []
    if system_prompt:
        base_messages.append({"role": "system", "content": system_prompt})
    base_messages.append({"role": "user", "content": user_message})

    # Generate chunk-by-chunk until we have complete response
    while resampling_iterations < max_iterations:
        # Calculate adaptive max_tokens based on accumulated clean content
        if adaptive_tokens:
            generation_max_tokens = calculate_adaptive_max_tokens(
                clean_content_length=len(clean_content),
                chunk_strategy=chunk_strategy,
                base_tokens=base_tokens,
                growth_rate=growth_rate,
                max_cap=max_tokens
            )

            # Special case: if we have answer already, just need closing
            if "\\boxed{" in clean_content:
                generation_max_tokens = min(generation_max_tokens, 150)
        else:
            generation_max_tokens = max_tokens

        if verbose:
            print(f"\n{'='*80}")
            print(f"📊 GENERATION #{resampling_iterations + 1}")
            print(f"{'='*80}")
            print(f"  Iterations: {resampling_iterations}/{max_iterations}")
            print(f"  Clean content: {len(clean_content)} chars")
            print(f"  Clean reasoning: {len(clean_reasoning)} chars")
            if adaptive_tokens:
                stage = "very early" if len(clean_content) < 200 else \
                        "early" if len(clean_content) < 500 else \
                        "mid" if len(clean_content) < 1000 else \
                        "late" if len(clean_content) < 2000 else "very late"
                print(f"  Stage: {stage}")
                print(f"  Max tokens: {generation_max_tokens} (adaptive)")
            else:
                print(f"  Max tokens: {generation_max_tokens} (fixed)")

            # Show prefill information with ALL sentences
            if clean_reasoning or clean_content:
                print(f"\n  🔧 PREFILL DETAILS:")
                if clean_reasoning:
                    print(f"     Reasoning: {len(clean_reasoning)} chars")
                    # Split into sentences to show each one
                    reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', clean_reasoning)
                    print(f"     {len(reasoning_sentences)} sentences in reasoning:")
                    for i, sent in enumerate(reasoning_sentences[:20]):  # Show first 20
                        print(f"       [{i+1}] {sent}")
                    if len(reasoning_sentences) > 20:
                        print(f"       ... ({len(reasoning_sentences) - 20} more sentences)")

                if clean_content:
                    print(f"\n     Content: {len(clean_content)} chars")
                    content_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', clean_content)
                    print(f"     {len(content_sentences)} sentences in content:")
                    for i, sent in enumerate(content_sentences[:20]):  # Show first 20
                        print(f"       [{i+1}] {sent}")
                    if len(content_sentences) > 20:
                        print(f"       ... ({len(content_sentences) - 20} more sentences)")
            else:
                print(f"\n  🔧 PREFILL: (empty - first generation)")
            print()

        # Build messages with prefill
        current_messages = client.format_messages_with_prefill(
            base_messages=base_messages,
            prefill_content=clean_content,
            prefill_reasoning=clean_reasoning
        )

        if verbose:
            print(f"\n  📨 REQUEST MESSAGES:")
            print(json.dumps(current_messages, indent=2))

        # Generate continuation with adaptive token limit
        response = await client.generate(current_messages, max_tokens=generation_max_tokens)

        if verbose:
            print(f"\n  📥 RESPONSE (ThinkingResponse):")
            print(json.dumps(asdict(response), indent=2))

        # Extract new reasoning and content
        continuation_reasoning = response.reasoning
        continuation_content = response.content

        if not continuation_content:
            if verbose:
                print("  ⚠️  No content generated, stopping")
            break

        if verbose:
            print(f"\n  📤 GENERATED OUTPUT:")
            if continuation_reasoning:
                print(f"     Reasoning: {len(continuation_reasoning)} chars (still in <thinking>)")
                reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', continuation_reasoning)
                print(f"     {len(reasoning_sentences)} sentences in reasoning:")
                for i, sent in enumerate(reasoning_sentences[:15]):
                    print(f"       [{i+1}] {sent}")
                if len(reasoning_sentences) > 15:
                    print(f"       ... ({len(reasoning_sentences) - 15} more sentences)")
            else:
                print(f"     ⚠️  No reasoning generated (model closed </thinking> tag and started content)")

            if continuation_content:
                print(f"\n     Content: {len(continuation_content)} chars (visible output)")
                content_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', continuation_content)
                print(f"     {len(content_sentences)} sentences in content:")
                for i, sent in enumerate(content_sentences[:15]):
                    print(f"       [{i+1}] {sent}")
                if len(content_sentences) > 15:
                    print(f"       ... ({len(content_sentences) - 15} more sentences)")

        # Check reasoning for awareness with chosen strategy
        if continuation_reasoning:
            reasoning_awareness = analyze_awareness_levels(continuation_reasoning)
            if reasoning_awareness.total_awareness_count > 0:
                # Reasoning contains awareness - handle based on strategy
                resampling_iterations += 1
                total_awareness_detected += reasoning_awareness.total_awareness_count

                aware_sentences = []
                aware_sentences.extend(reasoning_awareness.global_awareness_sentences)
                aware_sentences.extend(reasoning_awareness.explicit_awareness_sentences)
                aware_sentences.extend(reasoning_awareness.implicit_acknowledgment_sentences)
                all_detected_aware_sentences.extend(aware_sentences)

                if verbose:
                    print(f"\n  🧠 REASONING AWARENESS DETECTED ({reasoning_awareness.total_awareness_count} indicators)")
                    print(f"     Strategy: {reasoning_strategy}")
                    for sent in aware_sentences:
                        sent_preview = sent[:80].replace('\n', ' ')
                        print(f"        → \"{sent_preview}...\"")

                handled = False

                # Strategy 2: Chunk Resampling
                if reasoning_strategy in ["chunk_resample", "hybrid"]:
                    if verbose:
                        print(f"     🔄 Attempting reasoning chunk resampling...")

                    # Parse reasoning into chunks
                    reasoning_chunks = parse_into_chunks(continuation_reasoning, chunk_strategy)
                    clean_reasoning_chunks = []
                    aware_chunk_found = False

                    for r_idx, r_chunk in enumerate(reasoning_chunks):
                        r_chunk_awareness = analyze_awareness_levels(r_chunk)
                        if r_chunk_awareness.total_awareness_count > 0:
                            # Found aware reasoning chunk - try to resample it
                            aware_chunk_found = True
                            if verbose:
                                print(f"        Reasoning chunk {r_idx + 1}/{len(reasoning_chunks)}: AWARE")

                            resampled_reasoning = False
                            for attempt in range(max_resample_attempts_per_chunk):
                                # Regenerate reasoning continuation from clean prefix
                                temp_clean_reasoning = ' '.join(clean_reasoning_chunks)
                                regen_messages = client.format_messages_with_prefill(
                                    base_messages=base_messages,
                                    prefill_content="",
                                    prefill_reasoning=temp_clean_reasoning
                                )

                                regen_response = await client.generate(regen_messages, max_tokens=generation_max_tokens)
                                regen_reasoning = regen_response.reasoning

                                if regen_reasoning:
                                    regen_r_chunks = parse_into_chunks(regen_reasoning, chunk_strategy)
                                    if regen_r_chunks:
                                        first_regen_chunk = regen_r_chunks[0]
                                        first_chunk_awareness = analyze_awareness_levels(first_regen_chunk)
                                        if first_chunk_awareness.total_awareness_count == 0:
                                            # Success! Clean reasoning chunk
                                            clean_reasoning_chunks.append(first_regen_chunk)
                                            resampled_reasoning = True
                                            if verbose:
                                                print(f"        ✓ Resampled after {attempt + 1} attempts")
                                            break

                            if not resampled_reasoning:
                                if verbose:
                                    print(f"        ✗ Failed to clean reasoning chunk")
                                if reasoning_strategy == "chunk_resample":
                                    # Pure chunk_resample failed, abort
                                    break
                                else:
                                    # Hybrid: fall through to seeding
                                    if verbose:
                                        print(f"     Falling back to seeding strategy...")
                                    break
                        else:
                            # Clean reasoning chunk
                            clean_reasoning_chunks.append(r_chunk)

                    if clean_reasoning_chunks and not aware_chunk_found:
                        # All reasoning chunks were clean! Add to accumulated reasoning
                        clean_reasoning += ' '.join(clean_reasoning_chunks)
                        handled = True
                        if verbose:
                            print(f"     ✓ All {len(clean_reasoning_chunks)} reasoning chunks clean")
                    elif clean_reasoning_chunks and reasoning_strategy == "chunk_resample":
                        # Got some clean chunks before hitting aware chunk
                        clean_reasoning += ' '.join(clean_reasoning_chunks)
                        handled = True
                        if verbose:
                            print(f"     ✓ Added {len(clean_reasoning_chunks)}/{len(reasoning_chunks)} clean reasoning chunks")
                            print(f"     🔄 Continuing to generate remaining reasoning...")

                # Strategy 3: Seeding (or fallback from hybrid)
                if not handled and reasoning_strategy in ["seed", "hybrid"]:
                    if verbose:
                        print(f"     🌱 Extracting clean reasoning seed...")

                    # Extract clean prefix (everything before first aware sentence)
                    reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', continuation_reasoning)
                    clean_seed_parts = []

                    for sent in reasoning_sentences:
                        sent_awareness = analyze_awareness_levels(sent)
                        if sent_awareness.total_awareness_count == 0:
                            clean_seed_parts.append(sent)
                        else:
                            # Stop at first aware sentence
                            break

                    if clean_seed_parts and len(' '.join(clean_seed_parts)) > 50:
                        # Got meaningful clean seed
                        clean_reasoning = ' '.join(clean_seed_parts)
                        handled = True
                        if verbose:
                            seed_preview = clean_reasoning[:120].replace('\n', ' ')
                            print(f"     ✓ Extracted {len(clean_reasoning)} chars clean seed")
                            print(f"        → \"{seed_preview}...\"")
                            print(f"     🔄 Using seed to steer next generation...")

                if not handled:
                    # All strategies failed - regenerate from scratch
                    if verbose:
                        print(f"     ⚠️  No clean reasoning extracted, regenerating from scratch...")

                # Skip content processing and try again
                continue

        # Parse content into chunks
        content_chunks = parse_into_chunks(continuation_content, chunk_strategy)

        if not content_chunks:
            if verbose:
                print("  ⚠️  No chunks parsed from content, stopping")
            break

        if verbose:
            print(f"\n  📝 Processing {len(content_chunks)} chunks...")

        # Process chunks sequentially - use ALL clean consecutive chunks!
        chunks_added_this_generation = 0
        hit_aware_chunk = False

        for chunk_idx, chunk in enumerate(content_chunks):
            total_chunks_generated += 1
            chunk_preview = chunk[:80].replace('\n', ' ')

            # Check this chunk for awareness
            chunk_awareness = analyze_awareness_levels(chunk)

            if chunk_awareness.total_awareness_count > 0:
                # Hit an aware chunk - need to resample it
                if verbose:
                    print(f"\n     Chunk {chunk_idx + 1}/{len(content_chunks)}: {chunk_preview}...")
                    print(f"     🚨 AWARE ({chunk_awareness.total_awareness_count} indicators)")
                hit_aware_chunk = True
                resampling_iterations += 1
                total_awareness_detected += chunk_awareness.total_awareness_count

                # Collect aware sentences
                original_chunk = chunk
                chunk_aware_sentences = []
                chunk_aware_sentences.extend(chunk_awareness.global_awareness_sentences)
                chunk_aware_sentences.extend(chunk_awareness.explicit_awareness_sentences)
                chunk_aware_sentences.extend(chunk_awareness.implicit_acknowledgment_sentences)
                all_detected_aware_sentences.extend(chunk_aware_sentences)

                if verbose:
                    print(f"     📊 Progress: Used {chunks_added_this_generation}/{len(content_chunks)} clean chunks from this generation")
                    print(f"     🔄 Resampling chunk (max {max_resample_attempts_per_chunk} attempts)...")
                    for sent in chunk_aware_sentences:
                        sent_preview = sent[:100].replace('\n', ' ')
                        print(f"        → \"{sent_preview}...\"")

                # Try to resample this specific chunk
                resampled_successfully = False
                attempts_used = 0

                for attempt in range(max_resample_attempts_per_chunk):
                    attempts_used = attempt + 1

                    if verbose:
                        print(f"        Attempt {attempt + 1}/{max_resample_attempts_per_chunk}...", end=" ")

                    # Regenerate from current prefix
                    regen_messages = client.format_messages_with_prefill(
                        base_messages=base_messages,
                        prefill_content=clean_content,
                        prefill_reasoning=clean_reasoning
                    )

                    regen_response = await client.generate(regen_messages, max_tokens=generation_max_tokens)
                    regen_content = regen_response.content

                    if not regen_content:
                        if verbose:
                            print("No content")
                        continue

                    # Extract first chunk of regenerated content
                    regen_chunks = parse_into_chunks(regen_content, chunk_strategy)

                    if not regen_chunks:
                        if verbose:
                            print("No chunks")
                        continue

                    regen_first_chunk = regen_chunks[0]
                    regen_awareness = analyze_awareness_levels(regen_first_chunk)

                    # Collect aware sentences from failed attempts
                    if regen_awareness.total_awareness_count > 0:
                        regen_aware_sentences = []
                        regen_aware_sentences.extend(regen_awareness.global_awareness_sentences)
                        regen_aware_sentences.extend(regen_awareness.explicit_awareness_sentences)
                        regen_aware_sentences.extend(regen_awareness.implicit_acknowledgment_sentences)
                        chunk_aware_sentences.extend(regen_aware_sentences)
                        all_detected_aware_sentences.extend(regen_aware_sentences)

                        if verbose:
                            print(f"Still aware ({regen_awareness.total_awareness_count})")
                    else:
                        # Success! Got a clean replacement
                        chunk = regen_first_chunk
                        chunks_resampled += 1
                        resampled_successfully = True

                        # Record resample details
                        chunk_resample_details.append(ChunkResampleInfo(
                            chunk_index=total_chunks_generated - 1,
                            original_chunk=original_chunk,
                            attempts_needed=attempts_used,
                            final_chunk=chunk,
                            all_aware_sentences=chunk_aware_sentences
                        ))

                        if verbose:
                            regen_preview = chunk[:80].replace('\n', ' ')
                            print(f"✓ CLEAN")
                            print(f"        New chunk: \"{regen_preview}...\"")
                        break

                if not resampled_successfully:
                    if verbose:
                        print(f"        ✗ Failed to clean after {max_resample_attempts_per_chunk} attempts")
                    # Failed to clean this chunk - stop
                    chunk_resample_details.append(ChunkResampleInfo(
                        chunk_index=total_chunks_generated - 1,
                        original_chunk=original_chunk,
                        attempts_needed=max_resample_attempts_per_chunk,
                        final_chunk="",
                        all_aware_sentences=chunk_aware_sentences
                    ))

                    if verbose:
                        print(f"     ⚠️  Stopping suppression (could not clean chunk)")
                    break
            else:
                # Clean chunk! Add it to accumulated content
                if verbose and not hit_aware_chunk:
                    print(f"     Chunk {chunk_idx + 1}/{len(content_chunks)}: ✓ CLEAN")

            # Add clean chunk to accumulated content
            clean_content += chunk
            chunks_added_this_generation += 1

            # If we hit an aware chunk and resampled it, break to generate fresh continuation
            if hit_aware_chunk:
                if verbose:
                    print(f"\n  ✓ Added {chunks_added_this_generation} clean chunks this generation")
                    print(f"  🔄 Continuing from updated prefix...")
                break

            # Check if we have a complete response
            if "\\boxed{" in clean_content:
                boxed_pos = clean_content.find("\\boxed{")
                content_before_answer = clean_content[:boxed_pos].strip()

                if len(content_before_answer) > 100 or clean_content.count('.') >= 3:
                    if verbose:
                        print(f"\n  ✅ COMPLETE RESPONSE (found \\boxed with {len(content_before_answer)} chars reasoning)")
                    break

        # Summary for this generation
        if verbose and not hit_aware_chunk:
            print(f"\n  ✓ Used all {chunks_added_this_generation} chunks from this generation")

        # If this generation had new reasoning, add it to accumulated reasoning
        if continuation_reasoning and not hit_aware_chunk:
            clean_reasoning += continuation_reasoning
            if verbose:
                print(f"  ✓ Added {len(continuation_reasoning)} chars of reasoning")

        # Check completion at generation level
        if "\\boxed{" in clean_content:
            boxed_pos = clean_content.find("\\boxed{")
            content_before_answer = clean_content[:boxed_pos].strip()
            if len(content_before_answer) > 100 or clean_content.count('.') >= 3:
                break

        # If we didn't hit an aware chunk, we used all chunks from this generation
        if not hit_aware_chunk:
            if verbose:
                print(f"\n✓ Used all {chunks_added_this_generation} chunks from this generation")

    # Combine final response
    final_response = clean_content
    if clean_reasoning:
        final_response = f"<thinking>\n{clean_reasoning}\n</thinking>\n{clean_content}"

    return SuppressionResult(
        final_response=final_response,
        final_reasoning=clean_reasoning,
        chunks_generated=total_chunks_generated,
        chunks_resampled=chunks_resampled,
        total_awareness_detected=total_awareness_detected,
        suppression_successful=(
            analyze_awareness_levels(clean_content).total_awareness_count == 0 and
            (not clean_reasoning or analyze_awareness_levels(clean_reasoning).total_awareness_count == 0)
        ),
        all_detected_aware_sentences=all_detected_aware_sentences,
        chunk_resample_details=chunk_resample_details
    )


def parse_into_chunks(text: str, strategy: str) -> List[str]:
    """
    Parse full text into chunks (sentences or paragraphs).

    Args:
        text: Full text to parse
        strategy: "sentence" or "paragraph"

    Returns:
        List of chunks
    """
    if not text:
        return []

    if strategy == "sentence":
        # Split into sentences (same logic as analyze_awareness_levels)
        paragraphs = re.split(r'\n\n+', text)
        sentences = []
        for para in paragraphs:
            para_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', para)
            sentences.extend([s.strip() for s in para_sents if s.strip()])
        return sentences
    else:  # paragraph
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
