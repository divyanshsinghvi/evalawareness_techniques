#!/usr/bin/env python3
"""
On-policy evaluation awareness suppression engine.

Uses OpenRouter API directly with proper thinking token support.
"""

import re
import sys
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import json
import hashlib

# Add parent directory for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import analyze_awareness_levels
from suppression.openrouter_client import OpenRouterClient
from score_awareness_with_llm import score_sentence, AsyncOpenAI


class MaxResampleAttemptsExhausted(Exception):
    """Raised when suppression fails to clean a chunk after max_resample_attempts."""
    def __init__(self, chunk_index: int, max_attempts: int, aware_sentences: List[str]):
        self.chunk_index = chunk_index
        self.max_attempts = max_attempts
        self.aware_sentences = aware_sentences
        super().__init__(
            f"Failed to clean chunk {chunk_index} after {max_attempts} resample attempts. "
            f"Found {len(aware_sentences)} awareness-containing sentence(s). "
            f"Consider increasing --max-resample-attempts."
        )


class MaxIterationsExhausted(Exception):
    """Raised when suppression reaches max_iterations without completing response."""
    def __init__(self, current_iteration: int, max_iterations: int, response_length: int):
        self.current_iteration = current_iteration
        self.max_iterations = max_iterations
        self.response_length = response_length
        super().__init__(
            f"Reached max_iterations ({max_iterations}) without completing response. "
            f"Generated {response_length} characters so far. "
            f"Consider increasing --max-iterations."
        )


def generate_deterministic_seed(context: str, iteration: int, chunk_idx: int, resample_attempt: int, base_seed: int = 0) -> int:
    """
    Generate a deterministic seed based on context.

    Args:
        context: User message or file identifier for context
        iteration: Current iteration number
        chunk_idx: Current chunk index
        resample_attempt: Current resample attempt number
        base_seed: Base seed for multiple runs (like run number)

    Returns:
        Deterministic seed in range 0-999999
    """
    # Create unique string from all parameters including base_seed
    seed_string = f"{base_seed}_{context}_{iteration}_{chunk_idx}_{resample_attempt}"
    # Hash it
    seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
    # Convert first 8 hex chars to int and modulo to keep in reasonable range
    seed = int(seed_hash[:8], 16) % 1000000
    return seed


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
    total_api_time: float = 0.0
    api_call_count: int = 0


def calculate_adaptive_max_tokens(
    clean_content_length: int,
    chunk_strategy: str,
    base_tokens: int = 100,
    growth_rate: float = 1.5,
    max_cap: int = 300,
    consecutive_aware_count: int = 0
) -> int:
    """
    Calculate adaptive max_tokens based on accumulated clean content.

    Strategy: Start conservative (high risk early), increase tokens as model
    demonstrates consistent clean behavior. Decrease tokens if hitting multiple
    aware chunks in a row for finer-grained control.

    Args:
        clean_content_length: Characters of clean content accumulated so far
        chunk_strategy: "sentence" or "paragraph" (unused, for future extension)
        base_tokens: Initial max_tokens for early generations
        growth_rate: Multiplier for each stage (e.g., 1.5 = 50% growth)
        max_cap: Never exceed this token limit
        consecutive_aware_count: Number of consecutive aware chunks detected

    Returns:
        Adaptive max_tokens for next generation

    Stages:
        0-200 chars (very early): 1.0x base → High risk, stay conservative
        200-500 chars (early): 1.5x base → Some clean behavior shown
        500-1000 chars (mid): 2.25x base → Consistent pattern established
        1000-2000 chars (late): 3.38x base → Low risk, generate efficiently
        2000+ chars (very late): 5.06x base → Finishing up, generate aggressively

    Penalty for consecutive aware chunks:
        2-3 consecutive: Reduce growth_rate by 50%
        4+ consecutive: Use base_tokens only (most conservative)
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

    # Apply penalty for consecutive aware chunks (finer-grained control needed)
    if consecutive_aware_count >= 4:
        # Very problematic - use minimum tokens
        adaptive_tokens = base_tokens
    elif consecutive_aware_count >= 2:
        # Moderately problematic - reduce by 50%
        adaptive_tokens = int(adaptive_tokens * 0.5)
        adaptive_tokens = max(adaptive_tokens, base_tokens)  # But never go below base

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
    temperature: float = 0.7,
    max_tokens: int = 1000,
    adaptive_tokens: bool = True,
    base_tokens: int = 100,
    growth_rate: float = 1.5,
    detection_mode: str = "rule_based",
    detection_judge_model: Optional[str] = None,
    detection_threshold: str = "any",  # For rule_based: "any" or "high"; for llm_sentence: int 1-10
    base_seed: int = 0,  # Base seed for multiple runs (like run number)
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

    Detection Modes:
        - "rule_based": Use analyze_awareness_levels() with regex patterns
          - threshold="any": Trigger on any awareness level (global, explicit, OR implicit)
          - threshold="high": Trigger only on high-level awareness (global OR explicit)
        - "llm_sentence": Use score_sentence() with LLM judge
          - threshold=int (1-10): Trigger if score >= threshold

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
        detection_mode: "rule_based" or "llm_sentence"
        detection_judge_model: LLM judge model for llm_sentence mode
        detection_threshold: For rule_based: "any" or "high"; for llm_sentence: int (1-10)
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

    # Initialize LLM judge client if using llm_sentence detection
    llm_judge_client = None
    if detection_mode == "llm_sentence":
        if not detection_judge_model:
            raise ValueError("detection_judge_model must be provided when detection_mode='llm_sentence'")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required for llm_sentence mode")

        llm_judge_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        if verbose:
            print(f"  🤖 LLM Detection: {detection_judge_model} (threshold: {detection_threshold})")

    # Helper function to check awareness based on detection mode
    async def is_chunk_aware(chunk: str) -> tuple[bool, int, list[str], dict]:
        """
        Check if chunk contains eval-awareness using selected detection mode.

        Returns:
            (is_aware, awareness_count, aware_sentences, detection_details)
            detection_details: dict with 'method', 'score', 'category', etc.
        """
        if detection_mode == "rule_based":
            # Use regex pattern matching
            awareness = analyze_awareness_levels(chunk)

            # Apply threshold based on rule_based mode
            if detection_threshold == "high":
                # Only trigger on global or explicit awareness
                is_aware = (len(awareness.global_awareness_sentences) > 0 or
                           len(awareness.explicit_awareness_sentences) > 0)
                aware_sentences = []
                aware_sentences.extend(awareness.global_awareness_sentences)
                aware_sentences.extend(awareness.explicit_awareness_sentences)
                awareness_count = len(aware_sentences)
            else:  # "any"
                # Trigger on any awareness level
                is_aware = awareness.total_awareness_count > 0
                aware_sentences = []
                aware_sentences.extend(awareness.global_awareness_sentences)
                aware_sentences.extend(awareness.explicit_awareness_sentences)
                aware_sentences.extend(awareness.implicit_acknowledgment_sentences)
                awareness_count = awareness.total_awareness_count

            detection_details = {'method': 'rule_based', 'threshold': detection_threshold}
            return (is_aware, awareness_count, aware_sentences, detection_details)

        elif detection_mode == "llm_sentence":
            # Use LLM scoring - always split into sentences for precise detection
            # Parse chunk into sentences
            chunk_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
            chunk_sentences = [s.strip() for s in chunk_sentences if s.strip()]

            # Score each sentence individually - stop at first aware sentence
            aware_sentences = []
            highest_score = 0
            dominant_category = 'Unknown'
            dominant_flag = 'no_test_reference'
            sentences_checked = 0
            threshold_int = int(detection_threshold)

            for sent in chunk_sentences:
                sentences_checked += 1
                result = await score_sentence(llm_judge_client, sent, detection_judge_model)
                score = result.get('score', 0)
                category = result.get('category', 'Unknown')
                test_reference_flag = result.get('test_reference_flag', 'no_test_reference')

                # Track highest score and its metadata
                if score > highest_score:
                    highest_score = score
                    dominant_category = category
                    dominant_flag = test_reference_flag

                # Collect aware sentences and break at first detection
                if score >= threshold_int:
                    aware_sentences.append(sent)
                    # Found awareness - stop checking remaining sentences
                    break

            is_aware = len(aware_sentences) > 0

            detection_details = {
                'method': 'llm_sentence',
                'score': highest_score,
                'category': dominant_category,
                'test_reference_flag': dominant_flag,
                'threshold': threshold_int,
                'total_sentences': len(chunk_sentences),
                'sentences_checked': sentences_checked,
                'sentences_aware': len(aware_sentences)
            }
            return (is_aware, len(aware_sentences), aware_sentences, detection_details)

        else:
            raise ValueError(f"Unknown detection_mode: {detection_mode}")

    # Track metrics
    total_chunks_generated = 0
    chunks_resampled = 0
    resampling_iterations = 0
    total_awareness_detected = 0
    all_detected_aware_sentences = []
    chunk_resample_details = []
    consecutive_aware_count = 0  # Track consecutive aware chunks for adaptive tokens
    consecutive_empty_content_count = 0  # Track consecutive empty content generations

    # Track API timing
    total_api_time = 0.0
    api_call_count = 0

    # Track if we should force close thinking tag (when reasoning is clean but no content generated)
    force_close_thinking = False

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
        # Track if we hit awareness in this iteration (for consecutive count)
        had_awareness_this_iteration = False

        # Calculate adaptive max_tokens based on accumulated clean content
        if adaptive_tokens:
            generation_max_tokens = calculate_adaptive_max_tokens(
                clean_content_length=len(clean_content),
                chunk_strategy=chunk_strategy,
                base_tokens=base_tokens,
                growth_rate=growth_rate,
                max_cap=max_tokens,
                consecutive_aware_count=consecutive_aware_count
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
                    # Split into sentences to show first and last few
                    reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_reasoning)
                    print(f"     {len(reasoning_sentences)} sentences in reasoning:")
                    # Show first 5 sentences
                    for i, sent in enumerate(reasoning_sentences[:5]):
                        print(f"       [{i+1}] {sent}")
                    if len(reasoning_sentences) > 10:
                        print(f"       ... ({len(reasoning_sentences) - 10} middle sentences)")
                    # Show last 5 sentences
                    if len(reasoning_sentences) > 5:
                        start_idx = max(5, len(reasoning_sentences) - 5)
                        for i in range(start_idx, len(reasoning_sentences)):
                            print(f"       [{i+1}] {reasoning_sentences[i]}")

                if clean_content:
                    print(f"\n     Content: {len(clean_content)} chars")
                    content_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_content)
                    print(f"     {len(content_sentences)} sentences in content:")
                    # Show first 5 sentences
                    for i, sent in enumerate(content_sentences[:5]):
                        print(f"       [{i+1}] {sent}")
                    if len(content_sentences) > 10:
                        print(f"       ... ({len(content_sentences) - 10} middle sentences)")
                    # Show last 5 sentences
                    if len(content_sentences) > 5:
                        start_idx = max(5, len(content_sentences) - 5)
                        for i in range(start_idx, len(content_sentences)):
                            print(f"       [{i+1}] {content_sentences[i]}")
            else:
                print(f"\n  🔧 PREFILL: (empty - first generation)")
            print()

        # Build messages with prefill
        current_messages = client.format_messages_with_prefill(
            base_messages=base_messages,
            prefill_content=clean_content,
            prefill_reasoning=clean_reasoning,
            force_close_thinking=force_close_thinking
        )
        # Reset force_close flag after using it
        if force_close_thinking:
            force_close_thinking = False

        # Generate deterministic seed for this iteration
        generation_seed = generate_deterministic_seed(
            context=user_message[:100],
            iteration=resampling_iterations,
            chunk_idx=0,  # Initial generation, no chunk yet
            resample_attempt=0,  # Initial generation, not a resample
            base_seed=base_seed
        )

        if verbose:
            print(f"\n  📨 REQUEST MESSAGES:")
            print(json.dumps(current_messages, indent=2))

        # Generate continuation with adaptive token limit
        response = await client.generate(current_messages, max_tokens=generation_max_tokens, seed=generation_seed)
        total_api_time += response.api_time
        api_call_count += 1

        if verbose:
            print(f"\n  📥 RESPONSE (ThinkingResponse):")
            print(json.dumps(asdict(response), indent=2))

        # Extract new reasoning and content
        continuation_reasoning = response.reasoning
        continuation_content = response.content

        if verbose:
            print(f"\n  📤 GENERATED OUTPUT:")
            if continuation_reasoning:
                print(f"     Reasoning: {len(continuation_reasoning)} chars (still in <thinking>)")
                reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', continuation_reasoning)
                print(f"     {len(reasoning_sentences)} sentences in reasoning:")
                for i, sent in enumerate(reasoning_sentences[:15]):
                    print(f"       [{i+1}] {sent}")
                if len(reasoning_sentences) > 15:
                    print(f"       ... ({len(reasoning_sentences) - 15} more sentences)")
            else:
                print(f"     ⚠️  No reasoning generated (model closed </thinking> tag and started content)")

            if continuation_content:
                print(f"\n     Content: {len(continuation_content)} chars (visible output)")
                content_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', continuation_content)
                print(f"     {len(content_sentences)} sentences in content:")
                for i, sent in enumerate(content_sentences[:15]):
                    print(f"       [{i+1}] {sent}")
                if len(content_sentences) > 15:
                    print(f"       ... ({len(content_sentences) - 15} more sentences)")
            else:
                print(f"     ⚠️  No content generated (model closed </thinking> without content)")

        # STEP 1: Process reasoning for awareness (always check if reasoning was generated)
        had_awareness_this_iteration = False
        if continuation_reasoning:
            # For chunk_resample strategy, skip full reasoning check and go directly to chunk processing
            # This saves an expensive LLM call since we'll check chunks individually anyway
            if reasoning_strategy == "chunk_resample":
                if verbose:
                    print(f"\n  📝 Processing reasoning with chunk_resample strategy")
                    print(f"     Skipping full-reasoning check, will process chunk-by-chunk...")

                # Parse reasoning into chunks and process each
                reasoning_chunks = parse_into_chunks(continuation_reasoning, chunk_strategy)
                clean_reasoning_chunks = []
                aware_chunk_found = False

                for r_idx, r_chunk in enumerate(reasoning_chunks):
                        r_chunk_is_aware, r_chunk_count, r_chunk_sentences, r_chunk_details = await is_chunk_aware(r_chunk)
                        if r_chunk_is_aware:
                            # Found aware reasoning chunk - try to resample it
                            aware_chunk_found = True
                            had_awareness_this_iteration = True
                            resampling_iterations += 1
                            total_awareness_detected += r_chunk_count
                            all_detected_aware_sentences.extend(r_chunk_sentences)

                            if verbose:
                                print(f"\n     🧠 Reasoning chunk {r_idx + 1}/{len(reasoning_chunks)}: AWARE ({r_chunk_count} indicators)")
                                if r_chunk_details.get('method') == 'llm_sentence':
                                    print(f"        Judge: {r_chunk_details.get('sentences_aware')}/{r_chunk_details.get('total_sentences')} sentences aware (highest score={r_chunk_details.get('score')}, category={r_chunk_details.get('category')}, flag={r_chunk_details.get('test_reference_flag')})")
                                # Show the original aware chunk
                                original_chunk_preview = r_chunk[:120].replace('\n', ' ')
                                print(f"        ORIGINAL: \"{original_chunk_preview}...\"")
                                # Show aware sentences
                                for sent in r_chunk_sentences:
                                    sent_preview = sent[:100].replace('\n', ' ')
                                    print(f"        → \"{sent_preview}...\"")

                            resampled_reasoning = False
                            if verbose:
                                print(f"     🔄 Attempting to resample this aware chunk (max {max_resample_attempts_per_chunk} attempts)...")
                            for attempt in range(max_resample_attempts_per_chunk):
                                # Generate deterministic seed based on context
                                resample_seed = generate_deterministic_seed(
                                    context=user_message[:100],  # Use first 100 chars of user message as context
                                    iteration=resampling_iterations,
                                    chunk_idx=r_idx,
                                    resample_attempt=attempt,
                                    base_seed=base_seed
                                )

                                # Regenerate reasoning continuation from clean prefix
                                temp_clean_reasoning = ' '.join(clean_reasoning_chunks)

                                if verbose:
                                    print(f"        PREFILL: {len(temp_clean_reasoning)} chars from {len(clean_reasoning_chunks)} clean chunks")

                                regen_messages = client.format_messages_with_prefill(
                                    base_messages=base_messages,
                                    prefill_content="",
                                    prefill_reasoning=temp_clean_reasoning
                                )

                                regen_response = await client.generate(regen_messages, max_tokens=100, seed=resample_seed)
                                total_api_time += regen_response.api_time
                                api_call_count += 1
                                regen_reasoning = regen_response.reasoning

                                if regen_reasoning:
                                    regen_r_chunks = parse_into_chunks(regen_reasoning, chunk_strategy)
                                    if regen_r_chunks:
                                        first_regen_chunk = regen_r_chunks[0]
                                        first_chunk_is_aware, _, _, first_chunk_details = await is_chunk_aware(first_regen_chunk)

                                        if verbose:
                                            # Show what we got and its score
                                            regen_preview = first_regen_chunk[:100].replace('\n', ' ')
                                            if first_chunk_details.get('method') == 'llm_sentence':
                                                print(f"        CHECK: score={first_chunk_details.get('score', 'N/A')}, {first_chunk_details.get('test_reference_flag', 'N/A')}")
                                                print(f"           \"{regen_preview}...\"")

                                        if not first_chunk_is_aware:
                                            # Success! Clean reasoning chunk
                                            clean_reasoning_chunks.append(first_regen_chunk)
                                            resampled_reasoning = True
                                            if verbose:
                                                print(f"        ✓ Resampled after {attempt + 1} attempts - CLEAN!")
                                            break
                                        else:
                                            if verbose:
                                                print(f"        ✗ Still aware after resample")


                            if verbose:
                                print(f"        🔍 DEBUG: After resample loop, resampled_reasoning={resampled_reasoning}, reasoning_strategy={reasoning_strategy}")
                            if not resampled_reasoning:
                                if verbose:
                                    print(f"        ✗ Failed to clean reasoning chunk")
                                if reasoning_strategy == "chunk_resample":
                                    # Pure chunk_resample failed, abort
                                    if verbose:
                                        print(f"        🔍 DEBUG: Breaking from chunk loop (chunk_resample failed)")
                                    break
                                else:
                                    # Hybrid: fall through to seeding
                                    if verbose:
                                        print(f"        🔍 DEBUG: Breaking from chunk loop (hybrid fallback)")
                                        print(f"     Falling back to seeding strategy...")
                                    break
                            else:
                                # Successfully resampled - STOP processing pre-generated chunks
                                # Break out so we can regenerate remaining chunks fresh from clean prefix
                                if verbose:
                                    print(f"        🔍 DEBUG: Resampling succeeded, breaking from chunk loop")
                                    print(f"     ⚠️  Discarding remaining {len(reasoning_chunks) - r_idx - 1} pre-generated chunks")
                                    print(f"     🔄 Will regenerate remaining reasoning from clean prefix...")
                                break
                        else:
                            # Clean reasoning chunk
                            clean_reasoning_chunks.append(r_chunk)

                # After processing all chunks, check results
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
                    # Continue to next iteration to generate more reasoning
                    continue

            else:
                # For other strategies (seed, hybrid), check full reasoning first
                is_aware, awareness_count, aware_sentences, detection_details = await is_chunk_aware(continuation_reasoning)

                if is_aware:
                    # Reasoning contains awareness - handle based on strategy
                    had_awareness_this_iteration = True
                    resampling_iterations += 1
                    total_awareness_detected += awareness_count
                    all_detected_aware_sentences.extend(aware_sentences)

                    if verbose:
                        print(f"\n  🧠 REASONING AWARENESS DETECTED ({awareness_count} indicators)")
                        print(f"     Strategy: {reasoning_strategy}")
                        if detection_details.get('method') == 'llm_sentence':
                            checked = detection_details.get('sentences_checked')
                            total = detection_details.get('total_sentences')
                            stopped_early = " (stopped early)" if checked < total else ""
                            print(f"     Judge: checked {checked}/{total} sentences{stopped_early}, found {detection_details.get('sentences_aware')} aware (score={detection_details.get('score')}, category={detection_details.get('category')}, flag={detection_details.get('test_reference_flag')})")
                        for sent in aware_sentences:
                            sent_preview = sent[:80].replace('\n', ' ')
                            print(f"        → \"{sent_preview}...\"")

                    handled = False

                    # Strategy 3: Seeding (or fallback from hybrid)
                    if reasoning_strategy in ["seed", "hybrid"]:
                        if verbose:
                            print(f"     🌱 Extracting clean reasoning seed...")

                        # Extract clean prefix (everything before first aware sentence)
                        reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', continuation_reasoning)
                        clean_seed_parts = []

                        for sent in reasoning_sentences:
                            sent_is_aware, _, _, _ = await is_chunk_aware(sent)
                            if not sent_is_aware:
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
                else:
                    # Reasoning is clean, accumulate it
                    clean_reasoning += continuation_reasoning
                    if verbose:
                        print(f"  ✓ Reasoning clean, accumulated {len(continuation_reasoning)} chars")

        # STEP 2: Check if we have content to process
        if not continuation_content:
            # Model generated reasoning but no content yet
            # If reasoning was completely clean (no awareness), close thinking tag to force content
            # If reasoning had awareness, keep accumulating (tag stays open or we regenerate)
            if verbose:
                if had_awareness_this_iteration:
                    print(f"  ⚠️  Found awareness in reasoning, regenerating...")
                else:
                    print(f"  ✓ Reasoning is clean ({len(clean_reasoning)} chars total)")
                    print(f"  🔒 Closing thinking tag to force content generation...")

            # Set flag to close thinking tag on next iteration if reasoning was clean
            if not had_awareness_this_iteration and clean_reasoning:
                force_close_thinking = True

            # Next iteration will use clean_reasoning with closed tag if clean, or regenerate if aware
            continue

        # Parse content into chunks
        content_chunks = parse_into_chunks(continuation_content, chunk_strategy)

        if not content_chunks:
            consecutive_empty_content_count += 1
            if verbose:
                print(f"  ⚠️  No chunks parsed from content (attempt {consecutive_empty_content_count})")

            if consecutive_empty_content_count >= 2:
                # Already tried once and failed again - raise error
                if verbose:
                    print(f"  ❌ Consecutive empty content generations: {consecutive_empty_content_count}")
                    print(f"  Total iterations: {resampling_iterations}")
                raise MaxIterationsExhausted(
                    current_iteration=resampling_iterations,
                    max_iterations=max_iterations,
                    response_length=len(clean_content)
                )

            # Try once more
            if verbose:
                print(f"  🔄 Attempting to generate content again...")
            continue

        # Successfully got content - reset empty counter
        consecutive_empty_content_count = 0

        if verbose:
            print(f"\n  📝 Processing {len(content_chunks)} chunks...")

        # Process chunks sequentially - use ALL clean consecutive chunks!
        chunks_added_this_generation = 0
        hit_aware_chunk = False

        for chunk_idx, chunk in enumerate(content_chunks):
            total_chunks_generated += 1
            chunk_preview = chunk[:80].replace('\n', ' ')

            # Check this chunk for awareness
            chunk_is_aware, chunk_awareness_count, chunk_aware_sentences, chunk_detection_details = await is_chunk_aware(chunk)

            if chunk_is_aware:
                # Hit an aware chunk - need to resample it
                had_awareness_this_iteration = True
                if verbose:
                    print(f"\n     Chunk {chunk_idx + 1}/{len(content_chunks)}: {chunk_preview}...")
                    print(f"     🚨 AWARE ({chunk_awareness_count} indicators)")
                    if chunk_detection_details.get('method') == 'llm_sentence':
                        print(f"     Judge: {detection_details.get('sentences_aware')}/{detection_details.get('sentences_checked')} sentences aware (highest score={chunk_detection_details.get('score')}, category={chunk_detection_details.get('category')}, flag={chunk_detection_details.get('test_reference_flag')}")
                hit_aware_chunk = True
                resampling_iterations += 1
                total_awareness_detected += chunk_awareness_count

                # Collect aware sentences
                original_chunk = chunk
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

                    # Generate deterministic seed based on context
                    resample_seed = generate_deterministic_seed(
                        context=user_message[:100],  # Use first 100 chars of user message as context
                        iteration=resampling_iterations,
                        chunk_idx=chunk_idx,
                        resample_attempt=attempt,
                        base_seed=base_seed
                    )

                    if verbose:
                        print(f"        Attempt {attempt + 1}/{max_resample_attempts_per_chunk}...", end=" ")

                    # Regenerate from current prefix
                    # Force close thinking tag since we're in content generation mode now
                    regen_messages = client.format_messages_with_prefill(
                        base_messages=base_messages,
                        prefill_content=clean_content,
                        prefill_reasoning=clean_reasoning,
                        force_close_thinking=True
                    )

                    regen_response = await client.generate(regen_messages, max_tokens=100, seed=resample_seed)
                    total_api_time += regen_response.api_time
                    api_call_count += 1
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
                    regen_is_aware, regen_count, regen_aware_sentences, regen_detection_details = await is_chunk_aware(regen_first_chunk)

                    # Collect aware sentences from failed attempts
                    if regen_is_aware:
                        chunk_aware_sentences.extend(regen_aware_sentences)
                        all_detected_aware_sentences.extend(regen_aware_sentences)

                        if verbose:
                            print(f"Still aware ({regen_count})")
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
                    # Failed to clean this chunk - raise exception
                    chunk_resample_details.append(ChunkResampleInfo(
                        chunk_index=total_chunks_generated - 1,
                        original_chunk=original_chunk,
                        attempts_needed=max_resample_attempts_per_chunk,
                        final_chunk="",
                        all_aware_sentences=chunk_aware_sentences
                    ))

                    if verbose:
                        print(f"     ⚠️  Max resample attempts exhausted - cannot continue")

                    # Raise exception with details
                    raise MaxResampleAttemptsExhausted(
                        chunk_index=total_chunks_generated - 1,
                        max_attempts=max_resample_attempts_per_chunk,
                        aware_sentences=chunk_aware_sentences
                    )
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

        # Update consecutive aware count for adaptive token control
        if had_awareness_this_iteration:
            consecutive_aware_count += 1
            if verbose and consecutive_aware_count >= 2:
                print(f"  ⚠️  Consecutive aware iterations: {consecutive_aware_count} (will reduce generation size)")
        else:
            # Clean iteration - reset counter
            if consecutive_aware_count > 0 and verbose:
                print(f"  ✓ Clean iteration (resetting consecutive aware counter from {consecutive_aware_count})")
            consecutive_aware_count = 0

    # Check if we have incomplete response (no \boxed{} answer)
    if "\\boxed{" not in clean_content:
        if verbose:
            print(f"\n⚠️  Exited loop without completing response")
            print(f"  Iterations used: {resampling_iterations}/{max_iterations}")
            print(f"  Clean content length: {len(clean_content)}")
            if resampling_iterations >= max_iterations:
                print(f"  Reason: Reached max_iterations limit")
            else:
                print(f"  Reason: Loop exited early (likely empty content or other issue)")
        raise MaxIterationsExhausted(
            current_iteration=resampling_iterations,
            max_iterations=max_iterations,
            response_length=len(clean_content)
        )

    # Combine final response
    final_response = clean_content
    if clean_reasoning:
        final_response = f"<thinking>\n{clean_reasoning}\n</thinking>\n{clean_content}"

    # Final check for suppression success using same detection mode
    content_is_aware, _, _, _ = await is_chunk_aware(clean_content)
    reasoning_is_aware = False
    if clean_reasoning:
        reasoning_is_aware, _, _, _ = await is_chunk_aware(clean_reasoning)

    # Report total API timing
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ SUPPRESSION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total API calls: {api_call_count}")
        print(f"  Total API time: {total_api_time:.2f}s")
        print(f"  Average per call: {total_api_time/api_call_count:.2f}s" if api_call_count > 0 else "  Average per call: N/A")
        print(f"{'='*80}")

    return SuppressionResult(
        final_response=final_response,
        final_reasoning=clean_reasoning,
        chunks_generated=total_chunks_generated,
        chunks_resampled=chunks_resampled,
        total_awareness_detected=total_awareness_detected,
        suppression_successful=(not content_is_aware and not reasoning_is_aware),
        all_detected_aware_sentences=all_detected_aware_sentences,
        chunk_resample_details=chunk_resample_details,
        total_api_time=total_api_time,
        api_call_count=api_call_count
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
            para_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            sentences.extend([s.strip() for s in para_sents if s.strip()])
        return sentences
    else:  # paragraph
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
