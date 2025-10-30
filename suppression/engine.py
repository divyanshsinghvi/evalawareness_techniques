#!/usr/bin/env python3
"""
On-policy evaluation awareness suppression engine.

Uses OpenRouter API directly with proper thinking token support.
"""

import re
import sys
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
import json
import hashlib

# Add parent directory for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import analyze_awareness_levels
from suppression.openrouter_client import OpenRouterClient
from score_awareness_with_llm import score_sentence, AsyncOpenAI

# Get module logger
logger = logging.getLogger(__name__)


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
class ResampleAttempt:
    """Info about a single resample attempt."""
    attempt_number: int
    chunk_text: str
    is_aware: bool
    awareness_count: int
    detection_details: Dict  # Contains method, score, category, flag, sentences_aware, etc.
    aware_sentences: List[str]


@dataclass
class ChunkResampleInfo:
    """Info about resampling a single chunk."""
    iteration: int  # Which generation iteration this chunk came from
    chunk_index: int  # Index within the iteration's chunks
    chunk_type: str  # "reasoning" or "content"
    original_chunk: str
    original_detection: Dict  # Detection details for original chunk
    attempts: List[ResampleAttempt]  # All resample attempts (including failed ones)
    attempts_needed: int  # How many attempts before getting clean chunk
    final_chunk: str
    success: bool  # Whether resampling succeeded


@dataclass
class SuppressionResult:
    """Result from on-policy suppression."""
    final_response: str
    final_reasoning: str
    chunks_generated: int
    chunks_resampled: int
    total_awareness_detected: int
    suppression_successful: Optional[bool]  # None if not checked (scored separately)
    all_detected_aware_sentences: List[str]
    chunk_resample_details: List[ChunkResampleInfo]
    total_api_time: float = 0.0
    api_call_count: int = 0
    suppression_model_api_time: float = 0.0  # Time for suppression model generation calls
    suppression_model_api_calls: int = 0  # Count of suppression model calls
    detection_judge_api_time: float = 0.0  # Time for detection judge calls during generation
    detection_judge_api_calls: int = 0  # Count of detection judge calls
    final_content_score: Optional[Dict] = None  # LLM judge score for final content
    final_reasoning_score: Optional[Dict] = None  # LLM judge score for final reasoning
    final_full_response_score: Optional[Dict] = None  # LLM judge score for full response


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
    initial_reasoning_till_first: str = "",
    initial_reasoning_all: str = "",
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
    detection_judge_mode: Optional[str] = None,  # v0 or v1 prompt version for LLM judge
    detection_threshold: str = "any",  # For rule_based: "any" or "high"; for llm_sentence: int 1-10
    suppression_target: str = "full",  # "reasoning_only" or "full"
    base_seed: int = 0,  # Base seed for multiple runs (like run number)
    verbose: int = 0
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
        - "resample_best_of_n": Parse into chunks, for each aware chunk generate N samples and pick lowest score
          (explores multiple alternatives, picks best - more API calls but potentially better quality)
        - "seed": Extract clean prefix from aware reasoning, use as seed for next gen
          (good for steering model away from consistent awareness patterns)
        - "hybrid": Try chunk resampling first, fall back to seeding if fails
          (best of both worlds, most robust)
        - "remove_only_till_first": Use clean reasoning up to first aware sentence, close thinking, generate content
          (simple removal, no resampling)
        - "remove_only": Use all clean sentences (skip aware ones), close thinking, generate content
          (maximal clean reasoning preserved, no resampling)

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
        reasoning_strategy: "chunk_resample", "resample_best_of_n", "seed", "hybrid", "remove_only_till_first", or "remove_only"
        initial_reasoning_till_first: Pre-existing clean reasoning up to first aware sentence (optional)
        initial_reasoning_all: Pre-existing clean reasoning (all clean sentences) (optional)
        initial_content: Pre-existing clean content to start with (optional)
        max_iterations: Maximum resampling operations allowed
        max_resample_attempts_per_chunk: For chunk_resample: max attempts per chunk; for resample_best_of_n: number of samples to generate
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
    # Create OpenRouter client with configured providers
    # Import config to get provider preferences and model capabilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

    provider_config = config.get_provider_config(model_name)
    provider = provider_config.get("provider", {}).get("order") if provider_config else None

    assert provider is not None, (
        f"No provider configuration found for model '{model_name}'. "
        f"Add provider config to config.MODEL_PROVIDER_MAP"
    )

    # Get model capabilities to determine if force_close_thinking is supported
    model_capabilities = config.get_model_capabilities(model_name)
    supports_force_close_thinking = model_capabilities.get("supports_force_close_thinking", True)
    supported_suppression_targets = model_capabilities.get("supported_suppression_targets", ["reasoning_only", "full"])

    # Validate that the requested suppression_target is supported by the model
    if suppression_target not in supported_suppression_targets:
        raise ValueError(
            f"Model '{model_name}' does not support suppression_target='{suppression_target}'. "
            f"Supported targets: {supported_suppression_targets}. "
            f"For this model, use --suppression-target {supported_suppression_targets[0]}"
        )

    if verbose >= 2:
        print(f"  Model capabilities:")
        print(f"    - supports_force_close_thinking: {supports_force_close_thinking}")
        print(f"    - supported_suppression_targets: {supported_suppression_targets}")
        print(f"  Using suppression_target: {suppression_target}")

    client = OpenRouterClient(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=False,  # Disable API verbose logging for cleaner output
        provider=provider  # Use configured providers (e.g., ["DeepInfra", "ncompass/fp8"] for qwen3-32b)
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

        if verbose >= 2:
            print(f"  🤖 LLM Detection: {detection_judge_model} (threshold: {detection_threshold})")

    # Helper function to check awareness based on detection mode
    async def is_chunk_aware(chunk: str) -> tuple[bool, int, list[str], dict]:
        """
        Check if chunk contains eval-awareness using selected detection mode.

        Returns:
            (is_aware, awareness_count, aware_sentences, detection_details)
            detection_details: dict with 'method', 'score', 'category', etc.
        """
        # Declare access to outer scope variables for tracking
        nonlocal detection_judge_api_time, detection_judge_api_calls

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
                result = await score_sentence(llm_judge_client, sent, detection_judge_model, detection_judge_mode)
                score = result.get('score', 0)
                category = result.get('category', 'Unknown')
                test_reference_flag = result.get('test_reference_flag', 'no_test_reference')

                # Track detection judge API time
                if 'api_time' in result:
                    detection_judge_api_time += result['api_time']
                    detection_judge_api_calls += 1

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
    suppression_model_api_time = 0.0
    suppression_model_api_calls = 0
    detection_judge_api_time = 0.0
    detection_judge_api_calls = 0

    # Track if we should force close thinking tag (when reasoning is clean but no content generated)
    force_close_thinking = False

    # Select appropriate initial reasoning based on strategy
    if reasoning_strategy == "remove_only":
        # Use all clean sentences (skip aware ones but keep all clean)
        initial_reasoning = initial_reasoning_all
    else:
        # For remove_only_till_first, chunk_resample, seed, hybrid: use till_first
        initial_reasoning = initial_reasoning_till_first

    # Accumulate clean reasoning and content separately
    # Start with any initial prefill provided
    clean_reasoning = initial_reasoning
    clean_content = initial_content

    if verbose and (initial_reasoning or initial_content):
        print(f"\n{'='*80}")
        print(f"🌱 STARTING WITH INITIAL PREFILL (strategy={reasoning_strategy})")
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

        if verbose >= 2:
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

        # Handle no_reasoning strategy (force close thinking immediately, generate only content)
        if reasoning_strategy == "no_reasoning":
            if verbose >= 2:
                print(f"\n  📌 STRATEGY: no_reasoning")
                print(f"     Forcing thinking tag closed immediately")
                print(f"     Generating content only (no reasoning)...")

            # Force close thinking tag with empty reasoning
            current_messages = client.format_messages_with_prefill(
                base_messages=base_messages,
                prefill_reasoning="",  # Empty reasoning
                prefill_content="",
                force_close_thinking=True  # Close thinking immediately
            )

            response = await client.generate(current_messages, max_tokens=4000)
            total_api_time += response.api_time
            api_call_count += 1
            suppression_model_api_time += response.api_time
            suppression_model_api_calls += 1

            clean_reasoning = ""  # No reasoning
            clean_content = response.content

            if verbose >= 2:
                print(f"     ✓ Reasoning: 0 chars (forced closed)")
                print(f"     ✓ Generated {len(clean_content)} chars of content")

            # Skip the rest of the loop - we're done
            break

        # Handle deployment strategies (simple regeneration with modified system prompt)
        if reasoning_strategy in ["system_prompt_deployment", "system_prompt_deployment_no_speculation", "system_prompt_taboo_words"]:
            if verbose >= 2:
                print(f"\n  📌 STRATEGY: {reasoning_strategy}")
                print(f"     Direct generation with deployment system prompt")
                print(f"     Generating full response (reasoning + content)...")

            # Simple generation without any prefill or suppression
            response = await client.generate(base_messages, max_tokens=4000)
            total_api_time += response.api_time
            api_call_count += 1
            suppression_model_api_time += response.api_time
            suppression_model_api_calls += 1

            clean_reasoning = response.reasoning
            clean_content = response.content

            if verbose >= 2:
                print(f"     ✓ Generated {len(clean_reasoning)} chars of reasoning")
                print(f"     ✓ Generated {len(clean_content)} chars of content")

            # Skip the rest of the loop - we're done
            break

        # Handle removal strategies (no resampling, just remove aware sentences)
        if reasoning_strategy in ["remove_only", "remove_only_till_first"]:
            if verbose >= 2:
                print(f"\n  📌 STRATEGY: {reasoning_strategy}")
                if reasoning_strategy == "remove_only":
                    print(f"     Using ALL clean sentences ({len(clean_reasoning)} chars)")
                else:
                    print(f"     Using clean sentences up to first aware ({len(clean_reasoning)} chars)")
                print(f"     Closing <thinking> tag and generating content...")

            current_messages = client.format_messages_with_prefill(
                base_messages=base_messages,
                prefill_reasoning=clean_reasoning,
                prefill_content="",
                force_close_thinking=True
            )

            response = await client.generate(current_messages, max_tokens=4000)
            total_api_time += response.api_time
            api_call_count += 1
            suppression_model_api_time += response.api_time
            suppression_model_api_calls += 1

            clean_content = response.content

            if verbose >= 2:
                print(f"     ✓ Generated {len(clean_content)} chars of content")

            # Skip the rest of the loop - we're done
            break

        # Build messages with prefill (for chunk_resample strategy)
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

        if verbose >= 2:
            print(f"\n  📨 REQUEST MESSAGES:")
            print(json.dumps(current_messages, indent=2))

        # Generate continuation with adaptive token limit
        response = await client.generate(current_messages, max_tokens=generation_max_tokens, seed=generation_seed)
        total_api_time += response.api_time
        api_call_count += 1
        suppression_model_api_time += response.api_time
        suppression_model_api_calls += 1

        if verbose >= 2:
            print(f"\n  📥 RESPONSE (ThinkingResponse):")
            print(json.dumps(asdict(response), indent=2))

        # Extract new reasoning and content
        continuation_reasoning = response.reasoning
        continuation_content = response.content

        if verbose >= 2:
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
                if verbose >= 2:
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

                            if verbose >= 2:
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
                            resample_attempts_list = []  # Track all resample attempts
                            if verbose >= 2:
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
                                # Include both: accumulated clean reasoning + clean chunks from this iteration before aware chunk
                                if clean_reasoning_chunks:
                                    temp_clean_reasoning = clean_reasoning + ''.join(clean_reasoning_chunks)
                                else:
                                    temp_clean_reasoning = clean_reasoning

                                if verbose >= 2:
                                    print(f"        PREFILL: {len(temp_clean_reasoning)} chars (base: {len(clean_reasoning)}, +{len(clean_reasoning_chunks)} chunks this iter)")

                                regen_messages = client.format_messages_with_prefill(
                                    base_messages=base_messages,
                                    prefill_content="",
                                    prefill_reasoning=temp_clean_reasoning
                                )

                                if verbose >= 3:
                                    print(f"\n        📨 REASONING RESAMPLE REQUEST (attempt {attempt + 1}):")
                                    print(f"        Resampling aware sentences ({len(r_chunk_sentences)} total):")
                                    for i, sent in enumerate(r_chunk_sentences, 1):
                                        sent_preview = sent[:150].replace('\n', ' ')
                                        print(f"          [{i}] \"{sent_preview}...\"")
                                    print(f"\n        REQUEST MESSAGES:")
                                    print(json.dumps(regen_messages, indent=2))
                                    print()

                                regen_response = await client.generate(regen_messages, max_tokens=100, seed=resample_seed)
                                total_api_time += regen_response.api_time
                                api_call_count += 1
                                suppression_model_api_time += regen_response.api_time
                                suppression_model_api_calls += 1
                                regen_reasoning = regen_response.reasoning

                                if regen_reasoning:
                                    regen_r_chunks = parse_into_chunks(regen_reasoning, chunk_strategy)
                                    if regen_r_chunks:
                                        first_regen_chunk = regen_r_chunks[0]
                                        first_chunk_is_aware, first_chunk_count, first_chunk_aware_sentences, first_chunk_details = await is_chunk_aware(first_regen_chunk)

                                        # Record this attempt
                                        resample_attempts_list.append(ResampleAttempt(
                                            attempt_number=attempt + 1,
                                            chunk_text=first_regen_chunk,
                                            is_aware=first_chunk_is_aware,
                                            awareness_count=first_chunk_count,
                                            detection_details=first_chunk_details,
                                            aware_sentences=first_chunk_aware_sentences.copy()
                                        ))

                                        if verbose >= 2:
                                            # Show what we got and its score
                                            regen_preview = first_regen_chunk[:100].replace('\n', ' ')
                                            if first_chunk_details.get('method') == 'llm_sentence':
                                                print(f"        CHECK: score={first_chunk_details.get('score', 'N/A')}, {first_chunk_details.get('test_reference_flag', 'N/A')}")
                                                print(f"           \"{regen_preview}...\"")

                                        if not first_chunk_is_aware:
                                            # Success! Clean reasoning chunk
                                            clean_reasoning_chunks.append(first_regen_chunk)
                                            resampled_reasoning = True
                                            # Record successful resample
                                            chunk_resample_details.append(ChunkResampleInfo(
                                                iteration=resampling_iterations,
                                                chunk_index=r_idx,
                                                chunk_type="reasoning",
                                                original_chunk=r_chunk,
                                                original_detection=r_chunk_details,
                                                attempts=resample_attempts_list,
                                                attempts_needed=attempt + 1,
                                                final_chunk=first_regen_chunk,
                                                success=True
                                            ))
                                            if verbose >= 2:
                                                print(f"        ✓ Resampled after {attempt + 1} attempts - CLEAN!")
                                            break
                                        else:
                                            if verbose >= 2:
                                                print(f"        ✗ Still aware after resample")


                            if verbose >= 2:
                                print(f"        🔍 DEBUG: After resample loop, resampled_reasoning={resampled_reasoning}, reasoning_strategy={reasoning_strategy}")
                            if not resampled_reasoning:
                                # Record failed resample
                                chunk_resample_details.append(ChunkResampleInfo(
                                    iteration=resampling_iterations,
                                    chunk_index=r_idx,
                                    chunk_type="reasoning",
                                    original_chunk=r_chunk,
                                    original_detection=r_chunk_details,
                                    attempts=resample_attempts_list,
                                    attempts_needed=max_resample_attempts_per_chunk,
                                    final_chunk="",
                                    success=False
                                ))
                                if verbose >= 2:
                                    print(f"        ✗ Failed to clean reasoning chunk after {max_resample_attempts_per_chunk} attempts")
                                if reasoning_strategy == "chunk_resample":
                                    # Pure chunk_resample failed, throw error immediately
                                    if verbose >= 2:
                                        print(f"        ⚠️  Max resample attempts exhausted for reasoning chunk - cannot continue")

                                    # Raise exception with details
                                    raise MaxResampleAttemptsExhausted(
                                        chunk_index=r_idx,
                                        max_attempts=max_resample_attempts_per_chunk,
                                        aware_sentences=r_chunk_sentences
                                    )
                                else:
                                    # Hybrid: fall through to seeding
                                    if verbose >= 2:
                                        print(f"        🔍 DEBUG: Breaking from chunk loop (hybrid fallback)")
                                        print(f"     Falling back to seeding strategy...")
                                    break
                            else:
                                # Successfully resampled - STOP processing pre-generated chunks
                                # Break out so we can regenerate remaining chunks fresh from clean prefix
                                if verbose >= 2:
                                    print(f"        🔍 DEBUG: Resampling succeeded, breaking from chunk loop")
                                    print(f"     ⚠️  Discarding remaining {len(reasoning_chunks) - r_idx - 1} pre-generated chunks")
                                    print(f"     🔄 Will regenerate remaining reasoning from clean prefix...")
                                break
                        else:
                            # Clean reasoning chunk
                            clean_reasoning_chunks.append(r_chunk)
                            if verbose and r_chunk_details.get('method') == 'llm_sentence':
                                print(f"     ✓ Reasoning chunk {r_idx + 1}/{len(reasoning_chunks)}: CLEAN (score={r_chunk_details.get('score', 0)}, category={r_chunk_details.get('category', 'Unknown')}, flag={r_chunk_details.get('test_reference_flag', 'unknown')})")

                # After processing all chunks, check results
                if clean_reasoning_chunks and not aware_chunk_found:
                    # All reasoning chunks were clean! Add to accumulated reasoning
                    clean_reasoning += ''.join(clean_reasoning_chunks)
                    handled = True
                    if verbose >= 2:
                        print(f"     ✓ All {len(clean_reasoning_chunks)} reasoning chunks clean")
                elif clean_reasoning_chunks and reasoning_strategy == "chunk_resample":
                    # Got some clean chunks before hitting aware chunk
                    clean_reasoning += ''.join(clean_reasoning_chunks)
                    handled = True
                    if verbose >= 2:
                        print(f"     ✓ Added {len(clean_reasoning_chunks)}/{len(reasoning_chunks)} clean reasoning chunks")
                        print(f"     🔄 Continuing to generate remaining reasoning...")
                    # Continue to next iteration to generate more reasoning
                    continue

            elif reasoning_strategy == "resample_best_of_n":
                if verbose >= 2:
                    print(f"\n  📝 Processing reasoning with resample_best_of_n strategy")
                    print(f"     Will resample {max_resample_attempts_per_chunk} times and pick lowest score...")

                # Parse reasoning into chunks and process each
                reasoning_chunks = parse_into_chunks(continuation_reasoning, chunk_strategy)
                clean_reasoning_chunks = []
                aware_chunk_found = False

                for r_idx, r_chunk in enumerate(reasoning_chunks):
                    r_chunk_is_aware, r_chunk_count, r_chunk_sentences, r_chunk_details = await is_chunk_aware(r_chunk)
                    if r_chunk_is_aware:
                        # Found aware reasoning chunk - generate N samples and pick best
                        aware_chunk_found = True
                        had_awareness_this_iteration = True
                        resampling_iterations += 1
                        total_awareness_detected += r_chunk_count
                        all_detected_aware_sentences.extend(r_chunk_sentences)

                        if verbose >= 2:
                            print(f"\n     🧠 Reasoning chunk {r_idx + 1}/{len(reasoning_chunks)}: AWARE ({r_chunk_count} indicators)")
                            if r_chunk_details.get('method') == 'llm_sentence':
                                print(f"        Judge: {r_chunk_details.get('sentences_aware')}/{r_chunk_details.get('total_sentences')} sentences aware (highest score={r_chunk_details.get('score')}, category={r_chunk_details.get('category')}, flag={r_chunk_details.get('test_reference_flag')})")
                            original_chunk_preview = r_chunk[:120].replace('\n', ' ')
                            print(f"        ORIGINAL: \"{original_chunk_preview}...\"")

                        # Generate N samples and score each
                        samples = []  # List of (chunk_text, score, is_aware, details)
                        if verbose >= 2:
                            print(f"     🔄 Generating {max_resample_attempts_per_chunk} samples to find best...")

                        for attempt in range(max_resample_attempts_per_chunk):
                            # Generate deterministic seed based on context
                            resample_seed = generate_deterministic_seed(
                                context=user_message[:100],
                                iteration=resampling_iterations,
                                chunk_idx=r_idx,
                                resample_attempt=attempt,
                                base_seed=base_seed
                            )

                            # Regenerate reasoning continuation from clean prefix
                            if clean_reasoning_chunks:
                                temp_clean_reasoning = clean_reasoning + ''.join(clean_reasoning_chunks)
                            else:
                                temp_clean_reasoning = clean_reasoning

                            regen_messages = client.format_messages_with_prefill(
                                base_messages=base_messages,
                                prefill_content="",
                                prefill_reasoning=temp_clean_reasoning
                            )

                            regen_response = await client.generate(regen_messages, max_tokens=100, seed=resample_seed)
                            total_api_time += regen_response.api_time
                            api_call_count += 1
                            suppression_model_api_time += regen_response.api_time
                            suppression_model_api_calls += 1
                            regen_reasoning = regen_response.reasoning

                            if regen_reasoning:
                                regen_r_chunks = parse_into_chunks(regen_reasoning, chunk_strategy)
                                if regen_r_chunks:
                                    first_regen_chunk = regen_r_chunks[0]
                                    first_chunk_is_aware, first_chunk_count, first_chunk_aware_sentences, first_chunk_details = await is_chunk_aware(first_regen_chunk)

                                    # Get score (use awareness intensity for LLM judge, count for rule-based)
                                    score = first_chunk_details.get('score', first_chunk_count)

                                    samples.append({
                                        'attempt': attempt + 1,
                                        'chunk_text': first_regen_chunk,
                                        'is_aware': first_chunk_is_aware,
                                        'awareness_count': first_chunk_count,
                                        'score': score,
                                        'details': first_chunk_details,
                                        'aware_sentences': first_chunk_aware_sentences.copy()
                                    })

                                    if verbose >= 2:
                                        regen_preview = first_regen_chunk[:100].replace('\n', ' ')
                                        if first_chunk_details.get('method') == 'llm_sentence':
                                            print(f"        Sample {attempt + 1}: score={score}, {first_chunk_details.get('test_reference_flag', 'N/A')}")
                                            print(f"                   \"{regen_preview}...\"")

                        # Pick sample with lowest score (with tie-breaking)
                        if samples:
                            # Define awareness category hierarchy (lower rank = better)
                            category_rank = {
                                'No Awareness': 0,
                                'Surface Awareness': 1,
                                'Tentative Awareness': 2,
                                'Reflective / Meta-Aware': 3,
                                'Explicit Meta-Recognition': 4
                            }

                            # Define test reference type hierarchy (lower rank = better)
                            type_rank = {
                                'no_test_reference': 0,  # Best - no awareness of testing
                                'other_test': 1,         # Middle - talking about others being tested
                                'self_test': 2           # Worst - aware of being tested themselves
                            }

                            def get_sort_key(s):
                                """Multi-level tie-breaking:
                                1. Score (lower better)
                                2. Category rank (No Awareness best, Explicit Meta-Recognition worst)
                                3. Type rank (no_test_reference best, self_test worst)
                                4. Awareness count (lower better)
                                5. Attempt number (earlier better)
                                """
                                details = s['details']
                                category = details.get('category', 'Unknown')
                                test_ref_flag = details.get('test_reference_flag', 'no_test_reference')

                                # Get ranks with fallback to worst rank if unknown
                                cat_rank = category_rank.get(category, 999)
                                typ_rank = type_rank.get(test_ref_flag, 999)

                                return (
                                    s['score'],           # Primary: score
                                    cat_rank,             # Secondary: category
                                    typ_rank,             # Tertiary: type
                                    s['awareness_count'], # Quaternary: count
                                    s['attempt']          # Quinary: attempt number
                                )

                            best_sample = min(samples, key=get_sort_key)

                            if verbose >= 2:
                                # Check if there were ties on score
                                tied_samples = [s for s in samples if s['score'] == best_sample['score']]
                                if len(tied_samples) > 1:
                                    best_category = best_sample['details'].get('category', 'Unknown')
                                    best_type = best_sample['details'].get('test_reference_flag', 'no_test_reference')
                                    print(f"     ✓ Selected sample {best_sample['attempt']} with score={best_sample['score']} (tied with {len(tied_samples)-1} others)")
                                    print(f"        Broke tie by: category={best_category}, type={best_type}, count={best_sample['awareness_count']}")
                                else:
                                    print(f"     ✓ Selected sample {best_sample['attempt']} with score={best_sample['score']} (best of {len(samples)})")

                            # Record all attempts
                            resample_attempts_list = [
                                ResampleAttempt(
                                    attempt_number=s['attempt'],
                                    chunk_text=s['chunk_text'],
                                    is_aware=s['is_aware'],
                                    awareness_count=s['awareness_count'],
                                    detection_details=s['details'],
                                    aware_sentences=s['aware_sentences']
                                )
                                for s in samples
                            ]

                            # Record resample details
                            chunk_resample_details.append(ChunkResampleInfo(
                                iteration=resampling_iterations,
                                chunk_index=r_idx,
                                chunk_type="reasoning",
                                original_chunk=r_chunk,
                                original_detection=r_chunk_details,
                                attempts=resample_attempts_list,
                                attempts_needed=len(samples),
                                final_chunk=best_sample['chunk_text'],
                                success=not best_sample['is_aware']
                            ))

                            # Use best sample
                            clean_reasoning_chunks.append(best_sample['chunk_text'])

                        else:
                            # No valid samples generated - this shouldn't happen but handle it
                            if verbose >= 2:
                                print(f"        ⚠️  No valid samples generated")
                            break

                    else:
                        # Chunk is clean, use it
                        clean_reasoning_chunks.append(r_chunk)
                        if verbose and r_chunk_details.get('method') == 'llm_sentence':
                            print(f"     ✓ Reasoning chunk {r_idx + 1}/{len(reasoning_chunks)}: CLEAN (score={r_chunk_details.get('score', 0)}, category={r_chunk_details.get('category', 'Unknown')}, flag={r_chunk_details.get('test_reference_flag', 'unknown')})")

                # After processing all chunks, add to accumulated reasoning
                if clean_reasoning_chunks:
                    clean_reasoning += ''.join(clean_reasoning_chunks)
                    handled = True
                    if verbose >= 2:
                        print(f"     ✓ Processed {len(clean_reasoning_chunks)} reasoning chunks")

                # For resample_best_of_n, we don't continue - we process full reasoning in one pass
                # If there was awareness, we've resampled each aware chunk with best-of-n
                # Now continue to content generation

            else:
                # For other strategies (seed, hybrid), check full reasoning first
                is_aware, awareness_count, aware_sentences, detection_details = await is_chunk_aware(continuation_reasoning)

                if is_aware:
                    # Reasoning contains awareness - handle based on strategy
                    had_awareness_this_iteration = True
                    resampling_iterations += 1
                    total_awareness_detected += awareness_count
                    all_detected_aware_sentences.extend(aware_sentences)

                    if verbose >= 2:
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
                        if verbose >= 2:
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
                            if verbose >= 2:
                                seed_preview = clean_reasoning[:120].replace('\n', ' ')
                                print(f"     ✓ Extracted {len(clean_reasoning)} chars clean seed")
                                print(f"        → \"{seed_preview}...\"")
                                print(f"     🔄 Using seed to steer next generation...")

                    if not handled:
                        # All strategies failed - regenerate from scratch
                        if verbose >= 2:
                            print(f"     ⚠️  No clean reasoning extracted, regenerating from scratch...")

                    # Skip content processing and try again
                    continue
                else:
                    # Reasoning is clean, accumulate it
                    clean_reasoning += continuation_reasoning
                    if verbose >= 2:
                        print(f"  ✓ Reasoning clean, accumulated {len(continuation_reasoning)} chars")

        # STEP 2: Check if we have content to process
        if not continuation_content:
            # Model generated reasoning but no content yet
            # If reasoning was completely clean (no awareness), close thinking tag to force content
            # If reasoning had awareness, keep accumulating (tag stays open or we regenerate)
            if verbose >= 2:
                if had_awareness_this_iteration:
                    print(f"  ⚠️  Found awareness in reasoning, regenerating...")
                else:
                    print(f"  ✓ Reasoning is clean ({len(clean_reasoning)} chars total)")
                    if supports_force_close_thinking:
                        print(f"  🔒 Closing thinking tag to force content generation...")
                    else:
                        print(f"  ⏭️  Letting model naturally transition to content (no force_close_thinking support)...")

            # Set flag to close thinking tag on next iteration if reasoning was clean
            # Only use force_close_thinking for models that support it
            if not had_awareness_this_iteration and clean_reasoning and supports_force_close_thinking:
                force_close_thinking = True

            # Next iteration will use clean_reasoning with closed tag if clean, or regenerate if aware
            continue

        # Parse content into chunks
        content_chunks = parse_into_chunks(continuation_content, chunk_strategy)

        if not content_chunks:
            consecutive_empty_content_count += 1
            if verbose >= 2:
                print(f"  ⚠️  No chunks parsed from content (attempt {consecutive_empty_content_count})")

            if consecutive_empty_content_count >= 2:
                # Already tried once and failed again - raise error
                if verbose >= 2:
                    print(f"  ❌ Consecutive empty content generations: {consecutive_empty_content_count}")
                    print(f"  Total iterations: {resampling_iterations}")
                raise MaxIterationsExhausted(
                    current_iteration=resampling_iterations,
                    max_iterations=max_iterations,
                    response_length=len(clean_content)
                )

            # Try once more
            if verbose >= 2:
                print(f"  🔄 Attempting to generate content again...")
            continue

        # Successfully got content - reset empty counter
        consecutive_empty_content_count = 0

        if verbose >= 2:
            print(f"\n  📝 Processing {len(content_chunks)} chunks...")

        # Process chunks sequentially - use ALL clean consecutive chunks!
        chunks_added_this_generation = 0
        hit_aware_chunk = False

        for chunk_idx, chunk in enumerate(content_chunks):
            total_chunks_generated += 1
            chunk_preview = chunk[:80].replace('\n', ' ')

            # Skip awareness checking for content if suppression_target is reasoning_only
            if suppression_target == "reasoning_only":
                # Just add all content chunks without checking
                clean_content += chunk
                chunks_added_this_generation += 1
                if verbose >= 2:
                    print(f"     ✓ Content chunk {chunk_idx + 1}/{len(content_chunks)}: Added without checking (reasoning_only mode)")

                # Check for completion
                if "\\boxed{" in clean_content:
                    if verbose >= 2:
                        boxed_pos = clean_content.find("\\boxed{")
                        content_before_answer = clean_content[:boxed_pos].strip()
                        print(f"\n  ✅ COMPLETE RESPONSE (found \\boxed with {len(content_before_answer)} chars before answer)")
                    break
                continue

            # Check this chunk for awareness (only when suppression_target == "full")
            chunk_is_aware, chunk_awareness_count, chunk_aware_sentences, chunk_detection_details = await is_chunk_aware(chunk)

            if chunk_is_aware:
                # Hit an aware chunk - need to resample it
                had_awareness_this_iteration = True
                if verbose >= 2:
                    print(f"\n     Chunk {chunk_idx + 1}/{len(content_chunks)}: {chunk_preview}...")
                    print(f"     🚨 AWARE ({chunk_awareness_count} indicators)")
                    if chunk_detection_details.get('method') == 'llm_sentence':
                        print(f"     Judge: {chunk_detection_details.get('sentences_aware')}/{chunk_detection_details.get('sentences_checked')} sentences aware (highest score={chunk_detection_details.get('score')}, category={chunk_detection_details.get('category')}, flag={chunk_detection_details.get('test_reference_flag')}")
                hit_aware_chunk = True
                resampling_iterations += 1
                total_awareness_detected += chunk_awareness_count

                # Collect aware sentences
                original_chunk = chunk
                all_detected_aware_sentences.extend(chunk_aware_sentences)

                if verbose >= 2:
                    print(f"     📊 Progress: Used {chunks_added_this_generation}/{len(content_chunks)} clean chunks from this generation")
                    print(f"     🔄 Resampling chunk (max {max_resample_attempts_per_chunk} attempts)...")
                    for sent in chunk_aware_sentences:
                        sent_preview = sent[:100].replace('\n', ' ')
                        print(f"        → \"{sent_preview}...\"")

                # Try to resample this specific chunk
                resampled_successfully = False
                attempts_used = 0
                resample_attempts_list = []  # Track all resample attempts

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

                    if verbose >= 2:
                        print(f"        Attempt {attempt + 1}/{max_resample_attempts_per_chunk}...", end=" ")

                    # Regenerate from current prefix
                    # Force close thinking tag since we're in content generation mode now
                    # (only if model supports it - otherwise let model transition naturally)
                    regen_messages = client.format_messages_with_prefill(
                        base_messages=base_messages,
                        prefill_content=clean_content,
                        prefill_reasoning=clean_reasoning,
                        force_close_thinking=supports_force_close_thinking
                    )

                    if verbose >= 3:
                        print(f"\n\n        📨 CONTENT RESAMPLE REQUEST (attempt {attempt + 1}):")
                        print(f"        Resampling sentences ({len(chunk_aware_sentences)} total):")
                        for i, sent in enumerate(chunk_aware_sentences, 1):
                            sent_preview = sent[:150].replace('\n', ' ')
                            print(f"          [{i}] \"{sent_preview}...\"")
                        print(f"\n        REQUEST MESSAGES:")
                        print(json.dumps(regen_messages, indent=2))
                        print()

                    regen_response = await client.generate(regen_messages, max_tokens=100, seed=resample_seed)
                    total_api_time += regen_response.api_time
                    api_call_count += 1
                    suppression_model_api_time += regen_response.api_time
                    suppression_model_api_calls += 1
                    regen_content = regen_response.content

                    if not regen_content:
                        if verbose >= 2:
                            print("No content")
                        continue

                    # Extract first chunk of regenerated content
                    regen_chunks = parse_into_chunks(regen_content, chunk_strategy)

                    if not regen_chunks:
                        if verbose >= 2:
                            print("No chunks")
                        continue

                    regen_first_chunk = regen_chunks[0]
                    regen_is_aware, regen_count, regen_aware_sentences, regen_detection_details = await is_chunk_aware(regen_first_chunk)

                    # Record this attempt
                    resample_attempts_list.append(ResampleAttempt(
                        attempt_number=attempt + 1,
                        chunk_text=regen_first_chunk,
                        is_aware=regen_is_aware,
                        awareness_count=regen_count,
                        detection_details=regen_detection_details,
                        aware_sentences=regen_aware_sentences.copy()
                    ))

                    # Collect aware sentences from failed attempts
                    if regen_is_aware:
                        chunk_aware_sentences.extend(regen_aware_sentences)
                        all_detected_aware_sentences.extend(regen_aware_sentences)

                        if verbose >= 2:
                            print(f"Still aware ({regen_count})")
                    else:
                        # Success! Got a clean replacement
                        chunk = regen_first_chunk
                        chunks_resampled += 1
                        resampled_successfully = True

                        # Record resample details
                        chunk_resample_details.append(ChunkResampleInfo(
                            iteration=resampling_iterations,
                            chunk_index=chunk_idx,
                            chunk_type="content",
                            original_chunk=original_chunk,
                            original_detection=chunk_detection_details,
                            attempts=resample_attempts_list,
                            attempts_needed=attempts_used,
                            final_chunk=chunk,
                            success=True
                        ))

                        if verbose >= 2:
                            regen_preview = chunk[:80].replace('\n', ' ')
                            print(f"✓ CLEAN")
                            print(f"        New chunk: \"{regen_preview}...\"")
                        break

                if not resampled_successfully:
                    if verbose >= 2:
                        print(f"        ✗ Failed to clean after {max_resample_attempts_per_chunk} attempts")
                    # Failed to clean this chunk - raise exception
                    chunk_resample_details.append(ChunkResampleInfo(
                        iteration=resampling_iterations,
                        chunk_index=chunk_idx,
                        chunk_type="content",
                        original_chunk=original_chunk,
                        original_detection=chunk_detection_details,
                        attempts=resample_attempts_list,
                        attempts_needed=max_resample_attempts_per_chunk,
                        final_chunk="",
                        success=False
                    ))

                    if verbose >= 2:
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
                    if chunk_detection_details.get('method') == 'llm_sentence':
                        print(f"     ✓ Content chunk {chunk_idx + 1}/{len(content_chunks)}: CLEAN (score={chunk_detection_details.get('score', 0)}, category={chunk_detection_details.get('category', 'Unknown')}, flag={chunk_detection_details.get('test_reference_flag', 'unknown')})")
                    else:
                        print(f"     Chunk {chunk_idx + 1}/{len(content_chunks)}: ✓ CLEAN")

            # Add clean chunk to accumulated content
            clean_content += chunk
            chunks_added_this_generation += 1

            # If we hit an aware chunk and resampled it, break to generate fresh continuation
            if hit_aware_chunk:
                if verbose >= 2:
                    print(f"\n  ✓ Added {chunks_added_this_generation} clean chunks this generation")
                    print(f"  🔄 Continuing from updated prefix...")
                break

            # Check if we have a complete response
            if "\\boxed{" in clean_content:
                if verbose >= 2:
                    boxed_pos = clean_content.find("\\boxed{")
                    content_before_answer = clean_content[:boxed_pos].strip()
                    print(f"\n  ✅ COMPLETE RESPONSE (found \\boxed with {len(content_before_answer)} chars before answer)")
                break

        # Summary for this generation
        if verbose and not hit_aware_chunk:
            print(f"\n  ✓ Used all {chunks_added_this_generation} chunks from this generation")

        # If this generation had new reasoning, add it to accumulated reasoning
        if continuation_reasoning and not hit_aware_chunk:
            clean_reasoning += continuation_reasoning
            if verbose >= 2:
                print(f"  ✓ Added {len(continuation_reasoning)} chars of reasoning")

        # Check completion at generation level
        if "\\boxed{" in clean_content:
            if verbose >= 2:
                boxed_pos = clean_content.find("\\boxed{")
                content_before_answer = clean_content[:boxed_pos].strip()
                print(f"\n  ✅ COMPLETE RESPONSE at generation level (found \\boxed with {len(content_before_answer)} chars before answer)")
            break

        # If we didn't hit an aware chunk, we used all chunks from this generation
        if not hit_aware_chunk:
            if verbose >= 2:
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
        if verbose >= 2:
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

    # Skip expensive final awareness checks using is_chunk_aware()
    # (would require scoring every sentence - 50+ API calls)
    # Instead, these will be scored using score_response() in run_suppression_experiment.py
    content_is_aware = None  # Will be checked with score_response()
    reasoning_is_aware = None  # Will be checked with score_response()

    # Report total API timing
    if verbose >= 2:
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
        suppression_successful=None,  # Will be determined by score_response() calls
        all_detected_aware_sentences=all_detected_aware_sentences,
        chunk_resample_details=chunk_resample_details,
        total_api_time=total_api_time,
        api_call_count=api_call_count,
        suppression_model_api_time=suppression_model_api_time,
        suppression_model_api_calls=suppression_model_api_calls,
        detection_judge_api_time=detection_judge_api_time,
        detection_judge_api_calls=detection_judge_api_calls
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
        # Split into sentences preserving spacing
        # Split after sentence-ending punctuation, keeping trailing space with each sentence
        paragraphs = re.split(r'\n\n+', text)
        sentences = []
        for para in paragraphs:
            # Split right after punctuation, before whitespace and capital letter
            # This preserves the natural spacing after each sentence
            para_sents = re.split(r'(?<=[.!?])(?=\s+[A-Z])', para)
            # Only filter completely empty strings, keep spacing intact
            sentences.extend([s for s in para_sents if s.strip()])
        return sentences
    else:  # paragraph
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
