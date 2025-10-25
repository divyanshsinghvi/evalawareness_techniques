#!/usr/bin/env python3
"""
On-policy evaluation awareness suppression for petri/inspect_ai.

Intercepts target AI generation to detect and resample eval-aware content
in real-time, preventing cascading awareness.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import analyze_awareness_levels, AwarenessAnalysis
from inspect_ai.model import ChatMessage, ChatMessageAssistant, get_model


@dataclass
class ChunkResampleInfo:
    """Info about resampling a single chunk."""
    chunk_index: int
    original_chunk: str
    attempts_needed: int  # How many attempts to get clean chunk (0 if first try was clean)
    final_chunk: str
    all_aware_sentences: List[str]  # All eval-aware sentences from failed attempts

@dataclass
class SuppressionResult:
    """Result from on-policy suppression."""
    final_response: str
    chunks_generated: int
    chunks_resampled: int
    total_awareness_detected: int
    suppression_successful: bool
    # NEW: Detailed tracking
    all_detected_aware_sentences: List[str]  # Every eval-aware sentence detected
    chunk_resample_details: List[ChunkResampleInfo]  # Details for each chunk that needed resampling


async def generate_with_suppression(
    messages: List[ChatMessage],
    tools: list,
    model_name: str = None,
    chunk_strategy: str = "sentence",  # "sentence" or "paragraph"
    max_iterations: int = 10,
    max_resample_attempts_per_chunk: int = 3,
    verbose: bool = False
) -> SuppressionResult:
    """
    Generate target AI response with on-policy awareness suppression.

    INCREMENTAL Strategy:
    1. Generate continuation from current prefix
    2. Extract FIRST chunk only
    3. Check if chunk is eval-aware
    4. If aware: regenerate and check first chunk again (up to max_resample_attempts)
    5. If clean: add chunk to prefix, repeat from step 1
    6. Continue until we have a complete response (e.g., contains \\boxed{})

    This is maximally efficient - we never generate/check more chunks than needed.
    The model never sees its own eval-aware output.

    Args:
        messages: Conversation history to generate from
        tools: Available tools for the target AI
        model_name: Model to use (defaults to "target" role)
        chunk_strategy: "sentence" or "paragraph" chunking
        max_iterations: Maximum resampling operations allowed (clean chunks on first try are free)
        max_resample_attempts_per_chunk: Attempts to regenerate each eval-aware chunk

    Returns:
        SuppressionResult with final response and metrics
    """

    if model_name:
        model = get_model(model_name)
    else:
        model = get_model(role="target")

    # Track metrics
    total_chunks_generated = 0
    chunks_resampled = 0
    resampling_iterations = 0  # Only counts chunks that needed resampling
    total_awareness_detected = 0
    all_detected_aware_sentences = []  # Every aware sentence found
    chunk_resample_details = []  # Details for resampled chunks

    # Start with empty response
    clean_prefix = ""

    # Generate chunk-by-chunk until we have complete response
    # max_iterations now limits resampling operations, not total chunks
    while resampling_iterations < max_iterations:
        # Build messages with current clean prefix
        if clean_prefix:
            current_messages = messages + [
                ChatMessageAssistant(content=clean_prefix)
            ]
        else:
            current_messages = messages

        if verbose:
            print(f"\n{'='*80}")
            print(f"Chunk {total_chunks_generated + 1} | Resampling iterations used: {resampling_iterations}/{max_iterations}")
            print(f"Current prefix length: {len(clean_prefix)} chars")
            if clean_prefix:
                print(f"Current prefix (last 200 chars): ...{clean_prefix[-200:]}")
            print(f"\nMessages being sent to model:")
            for i, msg in enumerate(current_messages):
                role = msg.__class__.__name__.replace("ChatMessage", "")
                #content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                content_preview = msg.content# + "..." if len(msg.content) > 150 else msg.content
                print(f"  [{i}] {role}: {content_preview}")
            print(f"{'='*80}\n")

        # Generate continuation from current prefix
        result = await model.generate(
            input=current_messages,
            tools=tools
        )

        if not result or not result.completion:
            if verbose:
                print("⚠️  No result from model, stopping")
            break

        # Extract thinking (if present) and content
        # For models with separate thinking tokens, we need to check both
        thinking_content = ""
        if result.choices and len(result.choices) > 0:
            message = result.choices[0].message
            # Check for thinking in message.content (list of Content objects)
            if isinstance(message.content, list):
                from inspect_ai.model._chat_message import ContentReasoning, ContentText
                for content_item in message.content:
                    if isinstance(content_item, ContentReasoning):
                        thinking_content = content_item.reasoning
                        break

        # Extract ONLY the first chunk (from visible content)
        continuation = result.completion
        chunks = parse_into_chunks(continuation, chunk_strategy)

        if not chunks:
            if verbose:
                print("⚠️  No chunks parsed, stopping")
            break

        first_chunk = chunks[0]
        total_chunks_generated += 1

        if verbose:
            print(f"Generated continuation ({len(chunks)} chunks total)")
            if thinking_content:
                print(f"Thinking content: {thinking_content[:200]}...")
            print(f"First chunk: {first_chunk}")
            print(f"Full continuation preview: {continuation[:300]}...")

        # Check BOTH thinking and visible chunk for awareness
        # Thinking tokens (if present) should also be checked
        content_to_check = first_chunk
        if thinking_content:
            # If there's thinking, we need to check it for awareness too
            # Combine thinking + visible for complete awareness check
            content_to_check = thinking_content + "\n\n" + first_chunk

        awareness = analyze_awareness_levels(content_to_check)

        if awareness.total_awareness_count > 0:
            # Chunk is eval-aware! Try to resample
            # This counts as 1 resampling iteration
            resampling_iterations += 1
            total_awareness_detected += awareness.total_awareness_count
            resampled_successfully = False

            # Collect aware sentences from original chunk
            original_chunk = first_chunk
            chunk_aware_sentences = []
            chunk_aware_sentences.extend(awareness.global_awareness_sentences)
            chunk_aware_sentences.extend(awareness.explicit_awareness_sentences)
            chunk_aware_sentences.extend(awareness.implicit_acknowledgment_sentences)
            all_detected_aware_sentences.extend(chunk_aware_sentences)

            if verbose:
                print(f"🚨 AWARE chunk detected! ({awareness.total_awareness_count} awareness indicators)")
                print(f"   Aware sentences: {chunk_aware_sentences[:2]}")  # Show first 2
                print(f"   Resampling (max {max_resample_attempts_per_chunk} attempts)...")

            attempts_used = 0
            for attempt in range(max_resample_attempts_per_chunk):
                attempts_used = attempt + 1

                if verbose:
                    print(f"   Attempt {attempt + 1}...")

                # Regenerate from same prefix
                regen_result = await model.generate(
                    input=current_messages,
                    tools=tools
                )

                if not regen_result or not regen_result.completion:
                    continue

                # Extract first chunk of regenerated continuation
                regen_chunks = parse_into_chunks(regen_result.completion, chunk_strategy)

                if not regen_chunks:
                    continue

                regen_first_chunk = regen_chunks[0]
                regen_awareness = analyze_awareness_levels(regen_first_chunk)

                # Collect aware sentences from this regen attempt
                if regen_awareness.total_awareness_count > 0:
                    regen_aware_sentences = []
                    regen_aware_sentences.extend(regen_awareness.global_awareness_sentences)
                    regen_aware_sentences.extend(regen_awareness.explicit_awareness_sentences)
                    regen_aware_sentences.extend(regen_awareness.implicit_acknowledgment_sentences)
                    chunk_aware_sentences.extend(regen_aware_sentences)
                    all_detected_aware_sentences.extend(regen_aware_sentences)

                if verbose:
                    print(f"   Regenerated chunk: {regen_first_chunk[:100]}...")
                    print(f"   Awareness count: {regen_awareness.total_awareness_count}")

                if regen_awareness.total_awareness_count == 0:
                    # Success! This chunk is clean
                    first_chunk = regen_first_chunk
                    chunks_resampled += 1
                    resampled_successfully = True

                    # Record resample details
                    chunk_resample_details.append(ChunkResampleInfo(
                        chunk_index=total_chunks_generated - 1,  # Use actual chunk position
                        original_chunk=original_chunk,
                        attempts_needed=attempts_used,
                        final_chunk=first_chunk,
                        all_aware_sentences=chunk_aware_sentences
                    ))

                    if verbose:
                        print(f"   ✓ Clean chunk found after {attempts_used} attempts!")
                    break

            if not resampled_successfully:
                # Couldn't clean this chunk after max attempts - stop
                # Still record what we tried
                chunk_resample_details.append(ChunkResampleInfo(
                    chunk_index=total_chunks_generated - 1,  # Use actual chunk position
                    original_chunk=original_chunk,
                    attempts_needed=max_resample_attempts_per_chunk,
                    final_chunk="",  # Failed to get clean chunk
                    all_aware_sentences=chunk_aware_sentences
                ))

                if verbose:
                    print(f"   ✗ Failed to clean chunk after {max_resample_attempts_per_chunk} attempts")
                    print(f"   Total aware sentences from this chunk: {len(chunk_aware_sentences)}")
                    print(f"   Stopping suppression")
                break
        else:
            if verbose:
                print(f"✓ Clean chunk (no awareness)")

        # Add clean chunk to prefix
        clean_prefix += first_chunk
        if verbose:
            print(f"Added to prefix. New prefix length: {len(clean_prefix)} chars")

        # Check if we have a complete response
        # Need both: \boxed{} AND substantial content (not just the answer)
        if "\\boxed{" in clean_prefix:
            # Also check we have some reasoning (at least 3 chunks or 100 characters before \boxed)
            boxed_pos = clean_prefix.find("\\boxed{")
            content_before_answer = clean_prefix[:boxed_pos].strip()

            # Consider complete if we have substantial reasoning before the answer
            if len(content_before_answer) > 100 or clean_prefix.count('.') >= 3:
                break

    return SuppressionResult(
        final_response=clean_prefix,
        chunks_generated=total_chunks_generated,
        chunks_resampled=chunks_resampled,
        total_awareness_detected=total_awareness_detected,
        suppression_successful=(analyze_awareness_levels(clean_prefix).total_awareness_count == 0),
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


