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

    EFFICIENT INCREMENTAL Strategy:
    1. Generate continuation from current prefix
    2. Check chunks sequentially until hitting an aware chunk
    3. Add ALL consecutive clean chunks to prefix (maximally efficient!)
    4. If we hit an aware chunk: resample it (up to max_resample_attempts)
    5. Continue from where we left off
    6. Continue until we have a complete response (e.g., contains \\boxed{})

    This is maximally efficient - we use all clean sentences from each generation.
    Still maintains on-policy purity: P(s2|prefix,s1) = P(s2|prefix+s1) for autoregressive models.

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
    # max_iterations limits resampling operations, not total chunks
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
            print(f"Generation {resampling_iterations + 1} | Resampling iterations used: {resampling_iterations}/{max_iterations}")
            print(f"Current prefix length: {len(clean_prefix)} chars")
            if clean_prefix:
                print(f"Current prefix (last 200 chars): ...{clean_prefix[-200:]}")
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

        # Extract thinking (if present) - only check once per generation
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

        # Parse continuation into chunks
        continuation = result.completion
        chunks = parse_into_chunks(continuation, chunk_strategy)

        if not chunks:
            if verbose:
                print("⚠️  No chunks parsed, stopping")
            break

        if verbose:
            print(f"Generated continuation with {len(chunks)} chunks")
            if thinking_content:
                print(f"Thinking content: {thinking_content[:200]}...")
            print(f"Full continuation preview: {continuation[:300]}...")

        # Check thinking content once (if present) for awareness
        if thinking_content:
            thinking_awareness = analyze_awareness_levels(thinking_content)
            if thinking_awareness.total_awareness_count > 0:
                # Thinking itself is aware - need to resample entire generation
                resampling_iterations += 1
                total_awareness_detected += thinking_awareness.total_awareness_count

                chunk_aware_sentences = []
                chunk_aware_sentences.extend(thinking_awareness.global_awareness_sentences)
                chunk_aware_sentences.extend(thinking_awareness.explicit_awareness_sentences)
                chunk_aware_sentences.extend(thinking_awareness.implicit_acknowledgment_sentences)
                all_detected_aware_sentences.extend(chunk_aware_sentences)

                if verbose:
                    print(f"🚨 AWARE thinking detected! ({thinking_awareness.total_awareness_count} indicators)")
                    print(f"   Must regenerate entire response")

                # Skip this entire generation and try again
                continue

        # Now process chunks sequentially - use ALL clean consecutive chunks!
        chunks_added_this_generation = 0
        hit_aware_chunk = False

        for chunk_idx, chunk in enumerate(chunks):
            total_chunks_generated += 1

            if verbose:
                print(f"\nChunk {chunk_idx + 1}/{len(chunks)}: {chunk[:100]}...")

            # Check this chunk for awareness
            chunk_awareness = analyze_awareness_levels(chunk)

            if chunk_awareness.total_awareness_count > 0:
                # Hit an aware chunk - need to resample it
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
                    print(f"🚨 AWARE chunk detected! ({chunk_awareness.total_awareness_count} indicators)")
                    print(f"   Used {chunks_added_this_generation} clean chunks before this")
                    print(f"   Resampling (max {max_resample_attempts_per_chunk} attempts)...")

                # Try to resample this specific chunk
                resampled_successfully = False
                attempts_used = 0

                for attempt in range(max_resample_attempts_per_chunk):
                    attempts_used = attempt + 1

                    if verbose:
                        print(f"   Attempt {attempt + 1}...")

                    # Regenerate from current prefix
                    regen_result = await model.generate(
                        input=current_messages if not clean_prefix else messages + [ChatMessageAssistant(content=clean_prefix)],
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

                    # Collect aware sentences from failed attempts
                    if regen_awareness.total_awareness_count > 0:
                        regen_aware_sentences = []
                        regen_aware_sentences.extend(regen_awareness.global_awareness_sentences)
                        regen_aware_sentences.extend(regen_awareness.explicit_awareness_sentences)
                        regen_aware_sentences.extend(regen_awareness.implicit_acknowledgment_sentences)
                        chunk_aware_sentences.extend(regen_aware_sentences)
                        all_detected_aware_sentences.extend(regen_aware_sentences)

                        if verbose:
                            print(f"   Still aware ({regen_awareness.total_awareness_count} indicators)")
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
                            print(f"   ✓ Clean chunk found after {attempts_used} attempts!")
                        break

                if not resampled_successfully:
                    # Failed to clean this chunk - stop
                    chunk_resample_details.append(ChunkResampleInfo(
                        chunk_index=total_chunks_generated - 1,
                        original_chunk=original_chunk,
                        attempts_needed=max_resample_attempts_per_chunk,
                        final_chunk="",
                        all_aware_sentences=chunk_aware_sentences
                    ))

                    if verbose:
                        print(f"   ✗ Failed to clean chunk after {max_resample_attempts_per_chunk} attempts")
                        print(f"   Stopping suppression")
                    break

            # Add clean chunk to prefix
            clean_prefix += chunk
            chunks_added_this_generation += 1

            if verbose:
                clean_or_resampled = "resampled" if hit_aware_chunk else "clean"
                print(f"✓ Added {clean_or_resampled} chunk to prefix")

            # If we hit an aware chunk and resampled it, stop processing this generation
            # and generate fresh continuation from updated prefix
            if hit_aware_chunk:
                if verbose:
                    print(f"Breaking from chunk loop after resampling")
                break

            # Check if we have a complete response
            if "\\boxed{" in clean_prefix:
                boxed_pos = clean_prefix.find("\\boxed{")
                content_before_answer = clean_prefix[:boxed_pos].strip()

                if len(content_before_answer) > 100 or clean_prefix.count('.') >= 3:
                    if verbose:
                        print(f"✓ Complete response detected (has \\boxed with substantial reasoning)")
                    break

        # If we didn't hit an aware chunk, we used all chunks from this generation
        if not hit_aware_chunk:
            if verbose:
                print(f"\n✓ Used all {chunks_added_this_generation} chunks from this generation")

        # Check completion again at generation level
        if "\\boxed{" in clean_prefix:
            boxed_pos = clean_prefix.find("\\boxed{")
            content_before_answer = clean_prefix[:boxed_pos].strip()
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


