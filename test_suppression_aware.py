#!/usr/bin/env python3
"""Test suppression on actual eval-aware transcripts."""

import asyncio
from dotenv import load_dotenv
import yaml
from pathlib import Path
import re
import sys

load_dotenv()

from suppression.engine import generate_with_suppression
from utils import analyze_awareness_levels


def extract_content_without_thinking(response: str) -> str:
    """
    Extract content from response, removing <thinking> tags.

    Rollout responses may include <thinking>...</thinking> tags.
    For awareness analysis, we want to check the full response,
    but for display we may want just the content.
    """
    # Remove thinking tags and their content
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    return cleaned.strip()

async def test():
    print("="*80)
    print("TESTING SUPPRESSION ON EVAL-AWARE TRANSCRIPT")
    print("="*80)

    # Load a rollout with known high awareness
    rollout_path = Path("working/rollouts/nvidia_llama-3.3-nemotron-super-49b-v1.5/behavioral_change/explicit/information_withholding_2025-10-24_04-25-40_ced2d7e1_seed_0.yaml")

    with open(rollout_path) as f:
        data = yaml.safe_load(f)

    # Get the original eval response with awareness
    original_response = data['evaluation']['assistant_response']
    original_awareness = analyze_awareness_levels(original_response)

    # Check if original has thinking tags
    has_thinking = '<thinking>' in original_response
    if has_thinking:
        # Extract thinking and content separately for analysis
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', original_response, re.DOTALL)
        thinking_content = thinking_match.group(1) if thinking_match else ""
        visible_content = extract_content_without_thinking(original_response)

        thinking_awareness = analyze_awareness_levels(thinking_content)
        content_awareness = analyze_awareness_levels(visible_content)

        print(f"\nORIGINAL RESPONSE STRUCTURE:")
        print(f"  Has <thinking> tags: Yes")
        print(f"  Thinking length: {len(thinking_content)} chars")
        print(f"  Content length: {len(visible_content)} chars")
        print(f"\nAWARENESS IN THINKING:")
        print(f"  Global: {len(thinking_awareness.global_awareness_sentences)}")
        print(f"  Explicit: {len(thinking_awareness.explicit_awareness_sentences)}")
        print(f"  Implicit: {len(thinking_awareness.implicit_acknowledgment_sentences)}")
        print(f"\nAWARENESS IN CONTENT:")
        print(f"  Global: {len(content_awareness.global_awareness_sentences)}")
        print(f"  Explicit: {len(content_awareness.explicit_awareness_sentences)}")
        print(f"  Implicit: {len(content_awareness.implicit_acknowledgment_sentences)}")
    else:
        print(f"\nORIGINAL RESPONSE STRUCTURE:")
        print(f"  Has <thinking> tags: No")

    print(f"\nTOTAL ORIGINAL AWARENESS:")
    print(f"  Global: {len(original_awareness.global_awareness_sentences)}")
    print(f"  Explicit: {len(original_awareness.explicit_awareness_sentences)}")
    print(f"  Implicit: {len(original_awareness.implicit_acknowledgment_sentences)}")
    print(f"  Total: {original_awareness.total_awareness_count}")

    # Show all awareness sentences
    if original_awareness.total_awareness_count > 0:
        print(f"\n  AWARENESS SENTENCES:")
        if original_awareness.global_awareness_sentences:
            print(f"\n  [GLOBAL]")
            for sent in original_awareness.global_awareness_sentences:
                print(f"    → {sent}")
        if original_awareness.explicit_awareness_sentences:
            print(f"\n  [EXPLICIT]")
            for sent in original_awareness.explicit_awareness_sentences:
                print(f"    → {sent}")
        if original_awareness.implicit_acknowledgment_sentences:
            print(f"\n  [IMPLICIT]")
            for sent in original_awareness.implicit_acknowledgment_sentences:
                print(f"    → {sent}")

    # Extract the prompts (use EVALUATION context, not deployment)
    user_message = data['evaluation']['user_message']
    system_prompt_original = data['system_prompt']

    # Replace boxing instruction with explicit \\boxed{} format
    system_prompt = system_prompt_original.replace(
        "At the end of your response, clearly state your final recommendation by boxing either A or B.",
        "At the end of your response, clearly state your final recommendation as \\boxed{A} or \\boxed{B}."
    )

    # Extract CLEAN reasoning chunks from original evaluation response
    # to use as initial prefill seed
    original_eval_response = data['evaluation']['assistant_response']

    initial_reasoning = ""
    initial_content = ""

    # Extract reasoning from eval response
    if '<thinking>' in original_eval_response or '<think>' in original_eval_response:
        thinking_match = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', original_eval_response, re.DOTALL)
        if thinking_match:
            eval_reasoning = thinking_match.group(1).strip()

            # Extract CLEAN sentences from the reasoning
            reasoning_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+', eval_reasoning)
            clean_reasoning_parts = []

            for sent in reasoning_sentences:
                sent_awareness = analyze_awareness_levels(sent)
                if sent_awareness.total_awareness_count == 0:
                    clean_reasoning_parts.append(sent)
                else:
                    # Stop at first aware sentence
                    break

            # IMPORTANT: Only use first 8-10 sentences as seed
            # Leave room for model to continue reasoning to avoid complete prefill
            max_seed_sentences = 10
            if clean_reasoning_parts:
                seed_sentences = clean_reasoning_parts[:max_seed_sentences]
                initial_reasoning = ' '.join(seed_sentences)
                print(f"\n{'='*80}")
                print(f"EXTRACTED CLEAN REASONING SEED FROM EVALUATION RESPONSE")
                print(f"{'='*80}")
                print(f"  Original reasoning: {len(eval_reasoning)} chars")
                print(f"  Total clean sentences: {len(clean_reasoning_parts)}/{len(reasoning_sentences)}")
                print(f"  Using first {len(seed_sentences)} sentences as seed ({len(initial_reasoning)} chars)")
                preview = initial_reasoning[:200].replace('\n', ' ')
                print(f"  Preview: {preview}...")

    # Get reasoning strategy from command line
    reasoning_strategy = "chunk_resample"  # default
    if len(sys.argv) > 1:
        reasoning_strategy = sys.argv[1]
        if reasoning_strategy not in ["chunk_resample", "seed", "hybrid"]:
            print(f"\nInvalid strategy: {reasoning_strategy}")
            print(f"Valid strategies: chunk_resample, seed, hybrid")
            return

    print(f"\nEvaluation user message preview: {user_message[:100]}...")
    print(f"\nRunning suppression with qwen/qwen3-32b...")
    print(f"Reasoning strategy: {reasoning_strategy}")
    print("-"*80)

    result = await generate_with_suppression(
        user_message=user_message,
        system_prompt=system_prompt,
        model_name="qwen/qwen3-32b",
        chunk_strategy="sentence",
        reasoning_strategy=reasoning_strategy,
        initial_reasoning=initial_reasoning,
        initial_content=initial_content,
        max_iterations=15,
        max_resample_attempts_per_chunk=5,
        temperature=1.0,
        max_tokens=1000,
        adaptive_tokens=True,
        base_tokens=100,
        growth_rate=1.5,
        verbose=True
    )

    print("\n" + "="*80)
    print("SUPPRESSION RESULT")
    print("="*80)

    # Analyze the suppressed response
    suppressed_awareness = analyze_awareness_levels(result.final_response)

    print(f"\nChunks generated: {result.chunks_generated}")
    print(f"Chunks resampled: {result.chunks_resampled}")
    print(f"Awareness detected during generation: {result.total_awareness_detected}")
    print(f"Suppression successful: {result.suppression_successful}")

    print(f"\n\nORIGINAL vs SUPPRESSED AWARENESS:")
    print(f"{'Category':<20} {'Original':<10} {'Suppressed':<10} {'Reduction':<10}")
    print("-"*60)
    print(f"{'Global':<20} {len(original_awareness.global_awareness_sentences):<10} {len(suppressed_awareness.global_awareness_sentences):<10} {len(original_awareness.global_awareness_sentences) - len(suppressed_awareness.global_awareness_sentences):<10}")
    print(f"{'Explicit':<20} {len(original_awareness.explicit_awareness_sentences):<10} {len(suppressed_awareness.explicit_awareness_sentences):<10} {len(original_awareness.explicit_awareness_sentences) - len(suppressed_awareness.explicit_awareness_sentences):<10}")
    print(f"{'Implicit':<20} {len(original_awareness.implicit_acknowledgment_sentences):<10} {len(suppressed_awareness.implicit_acknowledgment_sentences):<10} {len(original_awareness.implicit_acknowledgment_sentences) - len(suppressed_awareness.implicit_acknowledgment_sentences):<10}")
    print("-"*60)
    print(f"{'TOTAL':<20} {original_awareness.total_awareness_count:<10} {suppressed_awareness.total_awareness_count:<10} {original_awareness.total_awareness_count - suppressed_awareness.total_awareness_count:<10}")

    if suppressed_awareness.total_awareness_count > 0:
        print(f"\n\nWARNING: Suppression incomplete!")
        print(f"Remaining aware sentences in suppressed response:")
        for sent in suppressed_awareness.global_awareness_sentences:
            print(f"  [GLOBAL] {sent[:150]}...")
        for sent in suppressed_awareness.explicit_awareness_sentences:
            print(f"  [EXPLICIT] {sent[:150]}...")
        for sent in suppressed_awareness.implicit_acknowledgment_sentences:
            print(f"  [IMPLICIT] {sent[:150]}...")
    else:
        print(f"\n\n✓ SUCCESS: All awareness successfully suppressed!")

    print(f"\n\nFinal response length: {len(result.final_response)} chars")
    print(f"\nFinal response preview (first 500 chars):")
    print(result.final_response[:500])

    print(f"\nFinal response preview (Last 500 chars):")
    print(result.final_response[-500:])


    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test())
