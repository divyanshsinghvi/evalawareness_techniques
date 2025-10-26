#!/usr/bin/env python3
"""Test the OpenRouter client directly."""

import asyncio
from dotenv import load_dotenv
import json
from dataclasses import asdict

load_dotenv()

from suppression.openrouter_client import OpenRouterClient


async def test_client():
    """Test OpenRouter client prefill support across different providers."""

    # # TEST 1: Basic generation with system prompt (COMMENTED OUT)
    # print("="*80)
    # print("TEST 1: Basic generation with system prompt")
    # print("="*80)
    #
    # client = OpenRouterClient(
    #     model="qwen/qwen3-32b",
    #     provider="SiliconFlow",
    #     thinking_tag="thinking",
    #     temperature=1.0,
    #     max_tokens=500,
    #     verbose=True
    # )
    #
    # # Test 1: Basic message list with system prompt
    # messages = [
    #     {"role": "system", "content": "You are a math tutor. Be concise."},
    #     {"role": "user", "content": "What is 2+2?"}
    # ]
    #
    # response = await client.generate(messages)
    #
    # print(f"\nResponse:")
    # print(f"  Reasoning length: {len(response.reasoning)} chars")
    # print(f"  Content length: {len(response.content)} chars")
    # print(f"\n  Full Reasoning:")
    # print(response.reasoning)
    # print(f"\n  Full Content:")
    # print(response.content)

    # # TEST 2: With prefilled content (COMMENTED OUT)
    # print("\n" + "="*80)
    # print("TEST 2: With prefilled reasoning")
    # print("="*80)
    #
    # base_messages = [
    #     {"role": "system", "content": "You are a math tutor."},
    #     {"role": "user", "content": "What is 5+5?"}
    # ]
    #
    # prefilled_messages = client.format_messages_with_prefill(
    #     base_messages=base_messages,
    #     prefill_reasoning="Let me calculate this step by step. First, I need to add 5 and",
    #     prefill_content=""
    # )
    #
    # print(f"\nPrefilled messages (using <think> tags):")
    # for i, msg in enumerate(prefilled_messages):
    #     print(f"  {i+1}. {msg['role']}:")
    #     print(f"     {msg['content']}")
    #
    # response2 = await client.generate(prefilled_messages)
    #
    # print(f"\nResponse:")
    # print(f"  Full Reasoning:")
    # print(response2.reasoning)
    # print(f"\n  Full Content:")
    # print(response2.content)

    # Test 3: Check prefill continuation across different providers
    print("="*80)
    print("TEST 3: Verify prefill continuation across providers")
    print("="*80)

    providers = ["nCompass", "DeepInfra", "Nebius", "Novita", "SiliconFlow"]

    for provider in providers:
        print(f"\n{'='*80}")
        print(f"Testing provider: {provider}")
        print(f"{'='*80}")

        try:
            client = OpenRouterClient(
                #model="qwen/qwen3-32b",
                model="qwen/qwen3-32b",
                provider=provider,
                temperature=1.0,
                max_tokens=500,
                verbose=False  # Set to True for detailed output
            )

            base_messages = [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 5+5?"}
            ]

            prefilled_messages = client.format_messages_with_prefill(
                base_messages=base_messages,
                prefill_reasoning="Let me calculate this step by step. First, I need to add 5 and",
                prefill_content=""
            )

            print(f"\n📨 REQUEST MESSAGES:")
            print(json.dumps(prefilled_messages, indent=2))

            response = await client.generate(prefilled_messages)

            print(f"\n📥 RESPONSE (ThinkingResponse):")
            print(json.dumps(asdict(response), indent=2))

            # Check if response continues from prefill
            prefill_text = "Let me calculate this step by step. First, I need to add 5 and"

            # Check various continuation indicators
            continuation_words = ["and 5", "5 together", "together", "equals", "10"]
            found_continuation = any(word in response.reasoning[:200].lower() for word in continuation_words)

            # Check if it started fresh (ignoring prefill)
            fresh_starts = ["okay", "the user", "let me", "to solve", "we need"]
            started_fresh = any(response.reasoning.lower().startswith(word) for word in fresh_starts)

            print(f"\nPrefilled: '{prefill_text}'")
            print(f"Response start: '{response.reasoning[:150].strip()}...'")
            print(f"\nReasoning length: {len(response.reasoning)} chars")
            print(f"Content length: {len(response.content)} chars")

            # Verdict
            if found_continuation and not started_fresh:
                print(f"\n✓ {provider}: SUPPORTS prefilling (continuation detected)")
            elif len(response.reasoning) > 0 and not started_fresh:
                print(f"\n? {provider}: UNCLEAR (has reasoning but unclear continuation)")
            else:
                print(f"\n✗ {provider}: DOES NOT support prefilling (started fresh or no reasoning)")

        except Exception as e:
            print(f"\n✗ {provider}: ERROR - {str(e)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(test_client())
