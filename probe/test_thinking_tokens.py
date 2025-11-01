"""Test thinking token detection after fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from probe.load_cached_activations import find_thinking_token_range

def test_thinking_token_detection():
    """Test that thinking token ranges are correctly identified."""

    print("="*80)
    print("TESTING THINKING TOKEN DETECTION")
    print("="*80)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    print("✓ Tokenizer loaded")

    # Create test response with thinking tags
    test_response = """<thinking>
This is my internal reasoning.
I need to think carefully about this problem.
The answer should consider multiple factors.
</thinking>

Based on my analysis, the answer is B."""

    print("\n[2/4] Test response:")
    print("-" * 40)
    print(test_response)
    print("-" * 40)

    # Tokenize
    print("\n[3/4] Tokenizing...")
    tokens = tokenizer.encode(test_response, add_special_tokens=False)
    tokens_tensor = torch.tensor(tokens)

    print(f"Total tokens: {len(tokens)}")

    # Decode thinking tag tokens to verify
    print("\nThinking tag tokens:")
    print(f"  <thinking>: {tokenizer.encode('<thinking>', add_special_tokens=False)}")
    print(f"  </thinking>: {tokenizer.encode('</thinking>', add_special_tokens=False)}")

    # Find thinking token range
    print("\n[4/4] Finding thinking token range...")
    start_idx, end_idx = find_thinking_token_range(
        tokens_tensor,
        start_idx=0,
        end_idx=len(tokens)
    )

    print(f"\nResults:")
    print(f"  Thinking start index: {start_idx}")
    print(f"  Thinking end index: {end_idx}")
    print(f"  Thinking tokens count: {end_idx - start_idx}")

    # Verify by decoding the thinking content
    if start_idx >= 0 and end_idx > start_idx:
        thinking_tokens = tokens[start_idx:end_idx]
        thinking_text = tokenizer.decode(thinking_tokens)

        print(f"\n✓ SUCCESS: Found thinking tokens!")
        print("\nDecoded thinking content:")
        print("-" * 40)
        print(thinking_text)
        print("-" * 40)

        # Verify it doesn't include the tags themselves
        if '<thinking>' not in thinking_text and '</thinking>' not in thinking_text:
            print("\n✓ CORRECT: Tags are excluded from range")
        else:
            print("\n✗ WARNING: Tags are included in range (unexpected)")

        return True
    else:
        print("\n✗ FAILED: Could not find thinking tokens")
        print("Using fallback heuristic instead")
        return False

def test_think_tags():
    """Test <think> tags as well."""

    print("\n" + "="*80)
    print("TESTING <think> TAGS")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    test_response = """<think>Short reasoning here.</think>

The answer is A."""

    print("\nTest response:")
    print(test_response)

    tokens = tokenizer.encode(test_response, add_special_tokens=False)
    tokens_tensor = torch.tensor(tokens)

    start_idx, end_idx = find_thinking_token_range(
        tokens_tensor,
        start_idx=0,
        end_idx=len(tokens)
    )

    if start_idx >= 0 and end_idx > start_idx:
        thinking_text = tokenizer.decode(tokens[start_idx:end_idx])
        print(f"\n✓ Found <think> content: {repr(thinking_text)}")
        return True
    else:
        print("\n✗ Failed to find <think> tokens")
        return False

if __name__ == "__main__":
    result1 = test_thinking_token_detection()
    result2 = test_think_tags()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"<thinking> tags: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"<think> tags: {'✓ PASS' if result2 else '✗ FAIL'}")
    print("="*80)
