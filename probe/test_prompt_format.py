"""Test script to verify prompt formatting with chat template."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from probe.response_data_prep import load_all_response_pairs

def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    # Load one pair (from behavioral_change subdirectory)
    print("\nLoading response pairs...")
    pairs = load_all_response_pairs(
        "working/response_categorization/v1/qwen_qwen3-32b/behavioral_change",
        limit=1,
        verbose=True,
        tokenizer=tokenizer
    )

    if not pairs:
        print("No pairs loaded!")
        return

    pair = pairs[0]

    print("\n" + "="*80)
    print("DEPLOYMENT CONVERSATION (full context):")
    print("="*80)
    print(pair.deployment_full)

    print("\n" + "="*80)
    print("EVALUATION CONVERSATION (full context):")
    print("="*80)
    print(pair.evaluation_full)

    print("\n" + "="*80)
    print("DEPLOYMENT TEXT ONLY (assistant response):")
    print("="*80)
    print(pair.deployment_text[:200] + "...")

    print("\n" + "="*80)
    print("Fields populated:")
    print("="*80)
    print(f"deployment_full is None: {pair.deployment_full is None}")
    print(f"evaluation_full is None: {pair.evaluation_full is None}")
    print(f"deployment_text length: {len(pair.deployment_text)}")
    print(f"deployment_full length: {len(pair.deployment_full) if pair.deployment_full else 0}")

if __name__ == "__main__":
    main()
