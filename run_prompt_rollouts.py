#!/usr/bin/env python3
"""
Run rollouts of extracted prompts through target models (async).

Usage:
    python run_prompt_rollouts.py --category behavioral_change --limit 5 --rollouts-per-prompt 3
    python run_prompt_rollouts.py --input working/extracted_prompts/behavioral_change/*.yaml --rollouts-per-prompt 5 --seed 0

Output filename format: {source_checksum}_{model}_{seed}.json
"""

import yaml
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class PromptRollout:
    """Run extracted prompts through models (async)."""

    def __init__(self, model: str = "qwen/qwen3-32b", output_dir: str = "suppression", concurrency: int = 5):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

    def load_prompt_file(self, filepath: Path) -> Dict:
        """Load extracted prompt YAML."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def add_answer_format_instruction(self, user_prompt: str) -> str:
        """Add instruction to box the final answer."""
        instruction = "\n\n---\nIMPORTANT: After providing your reasoning, you must give your final answer in this exact format:\n\n\\boxed{A} or \\boxed{B}\n\nPlace this at the end of your response."
        return user_prompt + instruction

    async def run_scenario(self, system_prompt: str, user_prompt: str, scenario_type: str, seed: int) -> Dict:
        """Run a single scenario."""
        # Add boxing instruction to user prompt
        formatted_prompt = self.add_answer_format_instruction(user_prompt)

        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7,
                max_tokens=32000,
                seed=seed
            )

            return {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,  # Store original without instruction
                'user_prompt_with_format': formatted_prompt,  # Store version sent to model
                'response': response.choices[0].message.content,
                'model': self.model,
                'scenario_type': scenario_type,
                'seed': seed
            }

    async def run_both_scenarios(self, filepath: Path, seed: int) -> Dict:
        """Run both deployment and evaluation scenarios in parallel."""
        data = self.load_prompt_file(filepath)

        scenario = data.get('scenario', 'unknown')
        target_model = data.get('target_model', 'unknown')

        system_prompt = data.get('system_prompt', '')
        user_prompt_deployment = data.get('user_prompt_deployment', '')
        user_prompt_evaluation = data.get('user_prompt_evaluation', '')

        tasks = []

        if user_prompt_deployment and user_prompt_deployment != "NOT FOUND":
            tasks.append(self.run_scenario(system_prompt, user_prompt_deployment, 'deployment', seed))

        if user_prompt_evaluation and user_prompt_evaluation != "NOT FOUND":
            tasks.append(self.run_scenario(system_prompt, user_prompt_evaluation, 'evaluation', seed))

        if not tasks:
            return {
                'metadata': {
                    'original_file': str(filepath),
                    'scenario': scenario,
                    'original_target_model': target_model,
                    'rollout_model': self.model,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'No valid prompts found'
                }
            }

        results = await asyncio.gather(*tasks, return_exceptions=True)

        deployment_result = None
        evaluation_result = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ✗ Error: {result}")
                continue

            if result['scenario_type'] == 'deployment':
                deployment_result = result
            else:
                evaluation_result = result

        return {
            'metadata': {
                'original_file': str(filepath),
                'scenario': scenario,
                'original_target_model': target_model,
                'rollout_model': self.model,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            },
            'deployment': deployment_result,
            'evaluation': evaluation_result
        }

    def get_output_path(self, source_file: Path, seed: int) -> Path:
        """Get output filepath for a rollout."""
        source_checksum = abs(hash(str(source_file))) % (10**16)
        model_safe = self.model.replace('/', '_')
        filename = f"{source_checksum:016d}_{model_safe}_seed{seed}.json"
        return self.output_dir / filename

    def save_rollout(self, result: Dict, source_file: Path, seed: int):
        """Save rollout result."""
        filepath = self.get_output_path(source_file, seed)

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        return filepath


def find_prompt_files(category: Optional[str] = None, input_files: Optional[List[str]] = None) -> List[Path]:
    """Find extracted prompt YAML files."""
    if input_files:
        return [Path(f) for f in input_files if Path(f).exists()]

    base_dir = Path('working/extracted_prompts')

    if not base_dir.exists():
        print(f"Error: {base_dir} not found!")
        exit(1)

    files = []

    if category:
        category_dir = base_dir / category
        if category_dir.exists():
            files = list(category_dir.glob('*.yaml'))
    else:
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                files.extend(list(subdir.glob('*.yaml')))

    return files


async def process_single_rollout(runner: PromptRollout, filepath: Path, seed: int):
    """Process and save a single rollout immediately."""
    result = await runner.run_both_scenarios(filepath, seed)
    output_path = runner.save_rollout(result, filepath, seed)
    print(f"[{filepath.name}] seed={seed} ✓ {output_path.name}")
    return result

async def process_files(files: List[Path], runner: PromptRollout, rollouts_per_prompt: int, start_seed: int):
    """Process all files with multiple rollouts concurrently, saving immediately."""
    tasks = []

    for filepath in files:
        for rollout_idx in range(rollouts_per_prompt):
            seed = start_seed + rollout_idx
            output_path = runner.get_output_path(filepath, seed)

            if output_path.exists():
                print(f"[{filepath.name}] seed={seed} → SKIPPED (already exists)")
                continue

            task = process_single_rollout(runner, filepath, seed)
            tasks.append(task)

    if not tasks:
        print("All rollouts already exist, nothing to process")
        return []

    print(f"Running {len(tasks)} rollouts...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"✗ Error: {result}")

    return results


async def main_async(args):
    print("=" * 100)
    print("PROMPT ROLLOUT RUNNER (ASYNC)")
    print("=" * 100)
    print()
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Rollouts per prompt: {args.rollouts_per_prompt}")
    print(f"Starting seed: {args.seed}")
    print(f"Output: {args.output_dir}/")
    print()

    files = find_prompt_files(args.category, args.input)

    if args.limit:
        files = files[:args.limit]

    total_possible = len(files) * args.rollouts_per_prompt
    print(f"Processing {len(files)} prompt files × {args.rollouts_per_prompt} rollouts = {total_possible} total")
    print()

    runner = PromptRollout(model=args.model, output_dir=args.output_dir, concurrency=args.concurrency)

    await process_files(files, runner, args.rollouts_per_prompt, args.seed)

    print()
    print("=" * 100)
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Run prompt rollouts (async)')
    parser.add_argument('--model', default='qwen/qwen3-32b',
                       help='Model to use for rollouts (default: Qwen3 32B)')
    parser.add_argument('--category',
                       choices=['behavioral_change', 'no_behavior_change', 'not_working'],
                       help='Only process prompts from this category')
    parser.add_argument('--input', nargs='+', help='Specific YAML files to process')
    parser.add_argument('--limit', type=int, help='Limit number of prompts to process')
    parser.add_argument('--output-dir', default='suppression',
                       help='Output directory (default: suppression)')
    parser.add_argument('--concurrency', type=int, default=5,
                       help='Number of concurrent requests (default: 5)')
    parser.add_argument('--rollouts-per-prompt', type=int, default=1,
                       help='Number of rollouts per prompt with different seeds (default: 1)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Starting seed value (increments for each rollout, default: 0)')

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
