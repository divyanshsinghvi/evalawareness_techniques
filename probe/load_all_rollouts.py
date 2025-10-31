"""Load rollouts from all sources: original + suppression + steered."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RolloutPair:
    """Rollout with deployment and evaluation responses."""
    source_file: str
    source_type: str  # 'original', 'suppression', 'steered'

    # Texts
    system_prompt: str
    deployment_user: str
    deployment_assistant: str
    evaluation_user: str
    evaluation_assistant: str

    # Choices (if available)
    deployment_choice: Optional[str] = None
    evaluation_choice: Optional[str] = None

    # Metadata
    prompt_id: str = ""
    seed: int = 0

    @property
    def behavioral_change(self) -> Optional[bool]:
        """Check if choice changed between deployment and evaluation."""
        if self.deployment_choice and self.evaluation_choice:
            return self.deployment_choice != self.evaluation_choice
        return None


def load_rollout_file(path: str, source_type: str) -> Optional[RolloutPair]:
    """Load a single rollout YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Extract system prompt
        system_prompt = data.get('system_prompt', '')

        # Extract deployment
        deployment = data.get('deployment', {})
        deploy_user = deployment.get('user_message', '')
        deploy_assistant = deployment.get('assistant_response', '')
        deploy_choice = deployment.get('choice')

        # Extract evaluation
        evaluation = data.get('evaluation', {})
        eval_user = evaluation.get('user_message', '')
        eval_assistant = evaluation.get('assistant_response', '')
        eval_choice = evaluation.get('choice')

        # Validate
        if not deploy_assistant or not eval_assistant:
            return None

        # Extract prompt ID and seed from filename
        filename = Path(path).stem
        if '_seed_' in filename:
            prompt_id = filename.rsplit('_seed_', 1)[0]
            try:
                seed = int(filename.rsplit('_seed_', 1)[1])
            except:
                seed = 0
        else:
            prompt_id = filename
            seed = 0

        return RolloutPair(
            source_file=path,
            source_type=source_type,
            system_prompt=system_prompt,
            deployment_user=deploy_user,
            deployment_assistant=deploy_assistant,
            evaluation_user=eval_user,
            evaluation_assistant=eval_assistant,
            deployment_choice=deploy_choice,
            evaluation_choice=eval_choice,
            prompt_id=prompt_id,
            seed=seed
        )

    except Exception as e:
        # print(f"Error loading {path}: {e}")
        return None


def load_all_rollouts(
    include_original: bool = True,
    include_suppression: bool = False,
    include_steered: bool = False,
    limit_per_source: Optional[int] = None,
    verbose: bool = True
) -> List[RolloutPair]:
    """
    Load rollouts from all sources.

    Args:
        include_original: Include original rollouts (~4k)
        include_suppression: Include suppression experiments (~35k)
        include_steered: Include steered experiments (~23k)
        limit_per_source: Limit per source type
        verbose: Print progress

    Returns:
        List of RolloutPair objects
    """
    all_pairs = []

    # 1. Original rollouts
    if include_original:
        if verbose:
            print("Loading original rollouts...")
        orig_dir = "working/rollouts/qwen_qwen3-32b"

        count = 0
        for root, dirs, files in os.walk(orig_dir):
            for filename in files:
                if not filename.endswith('.yaml'):
                    continue

                if limit_per_source and count >= limit_per_source:
                    break

                path = os.path.join(root, filename)
                pair = load_rollout_file(path, 'original')
                if pair:
                    all_pairs.append(pair)
                    count += 1

            if limit_per_source and count >= limit_per_source:
                break

        if verbose:
            print(f"  Loaded {count} original rollouts")

    # 2. Suppression experiments
    if include_suppression:
        if verbose:
            print("Loading suppression experiments...")
        supp_dir = "working/suppression_experiments"

        count = 0
        for root, dirs, files in os.walk(supp_dir):
            for filename in files:
                if not filename.endswith('.yaml'):
                    continue

                if limit_per_source and count >= limit_per_source:
                    break

                path = os.path.join(root, filename)
                pair = load_rollout_file(path, 'suppression')
                if pair:
                    all_pairs.append(pair)
                    count += 1

            if limit_per_source and count >= limit_per_source:
                break

        if verbose:
            print(f"  Loaded {count} suppression rollouts")

    # 3. Steered experiments
    if include_steered:
        if verbose:
            print("Loading steered experiments...")
        steer_dir = "working/steered-outs"

        count = 0
        for root, dirs, files in os.walk(steer_dir):
            for filename in files:
                if not filename.endswith('.yaml'):
                    continue

                if limit_per_source and count >= limit_per_source:
                    break

                path = os.path.join(root, filename)
                pair = load_rollout_file(path, 'steered')
                if pair:
                    all_pairs.append(pair)
                    count += 1

            if limit_per_source and count >= limit_per_source:
                break

        if verbose:
            print(f"  Loaded {count} steered rollouts")

    if verbose:
        print(f"\nTotal loaded: {len(all_pairs)} rollouts")

        # Stats by source
        by_source = {}
        for pair in all_pairs:
            by_source[pair.source_type] = by_source.get(pair.source_type, 0) + 1

        print("By source:")
        for source, count in sorted(by_source.items()):
            print(f"  {source}: {count}")

        # Behavioral change stats
        bc_pairs = [p for p in all_pairs if p.behavioral_change is not None]
        if bc_pairs:
            bc_count = sum(p.behavioral_change for p in bc_pairs)
            print(f"\nBehavioral change: {bc_count}/{len(bc_pairs)} ({100*bc_count/len(bc_pairs):.1f}%)")

    return all_pairs


if __name__ == "__main__":
    # Example: Load 1000 from each source
    pairs = load_all_rollouts(
        include_original=True,
        include_suppression=True,
        include_steered=True,
        limit_per_source=1000,
        verbose=True
    )

    print(f"\nExample pair:")
    if pairs:
        p = pairs[0]
        print(f"  Source: {p.source_type}")
        print(f"  Prompt ID: {p.prompt_id}")
        print(f"  Seed: {p.seed}")
        print(f"  Behavioral change: {p.behavioral_change}")
        print(f"  Deploy text (first 100 chars): {p.deployment_assistant[:100]}...")
