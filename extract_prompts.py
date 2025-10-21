#!/usr/bin/env python3
"""
Extract system_prompt, user_prompt_deployment, and user_prompt_evaluation from Petri transcript JSON.

Usage:
    python extract_prompts.py <transcript_file.json> [output.yaml]
"""

import json
import yaml
import sys

def extract_prompts(transcript_path):
    with open(transcript_path) as f:
        data = json.load(f)

    events = data.get('events', [])

    system_prompt = None
    all_deployment_messages = []
    all_evaluation_messages = []
    auditor_text_content = []  # For auditor's analysis/summary text
    current_branch = 'deployment'  # Track which branch we're in

    # Search through events
    for i, event in enumerate(events):
        if isinstance(event, dict):
            event_type = event.get('type')

            # Check transcript_event for auditor's final summary
            if event_type == 'transcript_event' and event.get('view') == 'auditor':
                edit = event.get('edit', {})
                if isinstance(edit, dict):
                    patch = edit.get('patch', [])
                    if isinstance(patch, list):
                        for p in patch:
                            if isinstance(p, dict):
                                value = p.get('value', {})
                                if isinstance(value, dict) and value.get('role') == 'assistant':
                                    text_content = value.get('content', '')
                                    if isinstance(text_content, str) and len(text_content) > 100:
                                        auditor_text_content.append((i, text_content))

            # Also check decision_event
            elif event_type == 'decision_event':
                content = event.get('content')
                if isinstance(content, dict):
                    role = content.get('role')

                    # Extract auditor's text content (analysis/summary)
                    if role == 'assistant' and content.get('source') == 'generate':
                        text_content = content.get('content', '')
                        if isinstance(text_content, str) and text_content.strip():
                            # Skip attachment references
                            if not text_content.startswith('attachment://'):
                                auditor_text_content.append((i, text_content))

                    tool_calls = content.get('tool_calls')
                    if tool_calls and isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                func_name = tc.get('function')
                                args = tc.get('arguments')

                                if not isinstance(args, dict):
                                    continue

                                # Extract system prompt/message
                                if func_name in ['set_target_system_prompt', 'set_target_system_message']:
                                    sp = args.get('system_prompt') or args.get('system_message')
                                    if sp and not system_prompt:
                                        system_prompt = sp

                                # Detect rollback - this likely indicates switching from deployment to evaluation branch
                                elif func_name == 'rollback':
                                    # After a rollback that goes back significantly, we're likely entering eval branch
                                    if all_deployment_messages:
                                        current_branch = 'evaluation'

                                # Extract user messages
                                elif func_name == 'send_message':
                                    msg = args.get('message', '')
                                    if msg:
                                        # Check for evaluation cues
                                        eval_cues = [
                                            'DEBUG_LOG',
                                            'ai_recommendation_quality_eval',
                                            'advice_eval_',
                                            'response_quality_test',
                                            '/evals/',
                                            'qa-testing@',
                                            '_eval.xlsx',
                                            '_eval.csv',
                                            'response_analysis',
                                            'test_id',
                                            'experiment_id'
                                        ]

                                        has_eval_cues = any(cue.lower() in msg.lower() for cue in eval_cues)

                                        # Determine branch based on cues AND current branch tracking
                                        if has_eval_cues or current_branch == 'evaluation':
                                            all_evaluation_messages.append((i, msg))
                                            if has_eval_cues:
                                                current_branch = 'evaluation'
                                        else:
                                            all_deployment_messages.append((i, msg))

    # Separate main prompts from subprompts
    deployment_prompt = all_deployment_messages[0][1] if all_deployment_messages else None
    deployment_subprompts = [msg for i, msg in all_deployment_messages[1:]] if len(all_deployment_messages) > 1 else []

    evaluation_prompt = all_evaluation_messages[0][1] if all_evaluation_messages else None
    evaluation_subprompts = [msg for i, msg in all_evaluation_messages[1:]] if len(all_evaluation_messages) > 1 else []

    # Filter out acknowledgment messages (short messages like "Thanks for the advice!")
    # These are typically < 200 chars and don't expect a substantial response
    deployment_subprompts = [msg for msg in deployment_subprompts if len(msg) > 200]
    evaluation_subprompts = [msg for msg in evaluation_subprompts if len(msg) > 200]

    # Extract auditor's closing analysis (last few text content items that are substantial)
    # Filter for text that looks like analysis/summary (longer than 100 chars)
    auditor_analysis = [text for i, text in auditor_text_content if len(text) > 100]

    # Get the final analysis text (typically at the end)
    auditor_closing = auditor_analysis[-1] if auditor_analysis else ""

    return {
        'system_prompt': system_prompt if system_prompt else "NOT FOUND",
        'user_prompt_deployment': deployment_prompt if deployment_prompt else "NOT FOUND",
        'user_subprompts_deployment': deployment_subprompts,
        'user_prompt_evaluation': evaluation_prompt if evaluation_prompt else "NOT FOUND",
        'user_subprompts_evaluation': evaluation_subprompts,
        'auditor_closing_summary': auditor_closing
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    transcript_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'extracted_prompts.yaml'

    prompts = extract_prompts(transcript_file)

    # Write to YAML file
    # Use custom representer for multi-line strings to avoid quote escaping
    def str_representer(dumper, data):
        if '\n' in data or len(data) > 80:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer)

    with open(output_file, 'w') as f:
        yaml.dump(prompts, f, default_flow_style=False, allow_unicode=True, width=1000, sort_keys=False)

    print(f"✓ Extracted prompts from {transcript_file}")
    print(f"✓ Wrote to {output_file}")
    print(f"\nFound:")
    print(f"  - System prompt: {'YES' if prompts['system_prompt'] != 'NOT FOUND' else 'NO'}")
    print(f"  - Deployment prompt: {'YES' if prompts['user_prompt_deployment'] != 'NOT FOUND' else 'NO'}")
    print(f"    - Deployment subprompts: {len(prompts['user_subprompts_deployment'])}")
    print(f"  - Evaluation prompt: {'YES' if prompts['user_prompt_evaluation'] != 'NOT FOUND' else 'NO'}")
    print(f"    - Evaluation subprompts: {len(prompts['user_subprompts_evaluation'])}")
    print(f"  - Auditor closing summary: {'YES' if prompts['auditor_closing_summary'] else 'NO'} ({len(prompts['auditor_closing_summary'])} chars)")
