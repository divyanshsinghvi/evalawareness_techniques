#!/usr/bin/env python3
"""
Simple OpenRouter client with thinking token support.

Handles <think> tag prefilling for models like qwen3-30b-a3b-thinking-2507.
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import httpx


@dataclass
class ThinkingResponse:
    """Response from a thinking model."""
    content: str  # Visible content (without thinking)
    reasoning: str  # Thinking/reasoning content
    raw_response: str  # Full response including thinking tags


# Model name to thinking tag mapping
MODEL_THINKING_TAG_MAP = {
    # Qwen models
    "qwen/qwen3-32b": "think",  # qwen3-32b uses <think>
    "qwen/qwen3-30b-a3b-thinking-2507": "thinking",  # qwen3-30b uses <thinking>
    "qwen/qwq-32b-preview": "think",
    # DeepSeek models use <think>
    "deepseek/deepseek-r1": "think",
    # Add more models here as needed
}


class OpenRouterClient:
    """Simple async OpenRouter client with thinking support."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1,
        max_tokens: int = 4000,
        seed: int = 0,
        timeout: int = 300,
        verbose: bool = False,
        provider: Optional[Union[str, List[str]]] = None,
        thinking_tag: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.verbose = verbose
        self.provider = provider  # Can be str or List[str]
        self.seed = seed

        # Auto-detect thinking tag from model name if not provided
        if thinking_tag is None:
            self.thinking_tag = MODEL_THINKING_TAG_MAP.get(model, "think")
            if verbose:
                print(f"Auto-detected thinking tag: <{self.thinking_tag}> for model {model}")
        else:
            raise("Don't use put it in map the thinking token -_-")
            self.thinking_tag = thinking_tag
            if verbose:
                print(f"Using specified thinking tag: <{self.thinking_tag}>")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None
    ) -> ThinkingResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override default max_tokens

        Returns:
            ThinkingResponse with content, reasoning, and raw response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "seed": seed if seed is not None else None
        }

        # Add provider preference with no fallback if specified
        if self.provider:
            # Convert string to list if needed
            provider_list = [self.provider] if isinstance(self.provider, str) else self.provider
            payload["provider"] = {
                "order": provider_list,
                "allow_fallbacks": False
            }

        if self.verbose:
            print(f"Calling OpenRouter API:")
            print(f"  Model: {self.model}")
            if self.provider:
                provider_str = ", ".join(provider_list) if isinstance(provider_list, list) else str(self.provider)
                print(f"  Provider(s): {provider_str} (no fallback)")
            print(f"  Messages: {len(messages)} messages")
            for i, msg in enumerate(messages):
                print(f"    {i+1}. {msg['role']}:")
                print(f"       {msg['content']}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"OpenRouter API error {response.status_code}: {error_text}"
                )

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                raise RuntimeError(f"No choices in API response: {result}")

            if self.verbose:
                print(f"\n  Full API response:")
                import json
                print(json.dumps(result, indent=2))

            message = result["choices"][0]["message"]

            if self.verbose:
                print(f"\n  Message object keys: {list(message.keys())}")
                print(f"  Message content:")
                for key, value in message.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"    {key}: {value[:200]}... (truncated, length={len(value)})")
                    else:
                        print(f"    {key}: {value}")

            # Extract reasoning and content directly from OpenRouter response
            # OpenRouter returns these in separate fields, no need to parse tags
            reasoning = message.get("reasoning", "")
            content = message.get("content", "")
            raw_response = f"<{self.thinking_tag}>{reasoning}</{self.thinking_tag}>{content}" if reasoning else content

            return ThinkingResponse(
                content=content,
                reasoning=reasoning,
                raw_response=raw_response
            )

    def format_messages_with_prefill(
        self,
        base_messages: List[Dict[str, str]],
        prefill_content: str = "",
        prefill_reasoning: str = ""
    ) -> List[Dict[str, str]]:
        """
        Format messages with optional prefilled content/reasoning.

        Uses thinking tags to prefill reasoning, then adds content.

        Args:
            base_messages: Base messages (system + user)
            prefill_content: Content to prefill in assistant message
            prefill_reasoning: Reasoning to prefill in thinking tags

        Returns:
            Messages with assistant prefill added if needed
        """
        messages = base_messages.copy()

        if prefill_reasoning or prefill_content:
            assistant_content = ""

            if prefill_reasoning:
                # Add thinking in tags (no closing tag = model continues)
                # Add newline after opening tag for readability
                assistant_content = f"<{self.thinking_tag}>\n{prefill_reasoning}"
                if prefill_content:
                    # Close thinking tag and add content
                    assistant_content += f"\n</{self.thinking_tag}>\n{prefill_content}"
            elif prefill_content:
                # Only content, no thinking
                assistant_content = prefill_content

            messages.append({
                "role": "assistant",
                "content": assistant_content
            })

        return messages
