"""
On-policy suppression package.

Clean OpenRouter-based implementation with thinking token support.
"""

from .engine import generate_with_suppression, SuppressionResult
from .openrouter_client import OpenRouterClient, ThinkingResponse

__all__ = [
    'generate_with_suppression',
    'SuppressionResult',
    'OpenRouterClient',
    'ThinkingResponse',
]
