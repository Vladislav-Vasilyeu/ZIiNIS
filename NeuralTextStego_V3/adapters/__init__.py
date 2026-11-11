"""
Адаптеры к API языковых моделей.
"""

from .base import BaseLLMAdapter, CandidateResult
from .classic import ClassicSynonymAdapter
from .openrouter import OpenRouterAdapter
from .ollama import OllamaAdapter

__all__ = [
    "BaseLLMAdapter",
    "CandidateResult",
    "ClassicSynonymAdapter",
    "OpenRouterAdapter",
    "OllamaAdapter",
]
