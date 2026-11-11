"""
Фабрика адаптеров.
"""

from __future__ import annotations

from adapters.base import BaseLLMAdapter
from adapters.classic import ClassicSynonymAdapter
from adapters.ollama import OllamaAdapter

# OpenRouter отключён в приложении (модуль adapters/openrouter.py сохранён):
# from adapters.openrouter import OpenRouterAdapter


def create_adapter(provider: str) -> BaseLLMAdapter:
    """Создать адаптер по имени провайдера."""
    provider = provider.lower().strip()
    # if provider == "openrouter":
    #     return OpenRouterAdapter()
    if provider == "openrouter":
        raise ValueError("Провайдер OpenRouter отключён. Доступны: ollama, classic")
    if provider == "ollama":
        return OllamaAdapter()
    if provider == "classic":
        return ClassicSynonymAdapter()
    raise ValueError(
        f"Неизвестный провайдер: {provider}. Доступны: ollama, classic"
    )
