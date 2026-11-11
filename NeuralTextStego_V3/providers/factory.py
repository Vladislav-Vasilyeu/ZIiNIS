"""Фабрика logits-провайдеров."""

from __future__ import annotations

from providers.ollama_logits import OllamaLogitsProvider

from providers.tooken_club import TookenClubProvider

LOGITS_PROVIDERS = {
    "ollama": "Ollama",
    "tooken_club": "Tooken Club",
}


def create_logits_provider(
    name: str,
    *,
    model: str | None = None,
) -> OllamaLogitsProvider | TookenClubProvider:
    key = (name or "ollama").lower().strip()
    if key == "ollama":
        return OllamaLogitsProvider(model=model)
    if key == "tooken_club":
        return TookenClubProvider(model=model)
    if key == "openrouter":
        raise ValueError("Провайдер OpenRouter отключён.")
    raise ValueError(
        f"Неизвестный провайдер: {name}. Доступны: {', '.join(LOGITS_PROVIDERS)}"
    )
