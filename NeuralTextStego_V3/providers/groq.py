"""GroqCloud через OpenAI-совместимый Chat Completions API."""

from typing import Any

from providers.base import TokenCandidate
from providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    label = "Groq"
    API_BASE = "https://api.groq.com/openai/v1"
    api_key_setting = "groq_api_key"
    model_setting = "groq_model"

    async def _chat(self, prompt: str, **options: Any) -> dict[str, Any]:
        options["max_completion_tokens"] = options.pop("max_tokens")
        if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            options["reasoning_effort"] = "low"
        return await super()._chat(prompt, **options)

    async def next_token_distribution(
        self, prompt: str, *, top_k: int = 20, temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        raise RuntimeError(
            "Groq пока не поддерживает logprobs и top_logprobs. "
            "Groq доступен для генерации контейнера. "
            "Для нейросетевого внедрения и извлечения выберите Ollama "
            "и модель с поддержкой logprobs."
        )
