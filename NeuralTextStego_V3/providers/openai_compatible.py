"""Общий клиент OpenAI-совместимых сервисов для текста и top-logprobs."""

from __future__ import annotations

from math import isfinite
from typing import Any

import httpx

from config import settings
from providers.base import LogitsProvider, TokenCandidate


class OpenAICompatibleProvider(LogitsProvider):
    label: str
    API_BASE: str
    api_key_setting: str
    model_setting: str

    def __init__(self, model: str | None = None) -> None:
        self.model = model or getattr(settings, self.model_setting)
        self._client: httpx.AsyncClient | None = None

    def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            key = getattr(settings, self.api_key_setting).strip()
            if not key:
                raise RuntimeError(f"Укажите {self.api_key_setting.upper()} в файле .env.")
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={"Authorization": f"Bearer {key}"},
                timeout=settings.api_timeout,
            )
        return self._client

    async def list_models(self) -> list[str]:
        response = await self._ensure().get("models")
        response.raise_for_status()
        return sorted({item["id"] for item in response.json()["data"]})

    async def _chat(self, prompt: str, **options: Any) -> dict[str, Any]:
        client = self._ensure()
        try:
            response = await client.post(
                "chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    **options,
                },
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"Не удалось выполнить запрос к {self.label}.") from exc

        try:
            data = response.json()
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not response.is_success:
            error = data.get("error")
            if options.get("logprobs") and isinstance(error, dict) and error.get("code") == "unsupported_feature":
                raise self._missing_logprobs()
            raise RuntimeError(
                f"{self.label}: HTTP {response.status_code} для модели {self.model}."
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"{self.label} вернул ответ без choices.")
        return choices[0]

    def _missing_logprobs(self) -> RuntimeError:
        return RuntimeError(
            f"{self.label} не вернул top_logprobs для модели {self.model}. "
            "Для нейросетевого метода выберите модель с поддержкой logprobs или Ollama."
        )

    async def next_token_distribution(
        self, prompt: str, *, top_k: int = 20, temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        limit = max(1, min(top_k, 20))
        choice = await self._chat(
            prompt, max_tokens=1, temperature=temperature, logprobs=True, top_logprobs=limit,
        )
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            raise self._missing_logprobs()
        alternatives = content[0].get("top_logprobs")
        if not isinstance(alternatives, list):
            raise self._missing_logprobs()

        scores: dict[str, float] = {}
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            token, score = item.get("token"), item.get("logprob")
            if isinstance(token, str) and token and isinstance(score, (int, float)) and isfinite(score):
                scores[token] = max(score, scores.get(token, score))
        if not scores:
            raise self._missing_logprobs()
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        return [
            TokenCandidate(token=token, logprob=score, rank=rank)
            for rank, (token, score) in enumerate(ranked)
        ]

    async def complete_text(self, prompt: str, *, max_tokens: int = 400) -> str:
        choice = await self._chat(prompt, max_tokens=max_tokens, temperature=0.8)
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"{self.label} вернул пустой текст.")
        return text.strip()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
