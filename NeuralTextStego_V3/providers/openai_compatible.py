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
        try:
            response = await self._ensure().get("models")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._http_error(exc.response.status_code) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Нет соединения с {self.label}. Проверьте сеть и повторите запрос.") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.label}: некорректный JSON каталога моделей.") from exc
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"{self.label}: каталог моделей не содержит список data.")
        models = sorted({item["id"] for item in items
                         if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()})
        if not models:
            raise RuntimeError(f"{self.label}: список моделей пуст. ID можно ввести вручную.")
        return models

    def _http_error(self, status: int) -> RuntimeError:
        hint = {
            400: "Проверьте модель и поддержку параметров запроса.",
            401: "Проверьте API-ключ в .env и перезапустите приложение.",
            402: "Проверьте баланс и доступ к модели в кабинете провайдера.",
            403: "Ключу запрещён доступ к этой модели или сервису.",
            404: "Модель или endpoint не найдены. Обновите список моделей.",
            429: "Превышен лимит запросов или квота. Подождите и проверьте лимиты провайдера.",
        }.get(status, "Сервис временно недоступен. Повторите запрос позже." if status >= 500 else "Проверьте настройки провайдера.")
        return RuntimeError(f"{self.label}: HTTP {status} для модели {self.model}. {hint}")

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
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"{self.label}: превышено время ожидания. Повторите запрос или выберите другую модель.") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Не удалось выполнить запрос к {self.label}. Проверьте подключение к сети.") from exc

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
            raise self._http_error(response.status_code)
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
