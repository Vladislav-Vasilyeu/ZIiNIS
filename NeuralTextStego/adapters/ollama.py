"""Адаптер локальных языковых моделей через HTTP API Ollama."""

from __future__ import annotations

import json
from typing import Any

import httpx

from adapters.base import BaseLLMAdapter
from config import settings


class OllamaAdapter(BaseLLMAdapter):
    name = "ollama"
    allows_partial_candidates = True

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.ollama_base_url.rstrip("/"),
                headers={"Content-Type": "application/json"},
                timeout=settings.api_timeout,
            )

    async def _request_candidates(
        self,
        text: str,
        word: str,
        num_candidates: int,
        *,
        temperature: float,
    ) -> list[str]:
        requested_count = num_candidates
        prompt = (
            f"Контекст: «{text}»\n"
            f"Предложи до {requested_count} естественных замен слова «{word}», "
            "подходящих по смыслу, стилю, части речи и грамматической форме. "
            "Не выдумывай редкие или неестественные варианты. Если подходящих замен "
            "меньше, верни только хорошие варианты; исходное слово не включай."
        )
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ты подбираешь естественные замены слов в русском тексте. "
                                "Отвечай только по заданной JSON-схеме."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "format": schema,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": settings.ollama_num_ctx,
                        "num_predict": 512,
                    },
                    "stream": False,
                    "keep_alive": "5m",
                },
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(
                f"Не удалось подключиться к Ollama по {settings.ollama_base_url}. "
                "Установите и запустите Ollama, затем скачайте модель командой "
                f"`ollama pull {settings.ollama_model}`"
            ) from exc
        if response.status_code != 200:
            details = response.text[:500]
            raise RuntimeError(
                f"Ollama API error {response.status_code}: {details}"
            )

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Ollama вернула ответ в неожиданном формате") from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Локальная модель Ollama вернула пустой ответ")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Локальная модель Ollama вернула некорректный JSON") from exc

        candidates = result.get("candidates") if isinstance(result, dict) else None
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) for candidate in candidates
        ):
            raise RuntimeError("Ollama вернула JSON без корректного списка candidates")
        return candidates
