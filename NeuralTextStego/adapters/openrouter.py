"""
Адаптер OpenRouter (OpenAI-совместимый, доступ к множеству моделей).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from adapters.base import BaseLLMAdapter
from config import settings


class OpenRouterAdapter(BaseLLMAdapter):
    name = "openrouter"
    allows_partial_candidates = True
    API_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openrouter/free"

    async def _ensure_client(self) -> None:
        if self._client is None:
            if not settings.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY не задан в .env")
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/neuraltextstego",
                    "X-Title": "NeuralTextStego",
                },
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
            f"Подбери до {requested_count} разных естественных синонимов слова или выражения "
            f"«{word}», подходящих по смыслу, стилю, части речи и грамматической форме "
            f"к этому контексту. Не повторяй исходное слово и не выдумывай редкие "
            "или неестественные варианты. Если хороших вариантов меньше, верни только их.\n"
            "Верни JSON-объект с одним полем candidates — массивом строк. "
            "Не добавляй другие поля, markdown или пояснения."
        )

        body: dict[str, Any] = {
            "model": self.DEFAULT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты подбираешь естественные замены слов в русском тексте. "
                        "Соблюдай указанный формат ответа и не добавляй пояснений."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": False,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "synonym_candidates",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["candidates"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        resp = await self._client.post("/chat/completions", json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter вернул ответ в неожиданном формате") from exc
        if not isinstance(content, str) or not content.strip():
            finish_reason = data["choices"][0].get("finish_reason")
            model = data.get("model", "unknown")
            raise RuntimeError(
                f"Модель {model} не вернула текст вариантов "
                f"(finish_reason={finish_reason!r}); возможно, лимита токенов не хватило"
            )

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            model = data.get("model", "unknown")
            raise RuntimeError(f"Модель {model} вернула некорректный JSON") from exc

        candidates = result.get("candidates") if isinstance(result, dict) else None
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) for candidate in candidates
        ):
            raise RuntimeError("OpenRouter вернул JSON без корректного списка candidates")
        return candidates
