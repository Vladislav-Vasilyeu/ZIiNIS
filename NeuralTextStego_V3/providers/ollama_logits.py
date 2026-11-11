"""Ollama logprobs provider — корректные top-level поля logprobs/top_logprobs."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from config import settings
from providers.base import LogitsProvider, TokenCandidate


class OllamaLogitsProvider(LogitsProvider):
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=settings.api_timeout,
            )
        return self._client

    def _parse_logprobs_payload(self, data: dict[str, Any], top_k: int) -> list[TokenCandidate]:
        """Разбор ответа Ollama (generate/chat) с logprobs."""
        # Формат с Ollama >= 0.12: data["logprobs"] = [{token, logprob, top_logprobs: [...]}]
        logprobs = data.get("logprobs")
        if logprobs is None and isinstance(data.get("message"), dict):
            # на всякий случай — вложенность
            logprobs = data["message"].get("logprobs")

        if not logprobs or not isinstance(logprobs, list):
            return []

        entry = logprobs[0] if logprobs else {}
        top = entry.get("top_logprobs") if isinstance(entry, dict) else None
        if not top and isinstance(entry, dict) and entry.get("token"):
            # только выбранный токен без top — недостаточно
            top = [entry]

        if not top or not isinstance(top, list):
            return []

        result: list[TokenCandidate] = []
        seen: set[str] = set()
        for rank, item in enumerate(top[:top_k]):
            if not isinstance(item, dict):
                continue
            token = str(item.get("token") or "")
            if not token or token in seen:
                continue
            seen.add(token)
            lp = float(item.get("logprob", -99.0))
            result.append(TokenCandidate(token=token, logprob=lp, rank=rank))
        return result

    async def next_token_distribution(
        self,
        prompt: str,
        *,
        top_k: int = 20,
        temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        client = await self._ensure()
        top_k = max(2, min(int(top_k), 20))

        # Важно: logprobs и top_logprobs — поля ВЕРХНЕГО уровня, не options
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {
                "temperature": temperature,
                "num_predict": 1,
                "top_k": max(top_k, 40),
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        last_err: Exception | None = None
        for attempt in range(1, settings.max_retries + 1):
            try:
                try:
                    resp = await client.post("/api/generate", json=body)
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    raise RuntimeError(
                        f"Не удалось подключиться к Ollama ({self.base_url}). "
                        f"Запустите Ollama и выполните: ollama pull {self.model}"
                    ) from exc

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Ollama error {resp.status_code}: {resp.text[:400]}"
                    )

                data = resp.json()
                result = self._parse_logprobs_payload(data, top_k)
                if len(result) >= 2:
                    return result

                # Fallback: /api/chat с logprobs
                chat_body = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": top_k,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 1,
                        "num_ctx": settings.ollama_num_ctx,
                    },
                }
                resp2 = await client.post("/api/chat", json=chat_body)
                if resp2.status_code == 200:
                    data2 = resp2.json()
                    result2 = self._parse_logprobs_payload(data2, top_k)
                    if len(result2) >= 2:
                        return result2

                # Последний fallback: кандидаты через обычный chat (без logprobs)
                return await self._candidates_via_chat(client, prompt, top_k, temperature)

            except RuntimeError:
                raise
            except Exception as exc:
                last_err = exc
                logger.warning(f"[ollama] попытка {attempt}: {exc}")
                if attempt < settings.max_retries:
                    import asyncio
                    await asyncio.sleep(settings.retry_base_delay * (2 ** (attempt - 1)))

        raise RuntimeError(str(last_err) if last_err else "Ollama: нет кандидатов") from last_err

    async def _candidates_via_chat(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        top_k: int,
        temperature: float,
    ) -> list[TokenCandidate]:
        """Запросить у модели список альтернативных продолжений (без logprobs)."""
        n = min(max(top_k, 4), 8)
        ask = (
            "Продолжи текст ОДНИМ коротким фрагментом (1–3 слова). "
            f"Дай ровно {n} РАЗНЫХ вариантов продолжения через точку с запятой, "
            "без нумерации и пояснений.\n\n"
            f"Текст:\n{prompt[-800:]}"
        )
        resp = await client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Отвечай только списком вариантов через ;"},
                    {"role": "user", "content": ask},
                ],
                "stream": False,
                "options": {"temperature": max(0.3, temperature), "num_predict": 120},
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama chat fallback error {resp.status_code}: {resp.text[:300]}")
        content = (resp.json().get("message") or {}).get("content") or ""
        parts = []
        for p in content.replace("\n", ";").split(";"):
            p = p.strip().strip(".-•)0123456789 ").strip()
            if p and p not in parts:
                parts.append(p)
            if len(parts) >= n:
                break
        if len(parts) < 2:
            raise RuntimeError(
                "Ollama не вернула logprobs и не смогла предложить варианты продолжения. "
                f"Обновите Ollama (нужен logprobs) или смените модель. Сейчас: {self.model}"
            )
        # Искусственные logprob по рангу (убывание)
        return [
            TokenCandidate(token=t if t.startswith(" ") or t[:1].isalnum() else f" {t}", logprob=-float(i), rank=i)
            for i, t in enumerate(parts)
        ]

    async def complete_text(self, prompt: str, *, max_tokens: int = 400) -> str:
        """Обычная генерация текста (для стегоконтейнера)."""
        client = await self._ensure()
        resp = await client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": max_tokens,
                    "num_ctx": settings.ollama_num_ctx,
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama complete error {resp.status_code}: {resp.text[:300]}")
        return ((resp.json().get("message") or {}).get("content") or "").strip()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
