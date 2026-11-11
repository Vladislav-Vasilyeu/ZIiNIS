"""OpenRouter provider: logprobs + fallback через chat-кандидаты."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from config import settings
from providers.base import LogitsProvider, TokenCandidate


class OpenRouterLogitsProvider(LogitsProvider):
    name = "openrouter"
    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.openrouter_model
        self._client: httpx.AsyncClient | None = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            if not settings.openrouter_api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY не задан. Укажите ключ в файле .env"
                )
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Vladislav-Vasilyeu/ZIiNIS",
                    "X-Title": "NeuralTextStego",
                },
                timeout=settings.api_timeout,
            )
        return self._client

    async def next_token_distribution(
        self,
        prompt: str,
        *,
        top_k: int = 20,
        temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        client = await self._ensure()
        top_k = max(2, min(top_k, 20))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": top_k,
        }
        last_err: Exception | None = None
        for attempt in range(1, settings.max_retries + 1):
            try:
                resp = await client.post("/chat/completions", json=body)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter error {resp.status_code}: {resp.text[:400]}"
                    )
                data = resp.json()
                choice = data["choices"][0]
                logprobs = choice.get("logprobs") or {}
                content = logprobs.get("content") or []
                if content:
                    top = content[0].get("top_logprobs") or []
                    result: list[TokenCandidate] = []
                    seen: set[str] = set()
                    for rank, item in enumerate(top[:top_k]):
                        token = item.get("token") or item.get("text") or ""
                        if not token or token in seen:
                            continue
                        seen.add(token)
                        lp = float(item.get("logprob", item.get("log_prob", -99.0)))
                        result.append(TokenCandidate(token=token, logprob=lp, rank=rank))
                    if len(result) >= 2:
                        return result

                # Fallback без logprobs
                return await self._candidates_via_chat(client, prompt, top_k, temperature)
            except Exception as exc:
                last_err = exc
                logger.warning(f"[openrouter] попытка {attempt}: {exc}")
                if attempt < settings.max_retries:
                    import asyncio
                    await asyncio.sleep(settings.retry_base_delay * (2 ** (attempt - 1)))
        raise RuntimeError(str(last_err)) from last_err

    async def _candidates_via_chat(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        top_k: int,
        temperature: float,
    ) -> list[TokenCandidate]:
        n = min(max(top_k, 4), 8)
        ask = (
            "Continue the text with ONE short fragment (1-3 words). "
            f"Give exactly {n} DIFFERENT continuation variants separated by semicolons, "
            "no numbering or explanations.\n\n"
            f"Text:\n{prompt[-800:]}"
        )
        resp = await client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Reply only with variants separated by ;"},
                    {"role": "user", "content": ask},
                ],
                "max_tokens": 120,
                "temperature": max(0.3, temperature),
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter chat fallback {resp.status_code}: {resp.text[:300]}")
        content = resp.json()["choices"][0]["message"]["content"].strip()
        parts: list[str] = []
        for p in content.replace("\n", ";").split(";"):
            p = p.strip().strip(".-•)0123456789 ").strip()
            if p and p not in parts:
                parts.append(p)
            if len(parts) >= n:
                break
        if len(parts) < 2:
            raise RuntimeError(
                "Модель не вернула logprobs и не дала список вариантов. "
                "Смените модель OpenRouter на поддерживающую logprobs."
            )
        return [
            TokenCandidate(token=t if t.startswith(" ") or t[:1].isalnum() else f" {t}", logprob=-float(i), rank=i)
            for i, t in enumerate(parts)
        ]

    async def complete_text(self, prompt: str, *, max_tokens: int = 400) -> str:
        client = await self._ensure()
        resp = await client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.8,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter complete error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
