"""
Базовый асинхронный адаптер источника кандидатов на замену.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from loguru import logger

from config import settings


@dataclass
class CandidateResult:
    """Результат запроса кандидатов на замену."""

    original: str
    candidates: list[str]
    context: str = ""


class BaseLLMAdapter(ABC):
    """
    Абстрактный асинхронный адаптер источника кандидатов.

    Сетевые запросы выполняются асинхронно с retry и экспоненциальной задержкой.
    """

    name: str = "base"
    supports_cancellation: bool = True
    allows_partial_candidates: bool = False

    def __init__(self) -> None:
        self._client = None
        self._closed = False

    @abstractmethod
    async def _ensure_client(self) -> None:
        """Инициализация HTTP-клиента при необходимости."""
        ...

    def supports_word(self, word: str) -> bool:
        """Сообщить, может ли адаптер предложить замену этому слову."""
        return True

    @abstractmethod
    async def _request_candidates(
        self,
        text: str,
        word: str,
        num_candidates: int,
        *,
        temperature: float,
    ) -> list[str]:
        """
        Низкоуровневый запрос к API.

        Должен вернуть список кандидатов (без исходного слова или с ним — нормализуется выше).
        """
        ...

    async def get_replacement_candidates(
        self,
        text: str,
        word: str,
        num_candidates: int | None = None,
        *,
        temperature: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> CandidateResult:
        """
        Получить список естественных вариантов замены слова/фразы.

        :param text: Контекст (предложение или абзац).
        :param word: Слово/фраза, которую нужно заменить.
        :param num_candidates: Сколько вариантов нужно (по умолчанию из настроек).
        :param temperature: Температура генерации.
        :param cancel_event: Event для отмены операции.
        """
        n = num_candidates or settings.num_candidates
        temp = temperature if temperature is not None else settings.temperature

        await self._ensure_client()

        last_exc: Exception | None = None
        for attempt in range(1, settings.max_retries + 1):
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("Операция отменена пользователем")

            try:
                candidates = await self._request_candidates(
                    text=text,
                    word=word,
                    num_candidates=n,
                    temperature=temp,
                )
                # Нормализация: убираем дубликаты и исходное слово, сохраняем порядок
                seen: set[str] = set()
                clean: list[str] = []
                for c in candidates:
                    c_norm = c.strip()
                    if not c_norm:
                        continue
                    key = c_norm.lower()
                    if key == word.lower() or key in seen:
                        continue
                    seen.add(key)
                    clean.append(c_norm)
                    if len(clean) >= n:
                        break

                if len(clean) < n:
                    if self.allows_partial_candidates and clean:
                        logger.warning(
                            f"[{self.name}] Получено {len(clean)} из {n} запрошенных "
                            "вариантов; продолжаю с доступными кандидатами"
                        )
                        return CandidateResult(
                            original=word,
                            candidates=clean,
                            context=text,
                        )
                    raise RuntimeError(
                        f"Модель вернула только {len(clean)} из {n} уникальных вариантов"
                    )

                return CandidateResult(original=word, candidates=clean[:n], context=text)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                delay = settings.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{self.name}] Попытка {attempt}/{settings.max_retries} не удалась: {exc}. "
                    f"Повтор через {delay:.1f} с"
                )
                if attempt < settings.max_retries:
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"[{self.name}] Не удалось получить {n} вариантов после "
            f"{settings.max_retries} попыток: {last_exc}"
        ) from last_exc

    async def close(self) -> None:
        """Закрыть ресурсы клиента."""
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
