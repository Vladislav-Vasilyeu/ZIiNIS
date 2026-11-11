"""Интерфейс провайдера top-logprobs для нейросетевых методов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TokenCandidate:
    token: str
    logprob: float
    rank: int


class LogitsProvider(ABC):
    """
    Источник распределения следующего токена.

    OpenRouter: logprobs + top_logprobs в Chat Completions.
    Ollama: попытка native logprobs; при отсутствии — понятная ошибка
    с рекомендацией llama-cpp-python / GGUF.
    """

    name: str = "base"

    @abstractmethod
    async def next_token_distribution(
        self,
        prompt: str,
        *,
        top_k: int = 20,
        temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        """Вернуть top_k кандидатов, отсортированных по убыванию вероятности (rank 0 = самый вероятный)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
