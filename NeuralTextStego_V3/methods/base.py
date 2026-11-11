"""Базовые типы стеганографических методов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class MethodKind(str, Enum):
    """Идентификаторы пяти методов."""

    CLASSIC_SYNONYMS = "classic_synonyms"
    CLASSIC_WHITESPACE = "classic_whitespace"
    NEURAL_RANK = "neural_rank"  # innocuous-style: кодирование по рангу
    NEURAL_ARITHMETIC = "neural_arithmetic"  # subtext-style: арифметическое кодирование
    NEURAL_SAMPLING = "neural_sampling"  # сэмплирование / binning токенов


ProgressCallback = Callable[[float, str], Awaitable[None] | None]


@dataclass
class MethodResult:
    success: bool
    text: str = ""
    secret: str = ""
    bits_embedded: int = 0
    bits_extracted: int = 0
    positions_used: int = 0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0


class StegoMethod(ABC):
    """Единый интерфейс метода внедрения/извлечения."""

    kind: MethodKind
    display_name: str
    is_neural: bool
    # Генеративные методы создают текст с нуля (нужен prompt).
    # Edit-based модифицируют cover_text.
    is_generative: bool = False

    @abstractmethod
    async def embed(
        self,
        *,
        secret: str,
        cover_text: str = "",
        prompt: str = "",
        password: str | None = None,
        use_marker: bool = True,
        progress: ProgressCallback | None = None,
        cancel_event: Any = None,
    ) -> MethodResult:
        ...

    @abstractmethod
    async def extract(
        self,
        *,
        stego_text: str,
        prompt: str = "",
        password: str | None = None,
        expected_bits: int | None = None,
        use_marker: bool = True,
        progress: ProgressCallback | None = None,
        cancel_event: Any = None,
    ) -> MethodResult:
        ...

    async def close(self) -> None:
        return None
