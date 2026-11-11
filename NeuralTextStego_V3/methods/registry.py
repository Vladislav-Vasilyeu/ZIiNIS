"""Реестр и фабрика методов."""

from __future__ import annotations

from typing import Any

from methods.base import MethodKind, StegoMethod
from methods.classic_synonyms import ClassicSynonymsMethod
from methods.classic_whitespace import ClassicWhitespaceMethod
from methods.neural_methods import (
    NeuralArithmeticMethod,
    NeuralRankMethod,
    NeuralSamplingMethod,
)
from providers.factory import create_logits_provider

METHOD_LABELS: dict[str, str] = {
    MethodKind.CLASSIC_SYNONYMS.value: "Классический: синонимы",
    MethodKind.CLASSIC_WHITESPACE.value: "Классический: пробелы (space/NBSP)",
    MethodKind.NEURAL_RANK.value: "Нейросеть: ранг логитов (innocuous)",
    MethodKind.NEURAL_ARITHMETIC.value: "Нейросеть: арифметическое кодирование",
    MethodKind.NEURAL_SAMPLING.value: "Нейросеть: сэмплирование токенов",
}

NEURAL_KINDS = {
    MethodKind.NEURAL_RANK.value,
    MethodKind.NEURAL_ARITHMETIC.value,
    MethodKind.NEURAL_SAMPLING.value,
}


def list_methods() -> list[tuple[str, str]]:
    return list(METHOD_LABELS.items())


def create_method(
    kind: str,
    *,
    logits_provider: str = "ollama",
    model: str | None = None,
    num_candidates: int = 4,
    **kwargs: Any,
) -> StegoMethod:
    key = kind.lower().strip()
    if key == MethodKind.CLASSIC_SYNONYMS.value:
        return ClassicSynonymsMethod(num_candidates=num_candidates)
    if key == MethodKind.CLASSIC_WHITESPACE.value:
        return ClassicWhitespaceMethod()
    if key in NEURAL_KINDS:
        provider = create_logits_provider(logits_provider, model=model)
        codec_kwargs = {
            "chunk_bits": kwargs.get("chunk_bits", 2),
            "top_k": kwargs.get("top_k", 20),
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        if key == MethodKind.NEURAL_RANK.value:
            return NeuralRankMethod(provider, **codec_kwargs)
        if key == MethodKind.NEURAL_ARITHMETIC.value:
            return NeuralArithmeticMethod(provider, **codec_kwargs)
        return NeuralSamplingMethod(provider, **codec_kwargs)
    raise ValueError(f"Неизвестный метод: {kind}")
