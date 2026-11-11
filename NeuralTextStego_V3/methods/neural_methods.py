"""Три нейросетевых метода на общем codec."""

from __future__ import annotations

from typing import Any

from methods.base import MethodKind, MethodResult, ProgressCallback, StegoMethod
from methods.neural_common import NeuralGenerativeCodec
from providers.base import LogitsProvider


class _NeuralBase(StegoMethod):
    is_neural = True
    is_generative = True
    strategy: str = "rank"

    def __init__(self, provider: LogitsProvider, **codec_kwargs: Any) -> None:
        self._provider = provider
        self._codec = NeuralGenerativeCodec(provider, self.strategy, **codec_kwargs)

    async def embed(self, **kwargs: Any) -> MethodResult:
        return await self._codec.embed(
            secret=kwargs.get("secret", ""),
            prompt=kwargs.get("prompt") or kwargs.get("cover_text") or "",
            password=kwargs.get("password"),
            use_marker=kwargs.get("use_marker", True),
            progress=kwargs.get("progress"),
            cancel_event=kwargs.get("cancel_event"),
            method_name=self.kind.value,
        )

    async def extract(self, **kwargs: Any) -> MethodResult:
        return await self._codec.extract(
            stego_text=kwargs.get("stego_text", ""),
            prompt=kwargs.get("prompt") or "",
            password=kwargs.get("password"),
            expected_bits=kwargs.get("expected_bits"),
            use_marker=kwargs.get("use_marker", True),
            progress=kwargs.get("progress"),
            cancel_event=kwargs.get("cancel_event"),
            method_name=self.kind.value,
        )

    async def close(self) -> None:
        await self._provider.close()


class NeuralRankMethod(_NeuralBase):
    """Innocuous-style: кодирование по рангу токена в top-logprobs."""

    kind = MethodKind.NEURAL_RANK
    display_name = "Нейросеть: ранг логитов (innocuous)"
    strategy = "rank"


class NeuralArithmeticMethod(_NeuralBase):
    """Subtext-style: упрощённое арифметическое кодирование."""

    kind = MethodKind.NEURAL_ARITHMETIC
    display_name = "Нейросеть: арифметическое кодирование"
    strategy = "arithmetic"


class NeuralSamplingMethod(_NeuralBase):
    """Сэмплирование / binning токенов по рангу."""

    kind = MethodKind.NEURAL_SAMPLING
    display_name = "Нейросеть: сэмплирование токенов"
    strategy = "sampling"
