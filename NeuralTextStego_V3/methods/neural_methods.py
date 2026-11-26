"""Три нейросетевых метода на общем codec."""

from __future__ import annotations

from typing import Any
import time

from methods.base import MethodKind, MethodResult, ProgressCallback, StegoMethod
from methods.neural_common import NeuralGenerativeCodec
from providers.base import LogitsProvider
from providers.ollama_logits import OllamaLogitsProvider


class _NeuralBase(StegoMethod):
    is_neural = True
    is_generative = True
    strategy: str = "rank"

    def __init__(self, provider: LogitsProvider, **codec_kwargs: Any) -> None:
        self._provider = provider
        self._codec = NeuralGenerativeCodec(provider, self.strategy, **codec_kwargs)

    async def embed(self, **kwargs: Any) -> MethodResult:
        started = time.perf_counter()
        result = await self._codec.embed(
            secret=kwargs.get("secret", ""),
            prompt=kwargs.get("prompt") or kwargs.get("cover_text") or "",
            password=kwargs.get("password"),
            use_marker=kwargs.get("use_marker", True),
            progress=kwargs.get("progress"),
            cancel_event=kwargs.get("cancel_event"),
            method_name=self.kind.value,
        )
        if result.success and not kwargs.get("use_marker", True):
            progress = kwargs.get("progress")
            if progress:
                pending = progress(0.99, "Проверка обратного извлечения…")
                if hasattr(pending, "__await__"):
                    await pending
            verified = await self._extract_with_current_provider(
                stego_text=result.text,
                prompt=kwargs.get("prompt") or kwargs.get("cover_text") or "",
                password=kwargs.get("password"), use_marker=False,
                cancel_event=kwargs.get("cancel_event"),
            )
            result.details["roundtrip_verified"] = verified.success and verified.secret == kwargs.get("secret", "")
            if not result.details["roundtrip_verified"]:
                result.success = False
                result.details["verification_error"] = verified.message
                result.message = (
                    "Контрольное извлечение не подтвердило сообщение. "
                    "Распределения модели оказались несовместимыми при повторном запросе. "
                    "Повторите операцию или выберите другой контекст/модель."
                )
        result.elapsed_sec = time.perf_counter() - started
        return result

    async def extract(self, **kwargs: Any) -> MethodResult:
        result = await self._extract_with_current_provider(**kwargs)
        cancelled = kwargs.get("cancel_event")
        if (
            not result.success
            and isinstance(self._provider, OllamaLogitsProvider)
            and self._provider.raw_continuation
            and not (cancelled and cancelled.is_set())
        ):
            # Older unmarked texts used a fresh chat template at every step.
            self._provider.raw_continuation = False
            try:
                legacy = await self._extract_with_current_provider(**kwargs)
                if legacy.success:
                    legacy.details["legacy_chat_context"] = True
                    return legacy
            finally:
                self._provider.raw_continuation = True
        return result

    async def _extract_with_current_provider(self, **kwargs: Any) -> MethodResult:
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
