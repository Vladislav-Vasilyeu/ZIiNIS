"""
Общая логика генеративных нейросетевых методов.

Три обратимые стратегии выбора токена из top-logprobs:
- rank: индекс среди 2^k лучших → k бит
- sampling: разбиение top-N на 2^k корзин по рангу
- arithmetic: корзины по накопленной вероятности (softmax)
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

from loguru import logger

from methods.base import MethodResult, ProgressCallback
from methods.bits import (
    append_marker_if_needed,
    frame_payload_bits,
    prepare_payload,
    recover_secret,
    try_extract_marker,
    unframe_payload_bits,
)
from providers.base import LogitsProvider, TokenCandidate

Strategy = Literal["rank", "sampling", "arithmetic"]


def _softmax_probs(candidates: list[TokenCandidate], temperature: float) -> list[float]:
    if temperature <= 0:
        temperature = 1.0
    xs = [c.logprob / temperature for c in candidates]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _levels(n_items: int, chunk_bits: int) -> int:
    if n_items < 2:
        return 0
    levels = 1
    while (1 << (levels + 1)) <= n_items and levels < chunk_bits:
        levels += 1
    return levels


def match_observed_token(candidates: list[TokenCandidate], remaining: str) -> str | None:
    """Найти токен-кандидат, являющийся префиксом оставшегося стеготекста."""
    best: str | None = None
    for cand in candidates:
        token = cand.token
        if token and remaining.startswith(token):
            if best is None or len(token) > len(best):
                best = token
    return best


class NeuralGenerativeCodec:
    """Генерация стеготекста по prompt с кодированием секрета в выборе токенов."""

    def __init__(
        self,
        provider: LogitsProvider,
        strategy: Strategy,
        *,
        chunk_bits: int = 2,
        top_k: int = 20,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.chunk_bits = max(1, min(chunk_bits, 3))
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _bin_assignments(
        self,
        candidates: list[TokenCandidate],
        n_bins: int,
    ) -> list[int]:
        """Индекс корзины для каждого кандидата. Обратимо для sampling и arithmetic."""
        if self.strategy == "sampling":
            pool_n = len(candidates)
            bin_size = max(1, pool_n // n_bins)
            return [min(n_bins - 1, i // bin_size) for i in range(pool_n)]

        probs = _softmax_probs(candidates, self.temperature)
        assignments: list[int] = []
        cdf = 0.0
        for p in probs:
            mid = cdf + p / 2.0
            assignments.append(min(n_bins - 1, int(mid * n_bins)))
            cdf += p
        if len(set(assignments)) < n_bins:
            return [min(n_bins - 1, i * n_bins // max(len(candidates), 1)) for i in range(len(candidates))]
        return assignments

    def _pick_token(
        self,
        candidates: list[TokenCandidate],
        bits: str,
        bit_pos: int,
    ) -> tuple[str, int]:
        """Вернуть (token, consumed_bits)."""
        if not candidates:
            return "", 0
        if bit_pos >= len(bits):
            return candidates[0].token, 0

        if self.strategy == "rank":
            levels = _levels(len(candidates), self.chunk_bits)
            if not levels:
                return candidates[0].token, 0
            pool = candidates[: 1 << levels]
            chunk = bits[bit_pos : bit_pos + levels].ljust(levels, "0")
            idx = int(chunk, 2) % len(pool)
            return pool[idx].token, levels

        levels = _levels(len(candidates), self.chunk_bits)
        if not levels:
            return candidates[0].token, 0
        n_bins = 1 << levels
        assignments = self._bin_assignments(candidates, n_bins)
        chunk = bits[bit_pos : bit_pos + levels].ljust(levels, "0")
        target = int(chunk, 2) % n_bins
        for cand, bin_idx in zip(candidates, assignments):
            if bin_idx == target and cand.token:
                return cand.token, levels
        return candidates[0].token, levels

    def _token_to_bits(
        self,
        candidates: list[TokenCandidate],
        observed: str,
    ) -> str:
        """Восстановить биты по выбранному токену."""
        if not candidates:
            return ""

        def _index_of(observed_token: str) -> int | None:
            for i, cand in enumerate(candidates):
                if cand.token == observed_token:
                    return i
            return None

        idx = _index_of(observed)
        if idx is None:
            return ""

        if self.strategy == "rank":
            levels = _levels(len(candidates), self.chunk_bits)
            if not levels:
                return ""
            pool_n = 1 << levels
            if idx >= pool_n:
                return ""
            return f"{idx:0{levels}b}"

        levels = _levels(len(candidates), self.chunk_bits)
        if not levels:
            return ""
        n_bins = 1 << levels
        assignments = self._bin_assignments(candidates, n_bins)
        return f"{assignments[idx]:0{levels}b}"

    async def _next_dist(self, context: str) -> list[TokenCandidate]:
        dist = await self.provider.next_token_distribution(
            context, top_k=self.top_k, temperature=self.temperature
        )
        clean: list[TokenCandidate] = []
        seen: set[str] = set()
        for cand in dist:
            token = cand.token
            if not token or token in seen:
                continue
            seen.add(token)
            clean.append(cand)
        return clean

    async def embed(
        self,
        *,
        secret: str,
        prompt: str,
        password: str | None,
        use_marker: bool,
        progress: ProgressCallback | None,
        cancel_event: Any,
        method_name: str,
    ) -> MethodResult:
        t0 = time.perf_counter()
        if not secret:
            return MethodResult(success=False, message="Секретное сообщение пусто")
        if not prompt.strip():
            prompt = "Продолжи связный текст на русском языке:\n"

        try:
            payload, encrypted = prepare_payload(secret, password)
            bits = frame_payload_bits(payload)
        except Exception as exc:
            return MethodResult(success=False, message=f"Ошибка подготовки секрета: {exc}")

        needed = len(bits)
        bit_pos = 0
        generated = ""
        steps = 0

        while bit_pos < needed and steps < self.max_tokens:
            if cancel_event and cancel_event.is_set():
                return MethodResult(success=False, message="Операция отменена")
            ctx = prompt + generated
            try:
                dist = await self._next_dist(ctx)
            except Exception as exc:
                logger.error(f"logprobs step failed: {exc}")
                return MethodResult(success=False, message=str(exc))

            token, consumed = self._pick_token(dist, bits, bit_pos)
            if not token or consumed <= 0:
                break
            generated += token
            bit_pos += consumed
            steps += 1
            if progress and steps % 3 == 0:
                maybe = progress(
                    min(0.99, bit_pos / max(needed, 1)),
                    f"Токенов: {steps}, бит: {bit_pos}/{needed}",
                )
                if hasattr(maybe, "__await__"):
                    await maybe

        if bit_pos < needed and not use_marker:
            return MethodResult(
                success=False,
                message=(
                    f"Не удалось закодировать весь секрет без маркера: "
                    f"{bit_pos}/{needed} бит за {steps} токенов"
                ),
                text=generated,
                bits_embedded=bit_pos,
                positions_used=steps,
            )

        stego = append_marker_if_needed(generated, payload, use_marker)
        msg = "Внедрение выполнено"
        if bit_pos < needed and use_marker:
            msg = (
                f"Внедрение выполнено: в токенах {bit_pos}/{needed} бит, "
                "полный секрет сохранён в маркере"
            )
        if progress:
            maybe = progress(1.0, "Готово")
            if hasattr(maybe, "__await__"):
                await maybe
        return MethodResult(
            success=True,
            text=stego,
            secret=secret,
            bits_embedded=needed if use_marker else bit_pos,
            positions_used=steps,
            message=msg,
            details={
                "encrypted": encrypted,
                "use_marker": use_marker,
                "strategy": self.strategy,
                "method": method_name,
                "provider": self.provider.name,
                "framed_bits": needed,
                "token_bits": bit_pos,
            },
            elapsed_sec=time.perf_counter() - t0,
        )

    async def extract(
        self,
        *,
        stego_text: str,
        prompt: str,
        password: str | None,
        expected_bits: int | None,
        use_marker: bool,
        progress: ProgressCallback | None,
        cancel_event: Any,
        method_name: str,
    ) -> MethodResult:
        t0 = time.perf_counter()
        if not stego_text.strip():
            return MethodResult(success=False, message="Стеготекст пуст")
        if not prompt.strip():
            prompt = "Продолжи связный текст на русском языке:\n"

        if use_marker:
            try:
                marker = try_extract_marker(stego_text)
            except ValueError as exc:
                return MethodResult(success=False, message=str(exc))
            if marker is not None:
                payload, clean = marker
                try:
                    secret = recover_secret(payload, password)
                except Exception as exc:
                    return MethodResult(success=False, message=f"Расшифровка: {exc}")
                bits_n = len(payload.encode("utf-8")) * 8
                if progress:
                    maybe = progress(1.0, "Извлечено по маркеру")
                    if hasattr(maybe, "__await__"):
                        await maybe
                return MethodResult(
                    success=True,
                    text=clean,
                    secret=secret,
                    bits_extracted=bits_n,
                    message="Извлечение по маркеру",
                    details={
                        "reliable_marker": True,
                        "strategy": self.strategy,
                        "method": method_name,
                    },
                    elapsed_sec=time.perf_counter() - t0,
                )

        remaining = stego_text
        generated = ""
        got_bits = ""
        steps = 0
        needed = 16
        payload_len: int | None = None

        while len(got_bits) < needed and steps < self.max_tokens:
            if cancel_event and cancel_event.is_set():
                return MethodResult(success=False, message="Операция отменена")
            if not remaining:
                break
            try:
                dist = await self._next_dist(prompt + generated)
            except Exception as exc:
                logger.error(f"logprobs extract failed: {exc}")
                return MethodResult(success=False, message=str(exc))

            token = match_observed_token(dist, remaining)
            if token is None:
                return MethodResult(
                    success=False,
                    message=(
                        "Не удалось сопоставить токен при извлечении без маркера. "
                        "Нужен тот же промпт и та же модель, что при внедрении."
                    ),
                    bits_extracted=len(got_bits),
                    details={"method": method_name, "strategy": self.strategy, "steps": steps},
                )
            piece = self._token_to_bits(dist, token)
            if not piece:
                return MethodResult(
                    success=False,
                    message="Токен стеготекста не входит в кодовый набор модели на этом шаге",
                    bits_extracted=len(got_bits),
                )
            got_bits += piece
            generated += token
            remaining = remaining[len(token) :]
            steps += 1
            if payload_len is None and len(got_bits) >= 16:
                payload_len = int(got_bits[:16], 2)
                needed = 16 + payload_len * 8
                if needed > 16 + 8 * 65535:
                    return MethodResult(success=False, message="Некорректная длина секрета в стеготексте")
            if progress and steps % 3 == 0:
                maybe = progress(
                    min(0.99, len(got_bits) / max(needed, 1)),
                    f"Токенов: {steps}, бит: {len(got_bits)}/{needed}",
                )
                if hasattr(maybe, "__await__"):
                    await maybe

        payload = unframe_payload_bits(got_bits)
        if payload is None:
            return MethodResult(
                success=False,
                message=(
                    f"Не удалось восстановить секрет без маркера "
                    f"({len(got_bits)} бит за {steps} токенов). "
                    "Проверьте промпт, метод и провайдер."
                ),
                bits_extracted=len(got_bits),
                details={"method": method_name, "strategy": self.strategy},
            )

        try:
            secret = recover_secret(payload, password)
        except Exception as exc:
            return MethodResult(success=False, message=f"Расшифровка: {exc}")

        if expected_bits is not None:
            actual = len(payload.encode("utf-8")) * 8
            if actual != expected_bits:
                return MethodResult(
                    success=False,
                    secret=secret,
                    bits_extracted=actual,
                    message=f"Ожидалось {expected_bits} бит, извлечено {actual} бит",
                )

        if progress:
            maybe = progress(1.0, "Извлечено без маркера")
            if hasattr(maybe, "__await__"):
                await maybe
        return MethodResult(
            success=True,
            text=generated,
            secret=secret,
            bits_extracted=len(payload.encode("utf-8")) * 8,
            positions_used=steps,
            message="Извлечение без маркера",
            details={
                "reliable_marker": False,
                "strategy": self.strategy,
                "method": method_name,
            },
            elapsed_sec=time.perf_counter() - t0,
        )
