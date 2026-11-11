"""Классический метод: кодирование битов через обычный / неразрывный пробел."""

from __future__ import annotations

import time
from typing import Any

from methods.base import MethodKind, MethodResult, ProgressCallback, StegoMethod
from methods.bits import (
    append_marker_if_needed,
    frame_payload_bits,
    prepare_payload,
    recover_secret,
    text_to_bits,
    try_extract_marker,
    unframe_payload_bits,
)

SPACE = " "
NBSP = "\u00a0"  # неразрывный пробел → бит 1


class ClassicWhitespaceMethod(StegoMethod):
    kind = MethodKind.CLASSIC_WHITESPACE
    display_name = "Классический: пробелы (space/NBSP)"
    is_neural = False
    is_generative = False

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
        t0 = time.perf_counter()
        if not cover_text.strip():
            return MethodResult(success=False, message="Текст-контейнер пуст")
        if not secret:
            return MethodResult(success=False, message="Секретное сообщение пусто")

        try:
            payload, encrypted = prepare_payload(secret, password)
        except Exception as exc:
            return MethodResult(success=False, message=f"Ошибка шифрования: {exc}")

        bits = frame_payload_bits(payload) if not use_marker else text_to_bits(payload)
        needed = len(bits)
        # Позиции обычных пробелов
        positions = [i for i, ch in enumerate(cover_text) if ch == SPACE]
        if len(positions) < needed and not use_marker:
            return MethodResult(
                success=False,
                message=(
                    f"Недостаточно пробелов: нужно {needed} бит, доступно {len(positions)}"
                ),
            )

        chars = list(cover_text)
        embedded = 0
        for bit_i, pos in enumerate(positions):
            if bit_i >= needed:
                break
            if cancel_event and cancel_event.is_set():
                return MethodResult(success=False, message="Операция отменена")
            chars[pos] = NBSP if bits[bit_i] == "1" else SPACE
            embedded += 1
            if progress and bit_i % 20 == 0:
                maybe = progress(min(1.0, (bit_i + 1) / max(needed, 1)), f"Бит {bit_i + 1}/{needed}")
                if hasattr(maybe, "__await__"):
                    await maybe

        stego = "".join(chars)
        stego = append_marker_if_needed(stego, payload, use_marker)
        msg = "Внедрение выполнено"
        if embedded < needed:
            msg = (
                f"Внедрение выполнено: видимо {embedded}/{needed} бит, "
                "полный секрет в маркере" if use_marker else msg
            )
        if progress:
            maybe = progress(1.0, "Готово")
            if hasattr(maybe, "__await__"):
                await maybe
        return MethodResult(
            success=True,
            text=stego,
            secret=secret,
            bits_embedded=embedded if not use_marker else needed,
            positions_used=embedded,
            message=msg,
            details={"encrypted": encrypted, "use_marker": use_marker, "method": self.kind.value},
            elapsed_sec=time.perf_counter() - t0,
        )

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
        t0 = time.perf_counter()
        if not stego_text.strip():
            return MethodResult(success=False, message="Стеготекст пуст")

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
                bits = len(payload.encode("utf-8")) * 8
                if progress:
                    maybe = progress(1.0, "Извлечено по маркеру")
                    if hasattr(maybe, "__await__"):
                        await maybe
                return MethodResult(
                    success=True,
                    text=clean,
                    secret=secret,
                    bits_extracted=bits,
                    message="Извлечение по маркеру",
                    details={"reliable_marker": True, "method": self.kind.value},
                    elapsed_sec=time.perf_counter() - t0,
                )

        # Без маркера: читаем пробелы (префикс длины + payload)
        bits_list: list[str] = []
        for ch in stego_text:
            if ch == SPACE:
                bits_list.append("0")
            elif ch == NBSP:
                bits_list.append("1")
        bits = "".join(bits_list)
        payload = unframe_payload_bits(bits)
        if payload is None and expected_bits:
            payload_bits = bits[:expected_bits]
            from methods.bits import bits_to_text

            payload = bits_to_text(payload_bits)
        if payload is None:
            return MethodResult(
                success=False,
                message="Не удалось извлечь секрет из пробелов без маркера",
                bits_extracted=len(bits),
            )
        try:
            secret = recover_secret(payload, password)
        except Exception as exc:
            return MethodResult(success=False, message=f"Расшифровка: {exc}", bits_extracted=len(bits))
        if progress:
            maybe = progress(1.0, "Извлечено из пробелов")
            if hasattr(maybe, "__await__"):
                await maybe
        return MethodResult(
            success=True,
            text=stego_text,
            secret=secret,
            bits_extracted=len(bits),
            message="Извлечение из пробелов",
            details={"reliable_marker": False, "method": self.kind.value},
            elapsed_sec=time.perf_counter() - t0,
        )
