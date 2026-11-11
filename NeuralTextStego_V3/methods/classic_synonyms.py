"""Классический метод: замена слов по словарю синонимов (edit-based)."""

from __future__ import annotations

import time
from typing import Any

from adapters.classic import ClassicSynonymAdapter, synonym_codebook
from core.stego_engine import WORD_PATTERN
from methods.base import MethodKind, MethodResult, ProgressCallback, StegoMethod
from methods.bits import (
    append_marker_if_needed,
    frame_payload_bits,
    prepare_payload,
    recover_secret,
    try_extract_marker,
    unframe_payload_bits,
)


class ClassicSynonymsMethod(StegoMethod):
    kind = MethodKind.CLASSIC_SYNONYMS
    display_name = "Классический: синонимы"
    is_neural = False
    is_generative = False

    def __init__(self, num_candidates: int = 4) -> None:
        self.num_candidates = num_candidates
        self._adapter = ClassicSynonymAdapter()

    def _positions(self, text: str) -> list[tuple[int, int, str, list[str]]]:
        found: list[tuple[int, int, str, list[str]]] = []
        for match in WORD_PATTERN.finditer(text):
            word = match.group()
            codebook = synonym_codebook(word, self.num_candidates)
            if len(codebook) < 2:
                continue
            found.append((match.start(), match.end(), word, codebook))
        return found

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
            bits = frame_payload_bits(payload)
        except Exception as exc:
            return MethodResult(success=False, message=f"Ошибка подготовки секрета: {exc}")

        positions = self._positions(cover_text)
        if not positions:
            return MethodResult(
                success=False,
                message="Во встроенном словаре нет подходящих слов для замены",
            )

        capacity = sum(1 if len(codebook) < 4 else 2 for *_, codebook in positions)
        if capacity < len(bits) and not use_marker:
            return MethodResult(
                success=False,
                message=(
                    f"Недостаточно позиций для режима без маркера: "
                    f"нужно {len(bits)} бит, доступно {capacity}"
                ),
            )

        chars = list(cover_text)
        offset = 0
        bit_pos = 0
        used = 0
        needed = len(bits)
        for start, end, word, codebook in positions:
            if bit_pos >= needed:
                break
            if cancel_event and cancel_event.is_set():
                return MethodResult(success=False, message="Операция отменена")
            levels = 1 if len(codebook) < 4 else 2
            chunk = bits[bit_pos : bit_pos + levels].ljust(levels, "0")
            idx = int(chunk, 2)
            chosen = codebook[idx]
            adj_start = start + offset
            adj_end = end + offset
            chars[adj_start:adj_end] = list(chosen)
            offset += len(chosen) - (end - start)
            bit_pos += levels
            used += 1
            if progress and used % 4 == 0:
                maybe = progress(
                    min(1.0, bit_pos / max(needed, 1)),
                    f"Внедрено {bit_pos}/{needed} бит",
                )
                if hasattr(maybe, "__await__"):
                    await maybe

        if used == 0:
            return MethodResult(success=False, message="Не удалось закодировать ни одного бита")

        stego = append_marker_if_needed("".join(chars), payload, use_marker)
        msg = "Внедрение выполнено"
        if bit_pos < needed and use_marker:
            msg = (
                f"Внедрение выполнено: видимо {bit_pos}/{needed} бит, "
                "полный секрет в маркере"
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
            positions_used=used,
            message=msg,
            details={
                "encrypted": encrypted,
                "use_marker": use_marker,
                "method": self.kind.value,
                "capacity_bits": capacity,
            },
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
                    details={"reliable_marker": True, "method": self.kind.value},
                    elapsed_sec=time.perf_counter() - t0,
                )

        got: list[str] = []
        for _start, _end, word, codebook in self._positions(stego_text):
            if cancel_event and cancel_event.is_set():
                return MethodResult(success=False, message="Операция отменена")
            try:
                idx = codebook.index(word)
            except ValueError:
                folded = [item.casefold() for item in codebook]
                try:
                    idx = folded.index(word.casefold())
                except ValueError:
                    continue
            levels = 1 if len(codebook) < 4 else min(2, 1 if self.num_candidates == 2 else 2)
            got.append(f"{idx:0{levels}b}")

        payload = unframe_payload_bits("".join(got))
        if payload is None:
            return MethodResult(
                success=False,
                message="Не удалось извлечь секрет из синонимов без маркера",
                details={"method": self.kind.value},
            )
        try:
            secret = recover_secret(payload, password)
        except Exception as exc:
            return MethodResult(success=False, message=f"Расшифровка: {exc}")
        if progress:
            maybe = progress(1.0, "Извлечено из синонимов")
            if hasattr(maybe, "__await__"):
                await maybe
        return MethodResult(
            success=True,
            text=stego_text,
            secret=secret,
            bits_extracted=len(payload.encode("utf-8")) * 8,
            message="Извлечение из синонимов без маркера",
            details={"reliable_marker": False, "method": self.kind.value},
            elapsed_sec=time.perf_counter() - t0,
        )

    async def close(self) -> None:
        await self._adapter.close()
