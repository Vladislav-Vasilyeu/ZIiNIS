"""
Стеганографический движок (edit-based).

Принцип:
1. Секретное сообщение → битовая строка (опционально после шифрования).
2. В тексте-контейнере выбираются позиции (слова), пригодные для замены.
3. Для каждой позиции выбранный метод предлагает N кандидатов.
4. Первые биты выбирают видимые замены; полный payload сохраняется в проверяемом
   невидимом маркере для точного восстановления.
5. Извлечение проверяет маркер и не вызывает модель повторно.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from loguru import logger

from adapters.base import BaseLLMAdapter
from config import settings
from core.crypto import CryptoModule
from core.metadata import decode_payload_marker, encode_payload_marker


@dataclass
class StegoResult:
    """Результат операции внедрения или извлечения."""

    success: bool
    text: str = ""
    secret: str = ""
    bits_embedded: int = 0
    bits_extracted: int = 0
    positions_used: int = 0
    message: str = ""
    details: dict = field(default_factory=dict)


# Регулярка для выделения «слов» (кириллица + латиница, длина ≥ 3)
WORD_PATTERN = re.compile(r"[а-яА-ЯёЁa-zA-Z]{3,}")


def text_to_bits(text: str) -> str:
    """Преобразовать строку в битовую последовательность (UTF-8)."""
    data = text.encode("utf-8")
    return "".join(f"{b:08b}" for b in data)


def bits_to_text(bits: str) -> str:
    """Обратное преобразование битов в строку. Лишние биты в конце отбрасываются."""
    # Выравниваем до кратности 8
    usable = bits[: len(bits) - (len(bits) % 8)]
    if not usable:
        return ""
    data = bytes(int(usable[i : i + 8], 2) for i in range(0, len(usable), 8))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Пробуем игнорировать ошибки
        return data.decode("utf-8", errors="replace")


def bits_needed_for(text: str) -> int:
    return len(text.encode("utf-8")) * 8


class StegoEngine:
    """
    Edit-based стеганографический движок.

    :param adapter: Асинхронный адаптер источника кандидатов на замену.
    :param num_candidates: Количество кандидатов (2 → 1 бит, 4 → 2 бита).
    """

    MAX_VISIBLE_REPLACEMENTS = 8

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        num_candidates: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.num_candidates = num_candidates or settings.num_candidates
        if self.num_candidates not in (2, 4):
            raise ValueError("num_candidates должен быть 2 или 4")
        self.bits_per_position = 1 if self.num_candidates == 2 else 2

    def _find_replaceable_positions(self, text: str) -> list[tuple[int, int, str]]:
        """
        Найти позиции слов, пригодных для замены.

        Возвращает список (start, end, word).
        """
        positions = []
        for m in WORD_PATTERN.finditer(text):
            word = m.group()
            # Пропускаем слишком короткие и совсем служебные
            if len(word) < 3:
                continue
            positions.append((m.start(), m.end(), word))
        return positions

    def _select_index(self, bits: str, pos: int) -> tuple[int, int]:
        """
        По текущей позиции в битовой строке выбрать индекс кандидата.

        Возвращает (index, сколько бит использовано).
        """
        if pos >= len(bits):
            return 0, 0
        chunk = bits[pos : pos + self.bits_per_position]
        if len(chunk) < self.bits_per_position:
            # Дополняем нулями
            chunk = chunk.ljust(self.bits_per_position, "0")
        return int(chunk, 2), self.bits_per_position

    async def embed(
        self,
        cover_text: str,
        secret: str,
        *,
        password: str | None = None,
        progress_callback: Callable[[float, str], Awaitable[None] | None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StegoResult:
        """
        Внедрить секретное сообщение в текст-контейнер.
        """
        if not cover_text.strip():
            return StegoResult(success=False, message="Текст-контейнер пуст")
        if not secret:
            return StegoResult(success=False, message="Секретное сообщение пусто")
        if len(secret) > settings.max_secret_length:
            return StegoResult(
                success=False,
                message=f"Секретное сообщение длиннее лимита {settings.max_secret_length} символов",
            )
        if "\u2063\u200b\u2063\u200b" in cover_text or "\u2063\u200c\u2063\u200c" in cover_text:
            return StegoResult(
                success=False,
                message="Текст-контейнер содержит зарезервированный маркер стеготекста",
            )

        # Опциональное шифрование
        payload = secret
        encrypted = False
        if password:
            try:
                crypto = CryptoModule(password)
                payload = crypto.encrypt(secret)
                encrypted = True
            except Exception as exc:
                return StegoResult(success=False, message=f"Ошибка шифрования: {exc}")

        bits = text_to_bits(payload)
        needed = len(bits)
        all_positions = self._find_replaceable_positions(cover_text)
        positions = [
            position
            for position in all_positions
            if self.adapter.supports_word(position[2])
        ]
        if not positions:
            message = (
                "Во встроенном словаре нет подходящих слов для замены"
                if self.adapter.name == "classic" and all_positions
                else "В тексте-контейнере не найдено слов для видимой замены"
            )
            return StegoResult(
                success=False,
                message=message,
            )

        capacity = len(positions) * self.bits_per_position
        # Строим новый текст, заменяя слова
        result_chars = list(cover_text)
        bit_pos = 0
        used = 0
        visible_bits = 0
        partial_candidate_positions = 0

        # Track replacements so later source offsets stay aligned.
        offset = 0
        total = min(len(positions), self.MAX_VISIBLE_REPLACEMENTS)

        for i, (start, end, word) in enumerate(positions[:total]):
            if bit_pos >= needed:
                break
            if cancel_event and cancel_event.is_set():
                return StegoResult(success=False, message="Операция отменена")

            # Контекст — предложение вокруг слова (упрощённо: ±80 символов)
            ctx_start = max(0, start - 80)
            ctx_end = min(len(cover_text), end + 80)
            context = cover_text[ctx_start:ctx_end]

            try:
                cand_result = await self.adapter.get_replacement_candidates(
                    text=context,
                    word=word,
                    num_candidates=self.num_candidates,
                    cancel_event=cancel_event,
                )
            except asyncio.CancelledError:
                return StegoResult(success=False, message="Операция отменена")
            except Exception as exc:
                logger.error(f"Ошибка получения кандидатов для «{word}»: {exc}")
                return StegoResult(
                    success=False,
                    message=f"Не удалось получить варианты для слова «{word}»: {exc}",
                )

            candidates = cand_result.candidates
            if len(candidates) < self.num_candidates:
                partial_candidate_positions += 1
            available_bits = min(
                self.bits_per_position,
                len(candidates).bit_length() - 1,
            )
            if not available_bits:
                logger.warning(
                    f"Пропуск слова «{word}»: нужен хотя бы один битовый выбор "
                    "между двумя заменами"
                )
                continue

            chunk = bits[bit_pos : bit_pos + available_bits].ljust(
                available_bits,
                "0",
            )
            idx = int(chunk, 2)
            chosen = candidates[idx]
            bit_pos += available_bits
            visible_bits += available_bits

            # Замена с учётом смещения
            adj_start = start + offset
            adj_end = end + offset
            result_chars[adj_start:adj_end] = list(chosen)
            offset += len(chosen) - (end - start)

            used += 1

            if progress_callback:
                prog = min(1.0, bit_pos / needed)
                maybe_coro = progress_callback(prog, f"Внедрено {bit_pos}/{needed} бит")
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro

        stego_text = "".join(result_chars)
        if used == 0:
            return StegoResult(
                success=False,
                message=(
                    "Не удалось закодировать ни одного видимого бита: "
                    "модель не предложила два разных варианта ни для одного слова"
                ),
            )

        stego_text += encode_payload_marker(payload)
        if progress_callback:
            maybe_coro = progress_callback(1.0, "Внедрение выполнено")
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        message = "Внедрение выполнено успешно"
        if visible_bits < needed:
            message = (
                "Внедрение выполнено: полный секрет записан в маркер, "
                f"видимо закодировано только {visible_bits} из {needed} бит "
                "из-за недостатка вариантов замены"
            )

        return StegoResult(
            success=True,
            text=stego_text,
            secret=secret,
            bits_embedded=needed,
            positions_used=used,
            message=message,
            details={
                "encrypted": encrypted,
                "needed_bits": needed,
                "capacity_bits": capacity,
                "bits_per_position": self.bits_per_position,
                "visible_bits_embedded": visible_bits,
                "positions_with_partial_candidates": partial_candidate_positions,
                "reliable_marker": True,
            },
        )

    async def extract(
        self,
        stego_text: str,
        *,
        expected_bits: int | None = None,
        password: str | None = None,
        progress_callback: Callable[[float, str], Awaitable[None] | None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> StegoResult:
        """
        Извлечь скрытое сообщение из стеготекста.

        :param expected_bits: Необязательная проверка ожидаемого количества бит payload.
        """
        if not stego_text.strip():
            return StegoResult(success=False, message="Стеготекст пуст")

        try:
            marker = decode_payload_marker(stego_text)
        except ValueError as exc:
            return StegoResult(success=False, message=str(exc))

        if cancel_event and cancel_event.is_set():
            return StegoResult(success=False, message="Операция отменена")
        if marker is None:
            return StegoResult(
                success=False,
                message=(
                    "Надёжный маркер не найден. Стеготекст создан старой версией "
                    "и не может быть надёжно извлечён."
                ),
            )

        payload, clean_text = marker
        payload_bits = len(payload.encode("utf-8")) * 8
        if expected_bits is not None and expected_bits != payload_bits:
            return StegoResult(
                success=False,
                text=clean_text,
                bits_extracted=payload_bits,
                message=(
                    f"Ожидалось {expected_bits} бит, в маркере найдено "
                    f"{payload_bits} бит"
                ),
                details={"expected_bits": expected_bits, "bits_extracted": payload_bits},
            )

        secret = payload
        if password:
            try:
                secret = CryptoModule(password).decrypt(payload)
            except Exception as exc:
                return StegoResult(
                    success=False,
                    text=clean_text,
                    message=f"Не удалось расшифровать: {exc}",
                    bits_extracted=payload_bits,
                )

        if progress_callback:
            maybe_coro = progress_callback(1.0, "Извлечение выполнено по маркеру")
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        return StegoResult(
            success=True,
            text=clean_text,
            secret=secret,
            bits_extracted=payload_bits,
            message="Извлечение выполнено",
            details={"reliable_marker": True, "encrypted": bool(password)},
        )
