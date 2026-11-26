"""Exact capacity for the two edit-based codecs, without encryption work."""
from dataclasses import dataclass
import math

from methods.classic_synonyms import ClassicSynonymsMethod


@dataclass(frozen=True)
class CoverPlan:
    payload_bytes: int
    required_bits: int
    minimum_positions: int
    target_words: int
    max_tokens: int


def plan_cover(secret: str, method: str, *, password: str | None = None,
               use_marker: bool = False) -> CoverPlan:
    """Compute payload size exactly; word/token lengths remain estimates."""
    size = payload_size(secret, password)
    bits = size * 8 + (0 if method == "classic_whitespace" and use_marker else 16)
    if method == "classic_whitespace":
        minimum = bits
        words = max(100, math.ceil(bits * 1.15) + 1)
    elif method == "classic_synonyms":
        minimum = math.ceil(bits / 2)  # Lower bound, only for four-entry codebooks.
        words = max(100, minimum * 3)
    else:
        from config import settings
        chunk_bits = max(1, min(settings.rank_chunk_bits, 3))
        minimum = math.ceil(bits / chunk_bits)
        words = 80  # This button prepares a prompt; embed produces the stegotext.
    return CoverPlan(size, bits, minimum, words, min(4096, max(500, words * 4)))


@dataclass(frozen=True)
class Capacity:
    available: int
    required: int

    @property
    def sufficient(self) -> bool:
        return self.available >= self.required

    def describe(self) -> str:
        return f"Ёмкость: {self.available} бит · требуется: {self.required} бит"


def payload_size(secret: str, password: str | None = None) -> int:
    size = len(secret.encode("utf-8"))
    if password:
        # salt 16 + nonce 12 + GCM tag 16, then base64 including padding.
        size = 4 * ((size + 44 + 2) // 3)
    if size > 65535:
        raise ValueError("Сообщение слишком длинное: полезная нагрузка не более 65535 байт")
    return size


def assess_capacity(cover: str, secret: str, method: str, *,
                    password: str | None = None, use_marker: bool = False) -> Capacity | None:
    if method == "classic_whitespace":
        available = cover.count(" ") + cover.count("\u00a0")
        header = 0 if use_marker else 16
    elif method == "classic_synonyms":
        available = sum(1 if len(book) < 4 else 2
                        for *_, book in ClassicSynonymsMethod()._positions(cover))
        header = 16
    else:
        return None
    return Capacity(available, payload_size(secret, password) * 8 + header)
