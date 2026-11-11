"""Общие преобразования битов и опциональный маркер."""

from __future__ import annotations

from core.crypto import CryptoModule
from core.metadata import decode_payload_marker, encode_payload_marker

# Длина полезной нагрузки в байтах (режим без маркера).
PAYLOAD_LEN_BITS = 16


def text_to_bits(text: str) -> str:
    data = text.encode("utf-8")
    return "".join(f"{b:08b}" for b in data)


def bits_to_text(bits: str) -> str:
    usable = bits[: len(bits) - (len(bits) % 8)]
    if not usable:
        return ""
    data = bytes(int(usable[i : i + 8], 2) for i in range(0, len(usable), 8))
    return data.decode("utf-8", errors="replace")


def frame_payload_bits(payload: str) -> str:
    """Префикс длины (16 бит) + UTF-8 биты. Нужен для извлечения без маркера."""
    data = payload.encode("utf-8")
    if len(data) > 0xFFFF:
        raise ValueError("Секрет слишком длинный для режима без маркера (лимит 65535 байт)")
    return f"{len(data):016b}" + "".join(f"{b:08b}" for b in data)


def unframe_payload_bits(bits: str) -> str | None:
    """Обратное к frame_payload_bits. None, если бит не хватает."""
    if len(bits) < PAYLOAD_LEN_BITS:
        return None
    nbytes = int(bits[:PAYLOAD_LEN_BITS], 2)
    need = PAYLOAD_LEN_BITS + nbytes * 8
    if len(bits) < need:
        return None
    raw = bits[PAYLOAD_LEN_BITS:need]
    data = bytes(int(raw[i : i + 8], 2) for i in range(0, len(raw), 8))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def prepare_payload(secret: str, password: str | None) -> tuple[str, bool]:
    if password:
        return CryptoModule(password).encrypt(secret), True
    return secret, False


def recover_secret(payload: str, password: str | None) -> str:
    if password:
        return CryptoModule(password).decrypt(payload)
    return payload


def append_marker_if_needed(text: str, payload: str, use_marker: bool) -> str:
    if use_marker:
        return text + encode_payload_marker(payload)
    return text


def try_extract_marker(text: str) -> tuple[str, str] | None:
    """Вернуть (payload, clean_text) или None."""
    try:
        return decode_payload_marker(text)
    except ValueError:
        raise
