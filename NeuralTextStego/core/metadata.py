"""Reversible invisible metadata used to recover the exact embedded payload."""

from __future__ import annotations

import hashlib

_START = "\u2063\u200b\u2063\u200b"
_END = "\u2063\u200c\u2063\u200c"
_ZERO = "\u200b"
_ONE = "\u200c"
_MAGIC = b"NTS1"
_CHECKSUM_SIZE = 8


def encode_payload_marker(payload: str) -> str:
    data = payload.encode("utf-8")
    header = _MAGIC + len(data).to_bytes(4, "big")
    checksum = hashlib.sha256(data).digest()[:_CHECKSUM_SIZE]
    bits = "".join(f"{byte:08b}" for byte in header + data + checksum)
    encoded_bits = bits.translate(str.maketrans({"0": _ZERO, "1": _ONE}))
    return f"{_START}{encoded_bits}{_END}"


def decode_payload_marker(text: str) -> tuple[str, str] | None:
    start = text.find(_START)
    end = text.find(_END)
    if start == -1 and end == -1:
        return None
    if (
        start == -1
        or end == -1
        or text.count(_START) != 1
        or text.count(_END) != 1
        or end < start + len(_START)
    ):
        raise ValueError("Невидимый маркер повреждён или продублирован")

    encoded_bits = text[start + len(_START) : end]
    if any(char not in (_ZERO, _ONE) for char in encoded_bits) or len(encoded_bits) % 8:
        raise ValueError("Невидимый маркер содержит повреждённые данные")

    bits = encoded_bits.translate(str.maketrans({_ZERO: "0", _ONE: "1"}))
    data = bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))
    header_size = len(_MAGIC) + 4
    if len(data) < header_size + _CHECKSUM_SIZE or data[: len(_MAGIC)] != _MAGIC:
        raise ValueError("Невидимый маркер имеет неизвестный формат")

    payload_size = int.from_bytes(data[len(_MAGIC) : header_size], "big")
    expected_size = header_size + payload_size + _CHECKSUM_SIZE
    if len(data) != expected_size:
        raise ValueError("Длина данных в невидимом маркере не совпадает")

    payload_bytes = data[header_size : header_size + payload_size]
    checksum = data[-_CHECKSUM_SIZE:]
    if hashlib.sha256(payload_bytes).digest()[:_CHECKSUM_SIZE] != checksum:
        raise ValueError("Контрольная сумма невидимого маркера не совпадает")

    try:
        payload = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Данные невидимого маркера не являются текстом UTF-8") from exc

    clean_text = text[:start] + text[end + len(_END) :]
    return payload, clean_text
