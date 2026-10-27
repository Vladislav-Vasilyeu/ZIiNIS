"""
Опциональный криптографический модуль (AES-256-GCM).
"""

from __future__ import annotations

import os
from base64 import urlsafe_b64decode, urlsafe_b64encode

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class CryptoModule:
    """
    Шифрование/расшифровка секретного сообщения перед внедрением.

    Использует AES-256-GCM + Scrypt для получения ключа из пароля.
    """

    SALT_SIZE = 16
    NONCE_SIZE = 12
    KEY_SIZE = 32

    def __init__(self, password: str) -> None:
        if not password:
            raise ValueError("Пароль не может быть пустым")
        self._password = password.encode("utf-8")

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = Scrypt(
            salt=salt,
            length=self.KEY_SIZE,
            n=2**14,
            r=8,
            p=1,
        )
        return kdf.derive(self._password)

    def encrypt(self, plaintext: str) -> str:
        """Вернуть base64url-строку: salt + nonce + ciphertext."""
        data = plaintext.encode("utf-8")
        salt = os.urandom(self.SALT_SIZE)
        key = self._derive_key(salt)
        nonce = os.urandom(self.NONCE_SIZE)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        payload = salt + nonce + ciphertext
        return urlsafe_b64encode(payload).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Расшифровать base64url-строку, полученную методом encrypt."""
        try:
            payload = urlsafe_b64decode(token.encode("ascii"))
        except Exception as exc:
            raise ValueError("Некорректный формат зашифрованных данных") from exc

        if len(payload) < self.SALT_SIZE + self.NONCE_SIZE + 16:
            raise ValueError("Слишком короткие зашифрованные данные")

        salt = payload[: self.SALT_SIZE]
        nonce = payload[self.SALT_SIZE : self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = payload[self.SALT_SIZE + self.NONCE_SIZE :]

        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        try:
            data = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise ValueError("Неверный пароль или повреждённые данные") from exc

        return data.decode("utf-8")
