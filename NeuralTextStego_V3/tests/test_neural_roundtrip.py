"""Round-trip нейросетевых методов на детерминированном fake-провайдере."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from methods.bits import frame_payload_bits, unframe_payload_bits
from methods.neural_methods import (
    NeuralArithmeticMethod,
    NeuralRankMethod,
    NeuralSamplingMethod,
)
from providers.base import LogitsProvider, TokenCandidate


class FakeLogitsProvider(LogitsProvider):
    name = "fake"

    def __init__(self) -> None:
        self.tokens = [chr(65 + i) for i in range(8)]  # A..H

    async def next_token_distribution(
        self,
        prompt: str,
        *,
        top_k: int = 20,
        temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        return [
            TokenCandidate(token=token, logprob=-float(i), rank=i)
            for i, token in enumerate(self.tokens[: max(2, min(top_k, len(self.tokens)))])
        ]

    async def close(self) -> None:
        return None


class FrameBitsTests(unittest.TestCase):
    def test_frame_roundtrip(self):
        for secret in ["A", "hi", "привет", "🔑"]:
            bits = frame_payload_bits(secret)
            self.assertEqual(unframe_payload_bits(bits), secret)

    def test_utf8_sig_csv_roundtrip(self):
        text = "run;success;message\n1;True;Внедрение выполнено\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp.csv"
            path.write_text(text, encoding="utf-8-sig")
            loaded = path.read_text(encoding="utf-8-sig")
            self.assertIn("Внедрение выполнено", loaded)
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))


class NeuralRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def _roundtrip(self, method_cls, *, use_marker: bool, secret: str = "hi"):
        method = method_cls(FakeLogitsProvider(), chunk_bits=2, top_k=8, max_tokens=256)
        prompt = "Научный текст:\n"
        emb = await method.embed(
            secret=secret,
            prompt=prompt,
            password=None,
            use_marker=use_marker,
            progress=None,
            cancel_event=None,
        )
        self.assertTrue(emb.success, emb.message)
        ext = await method.extract(
            stego_text=emb.text,
            prompt=prompt,
            password=None,
            use_marker=use_marker,
            progress=None,
            cancel_event=None,
        )
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)

    async def test_rank_with_marker(self):
        await self._roundtrip(NeuralRankMethod, use_marker=True)

    async def test_rank_without_marker(self):
        await self._roundtrip(NeuralRankMethod, use_marker=False, secret="Привет")

    async def test_unmarked_payload_falls_back_from_marker_detection(self):
        method = NeuralRankMethod(FakeLogitsProvider(), chunk_bits=2, top_k=8, max_tokens=256)
        prompt = "Prompt:\n"
        emb = await method.embed(
            secret="no marker", prompt=prompt, password=None,
            use_marker=False, progress=None, cancel_event=None,
        )
        self.assertTrue(emb.success, emb.message)
        ext = await method.extract(
            stego_text=emb.text, prompt=prompt, password=None,
            use_marker=True, progress=None, cancel_event=None,
        )
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, "no marker")
        self.assertFalse(ext.details["reliable_marker"])

    async def test_sampling_without_marker(self):
        await self._roundtrip(NeuralSamplingMethod, use_marker=False)

    async def test_arithmetic_without_marker(self):
        await self._roundtrip(NeuralArithmeticMethod, use_marker=False)

    async def test_rank_password_without_marker(self):
        method = NeuralRankMethod(FakeLogitsProvider(), chunk_bits=2, top_k=8, max_tokens=512)
        prompt = "Текст:\n"
        secret = "ok"
        emb = await method.embed(
            secret=secret,
            prompt=prompt,
            password="pw",
            use_marker=False,
            progress=None,
            cancel_event=None,
        )
        self.assertTrue(emb.success, emb.message)
        ext = await method.extract(
            stego_text=emb.text,
            prompt=prompt,
            password="pw",
            use_marker=False,
            progress=None,
            cancel_event=None,
        )
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)


if __name__ == "__main__":
    unittest.main()
