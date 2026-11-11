"""Реальные round-trip тесты классических методов (без сети)."""

from __future__ import annotations

import asyncio
import unittest

from methods.classic_synonyms import ClassicSynonymsMethod
from methods.classic_whitespace import ClassicWhitespaceMethod
from methods.bits import text_to_bits, bits_to_text
from core.metadata import encode_payload_marker, decode_payload_marker
from core.crypto import CryptoModule
from utils.cover_generator import generate_cover_local
from utils.file_handlers import load_text


class BitsTests(unittest.TestCase):
    def test_bits_roundtrip(self):
        for s in ["A", "привет", "hello world", "🔑"]:
            b = text_to_bits(s)
            self.assertEqual(bits_to_text(b)[: len(s)], s[: len(bits_to_text(b))])


class MarkerTests(unittest.TestCase):
    def test_marker_roundtrip(self):
        payload = "секретное сообщение"
        text = "Видимый текст."
        stego = text + encode_payload_marker(payload)
        got = decode_payload_marker(stego)
        self.assertIsNotNone(got)
        p, clean = got
        self.assertEqual(p, payload)
        self.assertEqual(clean, text)


class CryptoTests(unittest.TestCase):
    def test_aes_roundtrip(self):
        c = CryptoModule("пароль123")
        token = c.encrypt("hello")
        self.assertEqual(c.decrypt(token), "hello")


class WhitespaceRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def test_embed_extract_with_marker(self):
        secret = "hi"
        cover = generate_cover_local(secret, "classic_whitespace", seed=42)
        m = ClassicWhitespaceMethod()
        emb = await m.embed(secret=secret, cover_text=cover, use_marker=True)
        self.assertTrue(emb.success, emb.message)
        ext = await m.extract(stego_text=emb.text, use_marker=True)
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)

    async def test_embed_extract_without_marker(self):
        secret = "AB"
        cover = generate_cover_local(secret, "classic_whitespace", seed=1)
        m = ClassicWhitespaceMethod()
        emb = await m.embed(secret=secret, cover_text=cover, use_marker=False)
        self.assertTrue(emb.success, emb.message)
        bits = len(secret.encode("utf-8")) * 8
        ext = await m.extract(
            stego_text=emb.text, use_marker=False, expected_bits=bits
        )
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)


class SynonymsRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def test_embed_extract_with_marker(self):
        secret = "test"
        cover = generate_cover_local(secret, "classic_synonyms", seed=7)
        m = ClassicSynonymsMethod(num_candidates=4)
        emb = await m.embed(secret=secret, cover_text=cover, use_marker=True)
        self.assertTrue(emb.success, emb.message)
        ext = await m.extract(stego_text=emb.text, use_marker=True)
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)

    async def test_embed_extract_without_marker(self):
        secret = "Привет"
        cover = generate_cover_local(secret, "classic_synonyms", seed=11)
        m = ClassicSynonymsMethod(num_candidates=4)
        emb = await m.embed(secret=secret, cover_text=cover, use_marker=False)
        self.assertTrue(emb.success, emb.message)
        ext = await m.extract(stego_text=emb.text, use_marker=False)
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)

    async def test_expanded_sample_cover_round_trip_without_marker(self):
        cover = load_text("data/sample_cover.txt")
        method = ClassicSynonymsMethod(num_candidates=4)
        emb = await method.embed(secret="sample", cover_text=cover, use_marker=False)
        self.assertTrue(emb.success, emb.message)
        ext = await method.extract(stego_text=emb.text, use_marker=False)
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, "sample")


class OllamaParseTests(unittest.TestCase):
    def test_parse_logprobs_format(self):
        from providers.ollama_logits import OllamaLogitsProvider

        p = OllamaLogitsProvider()
        data = {
            "logprobs": [
                {
                    "token": " Hello",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": " Hello", "logprob": -0.1},
                        {"token": " Hi", "logprob": -1.2},
                        {"token": " Hey", "logprob": -2.0},
                    ],
                }
            ]
        }
        cands = p._parse_logprobs_payload(data, 5)
        self.assertGreaterEqual(len(cands), 2)
        self.assertEqual(cands[0].token, " Hello")


if __name__ == "__main__":
    unittest.main()
