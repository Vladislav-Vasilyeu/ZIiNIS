"""Сквозная проверка методов, фабрики и CSV."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from adapters.factory import create_adapter
from config import Settings
from methods.registry import NEURAL_KINDS, create_method
from providers.factory import create_logits_provider
from tests.test_neural_roundtrip import FakeLogitsProvider
from utils.cover_generator import generate_cover_local


class FactoryTests(unittest.TestCase):
    def test_legacy_openrouter_config_selects_ollama(self):
        self.assertEqual(Settings(default_logits_provider="openrouter").default_logits_provider, "ollama")

    def test_logits_openrouter_is_disabled(self):
        with self.assertRaises(ValueError):
            create_logits_provider("openrouter")

    def test_adapter_openrouter_is_disabled(self):
        with self.assertRaises(ValueError):
            create_adapter("openrouter")


class FunctionalClassic(unittest.IsolatedAsyncioTestCase):
    async def test_all_classic_modes(self):
        secret = "КП-тест"
        for method_key, marker in (
            ("classic_whitespace", True),
            ("classic_whitespace", False),
            ("classic_synonyms", True),
            ("classic_synonyms", False),
        ):
            with self.subTest(method=method_key, marker=marker):
                cover = generate_cover_local(secret, method_key, seed=3)
                method = create_method(method_key)
                emb = await method.embed(
                    secret=secret, cover_text=cover, use_marker=marker
                )
                self.assertTrue(emb.success, emb.message)
                ext = await method.extract(stego_text=emb.text, use_marker=marker)
                self.assertTrue(ext.success, ext.message)
                self.assertEqual(ext.secret, secret)
                await method.close()


class FunctionalNeural(unittest.IsolatedAsyncioTestCase):
    async def test_all_neural_modes_with_fake_provider(self):
        secret = "ok"
        prompt = "Промпт:\n"
        for kind in sorted(NEURAL_KINDS):
            for marker in (True, False):
                with self.subTest(method=kind, marker=marker):
                    method = create_method(kind, logits_provider="ollama")
                    method._provider = FakeLogitsProvider()
                    method._codec.provider = method._provider
                    method._codec.max_tokens = 256
                    emb = await method.embed(
                        secret=secret, prompt=prompt, cover_text=prompt, use_marker=marker
                    )
                    self.assertTrue(emb.success, emb.message)
                    ext = await method.extract(
                        stego_text=emb.text, prompt=prompt, use_marker=marker
                    )
                    self.assertTrue(ext.success, ext.message)
                    self.assertEqual(ext.secret, secret)
                    await method.close()

    async def test_unmarked_secret_extracts_with_marker_detection_enabled(self):
        method = create_method("neural_rank", logits_provider="ollama")
        method._provider = FakeLogitsProvider()
        method._codec.provider = method._provider
        method._codec.max_tokens = 256
        prompt = "Prompt:\n"
        secret = "unmarked"
        emb = await method.embed(
            secret=secret, prompt=prompt, cover_text=prompt, use_marker=False
        )
        self.assertTrue(emb.success, emb.message)
        ext = await method.extract(
            stego_text=emb.text, prompt=prompt, use_marker=True
        )
        self.assertTrue(ext.success, ext.message)
        self.assertEqual(ext.secret, secret)
        self.assertFalse(ext.details["reliable_marker"])
        await method.close()


class CsvEncoding(unittest.TestCase):
    def test_excel_friendly_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp.csv"
            from utils.experiment_results import write_experiment_csv

            message = "Внедрение; выполнено\nуспешно"
            write_experiment_csv(path, [["run", "success", "message"], [1, True, message]])
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
            import csv

            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.reader(stream, delimiter=";")), [
                    ["run", "success", "message"], ["1", "True", message]
                ])


if __name__ == "__main__":
    unittest.main()
