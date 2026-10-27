import unittest
from unittest.mock import patch

from adapters.base import BaseLLMAdapter
from config import settings
from core.stego_engine import StegoEngine


class FailingAdapter(BaseLLMAdapter):
    name = "failing-test-adapter"

    async def _ensure_client(self):
        return None

    async def _request_candidates(self, text, word, num_candidates, *, temperature):
        raise RuntimeError("model unavailable")


class StegoEngineFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_does_not_report_success_when_model_requests_fail(self):
        with (
            patch.object(settings, "max_retries", 1),
            patch.object(settings, "retry_base_delay", 0),
        ):
            result = await StegoEngine(FailingAdapter(), num_candidates=4).embed(
                "слово " * 4,
                "A",
            )

        self.assertFalse(result.success)
        self.assertEqual(result.bits_embedded, 0)
        self.assertIn("Не удалось получить варианты", result.message)

    async def test_extract_does_not_report_success_when_model_requests_fail(self):
        with (
            patch.object(settings, "max_retries", 1),
            patch.object(settings, "retry_base_delay", 0),
        ):
            result = await StegoEngine(FailingAdapter(), num_candidates=4).extract(
                "слово " * 4,
                expected_bits=8,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.bits_extracted, 0)
        self.assertIn("Надёжный маркер не найден", result.message)


if __name__ == "__main__":
    unittest.main()
