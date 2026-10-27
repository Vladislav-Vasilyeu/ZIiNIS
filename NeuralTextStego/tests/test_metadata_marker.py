import tempfile
import unittest
from pathlib import Path

from adapters.base import BaseLLMAdapter
from core.metadata import decode_payload_marker, encode_payload_marker
from core.stego_engine import StegoEngine
from utils.file_handlers import load_text, save_text


class StableSynonymAdapter(BaseLLMAdapter):
    name = "stable-test-adapter"

    def __init__(self):
        super().__init__()
        self.request_count = 0

    async def _ensure_client(self):
        return None

    async def _request_candidates(self, text, word, num_candidates, *, temperature):
        self.request_count += 1
        return ["влага", "жидкость", "субстанция", "молекула"]


class MetadataMarkerTests(unittest.IsolatedAsyncioTestCase):
    async def _embed(self, secret="A", password=None):
        adapter = StableSynonymAdapter()
        result = await StegoEngine(adapter, num_candidates=4).embed(
            "вода вода вода вода",
            secret,
            password=password,
        )
        self.assertTrue(result.success, result.message)
        return adapter, result

    async def test_embed_extract_round_trip_does_not_call_model_during_extraction(self):
        adapter, embedded = await self._embed()
        calls_after_embed = adapter.request_count

        extracted = await StegoEngine(adapter, num_candidates=4).extract(
            embedded.text,
            expected_bits=8,
        )

        self.assertTrue(extracted.success, extracted.message)
        self.assertEqual(extracted.secret, "A")
        self.assertEqual(extracted.bits_extracted, 8)
        self.assertTrue(extracted.details["reliable_marker"])
        self.assertEqual(adapter.request_count, calls_after_embed)

    async def test_long_payload_caps_visible_candidate_requests(self):
        adapter = StableSynonymAdapter()
        secret = "A" * 40
        embedded = await StegoEngine(adapter, num_candidates=4).embed(
            "вода " * 50,
            secret,
        )
        extracted = await StegoEngine(adapter, num_candidates=4).extract(embedded.text)

        self.assertTrue(embedded.success, embedded.message)
        self.assertEqual(adapter.request_count, 8)
        self.assertEqual(embedded.positions_used, 8)
        self.assertTrue(extracted.success, extracted.message)
        self.assertEqual(extracted.secret, secret)

    async def test_partial_candidate_sets_embed_only_their_actual_bit_capacity(self):
        class PartialAdapter(StableSynonymAdapter):
            allows_partial_candidates = True

            async def _request_candidates(self, text, word, num_candidates, *, temperature):
                self.request_count += 1
                return ["синоним", "замена"]

        adapter = PartialAdapter()
        embedded = await StegoEngine(adapter, num_candidates=4).embed(
            "текст текст текст текст",
            "A",
        )

        self.assertTrue(embedded.success, embedded.message)
        self.assertEqual(embedded.details["visible_bits_embedded"], 4)
        self.assertEqual(embedded.details["positions_with_partial_candidates"], 4)
        self.assertIn("4 из 8", embedded.message)

    async def test_single_candidate_is_skipped_in_favor_of_later_encodable_word(self):
        class MixedAdapter(StableSynonymAdapter):
            allows_partial_candidates = True

            async def _request_candidates(self, text, word, num_candidates, *, temperature):
                self.request_count += 1
                if word == "обзор":
                    return ["сводка", "резюме"]
                return ["аналитичный"]

        embedded = await StegoEngine(MixedAdapter(), num_candidates=4).embed(
            "Аналитический обзор текста",
            "A",
        )

        self.assertTrue(embedded.success, embedded.message)
        self.assertEqual(embedded.positions_used, 1)
        self.assertEqual(embedded.details["visible_bits_embedded"], 1)
        self.assertTrue(embedded.text.startswith("Аналитический "))

    async def test_encrypted_payload_round_trips_with_password(self):
        _, embedded = await self._embed(password="demo-password")
        extracted = await StegoEngine(StableSynonymAdapter(), num_candidates=4).extract(
            embedded.text,
            password="demo-password",
        )

        self.assertTrue(extracted.success, extracted.message)
        self.assertEqual(extracted.secret, "A")
        self.assertTrue(extracted.details["encrypted"])

    async def test_marker_survives_txt_and_docx_round_trips(self):
        _, embedded = await self._embed()
        with tempfile.TemporaryDirectory() as temp_dir:
            for suffix in (".txt", ".docx"):
                with self.subTest(suffix=suffix):
                    path = Path(temp_dir) / f"stego{suffix}"
                    save_text(path, embedded.text)
                    loaded = load_text(path)
                    extracted = await StegoEngine(
                        StableSynonymAdapter(),
                        num_candidates=4,
                    ).extract(loaded, expected_bits=8)
                    self.assertTrue(extracted.success, extracted.message)
                    self.assertEqual(extracted.secret, "A")

    async def test_tampered_marker_is_rejected(self):
        marker = encode_payload_marker("A")
        payload_start = len("\u2063\u200b\u2063\u200b") + 8 * 8
        first_bit = marker[payload_start]
        replacement = "\u200c" if first_bit == "\u200b" else "\u200b"
        tampered = (
            marker[:payload_start]
            + replacement
            + marker[payload_start + 1 :]
        )

        with self.assertRaisesRegex(ValueError, "Контрольная сумма"):
            decode_payload_marker(tampered)

        extracted = await StegoEngine(StableSynonymAdapter()).extract(tampered)
        self.assertFalse(extracted.success)

    async def test_unmarked_legacy_text_is_rejected_explicitly(self):
        extracted = await StegoEngine(StableSynonymAdapter()).extract("старый стеготекст")

        self.assertFalse(extracted.success)
        self.assertIn("маркер не найден", extracted.message)

    async def test_marker_expected_bit_count_is_validated(self):
        _, embedded = await self._embed()
        extracted = await StegoEngine(StableSynonymAdapter()).extract(
            embedded.text,
            expected_bits=16,
        )

        self.assertFalse(extracted.success)
        self.assertIn("Ожидалось 16 бит", extracted.message)


if __name__ == "__main__":
    unittest.main()
