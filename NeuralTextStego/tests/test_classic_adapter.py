import unittest

from adapters.classic import ClassicSynonymAdapter
from core.stego_engine import StegoEngine


class ClassicSynonymAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_deterministic_case_aware_candidates(self):
        adapter = ClassicSynonymAdapter()
        first = await adapter.get_replacement_candidates(
            text="Контекст",
            word="Современные",
            num_candidates=4,
        )
        second = await adapter.get_replacement_candidates(
            text="Другой контекст",
            word="Современные",
            num_candidates=4,
        )

        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.candidates, ["Актуальные", "Новейшие", "Нынешние", "Сегодняшние"])

    async def test_unsupported_words_are_skipped_during_embedding(self):
        adapter = ClassicSynonymAdapter()
        result = await StegoEngine(adapter, num_candidates=4).embed(
            "Неизвестныйсловарю современные",
            "A",
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.positions_used, 1)
        self.assertTrue(
            any(
                candidate.casefold() in result.text.casefold()
                for candidate in ("актуальные", "новейшие", "нынешние", "сегодняшние")
            )
        )

    async def test_no_dictionary_match_returns_a_clear_error(self):
        result = await StegoEngine(ClassicSynonymAdapter()).embed(
            "неизвестно",
            "A",
        )

        self.assertFalse(result.success)
        self.assertIn("словаре", result.message)
