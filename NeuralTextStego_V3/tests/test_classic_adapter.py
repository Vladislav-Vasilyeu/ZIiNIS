import unittest

from adapters.classic import (
    _SYNONYM_GROUPS,
    ClassicSynonymAdapter,
    synonym_codebook,
)
from core.stego_engine import WORD_PATTERN, StegoEngine
from utils.file_handlers import load_text


class ClassicSynonymAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_codebook_variants_keep_the_same_decoding_order(self):
        for candidate_count in (2, 4):
            for group in _SYNONYM_GROUPS:
                for word in group:
                    codebook = synonym_codebook(word, candidate_count)
                    if len(codebook) < 2:
                        continue
                    expected = [candidate.casefold() for candidate in codebook]
                    for variant in codebook:
                        with self.subTest(word=word, variant=variant, candidates=candidate_count):
                            actual = [
                                candidate.casefold()
                                for candidate in synonym_codebook(variant, candidate_count)
                            ]
                            self.assertEqual(actual, expected)

    def test_sample_cover_contains_expanded_dictionary_vocabulary(self):
        cover = load_text("data/sample_cover.txt")
        words = [match.group() for match in WORD_PATTERN.finditer(cover)]
        supported = {word.casefold() for word in words if len(synonym_codebook(word, 4)) >= 2}
        self.assertGreaterEqual(len(supported), 35)

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
