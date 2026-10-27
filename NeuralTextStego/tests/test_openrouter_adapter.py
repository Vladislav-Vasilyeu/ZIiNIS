import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from adapters.base import BaseLLMAdapter
from adapters.classic import ClassicSynonymAdapter
from adapters.factory import create_adapter
from adapters.openrouter import OpenRouterAdapter
from adapters.ollama import OllamaAdapter
from config import Settings, settings


class StubAdapter(BaseLLMAdapter):
    name = "stub"

    async def _ensure_client(self):
        return None

    async def _request_candidates(self, text, word, num_candidates, *, temperature):
        return ["вариант один", "вариант два"]


class OpenRouterAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_uses_configured_bearer_token(self):
        client = Mock()
        with (
            patch.object(settings, "openrouter_api_key", "test-openrouter-key"),
            patch(
                "adapters.openrouter.httpx.AsyncClient",
                return_value=client,
            ) as client_factory,
        ):
            adapter = OpenRouterAdapter()
            await adapter._ensure_client()

        self.assertEqual(adapter._client, client)
        self.assertEqual(
            client_factory.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-openrouter-key",
        )
        self.assertEqual(client_factory.call_args.kwargs["base_url"], adapter.API_BASE)

    async def test_request_uses_free_router_and_parses_structured_output(self):
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"candidates":["ливень","ливень сильный",'
                                '"обильный дождь","проливной"]}'
                            )
                        }
                    }
                ]
            },
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)

        adapter = OpenRouterAdapter()
        adapter._client = client
        candidates = await adapter._request_candidates(
            text="На улице начался сильный дождь",
            word="дождь",
            num_candidates=4,
            temperature=0.1,
        )

        self.assertEqual(
            candidates,
            ["ливень", "ливень сильный", "обильный дождь", "проливной"],
        )
        client.post.assert_awaited_once()
        self.assertEqual(client.post.call_args.args[0], "/chat/completions")
        self.assertEqual(
            client.post.call_args.kwargs["json"]["model"],
            "openrouter/free",
        )
        self.assertIn(
            "до 4",
            client.post.call_args.kwargs["json"]["messages"][1]["content"],
        )
        self.assertEqual(
            client.post.call_args.kwargs["json"]["messages"][0]["role"],
            "system",
        )
        self.assertEqual(client.post.call_args.kwargs["json"]["stream"], False)
        self.assertTrue(
            client.post.call_args.kwargs["json"]["provider"]["require_parameters"]
        )
        request = client.post.call_args.kwargs["json"]
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["max_tokens"], 2048)

    async def test_request_reports_http_errors(self):
        response = httpx.Response(
            429,
            text='{"error":{"message":"rate limited"}}',
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OpenRouterAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "429"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_empty_model_response_is_rejected(self):
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": None}, "finish_reason": "length"}
                ]
            },
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OpenRouterAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "finish_reason='length'"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_non_json_model_response_is_rejected(self):
        response = httpx.Response(
            200,
            json={
                "model": "bad-model",
                "choices": [{"message": {"content": "не JSON"}}],
            },
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OpenRouterAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "bad-model.*JSON"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_too_few_unique_candidates_fail_instead_of_filling_with_original(self):
        with (
            patch.object(settings, "max_retries", 1),
            patch.object(settings, "retry_base_delay", 0),
        ):
            with self.assertRaisesRegex(RuntimeError, "только 2 из 4"):
                await StubAdapter().get_replacement_candidates(
                    text="Контекст",
                    word="исходное",
                    num_candidates=4,
                )

    def test_factory_exposes_all_supported_methods(self):
        self.assertIsInstance(create_adapter("openrouter"), OpenRouterAdapter)
        self.assertIsInstance(create_adapter("ollama"), OllamaAdapter)
        self.assertIsInstance(create_adapter("classic"), ClassicSynonymAdapter)
        with self.assertRaises(ValueError):
            create_adapter("unsupported")

    def test_openrouter_is_the_only_default_provider(self):
        self.assertEqual(Settings.model_fields["default_provider"].default, "openrouter")


if __name__ == "__main__":
    unittest.main()
