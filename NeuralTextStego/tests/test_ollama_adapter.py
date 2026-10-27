import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from adapters.ollama import OllamaAdapter
from config import settings


class OllamaAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_uses_local_server_without_api_key(self):
        client = Mock()
        with (
            patch.object(settings, "ollama_base_url", "http://127.0.0.1:11434"),
            patch("adapters.ollama.httpx.AsyncClient", return_value=client) as factory,
        ):
            adapter = OllamaAdapter()
            await adapter._ensure_client()

        self.assertEqual(adapter._client, client)
        self.assertEqual(factory.call_args.kwargs["base_url"], "http://127.0.0.1:11434")
        self.assertNotIn("Authorization", factory.call_args.kwargs["headers"])

    async def test_request_uses_ollama_chat_and_json_schema(self):
        response = httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"candidates": ["вариант один", "вариант два"]}),
                }
            },
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OllamaAdapter()
        adapter._client = client

        candidates = await adapter._request_candidates(
            text="Контекст",
            word="слово",
            num_candidates=2,
            temperature=0.1,
        )

        self.assertEqual(candidates, ["вариант один", "вариант два"])
        self.assertEqual(client.post.call_args.args[0], "/api/chat")
        request = client.post.call_args.kwargs["json"]
        self.assertEqual(request["model"], settings.ollama_model)
        self.assertFalse(request["stream"])
        self.assertEqual(request["options"]["num_ctx"], settings.ollama_num_ctx)
        self.assertEqual(request["format"]["required"], ["candidates"])

    async def test_http_errors_are_reported(self):
        response = httpx.Response(
            404,
            text="model not found",
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OllamaAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "Ollama API error 404"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_empty_model_response_is_rejected(self):
        response = httpx.Response(
            200,
            json={"message": {"content": ""}},
        )
        client = Mock()
        client.post = AsyncMock(return_value=response)
        adapter = OllamaAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "пустой ответ"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_missing_local_server_has_setup_guidance(self):
        client = Mock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        adapter = OllamaAdapter()
        adapter._client = client

        with self.assertRaisesRegex(RuntimeError, "ollama pull qwen2.5:3b"):
            await adapter._request_candidates("Контекст", "слово", 2, temperature=0.1)

    async def test_one_good_candidate_is_accepted_without_retries(self):
        adapter = OllamaAdapter()
        adapter._ensure_client = AsyncMock()
        adapter._request_candidates = AsyncMock(return_value=["аналитичный"])

        result = await adapter.get_replacement_candidates(
            text="аналитический обзор",
            word="аналитический",
            num_candidates=4,
        )

        self.assertEqual(result.candidates, ["аналитичный"])
        adapter._request_candidates.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
