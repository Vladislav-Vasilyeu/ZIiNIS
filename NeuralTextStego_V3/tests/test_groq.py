import json
from unittest.mock import patch

import httpx
import pytest

from config import Settings, settings
from providers.factory import create_logits_provider


@pytest.mark.asyncio
async def test_groq_models_and_completion_use_groq_credentials_and_endpoint():
    provider = create_logits_provider("groq", model="openai/gpt-oss-120b")
    client_factory = httpx.AsyncClient

    def handle(request):
        assert request.headers["Authorization"] == "Bearer test-groq-key"
        if request.method == "GET":
            assert str(request.url) == "https://api.groq.com/openai/v1/models"
            return httpx.Response(200, json={"data": [
                {"id": "openai/gpt-oss-120b"}, {"id": "openai/gpt-oss-20b"},
            ]})
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        body = json.loads(request.content)
        assert body == {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "test prompt"}],
            "temperature": 0.8,
            "max_completion_tokens": 123,
            "reasoning_effort": "low",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})

    with patch.object(settings, "groq_api_key", "test-groq-key"), patch(
        "providers.openai_compatible.httpx.AsyncClient",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handle), **kwargs),
    ):
        try:
            assert await provider.list_models() == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
            assert await provider.complete_text("test prompt", max_tokens=123) == "Ответ"
        finally:
            await provider.close()
    assert provider._client is None


@pytest.mark.asyncio
async def test_groq_logprobs_explains_limitation_without_request():
    provider = create_logits_provider("groq")
    with patch("providers.openai_compatible.httpx.AsyncClient") as client:
        with pytest.raises(RuntimeError, match="Groq.*logprobs"):
            await provider.next_token_distribution("test prompt")
        client.assert_not_called()


@pytest.mark.asyncio
async def test_groq_missing_key_names_correct_env_variable():
    provider = create_logits_provider("groq")
    with patch.object(settings, "groq_api_key", ""):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            await provider.complete_text("test prompt")


def test_groq_can_be_configured_as_default():
    assert Settings(default_logits_provider="groq").default_logits_provider == "groq"
