import json
from unittest.mock import patch

import httpx
import pytest

from config import settings
from providers.factory import create_logits_provider


@pytest.mark.asyncio
async def test_tooken_club_chat_and_ranked_logprobs():
    provider = create_logits_provider("tooken_club", model="test-model")
    requests = []

    def handle(request):
        assert str(request.url) == "https://tooken.club/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]
        requests.append(body)
        if body.get("logprobs"):
            return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{
                "top_logprobs": [
                    {"token": " б", "logprob": -2},
                    {"token": " а", "logprob": -1},
                    {"token": " а", "logprob": -3},
                ],
            }]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})

    client_factory = httpx.AsyncClient
    with patch.object(settings, "tooken_club_api_key", "test-key"), patch(
        "providers.tooken_club.httpx.AsyncClient",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handle), **kwargs),
    ):
        try:
            assert await provider.complete_text("test prompt", max_tokens=16) == "Ответ"
            ranked = await provider.next_token_distribution("test prompt", top_k=4)
            assert [(c.token, c.logprob, c.rank) for c in ranked] == [
                (" а", -1, 0), (" б", -2, 1),
            ]
            assert requests[0]["max_tokens"] == 16
            assert "logprobs" not in requests[0]
            assert requests[1]["max_tokens"] == 1
            assert requests[1]["top_logprobs"] == 4
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(400, json={"error": {"code": "unsupported_feature"}}),
    httpx.Response(200, json={"choices": [{"logprobs": None}]}),
])
async def test_missing_logprobs_are_reported(response):
    provider = create_logits_provider("tooken_club", model="test-model")
    provider._client = httpx.AsyncClient(
        base_url=provider.API_BASE, transport=httpx.MockTransport(lambda _: response),
    )
    try:
        with pytest.raises(RuntimeError, match="top_logprobs.*test-model"):
            await provider.next_token_distribution("test prompt")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_missing_key_is_reported_before_request():
    provider = create_logits_provider("tooken_club")
    with patch.object(settings, "tooken_club_api_key", ""):
        with pytest.raises(RuntimeError, match="TOOKEN_CLUB_API_KEY"):
            await provider.complete_text("test prompt")
