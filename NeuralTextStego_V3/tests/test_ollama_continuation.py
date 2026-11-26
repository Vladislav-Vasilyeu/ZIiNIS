from unittest.mock import AsyncMock

import httpx
import pytest

from methods.base import MethodResult
from methods.registry import create_method
from providers.ollama_logits import OllamaLogitsProvider
from providers.base import TokenCandidate


@pytest.mark.asyncio
async def test_distribution_keeps_prefix_in_raw_continuation():
    provider = OllamaLogitsProvider()
    provider.raw_continuation = True
    provider._client = AsyncMock()
    provider._client.post.return_value = httpx.Response(200, json={"logprobs": [{
        "top_logprobs": [
            {"token": " a", "logprob": -1}, {"token": " b", "logprob": -2},
        ],
    }]})
    result = await provider.next_token_distribution("Prompt plus generated prefix")
    body = provider._client.post.call_args.kwargs["json"]
    assert body["raw"] is True
    assert body["prompt"] == "Prompt plus generated prefix"
    assert len(result) == 2
    await provider.close()


@pytest.mark.asyncio
async def test_raw_distribution_requires_real_logprobs():
    provider = OllamaLogitsProvider()
    provider.raw_continuation = True
    provider._client = AsyncMock()
    provider._client.post.return_value = httpx.Response(200, json={"response": "text"})
    with pytest.raises(RuntimeError, match="top_logprobs"):
        await provider.next_token_distribution("Prompt")
    assert provider._client.post.await_count == 1
    await provider.close()


@pytest.mark.asyncio
async def test_empty_eos_probes_reproducible_seeds_with_native_scores():
    provider = OllamaLogitsProvider()
    provider.raw_continuation = True
    provider._client = AsyncMock()
    seeds = []

    async def respond(path, *, json):
        seeds.append(json["options"]["seed"])
        if len(seeds) == 1:
            return httpx.Response(200, json={"done_reason": "stop", "response": ""})
        return httpx.Response(200, json={"logprobs": [{"top_logprobs": [
            {"token": " a", "logprob": -1}, {"token": " b", "logprob": -2}]}]})

    provider._client.post.side_effect = respond
    result = await provider.next_token_distribution("Prompt")
    assert seeds == [0, 1]
    assert [c.logprob for c in result] == [-1, -2]
    await provider.close()


@pytest.mark.asyncio
async def test_permanent_eos_is_bounded_and_actionable():
    provider = OllamaLogitsProvider()
    provider.raw_continuation = True
    provider._client = AsyncMock()
    provider._client.post.return_value = httpx.Response(200, json={"done_reason": "stop", "response": ""})
    with pytest.raises(RuntimeError, match="другой исходный контекст"):
        await provider.next_token_distribution("Prompt")
    assert provider._client.post.await_count == 8
    await provider.close()


@pytest.mark.asyncio
async def test_legacy_extraction_restores_provider_mode():
    method = create_method("neural_rank", logits_provider="ollama")
    method._provider.raw_continuation = True
    modes = []

    async def decode(**kwargs):
        modes.append(method._provider.raw_continuation)
        return MethodResult(
            success=not method._provider.raw_continuation,
            secret="Hi" if not method._provider.raw_continuation else "",
        )

    method._codec.extract = AsyncMock(side_effect=decode)
    result = await method.extract(stego_text="legacy text", use_marker=False)
    assert result.success and result.secret == "Hi"
    assert result.details["legacy_chat_context"] is True
    assert modes == [True, False]
    assert method._provider.raw_continuation is True
    await method.close()


@pytest.mark.asyncio
async def test_overlapping_tokens_use_prefix_free_codebook():
    method = create_method("neural_rank", logits_provider="ollama")
    method._provider.raw_continuation = True
    method._provider.next_token_distribution = AsyncMock(return_value=[
        TokenCandidate("a", -1, 0), TokenCandidate("ab", -2, 1),
        TokenCandidate("b", -3, 2), TokenCandidate("ba", -4, 3),
    ])
    embedded = await method.embed(secret="Hi", prompt="Prompt", use_marker=False)
    extracted = await method.extract(
        stego_text=embedded.text, prompt="Prompt", use_marker=False,
    )
    assert embedded.success and extracted.success and extracted.secret == "Hi"
    assert embedded.positions_used == 32
    await method.close()
