import asyncio
import math
from unittest.mock import AsyncMock, Mock

import pytest

from utils.capacity import plan_cover, assess_capacity
from utils.cover_generator import generate_cover_for_secret_async, ensure_cover_capacity, build_llm_cover_prompt


def test_plan_matches_frame_and_crypto_sizes():
    for method in ("classic_whitespace", "classic_synonyms"):
        plain = plan_cover("Секретное Сообщение", method)
        encrypted = plan_cover("Секретное Сообщение", method, password="password")
        assert plain.payload_bytes == 37 and plain.required_bits == 312
        assert encrypted.payload_bytes == 108 and encrypted.required_bits == 880
        assert encrypted.target_words > plain.target_words
    assert plan_cover("Hi", "classic_whitespace", use_marker=True).required_bits == 16
    assert plan_cover("Hi", "classic_synonyms", use_marker=True).required_bits == 32


def test_neural_generation_plan_prepares_short_context_not_a_large_container():
    small = plan_cover("Hi", "neural_rank")
    large = plan_cover("A" * 100, "neural_rank")
    assert small.target_words == large.target_words == 80
    assert large.minimum_positions > small.minimum_positions


def test_generation_requests_additional_model_text_and_checks_capacity(monkeypatch):
    secret = "Секретное Сообщение"
    enough = ensure_cover_capacity("", secret, "classic_synonyms")
    provider = Mock(complete_text=AsyncMock(side_effect=["Простой рассказ без подходящих замен. " * 3, enough]),
                    close=AsyncMock())
    monkeypatch.setattr("providers.factory.create_logits_provider", lambda *a, **k: provider)
    text, source = asyncio.run(generate_cover_for_secret_async(secret, return_report=True, allow_local_fallback=False))
    assert assess_capacity(text, secret, "classic_synonyms").sufficient
    assert source == "нейросеть: 2 запрос(а)"
    assert provider.complete_text.await_count == 2
    assert "не хватает" in provider.complete_text.call_args_list[1].args[0]
    provider.close.assert_awaited_once()


def test_secret_and_password_are_not_sent_to_cover_model(monkeypatch):
    secret, password = "PRIVATE-7f93020d-SECRET", "PASSWORD-a30c7c2f"
    provider = Mock(complete_text=AsyncMock(return_value="Нейтральный рассказ. " * 5), close=AsyncMock())
    monkeypatch.setattr("providers.factory.create_logits_provider", lambda *a, **k: provider)
    text, source = asyncio.run(generate_cover_for_secret_async(secret, password=password, return_report=True))
    assert assess_capacity(text, secret, "classic_synonyms", password=password).sufficient
    assert "локальное дополнение" in source
    assert provider.complete_text.await_count == 3
    for call in provider.complete_text.call_args_list:
        assert secret not in call.args[0] and password not in call.args[0]
    assert str(plan_cover(secret, "classic_synonyms", password=password).target_words) in build_llm_cover_prompt(secret, "classic_synonyms", password=password)


def test_generation_cancellation_closes_provider(monkeypatch):
    provider = Mock(complete_text=AsyncMock(side_effect=asyncio.CancelledError()), close=AsyncMock())
    monkeypatch.setattr("providers.factory.create_logits_provider", lambda *a, **k: provider)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generate_cover_for_secret_async("Hi"))
    provider.close.assert_awaited_once()


@pytest.mark.parametrize("strategy", ["rank", "arithmetic", "sampling"])
def test_single_token_step_carries_no_bits_but_does_not_end_coding(strategy):
    from methods.neural_common import NeuralGenerativeCodec
    from providers.base import TokenCandidate

    async def distribution(context, **kwargs):
        tokens = [" ", " word"] if context == "Prompt" else ["a", "b", "c", "d"]
        return [TokenCandidate(token=t, logprob=-float(i + 1), rank=i) for i, t in enumerate(tokens)]

    async def run():
        provider = Mock(raw_continuation=True, next_token_distribution=distribution)
        codec = NeuralGenerativeCodec(provider, strategy)
        embedded = await codec.embed(secret="Hi", prompt="Prompt", password=None,
            use_marker=False, progress=None, cancel_event=None, method_name=strategy)
        assert embedded.success and embedded.positions_used == 17
        assert embedded.text.startswith(" ")
        restored = await codec.extract(stego_text=embedded.text, prompt="Prompt", password=None,
            expected_bits=None, use_marker=False, progress=None, cancel_event=None, method_name=strategy)
        assert restored.success and restored.secret == "Hi"
    asyncio.run(run())


def test_teaching_example_matches_all_three_strategies():
    from methods.neural_common import NeuralGenerativeCodec
    from providers.base import TokenCandidate
    tokens = [" тепло", " холодно", " тихо", " темно", " светло", " ветрено", " пасмурно", " солнечно"]
    probabilities = [.24, .20, .16, .14, .10, .07, .05, .04]
    candidates = [TokenCandidate(token=t, logprob=math.log(p), rank=i)
                  for i, (t, p) in enumerate(zip(tokens, probabilities))]
    for strategy, expected in (("rank", " темно"), ("sampling", " пасмурно"), ("arithmetic", " светло")):
        codec = NeuralGenerativeCodec(Mock(), strategy, temperature=1)
        token, consumed = codec._pick_token(candidates, "11", 0)
        assert token == expected and consumed == 2
        assert codec._token_to_bits(candidates, token) == "11"


def test_neural_embed_cannot_report_success_when_self_verification_fails():
    from methods.neural_methods import NeuralRankMethod
    from methods.base import MethodResult
    method = NeuralRankMethod(Mock())
    method._codec.embed = AsyncMock(return_value=MethodResult(success=True, text="stego"))
    method._codec.extract = AsyncMock(return_value=MethodResult(success=False, message="distribution changed"))
    result = asyncio.run(method.embed(secret="Hi", prompt="Prompt", use_marker=False))
    assert not result.success and not result.details["roundtrip_verified"]
    assert result.details["verification_error"] == "distribution changed"


def test_neural_self_verification_requires_exact_secret():
    from methods.neural_methods import NeuralRankMethod
    from methods.base import MethodResult
    method = NeuralRankMethod(Mock())
    method._codec.embed = AsyncMock(return_value=MethodResult(success=True, text="stego"))
    method._codec.extract = AsyncMock(return_value=MethodResult(success=True, secret="Other"))
    assert not asyncio.run(method.embed(secret="Hi", use_marker=False)).success
