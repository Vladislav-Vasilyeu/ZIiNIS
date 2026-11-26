import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from methods.bits import frame_payload_bits, prepare_payload
from methods.registry import create_method
from utils.capacity import assess_capacity, payload_size
from utils.cover_generator import generate_cover_for_secret_async, ensure_cover_capacity
from utils.file_handlers import load_text


@pytest.mark.parametrize("password", [None, " пароль "])
@pytest.mark.parametrize("secret", ["Секретное Сообщение", " Hi \n", "🔒 белорусский тэкст"])
@pytest.mark.parametrize("method", ["classic_synonyms", "classic_whitespace"])
def test_insufficient_cloud_cover_is_expanded_and_roundtrips(monkeypatch, secret, password, method):
    provider = Mock()
    provider.complete_text = AsyncMock(return_value="Защита информации. " * 12)
    provider.close = AsyncMock()
    monkeypatch.setattr("providers.factory.create_logits_provider", lambda *a, **k: provider)

    async def run():
        cover, source = await generate_cover_for_secret_async(
            secret, method, provider_name="tooken_club", password=password,
            allow_local_fallback=False, return_report=True, max_rounds=1)
        assert "локальное дополнение" in source
        capacity = assess_capacity(cover, secret, method, password=password)
        assert capacity.sufficient
        payload, _ = prepare_payload(secret, password)
        assert capacity.required == len(frame_payload_bits(payload))
        codec = create_method(method)
        try:
            embedded = await codec.embed(secret=secret, cover_text=cover, password=password, use_marker=False)
            assert embedded.success, embedded.message
            extracted = await codec.extract(stego_text=embedded.text, password=password, use_marker=False)
            assert extracted.success, extracted.message
            assert extracted.secret == secret
        finally:
            await codec.close()
        provider.close.assert_awaited_once()
    asyncio.run(run())


def test_cloud_failure_is_not_silently_presented_as_generation(monkeypatch):
    provider = Mock(complete_text=AsyncMock(side_effect=RuntimeError("Service unavailable")), close=AsyncMock())
    monkeypatch.setattr("providers.factory.create_logits_provider", lambda *a, **k: provider)
    with pytest.raises(RuntimeError, match="Service unavailable"):
        asyncio.run(generate_cover_for_secret_async("Hi", provider_name="groq", allow_local_fallback=False))
    cover, source = asyncio.run(generate_cover_for_secret_async("Hi", return_report=True))
    assert "локальный шаблон" in source
    assert assess_capacity(cover, "Hi", "classic_synonyms").sufficient


def test_utf8_bom_and_limit(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_text("Текст", encoding="utf-8-sig")
    assert load_text(path) == "Текст"
    assert payload_size("A" * 65535) == 65535
    with pytest.raises(ValueError):
        payload_size("Я" * 32768)
    with pytest.raises(ValueError):
        payload_size("A" * 65500, "password")


def test_nbsp_container_can_be_reused_and_txt_preserves_line_endings(tmp_path):
    from utils.file_handlers import save_text
    cover = "слово\u00a0" * 80

    async def run():
        codec = create_method("classic_whitespace")
        embedded = await codec.embed(secret="Hi", cover_text=cover, use_marker=False)
        assert embedded.success
        extracted = await codec.extract(stego_text=embedded.text, use_marker=False)
        assert extracted.success and extracted.secret == "Hi"
    asyncio.run(run())
    path = tmp_path / "exact.txt"
    text = "a\r\nb\nc\r\n\u00a0"
    save_text(path, text)
    assert load_text(path) == text


def test_neural_capacity_bound_does_not_call_server():
    from methods.neural_common import NeuralGenerativeCodec
    provider = Mock(next_token_distribution=AsyncMock())
    codec = NeuralGenerativeCodec(provider, "rank", max_tokens=2)
    result = asyncio.run(codec.embed(secret="Hi", prompt="Text", password=None,
        use_marker=False, progress=None, cancel_event=None, method_name="neural_rank"))
    assert not result.success and "Сократите" in result.message
    provider.next_token_distribution.assert_not_awaited()


def test_marker_reports_zero_synonym_capacity():
    async def run():
        codec = create_method("classic_synonyms")
        result = await codec.embed(secret="Hi", cover_text="Hello world", use_marker=True)
        assert result.success and "0/32" in result.message and "маркер" in result.message
        extracted = await codec.extract(stego_text=result.text, use_marker=True)
        assert extracted.success and extracted.secret == "Hi"
    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 402, 403, 404, 429, 500])
def test_provider_http_failures_are_actionable(status):
    from providers.tooken_club import TookenClubProvider
    message = str(TookenClubProvider()._http_error(status))
    assert f"HTTP {status}" in message
    assert len(message.split(". ")) >= 2


@pytest.mark.parametrize("body", [{}, {"data": None}, {"data": []}, {"data": [None, {}]}])
def test_invalid_model_catalog_is_handled(body):
    import httpx
    from providers.tooken_club import TookenClubProvider
    provider = TookenClubProvider()
    response = httpx.Response(200, json=body, request=httpx.Request("GET", "https://example.test/models"))
    provider._client = Mock(get=AsyncMock(return_value=response))
    with pytest.raises(RuntimeError, match="каталог|список"):
        asyncio.run(provider.list_models())


@pytest.fixture
def app(monkeypatch, tmp_path):
    from ui.app import NeuralTextStegoApp, settings
    monkeypatch.setattr(settings, "ui_state_file", tmp_path / "state.json")
    monkeypatch.setattr("ui.app.messagebox.showerror", Mock())
    monkeypatch.setattr("ui.app.messagebox.showwarning", Mock())
    window = NeuralTextStegoApp()
    window.withdraw()
    yield window
    window._close_app()


def wait_for(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("GUI operation did not finish")


def test_gui_rejects_capacity_before_start_and_preserves_failed_load(app, monkeypatch):
    app.cover_text.insert("1.0", "Метод защиты информации.")
    app.secret_text.insert("1.0", "Секретное Сообщение")
    app.use_marker_var.set(False)
    runner = Mock()
    monkeypatch.setattr(app, "_run_async", runner)
    app._start_embed()
    runner.assert_not_called()
    from ui.app import messagebox
    assert "Сгенерировать" in messagebox.showwarning.call_args.args[1]
    monkeypatch.setattr("ui.app.filedialog.askopenfilename", lambda **kw: "missing.docx")
    app._load_cover()
    assert app._box_text(app.cover_text) == "Метод защиты информации."


def test_gui_roundtrip_keeps_secret_password_and_prompt(app):
    secret, password = " Секрет \n", " пароль "
    app.embed_method_var.set("Классический: синонимы")
    app.use_marker_var.set(False)
    app.secret_text.insert("1.0", secret)
    app.embed_password.insert(0, password)
    cover = ensure_cover_capacity("Начало.\n", secret, "classic_synonyms", password=password)
    app.cover_text.insert("1.0", cover)
    app._start_embed()
    assert app._busy
    wait_for(app, lambda: not app._busy)
    assert app._box_text(app.stego_result)
    app.cover_text.delete("1.0", "end")
    app.cover_text.insert("1.0", "Changed after embedding")
    app._send_stego_to_extract()
    assert app.extract_prompt.get() == cover
    app.extract_password.insert(0, password)
    app._start_extract()
    wait_for(app, lambda: not app._busy)
    assert app._box_text(app.extract_result) == secret


def test_gui_cancel_network_job_and_recover(app):
    cleaned = []
    started = []
    results = []

    async def slow():
        try:
            started.append(True)
            await asyncio.sleep(60)
        finally:
            cleaned.append(True)

    app._run_async(slow(), lambda result, error: results.append(error))
    assert app._busy
    wait_for(app, lambda: bool(started))
    app._start_extract()  # Cannot start another job or replace the cancellation event.
    app._cancel_operation()
    wait_for(app, lambda: not app._busy)
    assert cleaned and "отменена" in str(results[0])
    assert app.btn_embed.cget("state") == "normal"


def test_experiments_report_extraction_failure(app, monkeypatch, tmp_path):
    import csv
    from methods.base import MethodResult
    from ui.app import settings
    monkeypatch.setattr(settings, "project_root", tmp_path)
    codec = Mock(
        embed=AsyncMock(return_value=MethodResult(success=True, text="stego", message="embedded")),
        extract=AsyncMock(return_value=MethodResult(success=False, message="extraction failed")),
        close=AsyncMock())
    monkeypatch.setattr("ui.app.create_method", lambda *a, **k: codec)
    app.exp_runs.delete(0, "end")
    app.exp_runs.insert(0, "1")
    app._start_experiments()
    wait_for(app, lambda: not app._busy)
    path = next((tmp_path / "experiments/results").glob("*.csv"))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    assert rows[1][1] == "False" and rows[1][-1] == "extraction failed"


@pytest.mark.parametrize("value", ["x", "0", "-1", "101"])
def test_invalid_experiment_count_does_not_launch(app, monkeypatch, value):
    runner = Mock()
    monkeypatch.setattr(app, "_run_async", runner)
    app.exp_runs.delete(0, "end")
    app.exp_runs.insert(0, value)
    app._start_experiments()
    runner.assert_not_called()
