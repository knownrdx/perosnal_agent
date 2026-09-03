"""Voice-note transcription: API backend, local backend, and the failure modes.

Fully offline: httpx is mocked with ``MockTransport``, the vault is driven
through its real (SQLite backed) API, and faster-whisper is faked with a stub
module so the tests never download a model.
"""

from __future__ import annotations

import logging
import sys
import types

import httpx
import pytest

from app.integrations import transcription as tx
from app.integrations.transcription import (
    Transcriber,
    TranscriptionError,
    Transcript,
    get_transcriber,
    reset_transcriber,
)
from app.security import safe_path
from app.security.vault import get_vault, reset_vault

API_KEY = "sk-whisper-supersecret-do-not-log-0123456789"
BASE_URL = "https://whisper.test/v1"
AUDIO_BYTES = b"OggS\x00\x02fake-opus-voice-note-payload"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_singletons(environment):
    """The vault and transcriber are process singletons; isolate each test."""
    reset_vault()
    reset_transcriber()
    yield
    reset_vault()
    reset_transcriber()


@pytest.fixture
def voice_file(environment):
    path = safe_path("temp/voice.ogg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(AUDIO_BYTES)
    return path


@pytest.fixture
def no_local_backend(monkeypatch):
    """faster-whisper must look absent regardless of the dev machine."""
    monkeypatch.setattr(tx, "_faster_whisper_available", lambda: False)


async def configure_api(key: str = API_KEY, base_url: str = BASE_URL) -> None:
    vault = get_vault()
    await vault.set("whisper_api_key", key)
    await vault.set("whisper_base_url", base_url)


def make_transcriber(handler) -> Transcriber:
    """Transcriber wired to a MockTransport, with retries that do not sleep."""
    transcriber = Transcriber(retry_backoff_s=0.0)
    transcriber._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return transcriber


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_api_transcription_returns_text(voice_file, no_local_backend, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        return httpx.Response(200, json={"text": "kalke bank e jete hobe", "language": "bn"})

    transcriber = make_transcriber(handler)
    result = await transcriber.transcribe(voice_file)

    assert isinstance(result, Transcript)
    assert result.text == "kalke bank e jete hobe"
    assert result.backend == "whisper-api"
    assert result.language == "bn"
    assert result.duration_s >= 0.0
    await transcriber.close()


async def test_multipart_carries_file_bytes_and_model(
    voice_file, no_local_backend, environment
):
    await configure_api()
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["content_type"] = request.headers["content-type"].encode()
        return httpx.Response(200, json={"text": "ok"})

    transcriber = make_transcriber(handler)
    await transcriber.transcribe(voice_file, language="bn")

    body = seen["body"]
    assert b"multipart/form-data" in seen["content_type"]
    assert AUDIO_BYTES in body                     # the actual audio was uploaded
    assert b'name="file"' in body
    assert b"voice.ogg" in body
    assert b'name="model"' in body
    assert b"whisper-1" in body
    assert b'name="language"' in body
    assert b"bn" in body
    await transcriber.close()


async def test_relative_workspace_path_is_accepted(voice_file, no_local_backend, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "hello"})

    transcriber = make_transcriber(handler)
    result = await transcriber.transcribe("temp/voice.ogg")
    assert result.text == "hello"
    await transcriber.close()


# --------------------------------------------------------------------------- #
# HTTP failure taxonomy
# --------------------------------------------------------------------------- #
async def test_401_raises_authentication_error(voice_file, no_local_backend, environment):
    await configure_api()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError, match="authentication failed"):
        await transcriber.transcribe(voice_file)

    assert len(calls) == 1  # auth errors are permanent, no retry storm
    await transcriber.close()


async def test_403_is_also_authentication(voice_file, no_local_backend, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError, match="authentication failed"):
        await transcriber.transcribe(voice_file)
    await transcriber.close()


async def test_429_mentions_rate_limit(voice_file, no_local_backend, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError, match="rate limited"):
        await transcriber.transcribe(voice_file)
    await transcriber.close()


async def test_500_is_retried_three_times_then_fails(
    voice_file, no_local_backend, environment
):
    await configure_api()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, text="upstream exploded")

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError) as excinfo:
        await transcriber.transcribe(voice_file)

    assert len(calls) == 3
    assert "temporary" in str(excinfo.value).lower()
    assert excinfo.value.temporary is True
    await transcriber.close()


async def test_transient_500_then_success(voice_file, no_local_backend, environment):
    await configure_api()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="restarting")
        return httpx.Response(200, json={"text": "recovered"})

    transcriber = make_transcriber(handler)
    result = await transcriber.transcribe(voice_file)

    assert result.text == "recovered"
    assert len(calls) == 2
    await transcriber.close()


async def test_timeout_is_temporary_and_retried(voice_file, no_local_backend, environment):
    await configure_api()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("timed out", request=request)

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError) as excinfo:
        await transcriber.transcribe(voice_file)

    assert len(calls) == 3
    assert excinfo.value.temporary is True
    assert "temporary" in str(excinfo.value).lower()
    await transcriber.close()


async def test_empty_transcript_is_an_error(voice_file, no_local_backend, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "   "})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError, match="no text"):
        await transcriber.transcribe(voice_file)
    await transcriber.close()


# --------------------------------------------------------------------------- #
# Configuration / path safety
# --------------------------------------------------------------------------- #
async def test_no_backend_configured_explains_how_to_enable(
    voice_file, no_local_backend, environment
):
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError) as excinfo:
        await transcriber.transcribe(voice_file)

    message = str(excinfo.value)
    assert "pip install faster-whisper" in message
    assert "/setkey whisper" in message
    await transcriber.close()


async def test_path_traversal_is_rejected(no_local_backend, environment):
    await configure_api()
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError, match="rejected"):
        await transcriber.transcribe("../../etc/passwd")
    await transcriber.close()


async def test_missing_file_is_rejected(no_local_backend, environment):
    await configure_api()
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError, match="does not exist"):
        await transcriber.transcribe("temp/not-recorded.ogg")
    await transcriber.close()


async def test_empty_file_is_rejected(no_local_backend, environment):
    path = safe_path("temp/silent.ogg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    transcriber = Transcriber()
    with pytest.raises(TranscriptionError, match="empty"):
        await transcriber.transcribe(path)
    await transcriber.close()


async def test_oversized_file_rejected_before_any_http_call(
    voice_file, no_local_backend, monkeypatch, environment
):
    await configure_api()
    # Shrink the limit instead of writing 25 MB to disk on every test run.
    monkeypatch.setattr(tx, "MAX_AUDIO_BYTES", 8)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"text": "should never happen"})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError, match="too large"):
        await transcriber.transcribe(voice_file)

    assert calls == []  # nothing was uploaded
    await transcriber.close()


async def test_directory_path_is_rejected(no_local_backend, environment):
    safe_path("temp").mkdir(parents=True, exist_ok=True)
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError, match="not a file"):
        await transcriber.transcribe("temp")
    await transcriber.close()


# --------------------------------------------------------------------------- #
# Local faster-whisper backend
# --------------------------------------------------------------------------- #
class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    language = "bn"


class _FakeWhisperModel:
    created: list[tuple] = []

    def __init__(
        self, model_size: str, device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        _FakeWhisperModel.created.append((model_size, device, compute_type))

    def transcribe(self, path: str, language=None, vad_filter: bool = True):
        return [_FakeSegment(" ami"), _FakeSegment("kal asbo ")], _FakeInfo()


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    _FakeWhisperModel.created = []
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(tx, "_faster_whisper_available", lambda: True)
    return module


async def test_local_backend_used_when_no_api_key(
    voice_file, fake_faster_whisper, environment
):
    transcriber = Transcriber(local_model="small")
    result = await transcriber.transcribe(voice_file)

    assert result.backend == "faster-whisper"
    assert result.text == "ami kal asbo"
    assert result.language == "bn"
    assert _FakeWhisperModel.created == [("small", "cpu", "int8")]
    await transcriber.close()


async def test_api_backend_wins_over_local(voice_file, fake_faster_whisper, environment):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "from the api"})

    transcriber = make_transcriber(handler)
    result = await transcriber.transcribe(voice_file)

    assert result.backend == "whisper-api"
    assert _FakeWhisperModel.created == []  # the local model was never loaded
    await transcriber.close()


async def test_local_backend_failure_becomes_transcription_error(
    voice_file, fake_faster_whisper, monkeypatch, environment
):
    def boom(self, path, language=None, vad_filter=True):
        raise RuntimeError("ctranslate2 blew up")

    monkeypatch.setattr(_FakeWhisperModel, "transcribe", boom)
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError, match="local transcription failed"):
        await transcriber.transcribe(voice_file)
    await transcriber.close()


# --------------------------------------------------------------------------- #
# Health + singleton
# --------------------------------------------------------------------------- #
async def test_health_reports_reason_when_unconfigured(no_local_backend, environment):
    health = await Transcriber().health()

    assert health["ok"] is False
    assert health["backend"] == ""
    assert "faster-whisper" in health["reason"]


async def test_health_ok_with_api_key(no_local_backend, environment):
    await configure_api()
    health = await Transcriber().health()

    assert health["ok"] is True
    assert health["backend"] == "whisper-api"
    assert API_KEY not in health["reason"]


async def test_health_ok_with_local_model(fake_faster_whisper, environment):
    health = await Transcriber(local_model="base").health()

    assert health["ok"] is True
    assert health["backend"] == "faster-whisper"
    assert "base" in health["reason"]


async def test_get_transcriber_is_a_singleton(environment):
    first = get_transcriber()
    assert get_transcriber() is first
    reset_transcriber()
    assert get_transcriber() is not first


async def test_close_is_idempotent(environment):
    transcriber = Transcriber()
    await transcriber.close()
    await transcriber.close()


# --------------------------------------------------------------------------- #
# Secrets discipline
# --------------------------------------------------------------------------- #
async def test_api_key_and_transcript_never_reach_the_logs(
    voice_file, no_local_backend, caplog, environment
):
    await configure_api()
    secret_text = "gopon kotha bank password"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": secret_text, "language": "bn"})

    transcriber = make_transcriber(handler)
    with caplog.at_level(logging.DEBUG):
        result = await transcriber.transcribe(voice_file)

    assert result.text == secret_text
    assert API_KEY not in caplog.text
    assert secret_text not in caplog.text

    for record in caplog.records:
        blob = " ".join(str(value) for value in record.__dict__.values())
        assert API_KEY not in blob
        assert secret_text not in blob

    # The useful metadata is logged instead.
    events = [r for r in caplog.records if r.getMessage() == "voice_transcribed"]
    assert events and getattr(events[0], "chars") == len(secret_text)
    assert getattr(events[0], "backend") == "whisper-api"
    await transcriber.close()


async def test_error_messages_do_not_leak_the_key(
    voice_file, no_local_backend, environment
):
    await configure_api()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    transcriber = make_transcriber(handler)
    with pytest.raises(TranscriptionError) as excinfo:
        await transcriber.transcribe(voice_file)

    assert API_KEY not in str(excinfo.value)
    await transcriber.close()
