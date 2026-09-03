"""Voice-note transcription (speech -> text).

The owner writes in Banglish and talks faster than he types, so a Telegram
voice note is the cheapest way to hand the agent a task.  Telegram delivers
those notes as Opus ``.ogg``/``.oga`` files; this module turns such a file
into plain text that the rest of the agent can treat as an ordinary request.

Three backends are tried in order, all optional:

1. ``whisper-api``  - any OpenAI-compatible ``/audio/transcriptions`` endpoint.
   Preferred: no model download, no CPU burn on a small VPS.
2. ``faster-whisper`` - local CTranslate2 model, used when the package is
   installed.  Imported lazily so the container stays slim when it is not.
3. Nothing configured -> :class:`TranscriptionError` that tells the owner the
   exact command to run.  A missing optional dependency must never surface as
   an ImportError halfway through handling a message.

Secrets discipline (master prompt section 15): the API key is read from the
encrypted vault first, falls back to .env, and is never written to a log.  The
transcript itself is user content and is also kept out of the logs - only the
backend, the duration and the character count are recorded.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import UnsafePath, get_vault, rel_path, safe_path

log = get_logger(__name__)

# The OpenAI audio endpoint rejects anything above 25 MB; fail before the
# upload so the owner gets a useful message instead of a slow 413.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

DEFAULT_LOCAL_MODEL = "base"
DEFAULT_API_MODEL = "whisper-1"
LOCAL_PACKAGE = "faster_whisper"
INSTALL_HINT = (
    "no transcription backend is configured. Either set an API key with "
    "'/setkey whisper <key>' (or WHISPER_API_KEY in .env), or install the local "
    "model with 'pip install faster-whisper' and set WHISPER_LOCAL_MODEL=base."
)


class TranscriptionError(Exception):
    """Transcription failed. ``temporary`` drives the retry decision."""

    def __init__(
        self, message: str, *, temporary: bool = False, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.temporary = temporary
        self.status = status


@dataclass(slots=True)
class Transcript:
    """Result of one transcription. ``text`` is user content, never logged."""

    text: str
    language: str = ""
    duration_s: float = 0.0
    backend: str = ""


def _faster_whisper_available() -> bool:
    """True when the optional local backend can be imported.

    Kept as a function (not a module-level constant) so tests can flip it and
    so an install performed after startup is picked up without a restart.
    """
    try:
        return importlib.util.find_spec(LOCAL_PACKAGE) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


class Transcriber:
    """Turns an audio file inside the workspace into text."""

    def __init__(
        self,
        *,
        local_model: str = "",
        api_model: str = "",
        timeout_s: float = 120.0,
        retry_backoff_s: float = 1.0,
    ) -> None:
        settings = get_settings()
        self.local_model = local_model or settings.whisper_local_model or DEFAULT_LOCAL_MODEL
        self.api_model = api_model or settings.whisper_model or DEFAULT_API_MODEL
        self.timeout_s = timeout_s
        self.retry_backoff_s = retry_backoff_s
        self._client: httpx.AsyncClient | None = None
        self._vault_loaded = False
        self._local_engine: Any | None = None

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    async def _ensure_vault(self) -> None:
        """Populate the vault cache once; ``get``/``has`` are sync afterwards."""
        if self._vault_loaded:
            return
        try:
            await get_vault().load()
        except Exception as exc:  # noqa: BLE001 - vault problems must not block voice
            log.warning("transcriber_vault_load_failed", extra={"error": str(exc)[:200]})
        self._vault_loaded = True

    def _api_config(self) -> tuple[str, str]:
        """(api_key, base_url) with the vault taking precedence over .env."""
        settings = get_settings()
        vault = get_vault()
        api_key = vault.get("whisper_api_key", settings.whisper_api_key).strip()
        base_url = vault.get("whisper_base_url", settings.whisper_base_url).strip()
        return api_key, base_url.rstrip("/")

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Uploads are slow; give the connect phase its own short budget.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=10.0)
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    def _validate(self, path: str | Path) -> Path:
        try:
            resolved = safe_path(path, must_exist=True)
        except UnsafePath as exc:
            raise TranscriptionError(f"audio file rejected: {exc}") from exc

        if not resolved.is_file():
            raise TranscriptionError(f"audio path is not a file: {rel_path(resolved)}")

        size = resolved.stat().st_size
        if size == 0:
            raise TranscriptionError(f"audio file is empty: {rel_path(resolved)}")
        if size > MAX_AUDIO_BYTES:
            limit_mb = MAX_AUDIO_BYTES // (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise TranscriptionError(
                f"audio file is too large: {actual_mb:.1f} MB, the transcription "
                f"limit is {limit_mb} MB. Split the recording and try again."
            )
        return resolved

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def transcribe(self, path: str | Path, *, language: str = "") -> Transcript:
        """Transcribe ``path`` (workspace relative) and return the text."""
        audio = self._validate(path)
        await self._ensure_vault()

        api_key, base_url = self._api_config()
        started = time.perf_counter()

        if api_key and base_url:
            transcript = await self._transcribe_api(audio, api_key, base_url, language)
        elif _faster_whisper_available():
            transcript = await self._transcribe_local(audio, language)
        else:
            raise TranscriptionError(INSTALL_HINT)

        transcript.duration_s = round(time.perf_counter() - started, 3)
        # Deliberately no text and no key in the log record.
        log.info(
            "voice_transcribed",
            extra={
                "backend": transcript.backend,
                "duration": transcript.duration_s,
                "chars": len(transcript.text),
                "language": transcript.language,
                "path": rel_path(audio),
            },
        )
        return transcript

    async def health(self) -> dict[str, Any]:
        """Never raises: reports which backend a voice note would use."""
        try:
            await self._ensure_vault()
            api_key, base_url = self._api_config()
            if api_key and base_url:
                return {"ok": True, "backend": "whisper-api", "reason": f"api at {base_url}"}
            if _faster_whisper_available():
                return {
                    "ok": True,
                    "backend": "faster-whisper",
                    "reason": f"local model {self.local_model}",
                }
            return {"ok": False, "backend": "", "reason": INSTALL_HINT}
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "backend": "", "reason": str(exc)[:200]}

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._local_engine = None

    # ------------------------------------------------------------------ #
    # Backend 1: OpenAI-compatible HTTP API
    # ------------------------------------------------------------------ #
    async def _transcribe_api(
        self, audio: Path, api_key: str, base_url: str, language: str
    ) -> Transcript:
        url = f"{base_url}/audio/transcriptions"
        payload = audio.read_bytes()
        data: dict[str, str] = {"model": self.api_model}
        if language:
            data["language"] = language

        last_error: TranscriptionError | None = None
        for attempt in range(1, 4):
            files = {"file": (audio.name, payload, "application/octet-stream")}
            try:
                response = await self._http().post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files=files,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = TranscriptionError(
                    f"transcription service unreachable ({type(exc).__name__}); this is "
                    "temporary, retrying later should work",
                    temporary=True,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"transcription request failed: {exc}") from exc
            else:
                error = self._classify(response)
                if error is None:
                    return self._parse_api_response(response)
                if not error.temporary:
                    raise error
                last_error = error

            if attempt < 3:
                await asyncio.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))

        raise last_error or TranscriptionError("transcription failed", temporary=True)

    def _classify(self, response: httpx.Response) -> TranscriptionError | None:
        """Map a response onto the failure taxonomy. None means success."""
        status = response.status_code
        if status < 400:
            return None
        detail = _error_detail(response)
        if status in {401, 403}:
            return TranscriptionError(
                f"authentication failed ({status}) for the transcription API. "
                "Check the whisper API key with '/setkey whisper <key>'.",
                status=status,
            )
        if status == 429:
            return TranscriptionError(
                f"rate limited by the transcription API ({status}): {detail}. "
                "Wait a moment and send the voice note again.",
                temporary=True,
                status=status,
            )
        if status >= 500:
            return TranscriptionError(
                f"transcription service error {status}: temporary failure, retryable",
                temporary=True,
                status=status,
            )
        if status == 413:
            return TranscriptionError(
                f"the transcription API rejected the file as too large ({status})",
                status=status,
            )
        return TranscriptionError(f"transcription failed ({status}): {detail}", status=status)

    def _parse_api_response(self, response: httpx.Response) -> Transcript:
        try:
            data = response.json()
        except ValueError as exc:
            raise TranscriptionError("transcription API returned a non-JSON body") from exc

        if isinstance(data, str):
            data = {"text": data}
        if not isinstance(data, dict):
            raise TranscriptionError("transcription API returned an unexpected body")

        text = str(data.get("text") or "").strip()
        if not text:
            raise TranscriptionError(
                "the voice note produced no text; it may be silent or too short"
            )
        return Transcript(
            text=text,
            language=str(data.get("language") or ""),
            backend="whisper-api",
        )

    # ------------------------------------------------------------------ #
    # Backend 2: local faster-whisper
    # ------------------------------------------------------------------ #
    def _load_local_engine(self) -> Any:
        if self._local_engine is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dependency

            # int8 on CPU is the only setup that is fast enough on a small VPS.
            self._local_engine = WhisperModel(
                self.local_model, device="cpu", compute_type="int8"
            )
        return self._local_engine

    def _run_local(self, audio: Path, language: str) -> Transcript:
        engine = self._load_local_engine()
        segments, info = engine.transcribe(
            str(audio), language=language or None, vad_filter=True
        )
        text = " ".join(str(segment.text).strip() for segment in segments).strip()
        if not text:
            raise TranscriptionError(
                "the voice note produced no text; it may be silent or too short"
            )
        return Transcript(
            text=text,
            language=str(getattr(info, "language", "") or language or ""),
            backend="faster-whisper",
        )

    async def _transcribe_local(self, audio: Path, language: str) -> Transcript:
        try:
            # Model inference is blocking C++ work: keep the event loop free.
            return await asyncio.to_thread(self._run_local, audio, language)
        except TranscriptionError:
            raise
        except ImportError as exc:  # pragma: no cover - guarded by the spec check
            raise TranscriptionError(INSTALL_HINT) from exc
        except Exception as exc:  # noqa: BLE001 - model errors are opaque
            raise TranscriptionError(
                f"local transcription failed ({type(exc).__name__}): {str(exc)[:200]}"
            ) from exc


def _error_detail(response: httpx.Response) -> str:
    """Short human readable reason from an error body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200].strip() or "no detail"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:200]
        if error:
            return str(error)[:200]
        if payload.get("message"):
            return str(payload["message"])[:200]
    return str(payload)[:200]


_transcriber: Transcriber | None = None


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def set_transcriber(transcriber: Transcriber | None) -> None:
    """Injection point for tests and for the Telegram handler."""
    global _transcriber
    _transcriber = transcriber


def reset_transcriber() -> None:
    global _transcriber
    _transcriber = None


async def close_transcriber() -> None:
    global _transcriber
    if _transcriber is not None:
        try:
            await _transcriber.close()
        except Exception:  # noqa: BLE001
            pass
    _transcriber = None
