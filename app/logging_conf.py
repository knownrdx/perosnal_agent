"""Structured (JSON) logging with secret redaction.

Fields emitted: timestamp, level, service, event, plus any extra kwargs passed
through ``logger.info("event", extra={"task_id": ...})``.

Secrets are never allowed into the log stream: values that look like Telegram
bot tokens / API keys are replaced before the record is formatted.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

SERVICE = "agent"

_REDACT_PATTERNS = [
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),          # telegram bot token
    re.compile(r"\b(sk|pk|xoxb|ghp|gho|hf)_[A-Za-z0-9]{16,}\b"),  # common api keys
    re.compile(r"(?i)(authorization|api[_-]?token|password|secret)\"?\s*[:=]\s*\"?([^\s\",}]+)"),
]

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


def redact(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pattern in _REDACT_PATTERNS:
            out = pattern.sub(lambda m: m.group(0)[: max(4, len(m.group(0)) // 6)] + "***REDACTED***", out)
        return out
    if isinstance(value, dict):
        return {k: ("***REDACTED***" if _is_secret_key(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(
        marker in lowered
        for marker in ("token", "password", "secret", "api_key", "apikey", "cookie", "authorization")
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", SERVICE),
            "logger": record.name,
            "event": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in payload:
                continue
            payload[key] = redact(value)
        if record.exc_info:
            payload["error"] = redact(self.formatException(record.exc_info))[-4000:]
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            return json.dumps({"level": record.levelname, "event": "log_serialisation_failed"})


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third party noise control.
    for noisy in ("httpx", "httpcore", "aiosqlite", "asyncio", "aiogram.event"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
