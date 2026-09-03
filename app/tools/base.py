"""Tool framework.

Every tool declares: name, description, argument schema, permission level,
timeout, retry policy and whether it produces an externally visible side effect
(which makes it idempotency-guarded).

The framework - not the tool - handles validation, permission checks, timeouts,
retries with exponential backoff and failure classification.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.db.models import FailureKind
from app.logging_conf import get_logger
from app.security import Permission

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """Base tool failure with an explicit failure classification."""

    kind: FailureKind = FailureKind.UNKNOWN

    def __init__(self, message: str, kind: FailureKind | None = None) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind


class TemporaryToolError(ToolError):
    kind = FailureKind.TEMPORARY


class PermanentToolError(ToolError):
    kind = FailureKind.PERMANENT


class InvalidInput(ToolError):
    kind = FailureKind.INVALID_INPUT


class AuthError(ToolError):
    kind = FailureKind.AUTH


class RateLimited(TemporaryToolError):
    kind = FailureKind.RATE_LIMIT


class UserActionRequired(ToolError):
    kind = FailureKind.USER_ACTION_REQUIRED


# --------------------------------------------------------------------------- #
# Schema + context
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Arg:
    type: str = "string"          # string | integer | number | boolean | array | object
    required: bool = True
    description: str = ""
    default: Any = None
    choices: list[Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }
        if self.choices:
            out["choices"] = self.choices
        if self.default is not None:
            out["default"] = self.default
        return out


@dataclass(slots=True)
class ToolContext:
    """Everything a tool may need about the caller, without global state."""

    task_id: str | None = None
    chat_id: int | None = None
    user_id: int | None = None
    step: int = 0
    permission: Permission = Permission.WRITE
    extra: dict[str, Any] = field(default_factory=dict)

    def operation_key(self, tool: str, suffix: str) -> str:
        return f"{self.task_id or 'adhoc'}:{tool}:{suffix}"[:160]


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    kind: FailureKind | None = None
    attempts: int = 1
    duration_ms: float = 0.0

    def observation(self, limit: int = 2000) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "result": _truncate(self.data, limit)}
        return {"ok": False, "error": self.error[:limit], "failure_kind": (self.kind or FailureKind.UNKNOWN).value}


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [truncated {len(value)} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        head = [_truncate(v, limit) for v in value[:50]]
        if len(value) > 50:
            head.append(f"... [{len(value) - 50} more items]")
        return head
    return value


Handler = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args: dict[str, Arg]
    permission: Permission
    handler: Handler
    timeout_s: int = 120
    max_retries: int = 2
    retry_backoff_s: float = 2.0
    side_effect: bool = False       # externally visible -> idempotency guarded
    verify: Callable[[dict[str, Any]], bool] | None = None

    # -------------------------------------------------------------- #
    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission.value,
            "args": {name: arg.as_dict() for name, arg in self.args.items()},
            "timeout_s": self.timeout_s,
            "side_effect": self.side_effect,
        }

    def validate(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        raw = dict(raw or {})
        cleaned: dict[str, Any] = {}
        unknown = set(raw) - set(self.args)
        if unknown:
            raise InvalidInput(
                f"unknown argument(s) for {self.name}: {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(self.args)}"
            )
        for name, spec in self.args.items():
            if name in raw and raw[name] is not None:
                cleaned[name] = _coerce(self.name, name, raw[name], spec)
            elif spec.required:
                raise InvalidInput(f"{self.name}: missing required argument '{name}'")
            elif spec.default is not None:
                cleaned[name] = spec.default
        return cleaned

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate, execute with timeout, retry on temporary failures."""
        started = time.perf_counter()
        try:
            cleaned = self.validate(args)
        except ToolError as exc:
            return ToolResult(
                ok=False,
                error=str(exc),
                kind=exc.kind,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        attempts = 0
        last_error: ToolError | None = None
        while attempts < self.max_retries + 1:
            attempts += 1
            try:
                data = await self._invoke(cleaned, ctx)
                if self.verify is not None and not self.verify(data):
                    raise TemporaryToolError(f"{self.name}: post-execution verification failed")
                return ToolResult(
                    ok=True,
                    data=data,
                    attempts=attempts,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            except asyncio.TimeoutError:
                last_error = TemporaryToolError(f"{self.name}: timed out after {self.timeout_s}s")
            except ToolError as exc:
                last_error = exc
                if exc.kind not in {FailureKind.TEMPORARY, FailureKind.RATE_LIMIT}:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - unexpected -> unknown failure
                last_error = ToolError(f"{self.name}: {type(exc).__name__}: {exc}", FailureKind.UNKNOWN)
                log.exception("tool_unexpected_error", extra={"tool": self.name})
                break

            if attempts <= self.max_retries:
                delay = self.retry_backoff_s * (2 ** (attempts - 1))
                log.warning(
                    "tool_retry",
                    extra={
                        "tool": self.name,
                        "task_id": ctx.task_id,
                        "attempt": attempts,
                        "delay": delay,
                        "error": str(last_error)[:300],
                    },
                )
                await asyncio.sleep(delay)

        assert last_error is not None
        return ToolResult(
            ok=False,
            error=str(last_error),
            kind=last_error.kind,
            attempts=attempts,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def _invoke(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        params = inspect.signature(self.handler).parameters
        kwargs = dict(args)
        if "ctx" in params:
            kwargs["ctx"] = ctx

        if inspect.iscoroutinefunction(self.handler):
            result = await asyncio.wait_for(self.handler(**kwargs), timeout=self.timeout_s)
        else:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.handler(**kwargs)),
                timeout=self.timeout_s,
            )
        if not isinstance(result, dict):
            return {"value": result}
        return result


def _coerce(tool: str, name: str, value: Any, spec: Arg) -> Any:
    want = spec.type
    try:
        if want == "string":
            value = value if isinstance(value, str) else str(value)
        elif want == "integer":
            value = int(value)
        elif want == "number":
            value = float(value)
        elif want == "boolean":
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                value = bool(value)
        elif want == "array":
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",") if part.strip()]
            elif not isinstance(value, list):
                raise ValueError("expected a list")
        elif want == "object":
            if not isinstance(value, dict):
                raise ValueError("expected an object")
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"{tool}: argument '{name}' must be {want} ({exc})") from exc

    if spec.choices and value not in spec.choices:
        raise InvalidInput(f"{tool}: argument '{name}' must be one of {spec.choices}")
    return value
