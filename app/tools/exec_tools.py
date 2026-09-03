"""Code execution tools.

python_execute  - runs a script in a separate interpreter process, cwd pinned
                  to the workspace, wall-clock timeout, output size capped.
safe_shell_execute - allowlisted commands only, no shell interpretation, no
                  pipes/redirects, cwd restricted to the workspace.

Neither tool ever runs a string the model produced through a shell.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, safe_path
from app.tools.base import Arg, InvalidInput, PermanentToolError, TemporaryToolError
from app.tools.registry import tool

log = get_logger(__name__)

MAX_OUTPUT_CHARS = 12000
_BLOCKED_SHELL_CHARS = set(";|&><`$\n\r")


def _clip(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, total {len(text)} chars]"
    return text


def _child_env() -> dict[str, str]:
    """Minimal environment: no secrets from the agent process leak into children."""
    settings = get_settings()
    keep = {"PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["HOME"] = str(settings.workspace)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def _run(argv: list[str], cwd: Path, timeout_s: int) -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=_child_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise PermanentToolError(f"executable not found: {argv[0]}") from exc
    except OSError as exc:
        raise TemporaryToolError(f"failed to start process: {exc}") from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TemporaryToolError(f"process timed out after {timeout_s}s")

    return {
        "exit_code": proc.returncode,
        "stdout": _clip(stdout),
        "stderr": _clip(stderr),
    }


@tool(
    "python_execute",
    description=(
        "Run a short Python 3 script in a separate process. Working directory is the "
        "workspace root; print() output is returned. No network credentials are exposed."
    ),
    permission=Permission.WRITE,
    args={
        "code": Arg("string", True, "Python source to execute"),
        "timeout_s": Arg("integer", False, "Wall clock limit (max 600)", default=120),
    },
    timeout_s=660,
    max_retries=0,
    enabled=get_settings().enable_python_tool,
)
async def python_execute(code: str, timeout_s: int = 120) -> dict[str, Any]:
    if not code.strip():
        raise InvalidInput("code must not be empty")
    timeout_s = max(1, min(int(timeout_s), 600))

    settings = get_settings()
    settings.ensure_workspace()
    script = settings.workspace / "temp" / f"snippet_{os.getpid()}_{id(code) & 0xFFFF:x}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(code, encoding="utf-8")

    try:
        result = await _run([sys.executable, "-I", str(script)], settings.workspace, timeout_s)
    finally:
        script.unlink(missing_ok=True)

    result["ok"] = result["exit_code"] == 0
    log.info("python_execute", extra={"tool": "python_execute", "exit_code": result["exit_code"]})
    if result["exit_code"] != 0 and not result["stdout"]:
        raise PermanentToolError(f"python exited {result['exit_code']}: {result['stderr'][:800]}")
    return result


@tool(
    "safe_shell_execute",
    description=(
        "Run ONE allowlisted, non-destructive shell command inside the workspace. "
        "No pipes, redirects, globs or command chaining. Use python_execute for logic."
    ),
    permission=Permission.WRITE,
    args={
        "command": Arg("string", True, "Command line, e.g. 'ls -la downloads'"),
        "timeout_s": Arg("integer", False, "Wall clock limit (max 300)", default=60),
    },
    timeout_s=360,
    max_retries=0,
    enabled=get_settings().enable_shell_tool,
)
async def safe_shell_execute(command: str, timeout_s: int = 60) -> dict[str, Any]:
    settings = get_settings()
    command = command.strip()
    if not command:
        raise InvalidInput("command must not be empty")
    if any(char in _BLOCKED_SHELL_CHARS for char in command):
        raise InvalidInput(
            "command contains shell metacharacters (; | & > < ` $ newline) which are not allowed"
        )

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise InvalidInput(f"could not parse command: {exc}") from exc
    if not argv:
        raise InvalidInput("empty command")

    program = Path(argv[0]).name
    allowed = settings.shell_allowed_commands
    if program not in allowed:
        raise InvalidInput(
            f"command '{program}' is not allowlisted. Allowed: {', '.join(sorted(allowed))}"
        )

    # git: read-only subcommands only.
    if program == "git" and (len(argv) < 2 or argv[1] not in {"status", "diff", "log", "show", "branch"}):
        raise InvalidInput("only read-only git subcommands are allowed (status/diff/log/show/branch)")

    # Any path-looking argument must stay inside the workspace.
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        if "/" in arg or "\\" in arg or arg in {".", ".."} or Path(arg).exists():
            try:
                safe_path(arg)
            except UnsafePath as exc:
                raise InvalidInput(str(exc)) from exc

    settings.ensure_workspace()
    timeout_s = max(1, min(int(timeout_s), 300))
    result = await _run(argv, settings.workspace, timeout_s)
    result["ok"] = result["exit_code"] == 0
    result["command"] = command
    log.info(
        "safe_shell_execute",
        extra={"tool": "safe_shell_execute", "cmd": program, "exit_code": result["exit_code"]},
    )
    return result
