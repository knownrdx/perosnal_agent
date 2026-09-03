"""File tools: sandboxed to the workspace, size-capped, checksum verified."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, file_info, rel_path, safe_path, sha256_file
from app.tools.base import (
    Arg,
    InvalidInput,
    PermanentToolError,
    TemporaryToolError,
    ToolContext,
)
from app.tools.registry import tool

log = get_logger(__name__)


def _resolve(path: str, *, must_exist: bool = False) -> Path:
    try:
        return safe_path(path, must_exist=must_exist)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc


@tool(
    "file_write",
    description="Write UTF-8 text to a workspace file (creates parent dirs, overwrites).",
    permission=Permission.WRITE,
    args={
        "path": Arg("string", True, "Workspace-relative path, e.g. output/report.txt"),
        "content": Arg("string", True, "Text content to write"),
        "append": Arg("boolean", False, "Append instead of overwrite", default=False),
    },
    timeout_s=60,
    max_retries=0,
)
def file_write(path: str, content: str, append: bool = False) -> dict[str, Any]:
    target = _resolve(path)
    settings = get_settings()
    data = content.encode("utf-8")
    if len(data) > settings.max_file_bytes:
        raise InvalidInput(f"content exceeds max file size ({settings.max_file_mb} MB)")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab" if append else "wb") as handle:
        handle.write(data)
    info = file_info(target)
    log.info("file_write", extra={"tool": "file_write", "path": info["path"], "size": info["size_bytes"]})
    return {"written": True, **info}


@tool(
    "file_read",
    description="Read a UTF-8 text file from the workspace (truncated to max_chars).",
    permission=Permission.READ,
    args={
        "path": Arg("string", True, "Workspace-relative path"),
        "max_chars": Arg("integer", False, "Maximum characters to return", default=8000),
    },
    timeout_s=60,
    max_retries=0,
)
def file_read(path: str, max_chars: int = 8000) -> dict[str, Any]:
    target = _resolve(path, must_exist=True)
    if target.is_dir():
        raise InvalidInput(f"{rel_path(target)} is a directory, use file_list")
    raw = target.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "path": rel_path(target),
        "content": text[:max_chars],
        "truncated": truncated,
        "size_bytes": len(raw),
    }


@tool(
    "file_exists",
    description="Check whether a workspace path exists and return its metadata.",
    permission=Permission.READ,
    args={"path": Arg("string", True, "Workspace-relative path")},
    timeout_s=30,
    max_retries=0,
)
def file_exists(path: str) -> dict[str, Any]:
    target = _resolve(path)
    if not target.exists():
        return {"exists": False, "path": rel_path(target)}
    if target.is_dir():
        return {"exists": True, "is_dir": True, "path": rel_path(target)}
    return {"exists": True, "is_dir": False, **file_info(target)}


@tool(
    "file_list",
    description="List files in a workspace directory.",
    permission=Permission.READ,
    args={
        "path": Arg("string", False, "Workspace-relative directory", default="."),
        "pattern": Arg("string", False, "Glob pattern, e.g. *.pdf", default="*"),
        "limit": Arg("integer", False, "Maximum entries", default=100),
    },
    timeout_s=60,
    max_retries=0,
)
def file_list(path: str = ".", pattern: str = "*", limit: int = 100) -> dict[str, Any]:
    target = _resolve(path, must_exist=True)
    if not target.is_dir():
        raise InvalidInput(f"{rel_path(target)} is not a directory")
    entries = []
    for item in sorted(target.glob(pattern)):
        entries.append(
            {
                "path": rel_path(item),
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            }
        )
        if len(entries) >= limit:
            break
    return {"path": rel_path(target), "count": len(entries), "entries": entries}


@tool(
    "file_copy",
    description="Copy a file inside the workspace.",
    permission=Permission.WRITE,
    args={
        "source": Arg("string", True, "Existing workspace file"),
        "destination": Arg("string", True, "Destination workspace path"),
    },
    timeout_s=120,
    max_retries=0,
)
def file_copy(source: str, destination: str) -> dict[str, Any]:
    src = _resolve(source, must_exist=True)
    dst = _resolve(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"copied": True, **file_info(dst)}


@tool(
    "file_move",
    description="Move or rename a file inside the workspace.",
    permission=Permission.WRITE,
    args={
        "source": Arg("string", True, "Existing workspace file"),
        "destination": Arg("string", True, "Destination workspace path"),
    },
    timeout_s=120,
    max_retries=0,
)
def file_move(source: str, destination: str) -> dict[str, Any]:
    src = _resolve(source, must_exist=True)
    dst = _resolve(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"moved": True, **file_info(dst)}


@tool(
    "file_delete",
    description="Delete a workspace file. Destructive: requires approval.",
    permission=Permission.HIGH_RISK,
    args={"path": Arg("string", True, "Workspace-relative file to delete")},
    timeout_s=60,
    max_retries=0,
    side_effect=True,
)
def file_delete(path: str) -> dict[str, Any]:
    target = _resolve(path, must_exist=True)
    if target.is_dir():
        raise InvalidInput("refusing to delete a directory; delete individual files")
    checksum = sha256_file(target)
    target.unlink()
    log.info("file_delete", extra={"tool": "file_delete", "path": rel_path(target)})
    return {"deleted": True, "path": rel_path(target), "sha256": checksum}


@tool(
    "file_download",
    description="Download a URL into the workspace (http/https only) and verify the saved file.",
    permission=Permission.WRITE,
    args={
        "url": Arg("string", True, "http(s) URL to download"),
        "path": Arg("string", False, "Destination workspace path", default=""),
    },
    timeout_s=600,
    max_retries=2,
    verify=lambda data: bool(data.get("size_bytes", 0) > 0),
)
async def file_download(url: str, path: str = "", ctx: ToolContext | None = None) -> dict[str, Any]:
    if not url.lower().startswith(("http://", "https://")):
        raise InvalidInput("only http(s) URLs are supported")
    settings = get_settings()

    if not path:
        name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
        path = f"downloads/{name}"
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(target.suffix + ".part")
    total = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("GET", url) as response:
                if response.status_code == 429:
                    raise TemporaryToolError("rate limited by remote server")
                if response.status_code in {401, 403}:
                    raise PermanentToolError(f"access denied by remote server ({response.status_code})")
                if response.status_code >= 400:
                    raise PermanentToolError(f"download failed: HTTP {response.status_code}")
                with tmp.open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_size=256 * 1024):
                        total += len(chunk)
                        if total > settings.max_file_bytes:
                            raise PermanentToolError(
                                f"file exceeds max size ({settings.max_file_mb} MB)"
                            )
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise TemporaryToolError(f"network error: {exc}") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(target)
    info = file_info(target)
    log.info("file_download", extra={"tool": "file_download", "path": info["path"], "size": total})
    return {"downloaded": True, "url": url, **info}
