"""Workspace sandbox.

Every file path the agent touches is resolved through :func:`safe_path`, which
guarantees the result stays inside ``settings.workspace``.  Path traversal,
absolute escapes and symlink escapes are all rejected.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.config import get_settings


class UnsafePath(ValueError):
    """Raised when a requested path escapes the workspace."""


def workspace_root() -> Path:
    return get_settings().workspace


def safe_path(raw: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Resolve ``raw`` inside the workspace or raise :class:`UnsafePath`."""
    if raw is None or str(raw).strip() == "":
        raise UnsafePath("empty path")

    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)

    candidate = Path(str(raw).strip())
    if candidate.is_absolute():
        target = candidate
    else:
        target = root / candidate

    # ``strict=False`` so we can also validate not-yet-created files.
    resolved = target.resolve(strict=False)
    root_resolved = root.resolve(strict=False)

    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise UnsafePath(f"path escapes workspace: {raw}")

    if must_exist and not resolved.exists():
        raise UnsafePath(f"path does not exist: {rel_path(resolved)}")

    return resolved


def rel_path(path: Path) -> str:
    """Workspace relative representation, used in logs and LLM observations."""
    try:
        return str(path.resolve(strict=False).relative_to(workspace_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": rel_path(path),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "sha256": sha256_file(path) if stat.st_size <= 256 * 1024 * 1024 else None,
    }
