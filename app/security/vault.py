"""Encrypted credential vault.

Secrets that the owner sets at runtime (API keys, Teams client secrets) are
stored encrypted in the database, never in plaintext and never in logs.

Master prompt section 15/23: secrets belong in a secure configuration
mechanism, must not live in memory-only, and must never reach the LLM.

Encryption key resolution order:
  1. AGENT_SECRET_KEY environment variable
  2. <workspace>/.secret_key  (auto-generated once, chmod 600)

The vault degrades safely: if `cryptography` is unavailable the vault refuses
to store secrets rather than writing them in plaintext.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken

    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    Fernet = None  # type: ignore[assignment]
    InvalidToken = Exception  # type: ignore[assignment,misc]
    CRYPTO_AVAILABLE = False


class VaultError(Exception):
    """Raised when a secret cannot be stored or read."""


def _key_file() -> Path:
    return get_settings().workspace / ".secret_key"


# Key files this process had to invent because neither AGENT_SECRET_KEY nor
# the file existed. If stored credentials then fail to decrypt, a lost key
# file is almost certainly why, and the error log should say exactly that.
_generated_key_files: set[str] = set()


def _key_mismatch_cause() -> str:
    """Why stored credentials no longer decrypt. Never includes key material."""
    if os.environ.get("AGENT_SECRET_KEY", "").strip():
        return (
            "AGENT_SECRET_KEY is not the key these credentials were encrypted "
            "with (it was changed?)"
        )
    path = _key_file()
    if str(path) in _generated_key_files:
        return (
            f"the key file {path} was missing and AGENT_SECRET_KEY is not set, so "
            "a new key was generated; credentials saved under the old key cannot be read"
        )
    return (
        f"the key file {path} is not the key these credentials were encrypted "
        "with (replaced, or AGENT_SECRET_KEY was previously set and is now unset?)"
    )


_KEY_MISMATCH_FIX = (
    "restore the original .secret_key or set AGENT_SECRET_KEY to the original "
    "value and restart; otherwise re-enter the affected credentials (e.g. /tglogin)"
)


def _load_or_create_key() -> bytes:
    """Return the raw 32-byte master key, creating it on first use."""
    env_key = os.environ.get("AGENT_SECRET_KEY", "").strip()
    if env_key:
        return hashlib.sha256(env_key.encode("utf-8")).digest()

    path = _key_file()
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            return hashlib.sha256(raw.encode("utf-8")).digest()

    generated = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - windows / odd filesystems
        pass
    _generated_key_files.add(str(path))
    log.info("vault_key_created", extra={"path": str(path.name)})
    return hashlib.sha256(generated.encode("utf-8")).digest()


def _fernet() -> Any:
    if not CRYPTO_AVAILABLE:
        raise VaultError(
            "the 'cryptography' package is required to store credentials securely"
        )
    return Fernet(base64.urlsafe_b64encode(_load_or_create_key()))


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise VaultError("stored credential could not be decrypted (key changed?)") from exc


def mask(value: str) -> str:
    """Safe-to-display fingerprint of a secret."""
    if not value:
        return "(not set)"
    if len(value) <= 10:
        return value[:2] + "***"
    return f"{value[:6]}...{value[-4:]}"


class CredentialVault:
    """Async facade over the encrypted ``credentials`` table with a cache."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._loaded = False
        self._load_failed = False
        self._reload_task: asyncio.Task[bool] | None = None
        # Names whose stored value exists but no longer decrypts, so callers
        # can say "unreadable" instead of "not set".
        self._unreadable: set[str] = set()

    # ------------------------------------------------------------------ #
    async def load(self) -> bool:
        """Populate the in-memory cache from the database. True on success."""
        from app.db import repo
        from app.db.base import session_scope

        try:
            async with session_scope() as session:
                rows = await repo.list_credentials(session)
        except Exception as exc:  # noqa: BLE001 - startup must not fail on this
            # Left unloaded on purpose: the next access retries. Otherwise a
            # database that was merely slow at boot would leave every stored
            # credential (API keys, the Telegram session) missing until restart.
            self._load_failed = True
            log.warning(
                "vault_load_failed",
                extra={"error": str(exc)[:200], "retry": "on next access"},
            )
            return False

        cache: dict[str, str] = {}
        unreadable: list[str] = []
        for row in rows:
            try:
                cache[row.name] = decrypt(row.value)
            except VaultError:
                unreadable.append(row.name)
        self._cache = cache
        self._unreadable = set(unreadable)
        self._loaded = True
        self._load_failed = False
        if unreadable:
            # An ERROR, not a warning: the owner's credentials are effectively
            # gone and dependent features (the userbot above all) just look
            # "not linked". Names and the cause only - never values or keys.
            log.error(
                "vault_decrypt_failed",
                extra={
                    "cred_names": sorted(unreadable),
                    "count": len(unreadable),
                    "cause": _key_mismatch_cause(),
                    "fix": _KEY_MISMATCH_FIX,
                },
            )
        if cache:
            log.info("vault_loaded", extra={"count": len(cache)})
        return True

    async def ensure_loaded(self) -> bool:
        """Load now unless a load already succeeded. For async callers that
        must not act on an empty cache left behind by a failed boot load."""
        if self._loaded:
            return True
        task = self._reload_task
        if task is not None and not task.done():
            await asyncio.shield(task)
            if self._loaded:
                return True
        return await self.load()

    def _retry_failed_load(self) -> None:
        """Start a background reload if the last load failed.

        ``get`` is synchronous and called from sync code, so it cannot wait
        for the database: this call still answers from the cache, and later
        calls see the reloaded values.
        """
        if self._loaded or not self._load_failed:
            return
        if self._reload_task is not None and not self._reload_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop to run it on; ensure_loaded() will catch up
        self._reload_task = loop.create_task(self.load())

    async def set(self, name: str, value: str) -> None:
        from app.db import repo
        from app.db.base import session_scope

        name = name.strip().lower()
        if not name:
            raise VaultError("credential name must not be empty")
        if not value.strip():
            raise VaultError("credential value must not be empty")

        encrypted = encrypt(value)
        async with session_scope() as session:
            await repo.set_credential(session, name=name, value=encrypted)
        self._cache[name] = value
        self._unreadable.discard(name)
        # NOTE: the value itself is deliberately never logged. "name" is a
        # reserved LogRecord attribute - stdlib logging refuses `extra={"name": ...}`
        # with KeyError, so this is namespaced to avoid the collision.
        log.info("credential_set", extra={"cred_name": name})

    async def delete(self, name: str) -> bool:
        from app.db import repo
        from app.db.base import session_scope

        name = name.strip().lower()
        async with session_scope() as session:
            removed = await repo.delete_credential(session, name)
        self._cache.pop(name, None)
        self._unreadable.discard(name)
        if removed:
            log.info("credential_deleted", extra={"cred_name": name})
        return removed

    def get(self, name: str, fallback: str = "") -> str:
        """Cached lookup. Falls back to the value from .env."""
        self._retry_failed_load()
        return self._cache.get(name.strip().lower()) or fallback

    def has(self, name: str) -> bool:
        self._retry_failed_load()
        return bool(self._cache.get(name.strip().lower()))

    def names(self) -> list[str]:
        self._retry_failed_load()
        return sorted(self._cache)

    def unreadable(self, name: str) -> bool:
        """True if ``name`` is stored but could not be decrypted (key changed)."""
        return name.strip().lower() in self._unreadable

    def clear_cache(self) -> None:
        self._cache.clear()
        self._unreadable.clear()
        self._loaded = False


_vault: CredentialVault | None = None


def get_vault() -> CredentialVault:
    global _vault
    if _vault is None:
        _vault = CredentialVault()
    return _vault


def reset_vault() -> None:
    global _vault
    _vault = None
