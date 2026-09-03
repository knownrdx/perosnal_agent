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

    # ------------------------------------------------------------------ #
    async def load(self) -> None:
        """Populate the in-memory cache from the database."""
        from app.db import repo
        from app.db.base import session_scope

        try:
            async with session_scope() as session:
                rows = await repo.list_credentials(session)
        except Exception as exc:  # noqa: BLE001 - startup must not fail on this
            log.warning("vault_load_failed", extra={"error": str(exc)[:200]})
            return

        cache: dict[str, str] = {}
        for row in rows:
            try:
                cache[row.name] = decrypt(row.value)
            except VaultError:
                log.warning("vault_decrypt_failed", extra={"name": row.name})
        self._cache = cache
        self._loaded = True
        if cache:
            log.info("vault_loaded", extra={"count": len(cache)})

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
        # NOTE: the value itself is deliberately never logged.
        log.info("credential_set", extra={"name": name})

    async def delete(self, name: str) -> bool:
        from app.db import repo
        from app.db.base import session_scope

        name = name.strip().lower()
        async with session_scope() as session:
            removed = await repo.delete_credential(session, name)
        self._cache.pop(name, None)
        if removed:
            log.info("credential_deleted", extra={"name": name})
        return removed

    def get(self, name: str, fallback: str = "") -> str:
        """Cached lookup. Falls back to the value from .env."""
        return self._cache.get(name.strip().lower()) or fallback

    def has(self, name: str) -> bool:
        return bool(self._cache.get(name.strip().lower()))

    def names(self) -> list[str]:
        return sorted(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
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
