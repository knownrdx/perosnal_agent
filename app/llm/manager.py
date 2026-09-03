"""Runtime LLM manager.

Holds every configured provider, exposes the active one, and lets the owner
switch model/provider live from Telegram.  The choice is persisted in the
``settings`` table so it survives a restart.

Also implements an optional fallback chain: if the active provider fails
(cloud outage, local model down), the next configured provider is tried so a
long-running overnight task does not die because one API blipped.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.llm.base import LLMClient, LLMError, LLMResponse, Message
from app.logging_conf import get_logger

log = get_logger(__name__)

SETTING_KEY = "llm.active"
CUSTOM_KEY = "llm.custom_providers"

# Curated shortlist shown in Telegram /models. Free-form ids are still allowed.
KNOWN_MODELS: dict[str, list[str]] = {
    "omniroute": [
        "auto",             # gateway picks a working provider
        "auto/coding",      # routing strategy tuned for code/tools
        "auto/free",        # prefer free tiers
    ],
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-latest",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-5",
    ],
    "ollama": [
        "qwen2.5-coder:7b-instruct-q4_K_M",
        "qwen2.5-coder:14b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M",
    ],
}


def resolved_key(provider: str) -> str:
    """API key for a provider: runtime vault first, then .env."""
    from app.security.vault import get_vault

    settings = get_settings()
    env_value = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "omniroute": settings.omniroute_api_key,
        "custom": settings.custom_llm_api_key,
    }.get(provider, "")
    return get_vault().get(f"{provider}_api_key", env_value)


@dataclass(slots=True)
class ProviderInfo:
    key: str            # anthropic | openai | ollama | echo | <custom>
    label: str          # human name shown in Telegram
    model: str
    configured: bool    # has an API key / reachable endpoint


class LLMManager(LLMClient):
    """Owns every provider client and the currently active selection.

    Implements :class:`LLMClient` itself so the agent engine can treat it as an
    ordinary model client while it transparently routes + falls back.
    """

    name = "manager"

    def __init__(self) -> None:
        self._clients: dict[str, LLMClient] = {}
        self._active: str = ""
        self._model_override: dict[str, str] = {}
        # name -> {base_url, model}; the API key lives in the encrypted vault.
        self._custom: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build(self, key: str) -> LLMClient:
        from app.llm.anthropic_client import AnthropicClient
        from app.llm.echo_client import EchoClient
        from app.llm.ollama_client import OllamaClient
        from app.llm.openai_client import OpenAIClient

        settings = get_settings()
        model = self._model_override.get(key)

        if key == "anthropic":
            return AnthropicClient(api_key=resolved_key("anthropic"), model=model)
        if key == "openai":
            return OpenAIClient(api_key=resolved_key("openai"), model=model)
        if key == "ollama":
            return OllamaClient(model=model)
        if key == "omniroute":
            return OpenAIClient(
                api_key=resolved_key("omniroute") or "omniroute",
                base_url=settings.omniroute_base_url,
                model=model or settings.omniroute_model,
                provider_label="omniroute",
            )
        if key in {"echo", "stub", "fake"}:
            return EchoClient()

        # Curated presets: base_url + model already known.
        from app.llm.presets import PRESETS

        preset = PRESETS.get(key)
        if preset is not None and key not in {"ollama", "omniroute"}:
            from app.security.vault import get_vault

            api_key = get_vault().get(f"{key}_api_key", "")
            if preset.native == "anthropic":
                return AnthropicClient(
                    api_key=api_key, base_url=preset.base_url,
                    model=model or preset.default_model,
                )
            return OpenAIClient(
                api_key=api_key,
                base_url=preset.base_url,
                model=model or preset.default_model,
                provider_label=key,
            )

        # Providers registered at runtime from Telegram (/addllm).
        if key in self._custom:
            entry = self._custom[key]
            from app.security.vault import get_vault

            return OpenAIClient(
                api_key=get_vault().get(f"{key}_api_key", ""),
                base_url=entry["base_url"],
                model=model or entry.get("model", ""),
                provider_label=key,
            )

        # A single custom endpoint may also come from .env.
        if settings.custom_llm_base_url:
            return OpenAIClient(
                api_key=resolved_key("custom"),
                base_url=settings.custom_llm_base_url,
                model=model or settings.custom_llm_model,
                provider_label=key,
            )
        raise LLMError(f"unknown LLM provider: {key}")

    def client(self, key: str | None = None) -> LLMClient:
        name = (key or self.active_key()).lower()
        if name not in self._clients:
            self._clients[name] = self._build(name)
        return self._clients[name]

    # ------------------------------------------------------------------ #
    # Active selection
    # ------------------------------------------------------------------ #
    def active_key(self) -> str:
        if self._active:
            return self._active
        settings = get_settings()
        configured = settings.llm_provider.lower()

        # Auto-default: if the configured provider needs a key that is not set,
        # fall back to the OmniRoute gateway, which works without one.
        needs_key = {"anthropic": resolved_key("anthropic"), "openai": resolved_key("openai")}
        if configured in needs_key and not needs_key[configured] and settings.omniroute_enabled:
            return "omniroute"
        return configured

    def active_model(self) -> str:
        key = self.active_key()
        if key in self._model_override:
            return self._model_override[key]
        settings = get_settings()
        mapping = {
            "anthropic": settings.anthropic_model,
            "openai": settings.openai_model,
            "ollama": settings.llm_model,
            "omniroute": settings.omniroute_model,
            "echo": "echo",
        }
        if key in mapping:
            return mapping[key]

        from app.llm.presets import PRESETS

        preset = PRESETS.get(key)
        if preset is not None:
            return preset.default_model
        if key in self._custom:
            return self._custom[key].get("model", "")
        return settings.custom_llm_model or settings.llm_model

    def configured_providers(self) -> list[ProviderInfo]:
        settings = get_settings()
        out: list[ProviderInfo] = []
        if settings.omniroute_enabled:
            out.append(
                ProviderInfo(
                    "omniroute",
                    "OmniRoute gateway (no key needed)",
                    self._model_override.get("omniroute", settings.omniroute_model),
                    True,
                )
            )
        out += [
            ProviderInfo("anthropic", "Claude (Anthropic)",
                         self._model_override.get("anthropic", settings.anthropic_model),
                         bool(resolved_key("anthropic"))),
            ProviderInfo("openai", "ChatGPT (OpenAI)",
                         self._model_override.get("openai", settings.openai_model),
                         bool(resolved_key("openai"))),
            ProviderInfo("ollama", "Local (Ollama)",
                         self._model_override.get("ollama", settings.llm_model),
                         bool(settings.ollama_base_url)),
        ]
        if settings.custom_llm_base_url:
            out.append(
                ProviderInfo(
                    settings.custom_llm_name or "custom",
                    settings.custom_llm_name or "Custom endpoint",
                    self._model_override.get(
                        settings.custom_llm_name or "custom", settings.custom_llm_model
                    ),
                    True,
                )
            )
        from app.llm.presets import PRESETS
        from app.security.vault import get_vault

        vault = get_vault()

        # Presets the owner has actually supplied a key for.
        listed = {p.key for p in out}
        for name, preset in PRESETS.items():
            if name in listed or preset.auth == "none":
                continue
            if vault.has(f"{name}_api_key"):
                out.append(
                    ProviderInfo(
                        name,
                        preset.label,
                        self._model_override.get(name, preset.default_model),
                        True,
                    )
                )

        for name, entry in sorted(self._custom.items()):
            out.append(
                ProviderInfo(
                    name,
                    f"{name} ({entry['base_url']})",
                    self._model_override.get(name, entry.get("model", "")),
                    vault.has(f"{name}_api_key") or not entry.get("needs_key", True),
                )
            )
        return out

    async def set_active(self, key: str, model: str | None = None, *, persist: bool = True) -> ProviderInfo:
        from app.llm.presets import ALIASES, PRESETS

        key = ALIASES.get(key.lower().strip(), key.lower().strip())
        known = {p.key for p in self.configured_providers()} | {"echo"}
        if key not in known:
            if key in PRESETS:
                raise LLMError(
                    f"'{key}' has no key yet. Run /llm {key} to connect it."
                )
            raise LLMError(f"unknown provider '{key}'. Available: {', '.join(sorted(known))}")

        provider = next((p for p in self.configured_providers() if p.key == key), None)
        if provider is not None and not provider.configured:
            raise LLMError(f"provider '{key}' has no API key configured")

        async with self._lock:
            if model:
                self._model_override[key] = model
            self._clients.pop(key, None)   # rebuild with the new model
            self._active = key

        if persist:
            await self._persist()

        info = ProviderInfo(
            key, key, self.active_model(), True
        )
        log.info("llm_switched", extra={"provider": key, "model": info.model})
        return info

    async def set_model(self, model: str) -> str:
        """Change the model of the currently active provider."""
        return (await self.set_active(self.active_key(), model)).model

    async def connect_preset(self, name: str, api_key: str) -> dict[str, Any]:
        """Store a pasted key for a preset provider and verify it works."""
        from app.llm.presets import resolve, validate_key
        from app.security.vault import get_vault

        preset = resolve(name)
        if preset is None:
            raise LLMError(f"unknown provider '{name}'")
        if preset.auth == "none":
            raise LLMError(f"{preset.label} needs no key - it is ready to use")

        problem = validate_key(preset, api_key)
        if problem:
            raise LLMError(problem)

        await get_vault().set(f"{preset.key}_api_key", api_key.strip())
        self._clients.pop(preset.key, None)

        health = await self.client(preset.key).health()
        return {
            "provider": preset.key,
            "label": preset.label,
            "model": self.active_model() if self.active_key() == preset.key
                     else preset.default_model,
            "verified": bool(health.get("ok")),
            "error": str(health.get("error", ""))[:200],
        }

    # ------------------------------------------------------------------ #
    # Runtime provider registration (/addllm from Telegram)
    # ------------------------------------------------------------------ #
    async def add_custom_provider(
        self, name: str, base_url: str, model: str, api_key: str = ""
    ) -> ProviderInfo:
        """Register any OpenAI-compatible endpoint at runtime."""
        name = name.strip().lower()
        reserved = {"anthropic", "openai", "ollama", "omniroute", "echo", "custom"}
        if not name or name in reserved:
            raise LLMError(f"pick another name; '{name}' is reserved")
        if not base_url.startswith(("http://", "https://")):
            raise LLMError("base_url must start with http:// or https://")

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1") and "/v1" not in base_url:
            base_url = base_url + "/v1"

        if api_key:
            from app.security.vault import get_vault

            await get_vault().set(f"{name}_api_key", api_key)

        self._custom[name] = {
            "base_url": base_url,
            "model": model.strip(),
            "needs_key": bool(api_key),
        }
        self._clients.pop(name, None)
        await self._persist_custom()
        log.info("llm_custom_added", extra={"provider": name, "base_url": base_url})
        return ProviderInfo(name, name, model, True)

    async def remove_custom_provider(self, name: str) -> bool:
        name = name.strip().lower()
        if name not in self._custom:
            return False
        self._custom.pop(name)
        self._clients.pop(name, None)
        from app.security.vault import get_vault

        await get_vault().delete(f"{name}_api_key")
        if self._active == name:
            self._active = ""
        await self._persist_custom()
        return True

    async def _persist_custom(self) -> None:
        from app.db import repo
        from app.db.base import session_scope

        try:
            async with session_scope() as session:
                await repo.set_setting(session, CUSTOM_KEY, {"providers": self._custom})
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_custom_persist_failed", extra={"error": str(exc)[:200]})

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    async def _persist(self) -> None:
        from app.db import repo
        from app.db.base import session_scope

        try:
            async with session_scope() as session:
                await repo.set_setting(
                    session,
                    SETTING_KEY,
                    {"provider": self.active_key(), "models": dict(self._model_override)},
                )
        except Exception as exc:  # noqa: BLE001 - never break a switch on db issues
            log.warning("llm_persist_failed", extra={"error": str(exc)[:200]})

    async def load(self) -> None:
        """Restore the previously selected provider/model after a restart."""
        from app.db import repo
        from app.db.base import session_scope

        try:
            async with session_scope() as session:
                stored = await repo.get_setting(session, SETTING_KEY)
                custom = await repo.get_setting(session, CUSTOM_KEY)
        except Exception:  # noqa: BLE001
            stored, custom = None, None

        if custom and isinstance(custom.get("providers"), dict):
            self._custom = {
                str(k): dict(v) for k, v in custom["providers"].items() if isinstance(v, dict)
            }
        if not stored:
            return
        provider = str(stored.get("provider") or "").lower()
        models = stored.get("models") or {}
        if isinstance(models, dict):
            self._model_override.update({str(k): str(v) for k, v in models.items()})
        if provider:
            try:
                await self.set_active(provider, persist=False)
                log.info("llm_restored", extra={"provider": provider, "model": self.active_model()})
            except LLMError as exc:
                log.warning("llm_restore_failed", extra={"error": str(exc)[:200]})

    # ------------------------------------------------------------------ #
    # Execution with fallback
    # ------------------------------------------------------------------ #
    def fallback_order(self) -> list[str]:
        """Active provider first, then every other configured one.

        OmniRoute is tried last-but-one and the local model last, so a personal
        API key is always preferred over the shared gateway when present.
        """
        settings = get_settings()
        active = self.active_key()
        order = [active]
        if settings.llm_fallback_enabled:
            rank = {"anthropic": 0, "openai": 1, "omniroute": 2, "ollama": 3}
            others = [
                p.key for p in self.configured_providers()
                if p.key != active and p.configured
            ]
            order += sorted(others, key=lambda key: rank.get(key, 9))
        return order

    async def chat(self, messages: list[Message], *, temperature: float | None = None) -> LLMResponse:
        errors: list[str] = []
        for index, key in enumerate(self.fallback_order()):
            try:
                response = await self.client(key).chat(messages, temperature=temperature)
                if index > 0:
                    log.warning("llm_fallback_used", extra={"provider": key})
                return response
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{key}: {str(exc)[:200]}")
                log.warning("llm_provider_failed", extra={"provider": key, "error": str(exc)[:200]})
        raise LLMError("all LLM providers failed -> " + " | ".join(errors))

    async def chat_json(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> dict[str, Any]:
        from app.llm.base import extract_json

        response = await self.chat(messages, temperature=temperature)
        data = extract_json(response.content)
        if data is None:
            raise LLMError(f"model did not return JSON: {response.content[:400]!r}")
        return data

    async def health(self) -> dict[str, Any]:
        info = await self.client().health()
        info["active_provider"] = self.active_key()
        info["fallback"] = self.fallback_order()[1:]
        return info

    async def health_all(self) -> list[dict[str, Any]]:
        results = []
        for provider in self.configured_providers():
            if not provider.configured:
                results.append({"ok": False, "provider": provider.key, "model": provider.model,
                                "error": "not configured"})
                continue
            try:
                results.append(await self.client(provider.key).health())
            except Exception as exc:  # noqa: BLE001
                results.append({"ok": False, "provider": provider.key, "error": str(exc)[:200]})
        return results

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()


_manager: LLMManager | None = None


def get_manager() -> LLMManager:
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


def reset_manager() -> None:
    global _manager
    _manager = None
