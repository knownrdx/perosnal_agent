"""LLM package.

``get_llm()`` returns the *manager*, which behaves like an ``LLMClient`` but
routes to whichever provider is active (Claude / ChatGPT / local Ollama /
custom OpenAI-compatible endpoint) and falls back to another configured
provider if the active one fails.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient, LLMError, LLMResponse, Message, extract_json
from app.llm.echo_client import EchoClient
from app.llm.manager import KNOWN_MODELS, LLMManager, ProviderInfo, get_manager, reset_manager
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient

_override: LLMClient | None = None


def build_client(provider: str | None = None) -> LLMClient:
    """Build a single provider client (bypasses the manager)."""
    name = (provider or get_settings().llm_provider).lower()
    if name in {"echo", "stub", "fake"}:
        return EchoClient()
    return get_manager().client(name)


def get_llm() -> LLMClient:
    """The active LLM. Tests can pin a client with :func:`set_llm`."""
    if _override is not None:
        return _override
    settings = get_settings()
    # The echo provider is a plain client (used by tests / offline runs).
    if settings.llm_provider.lower() in {"echo", "stub", "fake"} and not get_manager()._active:
        return get_manager().client("echo")
    return get_manager()


def set_llm(client: LLMClient | None) -> None:
    """Inject a client (tests, or a future runtime model switch)."""
    global _override
    _override = client


async def close_llm() -> None:
    global _override
    if _override is not None:
        await _override.close()
    _override = None
    await get_manager().close()
    reset_manager()


__all__ = [
    "KNOWN_MODELS",
    "AnthropicClient",
    "EchoClient",
    "LLMClient",
    "LLMError",
    "LLMManager",
    "LLMResponse",
    "Message",
    "OllamaClient",
    "OpenAIClient",
    "ProviderInfo",
    "build_client",
    "close_llm",
    "extract_json",
    "get_llm",
    "get_manager",
    "reset_manager",
    "set_llm",
]
