"""LLM provider presets.

A curated catalogue so the owner never has to remember a base URL or a model
id.  Each preset knows where to get credentials, what endpoint to call, and
which models are worth defaulting to.

Auth kinds:
    key     - paste an API key from the provider's console (most providers)
    device  - real RFC 8628 device-code flow (needs a client_id)
    none    - works with no credentials at all
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    key: str                      # internal provider name
    label: str                    # shown in Telegram
    base_url: str                 # OpenAI-compatible endpoint (or native)
    models: tuple[str, ...]       # recommended models, best first
    auth: str = "key"             # key | device | none
    signup_url: str = ""          # where to create the credential
    key_prefix: str = ""          # sanity check on pasted keys
    native: str = ""              # "anthropic" for the Messages API
    free: bool = False
    notes: str = ""
    steps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def default_model(self) -> str:
        return self.models[0] if self.models else ""


PRESETS: dict[str, Preset] = {
    # ---------------------------------------------------------------- #
    # Zero-config
    # ---------------------------------------------------------------- #
    "omniroute": Preset(
        key="omniroute",
        label="OmniRoute gateway",
        base_url="http://omniroute:20128/v1",
        models=("auto", "auto/coding", "auto/free"),
        auth="none",
        free=True,
        notes="Already running. No key needed - this is the default.",
    ),
    "ollama": Preset(
        key="ollama",
        label="Local model (Ollama)",
        base_url="http://ollama:11434",
        models=(
            "qwen2.5-coder:7b-instruct-q4_K_M",
            "qwen2.5-coder:14b-instruct-q4_K_M",
            "llama3.1:8b-instruct-q4_K_M",
        ),
        auth="none",
        free=True,
        notes="Runs on this VPS. Private, no internet needed, slower.",
    ),

    # ---------------------------------------------------------------- #
    # Major providers (API key from their console)
    # ---------------------------------------------------------------- #
    "anthropic": Preset(
        key="anthropic",
        label="Claude (Anthropic)",
        base_url="https://api.anthropic.com",
        native="anthropic",
        models=(
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
        ),
        auth="key",
        signup_url="https://console.anthropic.com/settings/keys",
        key_prefix="sk-ant-",
        notes="Best for agentic tool use and long tasks.",
        steps=(
            "Open the link and sign in",
            "Click 'Create Key', copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "mwapi": Preset(
        key="mwapi",
        label="Claude via mwapi gateway",
        base_url="https://api.mwapi.dev/v1",
        native="anthropic",
        models=(
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
        ),
        auth="key",
        signup_url="https://api.mwapi.dev",
        key_prefix="sk-",
        notes="Anthropic-compatible gateway; opus/sonnet without an Anthropic account.",
        steps=(
            "Get your gateway key from the provider",
            "Send it back with:  /paste <key>",
        ),
    ),
    "openai": Preset(
        key="openai",
        label="ChatGPT (OpenAI)",
        base_url="https://api.openai.com/v1",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1"),
        auth="key",
        signup_url="https://platform.openai.com/api-keys",
        key_prefix="sk-",
        steps=(
            "Open the link and sign in",
            "Click 'Create new secret key', copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "gemini": Preset(
        key="gemini",
        label="Gemini (Google)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        models=("gemini-2.0-flash", "gemini-2.5-pro"),
        auth="key",
        signup_url="https://aistudio.google.com/apikey",
        free=True,
        notes="Has a usable free tier.",
        steps=(
            "Open the link and sign in with Google",
            "Click 'Create API key', copy it",
            "Send it back with:  /paste <key>",
        ),
    ),

    # ---------------------------------------------------------------- #
    # Aggregators and fast/free tiers (all OpenAI-compatible)
    # ---------------------------------------------------------------- #
    "openrouter": Preset(
        key="openrouter",
        label="OpenRouter (300+ models)",
        base_url="https://openrouter.ai/api/v1",
        models=(
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct:free",
        ),
        auth="key",
        signup_url="https://openrouter.ai/keys",
        key_prefix="sk-or-",
        free=True,
        notes="One key, every model. Some models are free.",
        steps=(
            "Open the link and sign in",
            "Click 'Create Key', copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "groq": Preset(
        key="groq",
        label="Groq (very fast, free tier)",
        base_url="https://api.groq.com/openai/v1",
        models=("llama-3.3-70b-versatile", "qwen-2.5-32b"),
        auth="key",
        signup_url="https://console.groq.com/keys",
        key_prefix="gsk_",
        free=True,
        notes="Fastest responses; generous free tier.",
        steps=(
            "Open the link and sign in",
            "Click 'Create API Key', copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "deepseek": Preset(
        key="deepseek",
        label="DeepSeek (very cheap)",
        base_url="https://api.deepseek.com/v1",
        models=("deepseek-chat", "deepseek-reasoner"),
        auth="key",
        signup_url="https://platform.deepseek.com/api_keys",
        key_prefix="sk-",
        notes="Strong at code, extremely cheap.",
        steps=(
            "Open the link and sign in",
            "Create an API key, copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "mistral": Preset(
        key="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        models=("mistral-large-latest", "mistral-small-latest"),
        auth="key",
        signup_url="https://console.mistral.ai/api-keys",
        free=True,
        steps=(
            "Open the link and sign in",
            "Create a key, copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "together": Preset(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        models=("meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"),
        auth="key",
        signup_url="https://api.together.xyz/settings/api-keys",
        steps=(
            "Open the link and sign in",
            "Copy your API key",
            "Send it back with:  /paste <key>",
        ),
    ),
    "xai": Preset(
        key="xai",
        label="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        models=("grok-4", "grok-3-mini"),
        auth="key",
        signup_url="https://console.x.ai",
        key_prefix="xai-",
        steps=(
            "Open the link and sign in",
            "Create an API key, copy it",
            "Send it back with:  /paste <key>",
        ),
    ),
    "github": Preset(
        key="github",
        label="GitHub Models (free with a GitHub account)",
        base_url="https://models.github.ai/inference",
        models=("openai/gpt-4o-mini", "openai/gpt-4o", "meta/Llama-3.3-70B-Instruct"),
        auth="key",
        signup_url="https://github.com/settings/tokens",
        key_prefix="gh",
        free=True,
        notes="Free tier for personal use; uses a GitHub token.",
        steps=(
            "Open the link, choose 'Generate new token (classic)'",
            "Tick the 'models' scope, generate, copy it",
            "Send it back with:  /paste <token>",
        ),
    ),
}

# Order shown in /llm
DISPLAY_ORDER = (
    "omniroute", "anthropic", "mwapi", "openai", "openrouter", "groq",
    "gemini", "deepseek", "github", "mistral", "together", "xai", "ollama",
)

ALIASES = {
    "claude": "anthropic",
    "chatgpt": "openai",
    "gpt": "openai",
    "google": "gemini",
    "grok": "xai",
    "local": "ollama",
    "gateway": "omniroute",
    "free": "omniroute",
    "router": "openrouter",
}


def resolve(name: str) -> Preset | None:
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    return PRESETS.get(key)


def catalogue() -> list[Preset]:
    return [PRESETS[name] for name in DISPLAY_ORDER if name in PRESETS]


def validate_key(preset: Preset, value: str) -> str | None:
    """Return an error string when the pasted value is obviously wrong."""
    value = value.strip()
    if not value:
        return "empty value"
    if len(value) < 12:
        return "that looks too short to be a key"
    if " " in value:
        return "a key should not contain spaces"
    if preset.key_prefix and not value.startswith(preset.key_prefix):
        return (
            f"{preset.label} keys normally start with '{preset.key_prefix}' - "
            "double-check you copied the whole key"
        )
    return None
