"""Environment based configuration.

Every knob of the system lives here.  Nothing else in the codebase is allowed
to read os.environ directly, so the whole configuration surface is auditable in
one file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- general ----------------------------------------------------------
    app_env: str = "dev"
    log_level: str = "INFO"

    # --- http api ---------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_token: str = ""

    # --- database ---------------------------------------------------------
    database_url: str = "postgresql+asyncpg://agent:agent@postgres:5432/agent"
    db_echo: bool = False

    # --- telegram ---------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""
    telegram_owner_chat_id: int | None = None
    telegram_api_base: str = "https://api.telegram.org"
    telegram_rate_limit_per_min: int = 30

    # --- llm --------------------------------------------------------------
    # ollama | openai | anthropic | echo | <custom name>
    llm_provider: str = "ollama"
    llm_timeout_s: int = 300
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_fallback_enabled: bool = True

    # local (Ollama)
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"

    # ChatGPT / OpenAI-compatible
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Claude / Anthropic
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-4-5"

    # OmniRoute: self-hosted OpenAI-compatible gateway that fans out to many
    # providers (incl. free tiers). Default when no personal API key is set.
    omniroute_enabled: bool = True
    omniroute_base_url: str = "http://omniroute:20128/v1"
    omniroute_api_key: str = ""
    omniroute_model: str = "auto"

    # Any other OpenAI-compatible endpoint (OpenRouter, Groq, DeepSeek, vLLM...)
    custom_llm_name: str = ""
    custom_llm_api_key: str = ""
    custom_llm_base_url: str = ""
    custom_llm_model: str = ""

    # --- workspace --------------------------------------------------------
    workspace_dir: str = "/data"
    max_file_mb: int = 45

    # --- engine / workers -------------------------------------------------
    max_task_steps: int = 14
    task_timeout_s: int = 3600
    worker_concurrency: int = 2
    worker_poll_interval_s: float = 2.0
    scheduler_poll_interval_s: float = 10.0
    max_task_retries: int = 2
    tool_default_timeout_s: int = 120

    # --- safety -----------------------------------------------------------
    require_approval_high_risk: bool = True
    # balanced: ask only about irreversible, unrequested actions (default)
    # high:     act without confirmation (still refuses protected paths)
    # paranoid: confirm every risky action
    autonomy_level: str = "balanced"
    shell_allowlist: str = (
        "ls,pwd,cat,head,tail,wc,grep,find,df,du,file,sha256sum,stat,python,python3,git"
    )
    enable_browser_tools: bool = False
    enable_python_tool: bool = True
    enable_shell_tool: bool = True

    # --- integrations: whatsapp + teams bridges (Go services) -------------
    whatsapp_enabled: bool = False
    whatsapp_bridge_url: str = "http://whatsapp-bridge:8081"
    teams_enabled: bool = False
    teams_bridge_url: str = "http://teams-bridge:8082"
    bridge_token: str = ""
    bridge_timeout_s: int = 120
    inbound_auto_task: bool = False
    inbound_allowed_senders: str = ""

    # --- voice ------------------------------------------------------------
    voice_enabled: bool = True
    whisper_base_url: str = "https://api.openai.com/v1"
    whisper_api_key: str = ""
    whisper_model: str = "whisper-1"
    whisper_local_model: str = ""          # e.g. "base" to use faster-whisper

    # --- email ------------------------------------------------------------
    email_enabled: bool = True
    email_poll_enabled: bool = False       # background inbox watching
    email_poll_interval_s: int = 300

    # --- web ---------------------------------------------------------------
    web_search_enabled: bool = True
    http_tools_enabled: bool = True
    http_user_agent: str = "personal-ai-agent/1.0"

    # --- documents ---------------------------------------------------------
    document_tools_enabled: bool = True

    # --- daily briefing -----------------------------------------------------
    briefing_enabled: bool = False
    briefing_cron: str = "0 8 * * *"       # 08:00 every day, owner's timezone
    briefing_period_hours: int = 24

    # --- reasoning quality ------------------------------------------------
    planning_enabled: bool = True
    verify_completion: bool = True

    # --- self-learning ----------------------------------------------------
    learning_enabled: bool = True
    telegram_user_enabled: bool = True

    # --- runtime toggles (mostly for tests) -------------------------------
    start_telegram_bot: bool = True
    start_worker: bool = True
    start_scheduler: bool = True

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("telegram_owner_chat_id", mode="before")
    @classmethod
    def _empty_chat_id_is_none(cls, value):  # noqa: ANN001, ANN206
        """An unset TELEGRAM_OWNER_CHAT_ID= line must not crash startup."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value):  # noqa: ANN001, ANN206
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        text = str(value or "INFO").upper().strip()
        return text if text in allowed else "INFO"

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def allowed_user_ids(self) -> set[int]:
        out: set[int] = set()
        for part in _split_csv(self.telegram_allowed_user_ids):
            try:
                out.add(int(part))
            except ValueError:
                continue
        return out

    @property
    def owner_chat_id(self) -> int | None:
        if self.telegram_owner_chat_id:
            return self.telegram_owner_chat_id
        ids = sorted(self.allowed_user_ids)
        return ids[0] if ids else None

    @property
    def shell_allowed_commands(self) -> set[str]:
        return set(_split_csv(self.shell_allowlist))

    @property
    def workspace(self) -> Path:
        return Path(self.workspace_dir).resolve()

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.allowed_user_ids)

    @property
    def inbound_sender_allowlist(self) -> set[str]:
        return {p.lower() for p in _split_csv(self.inbound_allowed_senders)}

    def sender_allowed(self, sender: str) -> bool:
        allowlist = self.inbound_sender_allowlist
        if not allowlist:
            return False
        normalised = (sender or "").lower()
        digits = "".join(c for c in normalised if c.isdigit())
        for entry in allowlist:
            if entry == "*":
                return True
            if entry == normalised:
                return True
            entry_digits = "".join(c for c in entry if c.isdigit())
            if entry_digits and digits and (digits.endswith(entry_digits) or entry_digits.endswith(digits)):
                return True
        return False

    def workspace_subdirs(self) -> list[Path]:
        base = self.workspace
        return [
            base / "downloads",
            base / "uploads",
            base / "tasks",
            base / "temp",
            base / "output",
        ]

    def ensure_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        for path in self.workspace_subdirs():
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache (used by tests after monkeypatching the environment)."""
    get_settings.cache_clear()
    return get_settings()
