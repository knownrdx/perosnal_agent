"""Tool package: importing it registers every enabled tool."""

from __future__ import annotations

from app.tools.base import (
    Arg,
    AuthError,
    InvalidInput,
    PermanentToolError,
    RateLimited,
    TemporaryToolError,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    UserActionRequired,
)
from app.tools.registry import registry, tool

# Import order defines nothing functionally; each module self-registers.
from app.tools import browser_tools  # noqa: F401,E402
from app.tools import contact_tools  # noqa: F401,E402
from app.tools import document_tools  # noqa: F401,E402
from app.tools import email_tools  # noqa: F401,E402
from app.tools import exec_tools  # noqa: F401,E402
from app.tools import file_tools  # noqa: F401,E402
from app.tools import memory_tools  # noqa: F401,E402
from app.tools import messaging_tools  # noqa: F401,E402
from app.tools import site_tools  # noqa: F401,E402
from app.tools import task_tools  # noqa: F401,E402
from app.tools import telegram_user_tools  # noqa: F401,E402
from app.tools import telegram_tools  # noqa: F401,E402
from app.tools import web_tools  # noqa: F401,E402

__all__ = [
    "Arg",
    "AuthError",
    "InvalidInput",
    "PermanentToolError",
    "RateLimited",
    "TemporaryToolError",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "UserActionRequired",
    "registry",
    "tool",
]
