"""Tool registry.

Tools register themselves at import time via the ``@tool`` decorator; the
registry is the single source of truth for what the LLM is allowed to call.
"""

from __future__ import annotations

from typing import Any, Callable

from app.security import Permission
from app.tools.base import Arg, Handler, Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self, *, max_permission: Permission | None = None) -> list[Tool]:
        tools = sorted(self._tools.values(), key=lambda t: t.name)
        if max_permission is not None:
            tools = [t for t in tools if t.permission.rank <= max_permission.rank]
        return tools

    def specs(self, *, max_permission: Permission | None = None) -> list[dict[str, Any]]:
        return [t.spec() for t in self.all(max_permission=max_permission)]

    def clear(self) -> None:
        self._tools.clear()


registry = ToolRegistry()


def tool(
    name: str,
    *,
    description: str,
    args: dict[str, Arg] | None = None,
    permission: Permission = Permission.READ,
    timeout_s: int = 120,
    max_retries: int = 2,
    side_effect: bool = False,
    verify: Callable[[dict[str, Any]], bool] | None = None,
    enabled: bool = True,
) -> Callable[[Handler], Handler]:
    """Decorator registering a coroutine/function as an agent tool."""

    def decorator(func: Handler) -> Handler:
        if enabled:
            registry.register(
                Tool(
                    name=name,
                    description=description,
                    args=args or {},
                    permission=permission,
                    handler=func,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    side_effect=side_effect,
                    verify=verify,
                )
            )
        return func

    return decorator
