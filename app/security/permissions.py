"""Security primitives: permission levels and the approval gate."""

from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    """Ordered permission levels. Higher ordinal == more dangerous."""

    READ = "READ"
    WRITE = "WRITE"
    HIGH_RISK = "HIGH_RISK"

    @property
    def rank(self) -> int:
        return {"READ": 0, "WRITE": 1, "HIGH_RISK": 2}[self.value]


class PermissionDenied(Exception):
    """Raised when a tool call exceeds the granted permission level."""


class ApprovalRequired(Exception):
    """Raised when a HIGH_RISK tool needs explicit human approval first."""

    def __init__(self, tool: str, reason: str = "high risk action requires approval") -> None:
        super().__init__(reason)
        self.tool = tool
        self.reason = reason


def check_permission(required: Permission, granted: Permission) -> None:
    """Raise :class:`PermissionDenied` when ``required`` exceeds ``granted``."""
    if required.rank > granted.rank:
        raise PermissionDenied(
            f"tool requires {required.value} but the task was granted {granted.value}"
        )
