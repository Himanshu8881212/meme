"""Process-wide runtime context for tools that need live handles.

Set once from the TUI's `on_mount`, read from `_model1_tool_dispatch`.
The alternative — threading `app`/`voice`/`video` through every function
signature — bloats the call site. Kept deliberately small: a dict + two
helpers. No magic, no lazy loading.

Nothing tests-facing depends on this; tests that exercise the tools pass
plain dicts directly.
"""
from __future__ import annotations

from typing import Any

_ctx: dict[str, Any] = {}


def set_context(**kwargs: Any) -> None:
    _ctx.update(kwargs)


def get(key: str, default: Any = None) -> Any:
    return _ctx.get(key, default)


def clear() -> None:
    _ctx.clear()
