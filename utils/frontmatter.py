from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read(path: str | Path) -> tuple[dict[str, Any], str]:
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    # Tolerant to bad YAML — a single malformed file shouldn't break every
    # retrieval that follows. Return an empty frontmatter; the body survives.
    try:
        fm = yaml.safe_load(parts[1]) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    body = parts[2]
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


def write(path: str | Path, fm: dict[str, Any], body: str) -> None:
    dumped = yaml.dump(fm, sort_keys=False, allow_unicode=True)
    if not body.startswith("\n"):
        body = "\n" + body
    content = f"---\n{dumped}---{body}"
    Path(path).write_text(content, encoding="utf-8")


def update(path: str | Path, updates: dict[str, Any]) -> None:
    fm, body = read(path)
    fm.update(updates)
    write(path, fm, body)
