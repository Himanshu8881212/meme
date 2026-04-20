from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import frontmatter, wikilinks

# Folders whose contents are NEVER indexed. Archived content and any other
# quarantine zone stays out of retrieval, backlink counts, monitor metrics.
HIDDEN_FOLDERS = {"_archive"}


def _is_hidden(path: Path, vault: Path) -> bool:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        return True
    return any(part in HIDDEN_FOLDERS for part in rel.parts)


def _safe_float(value: Any, default: float) -> float:
    """Tolerant float coercion. YAML sometimes parses a bare `2026-04-20`-style
    value into a datetime.date, which chokes float(). Fall back to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build(vault_path: str | Path) -> dict[str, dict[str, Any]]:
    vault = Path(vault_path)
    index: dict[str, dict[str, Any]] = {}

    for path in vault.rglob("*.md"):
        if path.name.startswith("_"):
            continue
        if _is_hidden(path, vault):
            continue
        fm, _ = frontmatter.read(path)
        name = path.stem
        index[name] = {
            "path": str(path),
            "type": fm.get("type"),
            "tags": fm.get("tags", []) or [],
            "decay_weight": _safe_float(fm.get("decay_weight", 0.5), 0.5),
            "last_accessed": fm.get("last_accessed"),
            "access_count": _safe_int(fm.get("access_count", 0), 0),
            "importance": _safe_float(fm.get("importance", 0.5), 0.5),
            "connection_count": _safe_int(fm.get("connection_count", 0), 0),
            "archived": bool(fm.get("archived", False)),
        }
    return index


def backlink_counts(vault_path: str | Path) -> dict[str, int]:
    vault = Path(vault_path)
    counts: dict[str, int] = {}
    for path in vault.rglob("*.md"):
        if _is_hidden(path, vault):
            continue
        _, body = frontmatter.read(path)
        for target in wikilinks.extract(body):
            counts[target] = counts.get(target, 0) + 1
    return counts
