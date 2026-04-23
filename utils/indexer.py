from __future__ import annotations

import threading
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


# ── mtime-signature cache ──────────────────────────────────────────────
# build() is a hot path — called ~5x per chat turn (retrieval, tool
# dispatch, monitor metrics, RLM-lite tools). Previously it scanned every
# markdown file on every call. Now we cache by (vault_path, signature)
# where signature = (max mtime of any .md, file count). A signature
# change (any file touched, added, or removed) auto-invalidates.
_CACHE: dict[Path, tuple[tuple[float, int], dict[str, dict[str, Any]]]] = {}
_BACKLINK_CACHE: dict[Path, tuple[tuple[float, int], dict[str, int]]] = {}
_CACHE_LOCK = threading.Lock()


def _signature(vault: Path) -> tuple[float, int]:
    """Cheap invalidation signature. Walks all .md files once with stat()
    but skips frontmatter parsing — ~10× faster than a full build."""
    max_mtime = 0.0
    n = 0
    for p in vault.rglob("*.md"):
        if _is_hidden(p, vault):
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > max_mtime:
            max_mtime = m
        n += 1
    return (max_mtime, n)


def invalidate(vault_path: str | Path | None = None) -> None:
    """Call this after any write to force a rebuild on the next read.
    Normally not needed — the mtime signature catches writes — but
    wipes the cache after bulk operations if you want immediate effect."""
    with _CACHE_LOCK:
        if vault_path is None:
            _CACHE.clear()
            _BACKLINK_CACHE.clear()
        else:
            v = Path(vault_path).resolve()
            _CACHE.pop(v, None)
            _BACKLINK_CACHE.pop(v, None)


def build(vault_path: str | Path) -> dict[str, dict[str, Any]]:
    vault = Path(vault_path).resolve()
    sig = _signature(vault)
    with _CACHE_LOCK:
        cached = _CACHE.get(vault)
        if cached is not None and cached[0] == sig:
            return cached[1]

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
    with _CACHE_LOCK:
        _CACHE[vault] = (sig, index)
    return index


def backlink_counts(vault_path: str | Path) -> dict[str, int]:
    vault = Path(vault_path).resolve()
    sig = _signature(vault)
    with _CACHE_LOCK:
        cached = _BACKLINK_CACHE.get(vault)
        if cached is not None and cached[0] == sig:
            return cached[1]
    counts: dict[str, int] = {}
    for path in vault.rglob("*.md"):
        if _is_hidden(path, vault):
            continue
        _, body = frontmatter.read(path)
        for target in wikilinks.extract(body):
            counts[target] = counts.get(target, 0) + 1
    with _CACHE_LOCK:
        _BACKLINK_CACHE[vault] = (sig, counts)
    return counts
