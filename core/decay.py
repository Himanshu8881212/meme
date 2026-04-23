from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer


def compute_strength(
    importance: float,
    connections: int,
    days_since_access: int,
    access_count: int,
    lambda_: float,
) -> float:
    access_divisor = math.log1p(max(access_count, 1))
    return (
        importance
        * math.log1p(max(connections, 1))
        * math.exp(-lambda_ * days_since_access / access_divisor)
    )


def _to_date(value: Any, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return default


def _safe_int(value: Any, default: int) -> int:
    """Tolerant int coercion. The LLM occasionally leaves literal
    placeholder strings like '<incremented>' in frontmatter; without
    this guard the whole decay pass crashes on one bad file."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run(vault_path: str | Path, lambda_: float, archive_threshold: float) -> dict[str, Any]:
    vault = Path(vault_path)
    today = date.today()
    backlinks = indexer.backlink_counts(vault)

    processed = 0
    archived = 0

    for path in vault.rglob("*.md"):
        if path.name.startswith("_"):
            continue
        if indexer._is_hidden(path, vault):
            continue
        fm, body = frontmatter.read(path)
        if not fm:
            continue
        # Transcripts are immutable — they are the verbatim source of truth.
        # Identity files are also exempt — they aren't graph nodes to be
        # retrieved and scored, they're always injected into the system prompt.
        if fm.get("type") in ("transcript", "identity") or fm.get("immutable"):
            continue

        # Pinned nodes skip decay entirely — for birthdays, formative
        # incidents, or any rare-but-critical fact that would otherwise be
        # archived by a 35-day half-life if not re-accessed. Their retrieval
        # weight stays at their importance (no recency penalty).
        if fm.get("pin"):
            fm["decay_weight"] = _safe_float(fm.get("importance", 1.0), 1.0)
            fm["archived"] = False
            frontmatter.write(path, fm, body)
            processed += 1
            continue

        name = path.stem
        connections = backlinks.get(name, _safe_int(fm.get("connection_count", 0), 0))
        last = _to_date(fm.get("last_accessed") or fm.get("created"), today)
        days = max((today - last).days, 0)

        strength = compute_strength(
            importance=_safe_float(fm.get("importance", 0.5), 0.5),
            connections=connections,
            days_since_access=days,
            access_count=_safe_int(fm.get("access_count", 1), 1),
            lambda_=lambda_,
        )

        fm["connection_count"] = connections
        fm["decay_weight"] = round(strength, 3)
        fm["archived"] = strength < archive_threshold

        frontmatter.write(path, fm, body)
        processed += 1
        if fm["archived"]:
            archived += 1

    return {"processed": processed, "archived": archived, "date": today.isoformat()}
