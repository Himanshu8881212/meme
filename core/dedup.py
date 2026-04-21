"""Near-duplicate detection across the vault.

Decay + tension reconciliation handle *stale* and *contradictory* content —
neither catches semantic twins where the scribe created parallel nodes
("Puppet" / "my dog" / "black lab at home") that should have been one.

Strategy: for every pair of same-type nodes, compute Jaccard similarity on
title tokens and body tokens (stopword-filtered). Flag pairs above a
threshold so the deep auditor can decide to merge. Deliberately conservative
— better to surface a candidate and let the auditor choose than to auto-merge
and destroy nuance.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "i", "me", "my", "we", "us",
    "our", "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "when",
    "where", "why", "how", "not", "no", "so", "if", "then", "than",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def find_duplicate_candidates(
    vault_path: str | Path,
    title_threshold: float = 0.6,
    body_threshold: float = 0.5,
    min_body_tokens: int = 3,
) -> list[dict[str, Any]]:
    """Return pairs of same-type nodes that look like near-duplicates.

    Args:
        title_threshold: minimum title Jaccard to flag regardless of body.
        body_threshold: minimum body Jaccard (combined with any title overlap)
            to flag even when titles differ.
        min_body_tokens: skip nodes whose body is too short to trust.

    Returns:
        List of {a, b, type, title_sim, body_sim, reason} dicts sorted by
        max(title_sim, body_sim) descending.
    """
    vault = Path(vault_path)
    idx = indexer.build(vault)

    # Collect (name, type, title_tokens, body_tokens) for every eligible node.
    bucket: dict[str, list[tuple[str, set[str], set[str]]]] = {}
    for name, meta in idx.items():
        ntype = meta.get("type")
        if not ntype or ntype in ("transcript", "identity"):
            continue
        path = Path(meta["path"])
        if not path.exists():
            continue
        fm, body = frontmatter.read(path)
        if fm.get("archived") or fm.get("immutable"):
            continue
        t_tokens = _tokens(name)
        b_tokens = _tokens(body)
        if len(b_tokens) < min_body_tokens:
            continue
        bucket.setdefault(ntype, []).append((name, t_tokens, b_tokens))

    candidates: list[dict[str, Any]] = []
    for ntype, items in bucket.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                n1, t1, b1 = items[i]
                n2, t2, b2 = items[j]
                title_sim = _jaccard(t1, t2)
                body_sim = _jaccard(b1, b2)
                reason: str | None = None
                if title_sim >= title_threshold:
                    reason = f"title overlap ({title_sim:.2f})"
                elif body_sim >= body_threshold:
                    reason = f"body overlap ({body_sim:.2f})"
                if reason:
                    candidates.append({
                        "a": n1, "b": n2, "type": ntype,
                        "title_sim": round(title_sim, 3),
                        "body_sim": round(body_sim, 3),
                        "reason": reason,
                    })

    candidates.sort(key=lambda c: max(c["title_sim"], c["body_sim"]), reverse=True)
    return candidates


def summarize(candidates: list[dict[str, Any]], limit: int = 10) -> str:
    """Format duplicate candidates into a human/model-readable block."""
    if not candidates:
        return "(no duplicate candidates)"
    lines = [f"found {len(candidates)} duplicate candidate(s) (showing up to {limit}):"]
    for c in candidates[:limit]:
        lines.append(
            f"  · [{c['type']}] {c['a']}  ↔  {c['b']}  "
            f"(title {c['title_sim']}, body {c['body_sim']}, {c['reason']})"
        )
    return "\n".join(lines)
