from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer, wikilinks

ALLOWED_FOLDERS = {
    "entities", "concepts", "decisions", "episodes",
    "tensions", "questions", "procedures", "_meta", "_transcripts", "_identity",
}

HEADING_RE = re.compile(r"^#+\s.*$", re.MULTILINE)
MIN_BODY_CHARS = 20


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def collect(vault_path: str | Path) -> dict[str, Any]:
    vault = Path(vault_path)
    index = indexer.build(vault)
    backlinks = indexer.backlink_counts(vault)

    total = len(index)
    tags: set[str] = set()
    archived = 0
    orphans = 0
    decay_sum = 0.0
    hubs: list[tuple[str, int]] = []
    stale_tensions: list[tuple[str, int]] = []
    today = date.today()

    for name, node in index.items():
        for t in node["tags"]:
            tags.add(str(t))
        if node["archived"]:
            archived += 1
        if backlinks.get(name, 0) == 0:
            orphans += 1
        decay_sum += node["decay_weight"]

        bc = backlinks.get(name, 0)
        if bc > 0:
            hubs.append((name, bc))

        if node["type"] == "tension":
            created = _to_date(frontmatter.read(node["path"])[0].get("created"))
            if created:
                stale_tensions.append((name, (today - created).days))

    type_counts: dict[str, int] = {}
    for node in index.values():
        t = node["type"] or "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1

    hubs.sort(key=lambda x: x[1], reverse=True)

    return {
        "total_nodes": total,
        "archived": archived,
        "archived_ratio": archived / total if total else 0.0,
        "orphans": orphans,
        "orphan_ratio": orphans / total if total else 0.0,
        "tag_vocabulary": len(tags),
        "avg_decay_weight": decay_sum / total if total else 0.0,
        "top_hubs": hubs[:10],
        "stale_tensions": [(n, d) for n, d in stale_tensions if d > 0],
        "type_counts": type_counts,
        "date": today.isoformat(),
    }


def find_broken_nodes(
    vault_path: str | Path, min_body_chars: int = MIN_BODY_CHARS,
) -> list[dict[str, str]]:
    """Structurally broken nodes that don't deserve model attention.

    A node is broken if it has no frontmatter, no `type`, no meaningful body,
    or lives outside the allowed folders. These are almost always stubs
    auto-created by Obsidian or model slips — safe to delete algorithmically.
    """
    vault = Path(vault_path)
    broken: list[dict[str, str]] = []

    for path in vault.rglob("*.md"):
        if path.name.startswith("_"):
            continue

        rel = path.relative_to(vault)
        top = rel.parts[0] if rel.parts else ""
        if top not in ALLOWED_FOLDERS:
            broken.append({"path": str(rel), "reason": f"outside allowed folders (/{top or 'root'})"})
            continue

        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            broken.append({"path": str(rel), "reason": "no frontmatter"})
            continue

        fm, body = frontmatter.read(path)
        if not fm.get("type"):
            broken.append({"path": str(rel), "reason": "no type field"})
            continue

        stripped = HEADING_RE.sub("", body).strip()
        if len(stripped) < min_body_chars:
            broken.append({"path": str(rel), "reason": "empty body"})

    return broken


def find_orphans(vault_path: str | Path) -> list[str]:
    """Truly isolated nodes — no inbound AND no outbound wikilinks.

    These are valid-looking nodes that never wired into the graph. The deep
    reflection pass should either link them, merge them, or mark them for
    archival. Transcripts and _meta pages are excluded — they live outside
    the regular graph by design.
    """
    vault = Path(vault_path)
    index = indexer.build(vault)
    backlinks = indexer.backlink_counts(vault)

    orphans: list[str] = []
    for name, node in index.items():
        node_type = (node.get("type") or "").lower()
        # Identity, transcript, and meta nodes are not graph members by design.
        if node_type in ("transcript", "meta", "identity"):
            continue
        _, body = frontmatter.read(node["path"])
        outbound = wikilinks.extract(body)
        inbound = backlinks.get(name, 0)
        if inbound == 0 and not outbound:
            orphans.append(name)
    return orphans


def cleanup_broken(
    vault_path: str | Path, min_body_chars: int = MIN_BODY_CHARS,
) -> list[dict[str, str]]:
    """Delete structurally broken nodes. Returns a list of what was removed.

    This is the algorithmic pre-pass for deep reflection — don't waste a
    reasoning model on files that are obviously trash.
    """
    vault = Path(vault_path)
    broken = find_broken_nodes(vault, min_body_chars=min_body_chars)
    for item in broken:
        (vault / item["path"]).unlink(missing_ok=True)
    return broken


def check_thresholds(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    m = config["monitor"]
    triggers: list[str] = []

    for name, count in metrics["top_hubs"]:
        if count > m["hub_backlink_limit"]:
            triggers.append(f"hub_split:{name} ({count} backlinks)")

    if metrics["tag_vocabulary"] > m["tag_vocabulary_limit"]:
        triggers.append(f"tag_consolidation ({metrics['tag_vocabulary']} tags)")

    if metrics["orphan_ratio"] > m["orphan_ratio_limit"]:
        triggers.append(f"orphan_review ({metrics['orphan_ratio']:.1%})")

    if metrics["archived_ratio"] > m["archived_ratio_limit"]:
        triggers.append(f"archive_audit ({metrics['archived_ratio']:.1%})")

    for name, age in metrics["stale_tensions"]:
        if age > m["tension_age_days_limit"]:
            triggers.append(f"stale_tension:{name} ({age}d)")

    return triggers
