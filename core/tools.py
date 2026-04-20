"""Deterministic tools the deep reflection model can call during /meta.

Every function here is pure computation over the vault — zero judgment, zero
model cost. The reasoning model invokes them instead of guessing at facts
like 'how many nodes have tag X?' or 'which nodes link to Y?'. This
eliminates a whole class of hallucinations.

Principle: if the answer can be computed, it MUST be computed.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer, wikilinks

# --------- Query functions (cheap, deterministic) ---------------------------


def list_nodes_by_tag(vault: Path, tag: str) -> list[str]:
    idx = indexer.build(vault)
    tag = tag.lower()
    return sorted(
        n for n, v in idx.items()
        if any(str(t).lower() == tag for t in v.get("tags", []))
    )


def count_nodes_by_tag(vault: Path, tag: str) -> int:
    return len(list_nodes_by_tag(vault, tag))


def list_nodes_by_type(vault: Path, type_: str) -> list[str]:
    idx = indexer.build(vault)
    return sorted(n for n, v in idx.items() if v.get("type") == type_)


def read_node(vault: Path, name: str) -> str:
    idx = indexer.build(vault)
    if name not in idx:
        matches = [n for n in idx if name.lower() in n.lower()]
        if not matches:
            return f"(no node matching '{name}')"
        name = matches[0]
    return Path(idx[name]["path"]).read_text(encoding="utf-8")


def backlinks_to(vault: Path, name: str) -> list[str]:
    vault = Path(vault)
    out: list[str] = []
    for path in vault.rglob("*.md"):
        if path.name.startswith("_") or path.stem == name:
            continue
        _, body = frontmatter.read(path)
        if name in wikilinks.extract(body):
            out.append(path.stem)
    return sorted(out)


def outbound_from(vault: Path, name: str) -> list[str]:
    idx = indexer.build(vault)
    if name not in idx:
        return []
    _, body = frontmatter.read(idx[name]["path"])
    return sorted(set(wikilinks.extract(body)))


def find_by_title_substring(vault: Path, query: str) -> list[str]:
    idx = indexer.build(vault)
    q = query.lower()
    return sorted(n for n in idx if q in n.lower())


def node_age_days(vault: Path, name: str) -> int:
    idx = indexer.build(vault)
    if name not in idx:
        return -1
    fm, _ = frontmatter.read(idx[name]["path"])
    created = fm.get("created")
    if not created:
        return -1
    try:
        return (date.today() - date.fromisoformat(str(created))).days
    except (ValueError, TypeError):
        return -1


def all_tags_with_counts(vault: Path) -> dict[str, int]:
    idx = indexer.build(vault)
    counts: dict[str, int] = {}
    for v in idx.values():
        for t in v.get("tags", []):
            counts[str(t)] = counts.get(str(t), 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# --------- Verbatim recall --------------------------------------------------
#
# These tools are the bridge between the synthesized graph and the lossless
# archive. When the question is *"what did I say on March 3rd"* or *"find
# the exact line where X first came up"*, the distilled layer can't answer
# — but the transcripts can.

import re as _re  # noqa: E402
from datetime import date as _date  # noqa: E402

_DATE_PREFIX = _re.compile(r"^(\d{4}-\d{2}-\d{2})")


def transcripts_by_date(
    vault: Path, start_date: str, end_date: str | None = None,
) -> list[str]:
    """List transcript names whose filename date is within [start, end].

    Date format: YYYY-MM-DD. If end_date is omitted, returns transcripts from
    that single day.
    """
    tdir = Path(vault) / "_transcripts"
    if not tdir.exists():
        return []

    try:
        start_d = _date.fromisoformat(start_date)
    except (ValueError, TypeError):
        return []
    end_d = start_d
    if end_date:
        try:
            end_d = _date.fromisoformat(end_date)
        except (ValueError, TypeError):
            pass
    if end_d < start_d:
        start_d, end_d = end_d, start_d

    out: list[str] = []
    for path in tdir.glob("*.md"):
        m = _DATE_PREFIX.match(path.stem)
        if not m:
            continue
        try:
            d = _date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if start_d <= d <= end_d:
            out.append(path.stem)
    return sorted(out)


def grep_vault(
    vault: Path,
    phrase: str,
    limit: int = 10,
    include_transcripts: bool = True,
) -> list[dict[str, Any]]:
    """Case-insensitive line-match across vault files. Returns
    [{path, line_no, snippet}, ...]. Skips _archive unconditionally."""
    if not phrase:
        return []
    vault = Path(vault)
    needle = phrase.lower()
    hits: list[dict[str, Any]] = []

    for path in vault.rglob("*.md"):
        if "_archive" in path.parts:
            continue
        if not include_transcripts and "_transcripts" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(path.relative_to(vault))
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                hits.append({
                    "path": rel,
                    "line_no": i,
                    "snippet": line.strip()[:240],
                })
                if len(hits) >= limit:
                    return hits
    return hits


# --------- OpenAI-compatible tool schemas -----------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_nodes_by_tag",
            "description": (
                "Return the names of every vault node that has the given tag. "
                "Use before consolidating tags, detecting synonyms, or auditing "
                "a cluster — never guess at counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string", "description": "The tag to filter by."}},
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_nodes_by_tag",
            "description": "Return the count of nodes carrying a tag.",
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string"}},
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_nodes_by_type",
            "description": "Return node names filtered by type (entity|concept|decision|episode|tension|question|procedure).",
            "parameters": {
                "type": "object",
                "properties": {"type": {"type": "string"}},
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_node",
            "description": "Fetch the full markdown content of a node by name. Use to verify a claim before rewriting or merging.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backlinks_to",
            "description": "List node names that contain a wikilink to the given target. Use to assess a hub before splitting.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outbound_from",
            "description": "List the wikilink targets emitted by this node's body.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_by_title_substring",
            "description": "Find nodes whose name contains the substring (case-insensitive). Use to spot duplicates before merging.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_age_days",
            "description": "Days since the node was created. Returns -1 if unknown.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "all_tags_with_counts",
            "description": "Return the full tag vocabulary with frequencies. Use to find sparse tags worth consolidating.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --------- Dispatch ---------------------------------------------------------

_DISPATCH = {
    "list_nodes_by_tag": list_nodes_by_tag,
    "count_nodes_by_tag": count_nodes_by_tag,
    "list_nodes_by_type": list_nodes_by_type,
    "read_node": read_node,
    "backlinks_to": backlinks_to,
    "outbound_from": outbound_from,
    "find_by_title_substring": find_by_title_substring,
    "node_age_days": node_age_days,
    "all_tags_with_counts": all_tags_with_counts,
    "transcripts_by_date": transcripts_by_date,
    "grep_vault": grep_vault,
}


def call(vault: Path, name: str, args: dict[str, Any]) -> Any:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(vault, **args)
    except TypeError as exc:
        return {"error": f"bad args for {name}: {exc}"}
    except Exception as exc:
        return {"error": f"{name} failed: {exc}"}
