"""MCP server — exposes the memory vault to any MCP-compatible chat client.

Tools:
  memory_search(query, tags?, limit?)  → top retrieved nodes as concatenated markdown
  memory_read(name)                    → full body of a specific node
  memory_reflect(transcript, task?)    → run routine reflection over a transcript
  memory_stats()                       → vault counts, hubs, triggers
  memory_list_tags()                   → canonical tag vocabulary

Launch:
  python mcp_server.py

Plug into Claude Desktop / Cursor / Zed via that stdio command.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import flagging, monitor, reflection, retrieval  # noqa: E402
from utils import env, indexer  # noqa: E402

env.load_dotenv(ROOT / ".env")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.stderr.write("mcp SDK not installed: `pip install mcp`\n")
    raise

CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VAULT = Path(CONFIG["vault_path"])
if not VAULT.is_absolute():
    VAULT = (ROOT / VAULT).resolve()

mcp = FastMCP(CONFIG["mcp"]["server_name"])


@mcp.tool()
def memory_search(query: str, tags: list[str] | None = None, limit: int | None = None) -> str:
    """Search the memory vault and return the most relevant nodes as markdown.

    Uses frontmatter scoring (tags, title, decay weight, recency) and wikilink
    graph expansion — no embeddings. Call this before answering the user to
    ground responses in the system's persistent memory.
    """
    files = retrieval.retrieve(VAULT, query, tags or [], CONFIG)
    cap = limit or CONFIG["mcp"]["default_search_limit"]
    files = files[:cap]
    if not files:
        return "(no matching nodes — the vault may be empty or unrelated)"
    return "\n\n".join(f"### {Path(p).relative_to(VAULT)}\n{c}" for p, c in files)


@mcp.tool()
def memory_read(name: str) -> str:
    """Read a specific vault node by its name (e.g. 'Memory System')."""
    idx = indexer.build(VAULT)
    if name not in idx:
        matches = [n for n in idx if name.lower() in n.lower()]
        if not matches:
            return f"(no node matching '{name}')"
        name = matches[0]
    return Path(idx[name]["path"]).read_text(encoding="utf-8")


@mcp.tool()
def memory_reflect(transcript: str, task: str = "") -> str:
    """Run the routine reflection pass over a transcript and write memory.

    Extracts inline flags (`[NOVEL: ...]`, `[SALIENT: ...]`, etc.) from the
    transcript, retrieves related vault context against the flag summary or
    the supplied task, invokes the routine reflection model, and applies any
    <<WRITE>> blocks it produces to the vault. Returns a summary of writes.
    """
    flags = flagging.extract(transcript)
    summary = flagging.summarize(flags)
    query = summary if flags else task
    files = retrieval.retrieve(VAULT, query, [], CONFIG)
    output = reflection.routine(
        flag_summary=summary,
        vault_files=files,
        session_notes=transcript[:4000],
        config=CONFIG,
    )
    writes = reflection.apply_writes(
        output, VAULT,
        similarity_threshold=float(
            (CONFIG.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
        ),
    )
    return (
        f"flags: {len(flags)}\n"
        f"writes: {len(writes)}\n"
        + "\n".join(f"  {w.get('action')}: {w.get('path')}" for w in writes)
    )


@mcp.tool()
def memory_stats() -> str:
    """Return vault counts, top hubs, and any monitor triggers."""
    metrics = monitor.collect(VAULT)
    triggers = monitor.check_thresholds(metrics, CONFIG)
    lines = [
        f"nodes: {metrics['total_nodes']}",
        f"archived: {metrics['archived']} ({metrics['archived_ratio']:.1%})",
        f"orphans: {metrics['orphans']} ({metrics['orphan_ratio']:.1%})",
        f"tags: {metrics['tag_vocabulary']}",
        f"avg decay weight: {metrics['avg_decay_weight']:.3f}",
        "top hubs:",
    ]
    for name, count in metrics["top_hubs"][:5]:
        lines.append(f"  {name} ({count})")
    if triggers:
        lines.append("triggers:")
        lines.extend(f"  - {t}" for t in triggers)
    return "\n".join(lines)


@mcp.tool()
def memory_list_transcripts(limit: int = 20) -> str:
    """List recent verbatim session transcripts.

    Transcripts are immutable, lossless records of what was actually said.
    Retrieval normally skips them (they're noise for distilled queries) but
    you can read one directly when you need to know exactly what was said —
    e.g. to resolve a contradiction by going back to source.
    """
    tdir = VAULT / "_transcripts"
    if not tdir.exists():
        return "(no transcripts yet)"
    files = sorted(tdir.glob("*.md"), reverse=True)[:limit]
    if not files:
        return "(no transcripts yet)"
    return "\n".join(p.stem for p in files)


@mcp.tool()
def memory_read_transcript(name: str) -> str:
    """Read a session transcript verbatim.

    Use `memory_list_transcripts` to discover names. Transcripts contain the
    raw USER/ASSISTANT dialogue that was distilled into vault nodes —
    invaluable when the distilled summary doesn't answer the question and
    you need to see the original exchange.
    """
    tdir = VAULT / "_transcripts"
    path = tdir / f"{name}.md"
    if not path.exists():
        matches = list(tdir.glob(f"*{name}*.md"))
        if not matches:
            return f"(no transcript matching '{name}')"
        path = matches[0]
    return path.read_text(encoding="utf-8")


@mcp.tool()
def memory_grep(phrase: str, limit: int = 10) -> str:
    """Case-insensitive exact-phrase search across all memory (distilled
    nodes + verbatim transcripts). Use for 'what did I say about X' or
    'find the line where Y came up'. Returns path:line: snippet per hit."""
    from core.tools import grep_vault
    hits = grep_vault(VAULT, phrase, limit=limit)
    if not hits:
        return "(no matches)"
    return "\n".join(f"{h['path']}:{h['line_no']}: {h['snippet']}" for h in hits)


@mcp.tool()
def memory_transcripts_by_date(start_date: str, end_date: str | None = None) -> str:
    """List transcripts in a date range. YYYY-MM-DD format. If end_date is
    omitted, returns transcripts from that single day. Pair with
    memory_read_transcript to fetch the full text."""
    from core.tools import transcripts_by_date
    names = transcripts_by_date(VAULT, start_date, end_date)
    if not names:
        return "(no transcripts in range)"
    return "\n".join(names)


@mcp.tool()
def memory_list_tags() -> str:
    """List the canonical tag vocabulary from the vault's tag registry."""
    idx = indexer.build(VAULT)
    tags: set[str] = set()
    for node in idx.values():
        for t in node["tags"]:
            tags.add(str(t))
    return ", ".join(sorted(tags))


# ── RLM-lite: aggregate enumeration + recursive summarization ─────────────

@mcp.tool()
def memory_list(tag: str | None = None, type: str | None = None, limit: int = 100) -> str:
    """Metadata-only enumeration of vault nodes — for 'all of X' / 'everything
    about Y' queries. Returns name + type + tags + importance per node, no
    bodies. Pair with `memory_summarize` for vault-scale answers."""
    from core.reflection import _model1_tool_dispatch
    args = {"limit": limit}
    if tag: args["tag"] = tag
    if type: args["type"] = type
    return _model1_tool_dispatch(VAULT, "memory_list", args, CONFIG)


@mcp.tool()
def memory_summarize(names: list[str], query: str) -> str:
    """Recursive distillation — feed up to 30 node names + a focus question;
    a sub-LM reads each node and returns one tight paragraph."""
    from core.reflection import _model1_tool_dispatch
    return _model1_tool_dispatch(
        VAULT, "memory_summarize",
        {"names": names, "query": query}, CONFIG,
    )


# ── Obsidian (user's external notebook) — only active if configured ──────

def _ext_ok() -> bool:
    from core import obsidian as _ob
    return _ob.resolve_vault_path(CONFIG) is not None


@mcp.tool()
def obsidian_create(rel_path: str, body: str, frontmatter: dict | None = None) -> str:
    """Create a new note in the user's external Obsidian vault. Writes to
    THEIR notebook, not the memory vault. Search/list first to avoid
    duplicates."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    r = _ob.create_note(
        _ob.resolve_vault_path(CONFIG), rel_path, body,
        frontmatter=frontmatter, config=CONFIG,
    )
    return r.get("preview") if r.get("ok") else f"error: {r.get('error')}"


@mcp.tool()
def obsidian_update(rel_path: str, body: str, mode: str = "replace") -> str:
    """Edit an existing note in the user's Obsidian vault.
    mode: 'replace' | 'append' | 'prepend'."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    r = _ob.update_note(
        _ob.resolve_vault_path(CONFIG), rel_path, body,
        mode=mode, config=CONFIG,
    )
    return r.get("preview") if r.get("ok") else f"error: {r.get('error')}"


@mcp.tool()
def obsidian_read(rel_path: str) -> str:
    """Read a note from the user's Obsidian vault."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    return _ob.read_note(_ob.resolve_vault_path(CONFIG), rel_path)


@mcp.tool()
def obsidian_search(query: str, limit: int = 10) -> str:
    """Phrase search across the user's Obsidian vault. Returns path, line,
    snippet per hit. Extract the core concept word from the user's question
    before searching — don't search the whole question verbatim."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    hits = _ob.search_notes(_ob.resolve_vault_path(CONFIG), query, limit=limit)
    if not hits:
        return "(no matches)"
    return "\n".join(f"{h['path']}:{h['line_no']}: {h['snippet']}" for h in hits)


@mcp.tool()
def obsidian_list(folder: str | None = None, limit: int = 50) -> str:
    """List notes in the user's Obsidian vault — all of them, or one folder."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    names = _ob.list_notes(_ob.resolve_vault_path(CONFIG), folder, limit=limit)
    return "\n".join(names) if names else "(empty)"


@mcp.tool()
def obsidian_link(rel_path: str, target: str, label: str | None = None) -> str:
    """Append a [[wikilink]] to an existing note in the user's Obsidian vault."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    r = _ob.add_wikilink(
        _ob.resolve_vault_path(CONFIG), rel_path, target,
        label=label, config=CONFIG,
    )
    return r.get("preview") if r.get("ok") else f"error: {r.get('error')}"


@mcp.tool()
def obsidian_rename(old_rel: str, new_rel: str) -> str:
    """Rename / move a note. Updates every incoming wikilink across the vault."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    r = _ob.rename_note(
        _ob.resolve_vault_path(CONFIG), old_rel, new_rel, config=CONFIG,
    )
    return r.get("preview") if r.get("ok") else f"error: {r.get('error')}"


@mcp.tool()
def obsidian_delete(rel_path: str) -> str:
    """Soft-delete a note (moves to _trash/, reversible)."""
    if not _ext_ok():
        return "(external Obsidian vault not configured)"
    from core import obsidian as _ob
    r = _ob.delete_note(_ob.resolve_vault_path(CONFIG), rel_path, config=CONFIG)
    return r.get("preview") if r.get("ok") else f"error: {r.get('error')}"


# ── Utility (free web search, persistent reminders, clock) ───────────────

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Free DuckDuckGo search. For current info beyond the model's cutoff —
    news, prices, live facts. Returns up to `max_results` title/URL/snippet."""
    from core.reflection import _model1_tool_dispatch
    return _model1_tool_dispatch(
        VAULT, "web_search",
        {"query": query, "max_results": max_results}, CONFIG,
    )


@mcp.tool()
def current_time() -> str:
    """Current wall-clock time + date in the user's local timezone."""
    from core.reflection import _model1_tool_dispatch
    return _model1_tool_dispatch(VAULT, "current_time", {}, CONFIG)


@mcp.tool()
def schedule_reminder(message: str, cron: str | None = None, once_at: str | None = None) -> str:
    """Schedule a persistent reminder. Provide either `cron` (5-field
    recurring) or `once_at` (ISO datetime, one-shot). Persisted in
    vault/_meta/schedule.json — survives restarts."""
    from core.reflection import _model1_tool_dispatch
    args = {"message": message}
    if cron: args["cron"] = cron
    if once_at: args["once_at"] = once_at
    return _model1_tool_dispatch(VAULT, "schedule_reminder", args, CONFIG)


@mcp.tool()
def list_reminders() -> str:
    """Every active scheduled reminder (id, next_fire, message)."""
    from core.reflection import _model1_tool_dispatch
    return _model1_tool_dispatch(VAULT, "list_reminders", {}, CONFIG)


@mcp.tool()
def cancel_reminder(id: str) -> str:
    """Remove a scheduled reminder by id."""
    from core.reflection import _model1_tool_dispatch
    return _model1_tool_dispatch(VAULT, "cancel_reminder", {"id": id}, CONFIG)


if __name__ == "__main__":
    mcp.run()
