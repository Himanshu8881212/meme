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


if __name__ == "__main__":
    mcp.run()
