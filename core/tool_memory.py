"""Tool-use memory — log every Model-1 tool call with its outcome so the
chat model can see what worked and what didn't next turn.

Stored as JSONL (append-only, compact) at `vault/_meta/tool_log.jsonl`.
Injected into the system prompt at session start so the model walks into
every turn with a short recent-history of its own tool-use patterns.

Privacy-sane: we only log tool name + truncated args + an outcome tag +
result length. The full response body stays in-conversation, not in the
log file.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_FILE = "tool_log.jsonl"
MAX_ARG_CHARS = 160         # truncate long args so the log stays small
MAX_TAIL = 15               # how many recent entries to show the model
MAX_LINES_IN_PROMPT = 12    # after outcome filtering, hard cap


def _path(vault: str | Path) -> Path:
    return Path(vault) / "_meta" / LOG_FILE


def _classify(result: str) -> tuple[str, int]:
    """Return (outcome_tag, result_len). Heuristic:
      - starts with `(no matches)`/`(no results)`/`(none)` → empty
      - starts with `error:`/`(...error...)` → error
      - otherwise → success
    """
    r = (result or "").strip()
    n = len(r)
    low = r.lower()
    if n == 0:
        return "empty", 0
    if low.startswith(("(no matches", "(no results", "(no ", "(empty", "(nothing")):
        return "empty", n
    if low.startswith("error:") or "error:" in low[:40]:
        return "error", n
    if "unavailable" in low[:60] or "not configured" in low[:60]:
        return "error", n
    return "success", n


def _truncate_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > MAX_ARG_CHARS:
            out[k] = v[:MAX_ARG_CHARS] + "…"
        elif isinstance(v, list) and len(v) > 6:
            out[k] = v[:6] + [f"… +{len(v) - 6} more"]
        else:
            out[k] = v
    return out


def log_call(vault: str | Path, tool: str, args: dict[str, Any], result: str) -> None:
    """Record one tool invocation. Silent on IO failure — we never want
    logging to break a chat turn."""
    try:
        outcome, n = _classify(result)
        entry = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": tool,
            "args": _truncate_args(args),
            "outcome": outcome,
            "result_len": n,
        }
        path = _path(vault)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def recent_calls(vault: str | Path, limit: int = MAX_TAIL) -> list[dict[str, Any]]:
    path = _path(vault)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def prompt_summary(vault: str | Path) -> str:
    """Format the tail of the log for injection into the system prompt.
    Returns '' if empty — caller should only include the section header
    when there's actual content."""
    calls = recent_calls(vault, limit=MAX_TAIL)
    if not calls:
        return ""
    # Show the most-recent-first, capped. Skip routine memory_search logs
    # that succeed loudly — they're not instructive for pattern learning.
    lines: list[str] = []
    for c in reversed(calls[-MAX_LINES_IN_PROMPT:]):
        hm = c.get("ts", "")[-14:-3]  # "HH:MM" roughly
        args = c.get("args") or {}
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        if len(args_str) > 90:
            args_str = args_str[:90] + "…"
        outcome = c.get("outcome", "?")
        badge = {"success": "✓", "empty": "∅", "error": "✗"}.get(outcome, "·")
        lines.append(f"- {hm}  {badge}  {c.get('tool','?')}({args_str})")
    return "\n".join(lines)


# ── pattern synthesis ────────────────────────────────────────────────────
#
# Raw history gets the model short-term memory within a session. For real
# long-term learning we need to promote repeated patterns into durable
# `procedures/` nodes: notes that live in the vault, get retrieved normally,
# and survive log truncation. This is done on demand via `/tools_audit`
# and automatically inside `/meta` deep reflection.

AUDIT_MIN_CALLS = 15  # below this, not enough signal to synthesize


def format_for_audit(vault: str | Path, limit: int = 100) -> str:
    """Compact log dump, newest first, for the synthesis prompt."""
    calls = recent_calls(vault, limit=limit)
    if not calls:
        return "(no tool calls logged)"
    lines: list[str] = []
    for c in reversed(calls):
        args = c.get("args") or {}
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 140:
            args_str = args_str[:140] + "…"
        lines.append(
            f"{c.get('ts','?')[11:19]}  "
            f"{c.get('outcome','?'):<7}  "
            f"{c.get('tool','?')}  {args_str}  "
            f"(→ {c.get('result_len', 0)} chars)"
        )
    return "\n".join(lines)


AUDIT_PROMPT = """You are auditing an agent's recent tool-use log. Your job
is to identify *repeated* patterns — things the agent did 3+ times with
consistent outcomes — and write them as procedure notes so the agent
remembers what worked and what didn't.

Write ONE `<<WRITE path="procedures/<name>.md" action="create">>` block
per pattern you see. Name the file after the WHAT-IT-DOES, not the tool
(e.g. `procedures/Looking up user's grocery list.md`, not `procedures/
obsidian_list.md`). Use YAML frontmatter with `type: procedure` and
`tags: [tool-use, <topic>]`.

Body should be tight and actionable. Structure:

```
When the user asks about X, the reliable pattern is:

  TOOL(args=...)  →  SUCCESS

Avoid `OTHER_TOOL(args=...)` — it returned empty N times in a row.
```

Hard rules:
- ONLY write a procedure if you see 3+ calls of the same shape (tool+arg
  pattern) with the same outcome. Single occurrences are noise.
- DO NOT invent tool names or patterns not in the log.
- DO NOT write a procedure for trivia — focus on patterns that cost time
  or fail silently.
- If there are no patterns worth writing, output exactly `(no patterns)`
  and nothing else. Silence is valid.
- Short procedures. Under 150 words each.

Here is the log (newest first):
"""


def audit_and_synthesize(
    vault: str | Path, config: dict[str, Any], role: str = "deep",
) -> dict[str, Any]:
    """Run the synthesis pass. Returns {'writes': [...], 'calls_seen': N,
    'skipped': reason_or_None}."""
    calls = recent_calls(vault, limit=200)
    if len(calls) < AUDIT_MIN_CALLS:
        return {
            "writes": [], "calls_seen": len(calls),
            "skipped": f"need {AUDIT_MIN_CALLS}+ calls, have {len(calls)}",
        }
    # Local import to avoid a circular dep at module load time.
    from core import reflection as _r
    log_block = format_for_audit(vault, limit=200)
    try:
        raw = _r.chat(
            role=role, system=AUDIT_PROMPT,
            messages=[{"role": "user", "content": log_block}],
            config=config, max_tokens=3072,
        )
    except Exception as exc:
        return {"writes": [], "calls_seen": len(calls), "skipped": f"model error: {exc}"}
    if "(no patterns)" in raw.lower() and "<<WRITE" not in raw:
        return {"writes": [], "calls_seen": len(calls), "skipped": "model saw no patterns"}
    writes = _r.apply_writes(
        raw, Path(vault),
        similarity_threshold=float(
            (config.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
        ),
        reconcile=False,
    )
    return {"writes": writes, "calls_seen": len(calls), "skipped": None}
