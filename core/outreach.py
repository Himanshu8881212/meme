"""Outreach drafting + logging + context assembly.

The proactive module (core/proactive.py) picks the candidate. This module
turns a candidate into a message (via the chat model, grounded in the node
body) and writes an auditable entry to `_meta/outreach_log.md`.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from core import proactive
from utils import frontmatter


PROMPT_REL = Path(__file__).resolve().parent.parent / "prompts" / "outreach.md"


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------
def _read_persona(vault: Path) -> str:
    p = vault / "_identity" / "persona.md"
    if not p.exists():
        return ""
    try:
        _, body = frontmatter.read(p)
        return body.strip()
    except Exception:
        return ""


def _load_prompt() -> str:
    try:
        return PROMPT_REL.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def draft_message(
    candidate: dict[str, Any],
    vault: Path,
    config: dict[str, Any],
) -> str:
    """Call Model 1 with Samantha's persona + the candidate node to draft.

    Falls back to a terse, node-grounded stub if the model returns nothing
    usable — callers (TUI) should still surface something when a candidate
    was chosen.
    """
    from core import reflection as _r

    vault = Path(vault)
    persona = _read_persona(vault)
    prompt = _load_prompt()
    node_path = Path(candidate["node_path"])
    try:
        _, node_body = frontmatter.read(node_path)
    except Exception:
        node_body = ""

    system = (
        f"{persona}\n\n---\n\n{prompt}".strip()
        if (persona or prompt)
        else "You are a concise companion."
    )
    user = (
        f"Candidate node: {candidate['node_name']} "
        f"(type={candidate.get('node_type')}, score={candidate.get('score')}).\n"
        f"Why surfaced: {', '.join(candidate.get('reasons', [])) or '—'}\n\n"
        f"Node body (ground truth — reference what's actually in here):\n"
        f"{node_body.strip()[:2000]}\n\n"
        f"Draft the nudge now. 1–3 sentences. Reference the node by its "
        f"specific subject. No greetings, no sign-off."
    )

    try:
        raw = _r.chat(
            role="model1",
            system=system,
            messages=[{"role": "user", "content": user}],
            config=config,
            max_tokens=256,
        )
        text = _r.strip_thinking(raw).strip()
    except Exception as exc:
        text = f"(draft failed: {exc})"

    if not text or text.startswith("[ECHO]"):
        # Echo backend or empty reply — surface a deterministic stub so the
        # TUI still shows a grounded message in tests / offline.
        subj = candidate["node_name"]
        text = (
            f"I've been thinking about {subj}. Want to pick it back up?"
        )
    return text


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_HEADER = "# Outreach log\n\n"


def _count_today(vault: Path, now: datetime | None = None) -> int:
    now = now or datetime.now()
    today = now.date()
    entries = proactive._read_log_entries(vault)
    return sum(1 for e in entries if e["ts"].date() == today and e.get("delivered", True))


def log_outreach(
    vault: Path,
    candidate: dict[str, Any],
    message: str,
    delivered: bool,
) -> bool:
    """Append a structured entry. Returns False if the daily cap already hit
    (prevents logging a forbidden 4th surface even if a caller tries)."""
    vault = Path(vault)
    if delivered:
        cap = int(
            ((proactive._pcfg({})  # default cap if unknown — the caller
              .get("daily_cap")) or proactive.DEFAULT_DAILY_CAP)
        )
        # Real cap lives in the config; we don't want to import it here.
        # Callers that pass a config should use build_context to know.
        # This is a defensive guard only: if >= DEFAULT_DAILY_CAP already
        # delivered today, refuse.
        if _count_today(vault) >= cap:
            return False

    path = vault / proactive.OUTREACH_LOG_REL
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat(timespec="seconds")
    reasons = "; ".join(candidate.get("reasons") or []) or "-"
    # Strip newlines from the message for log single-line hygiene; full msg
    # also stored in a fenced code block for readability.
    msg_line = (message or "").strip().replace("\n", " ")[:400]
    entry_line = (
        f"- {ts} | {candidate['node_name']} | score={candidate['score']} "
        f"| delivered={'true' if delivered else 'false'} | reasons={reasons}\n"
        f"    > {msg_line}\n"
    )
    head = "" if path.exists() else _LOG_HEADER
    with path.open("a", encoding="utf-8") as f:
        f.write(head + entry_line)
    return True


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def _last_transcript_mtime(vault: Path) -> datetime | None:
    tdir = vault / "_transcripts"
    if not tdir.exists():
        return None
    latest: float = 0.0
    for p in tdir.glob("*.md"):
        try:
            m = p.stat().st_mtime
        except Exception:
            continue
        if m > latest:
            latest = m
    if latest == 0.0:
        return None
    return datetime.fromtimestamp(latest)


def _last_log_mtime(vault: Path) -> datetime | None:
    path = vault / proactive.OUTREACH_LOG_REL
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def build_context(vault: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Assemble the dict consumed by should_reach_out()."""
    vault = Path(vault)
    now = datetime.now()
    entries = proactive._read_log_entries(vault)

    today = now.date()
    outreaches_today = sum(
        1 for e in entries if e["ts"].date() == today and e.get("delivered", True)
    )

    last_outreach_ts = max(
        (e["ts"] for e in entries if e.get("delivered", True)),
        default=None,
    )
    hours_since_last_outreach: float | None = None
    if last_outreach_ts is not None:
        hours_since_last_outreach = (now - last_outreach_ts).total_seconds() / 3600.0

    t_mtime = _last_transcript_mtime(vault)
    l_mtime = _last_log_mtime(vault)
    activity_ts = max([t for t in (t_mtime, l_mtime) if t is not None], default=None)
    hours_since_last_user_activity: float | None = None
    if activity_ts is not None:
        hours_since_last_user_activity = (now - activity_ts).total_seconds() / 3600.0

    return {
        "now": now,
        "outreaches_today": outreaches_today,
        "hours_since_last_outreach": hours_since_last_outreach,
        "hours_since_last_user_activity": hours_since_last_user_activity,
    }


# ---------------------------------------------------------------------------
# Pause + mute helpers (for the TUI)
# ---------------------------------------------------------------------------
def set_pause(vault: Path, until: datetime) -> None:
    path = Path(vault) / proactive.PAUSE_FILE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"until": until.isoformat(timespec="seconds")}),
        encoding="utf-8",
    )


def clear_pause(vault: Path) -> bool:
    path = Path(vault) / proactive.PAUSE_FILE_REL
    if path.exists():
        path.unlink()
        return True
    return False


def active_pause(vault: Path) -> datetime | None:
    p = proactive._active_pause(Path(vault))
    return p["until"] if p else None


def parse_pause_arg(arg: str, now: datetime | None = None) -> datetime | None:
    """Parse '/pause' argument. Returns target datetime or None if unparseable.

    Accepted forms:
      "24h"                       → now + 24h
      "90m" / "2d"                → minute/day offset
      "until 2026-05-01"          → midnight that date
      "until tomorrow"            → midnight tomorrow
      "until 2026-05-01 15:00"    → explicit dt
    """
    now = now or datetime.now()
    s = (arg or "").strip().lower()
    if not s:
        return None
    if s.startswith("until "):
        rest = s[len("until "):].strip()
        if rest == "tomorrow":
            tmr = (now + timedelta(days=1)).date()
            return datetime.combine(tmr, datetime.min.time())
        try:
            if " " in rest:
                return datetime.fromisoformat(rest)
            d = date.fromisoformat(rest)
            return datetime.combine(d, datetime.min.time())
        except ValueError:
            return None
    # offset form: NN(h|m|d)
    if len(s) >= 2 and s[-1] in ("h", "m", "d"):
        unit = s[-1]
        try:
            n = float(s[:-1])
        except ValueError:
            return None
        if unit == "h":
            return now + timedelta(hours=n)
        if unit == "m":
            return now + timedelta(minutes=n)
        if unit == "d":
            return now + timedelta(days=n)
    return None


def set_node_proactive(vault: Path, name: str, value: bool) -> bool:
    """Flip `proactive: <value>` on a node's frontmatter. Returns True on
    success, False if node not found."""
    from utils import indexer
    idx = indexer.build(Path(vault))
    if name not in idx:
        matches = [n for n in idx if name.lower() in n.lower()]
        if not matches:
            return False
        name = matches[0]
    path = Path(idx[name]["path"])
    fm, body = frontmatter.read(path)
    fm["proactive"] = bool(value)
    frontmatter.write(path, fm, body)
    return True


# ---------------------------------------------------------------------------
# Reading the log for display
# ---------------------------------------------------------------------------
def tail_log(vault: Path, n: int = 10) -> list[dict[str, Any]]:
    entries = proactive._read_log_entries(Path(vault))
    return entries[-n:][::-1]
