"""Scheduled reminders — cron and one-shot.

Stored as plain JSON at `vault/_meta/schedule.json` so the user can see
and edit them, and so they survive restarts. A background ticker in the
TUI reads this file and fires entries whose `next_fire` has passed.

Two kinds of entries:
  - cron expr (5-field standard) → recurring, `next_fire` recomputed after firing
  - once_at (ISO datetime)       → one-shot, deactivated after firing
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from croniter import croniter
    HAS_CRON = True
except Exception:  # pragma: no cover
    HAS_CRON = False

SCHEDULE_FILE = "schedule.json"


def _path(vault: str | Path) -> Path:
    return Path(vault) / "_meta" / SCHEDULE_FILE


def load(vault: str | Path) -> list[dict[str, Any]]:
    p = _path(vault)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save(vault: str | Path, entries: list[dict[str, Any]]) -> None:
    p = _path(vault)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _now() -> datetime:
    return datetime.now().astimezone()


def _compute_next(entry: dict[str, Any], now: datetime | None = None) -> str | None:
    """Return ISO of next fire time, or None if the entry is done."""
    now = now or _now()
    if entry.get("cron"):
        if not HAS_CRON:
            return None
        try:
            ci = croniter(entry["cron"], now)
            nxt = ci.get_next(datetime)
            # croniter returns naive; attach local tz.
            if nxt.tzinfo is None:
                nxt = nxt.astimezone()
            return nxt.isoformat(timespec="seconds")
        except Exception:
            return None
    once = entry.get("once_at")
    if once:
        try:
            dt = datetime.fromisoformat(str(once))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return dt.isoformat(timespec="seconds") if dt > now else None
        except Exception:
            return None
    return None


def add(
    vault: str | Path, message: str, *,
    cron: str | None = None, once_at: str | None = None,
) -> dict[str, Any]:
    """Append a new entry. Either cron OR once_at must be supplied."""
    if not message.strip():
        return {"ok": False, "error": "empty message"}
    if not cron and not once_at:
        return {"ok": False, "error": "need cron= or once_at="}
    if cron and not HAS_CRON:
        return {"ok": False, "error": "croniter not installed"}
    if cron:
        try:
            croniter(cron, _now())  # validation
        except Exception as exc:
            return {"ok": False, "error": f"bad cron: {exc}"}
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "message": message.strip(),
        "cron": cron, "once_at": once_at,
        "created": _now().isoformat(timespec="seconds"),
        "last_fired": None,
        "active": True,
    }
    nxt = _compute_next(entry)
    if nxt is None:
        return {"ok": False, "error": "next fire time is in the past or unparseable"}
    entry["next_fire"] = nxt
    entries = load(vault)
    entries.append(entry)
    save(vault, entries)
    return {"ok": True, "entry": entry}


def remove(vault: str | Path, entry_id: str) -> bool:
    entries = load(vault)
    keep = [e for e in entries if e.get("id") != entry_id]
    if len(keep) == len(entries):
        return False
    save(vault, keep)
    return True


def active(vault: str | Path) -> list[dict[str, Any]]:
    return [e for e in load(vault) if e.get("active")]


def due(vault: str | Path, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or _now()
    out: list[dict[str, Any]] = []
    for e in load(vault):
        if not e.get("active"):
            continue
        nxt_s = e.get("next_fire")
        if not nxt_s:
            continue
        try:
            nxt = datetime.fromisoformat(nxt_s)
            if nxt.tzinfo is None:
                nxt = nxt.astimezone()
            if nxt <= now:
                out.append(e)
        except Exception:
            continue
    return out


def mark_fired(vault: str | Path, entry_id: str) -> None:
    entries = load(vault)
    now = _now()
    for e in entries:
        if e.get("id") != entry_id:
            continue
        e["last_fired"] = now.isoformat(timespec="seconds")
        # One-shots always deactivate after firing — don't reschedule them
        # even if `once_at` is still in the future (we just fired manually).
        if e.get("once_at") and not e.get("cron"):
            e["active"] = False
            e["next_fire"] = None
            continue
        nxt = _compute_next(e, now)
        if nxt is None:
            e["active"] = False
            e["next_fire"] = None
        else:
            e["next_fire"] = nxt
    save(vault, entries)
