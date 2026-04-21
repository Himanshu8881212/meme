"""Proactive outreach — scoring + decision over the vault.

Pure functions over files. No I/O except reads (except for log-parsing in
build_context, which also only reads). Outreach drafting/logging lives in
core/outreach.py.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer, wikilinks

# ---------------------------------------------------------------------------
# Defaults — mirrored by config.proactive.* so overrides are easy.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "importance": 0.35,
    "urgency": 0.25,
    "salience": 0.20,
    "connection_density": 0.10,
    "freshness": 0.10,
}
DEFAULT_DAILY_CAP = 3
DEFAULT_MIN_SCORE = 0.55
DEFAULT_MIN_GAP_HOURS = 4
DEFAULT_MIN_SILENCE_HOURS = 2
DEFAULT_QUIET_HOURS = [22, 9]  # [start, end_exclusive], local clock
DEFAULT_TOP_N = 5

# Recency guard — don't re-surface a node already seen in the last N hours.
RECENT_SURFACE_HOURS = 48

OUTREACH_LOG_REL = Path("_meta") / "outreach_log.md"
PAUSE_FILE_REL = Path("_meta") / "proactive_pause.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _pcfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config or {}).get("proactive") or {}


def _weights(config: dict[str, Any]) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    w.update(_pcfg(config).get("weights") or {})
    return w


# ---------------------------------------------------------------------------
# Urgency curves
# ---------------------------------------------------------------------------
def _urgency(node_type: str | None, days: int) -> float:
    d = max(0, days)
    if node_type == "tension":
        return min(1.0, d / 14.0)
    if node_type == "question":
        return min(1.0, d / 7.0)
    if node_type == "decision":
        # bell: 0 before 3, 1.0 at 10, 0 after 45
        if d < 3 or d > 45:
            return 0.0
        if d <= 10:
            return (d - 3) / 7.0
        return max(0.0, 1.0 - (d - 10) / 35.0)
    if node_type == "episode":
        # bell: 0 before 1, 1.0 at 5, 0 after 14
        if d < 1 or d > 14:
            return 0.0
        if d <= 5:
            return (d - 1) / 4.0
        return max(0.0, 1.0 - (d - 5) / 9.0)
    return 0.0


def _freshness_penalty(days: int) -> float:
    # Reward nodes that haven't been touched recently (they're more likely
    # forgotten). 0 days → 0. 30+ days → 1.0.
    return min(1.0, max(0, days) / 30.0)


def _connection_density(connection_count: int) -> float:
    # Normalize connection count to [0, 1]. 0 conns → 0, 10+ → 1.0.
    return min(1.0, max(0, connection_count) / 10.0)


# ---------------------------------------------------------------------------
# Days-since helpers
# ---------------------------------------------------------------------------
def _node_last_activity(path: Path, fm: dict[str, Any]) -> date | None:
    """Best-effort 'when was this last touched'. Prefer last_accessed, then
    created, then filesystem mtime."""
    for key in ("last_accessed", "last_updated", "updated", "created"):
        v = fm.get(key)
        if not v:
            continue
        try:
            return date.fromisoformat(str(v)[:10])
        except (ValueError, TypeError):
            continue
    try:
        return date.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def _days_since(d: date | None) -> int:
    if d is None:
        return 0
    return max(0, (date.today() - d).days)


# ---------------------------------------------------------------------------
# Outreach log — recent-surface guard
# ---------------------------------------------------------------------------
_LOG_ENTRY_RE = re.compile(
    r"^-\s+(?P<ts>\S+)\s+\|\s+(?P<node>[^|]+?)\s+\|\s+score=(?P<score>[\d.]+)"
    r"(?:\s+\|\s+delivered=(?P<delivered>\w+))?",
    re.MULTILINE,
)


def _read_log_entries(vault: Path) -> list[dict[str, Any]]:
    path = vault / OUTREACH_LOG_REL
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for m in _LOG_ENTRY_RE.finditer(text):
        try:
            ts = datetime.fromisoformat(m.group("ts"))
        except ValueError:
            continue
        out.append({
            "ts": ts,
            "node": m.group("node").strip(),
            "score": float(m.group("score")),
            "delivered": (m.group("delivered") or "true").lower() == "true",
        })
    return out


def _recently_surfaced(vault: Path, hours: int = RECENT_SURFACE_HOURS) -> set[str]:
    cutoff = datetime.now() - timedelta(hours=hours)
    return {e["node"] for e in _read_log_entries(vault) if e["ts"] >= cutoff}


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------
_CANDIDATE_TYPES = ("tension", "question", "decision", "episode")


def _is_candidate_entity(fm: dict[str, Any]) -> bool:
    # `checkin: true` on an entity opts it into proactive surfacing.
    return bool(fm.get("checkin"))


def _backlink_counts(vault: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in vault.rglob("*.md"):
        if any(p in ("_archive",) for p in path.parts):
            continue
        try:
            _, body = frontmatter.read(path)
        except Exception:
            continue
        for target in wikilinks.extract(body):
            counts[target] = counts.get(target, 0) + 1
    return counts


def candidates(vault: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Ranked outreach candidates with provenance. Top-N by score."""
    vault = Path(vault)
    weights = _weights(config)
    top_n = int(_pcfg(config).get("top_n", DEFAULT_TOP_N))
    recent = _recently_surfaced(vault)
    backlinks = _backlink_counts(vault)
    # Pre-count outbound connections via the index (connection_count frontmatter
    # is the authoritative source, but fall back to wikilink scan).

    results: list[dict[str, Any]] = []
    idx = indexer.build(vault)

    for name, meta in idx.items():
        path = Path(meta["path"])
        try:
            fm, body = frontmatter.read(path)
        except Exception:
            continue

        if fm.get("archived"):
            continue
        # `proactive: false` — user muted this node.
        if fm.get("proactive") is False:
            continue
        # Pinned nodes are only eligible if explicitly opted-in via checkin.
        if fm.get("pin") and not fm.get("checkin"):
            continue
        if name in recent:
            continue

        ntype = fm.get("type")
        is_entity_checkin = ntype == "entity" and _is_candidate_entity(fm)
        if ntype not in _CANDIDATE_TYPES and not is_entity_checkin:
            continue

        last_act = _node_last_activity(path, fm)
        days = _days_since(last_act)

        # Per-type eligibility windows (skip candidates outside their range).
        if ntype == "tension" and days < 7:
            # let a young tension still be eligible — but cap at 0.3 urgency
            pass
        if ntype == "episode" and (days < 2 or days > 14):
            continue
        if ntype == "decision" and (days < 7 or days > 30):
            continue
        if ntype == "question" and days < 3:
            # eligible but urgency stays low
            pass

        imp = float(fm.get("importance", 0.5) or 0.5)
        sal = float(fm.get("salience", 0.5) or 0.5)
        urg = _urgency(ntype if ntype in _CANDIDATE_TYPES else "entity", days) if ntype in _CANDIDATE_TYPES else 0.7
        if is_entity_checkin:
            # check-in entities get a mild urgency ramp from the last_accessed
            # gap — the longer the silence, the more reason to surface.
            urg = min(1.0, days / 14.0) + 0.3
            urg = min(1.0, urg)

        bl = int(fm.get("connection_count", 0) or backlinks.get(name, 0))
        cd = _connection_density(bl)
        fresh = _freshness_penalty(days)

        score = (
            weights["importance"] * imp
            + weights["urgency"] * urg
            + weights["salience"] * sal
            + weights["connection_density"] * cd
            + weights["freshness"] * fresh
        )

        reasons: list[str] = []
        if ntype == "tension":
            reasons.append(f"unresolved tension, {days}d old")
        elif ntype == "question":
            reasons.append(f"open question, {days}d old")
        elif ntype == "decision":
            reasons.append(f"recent decision, {days}d old")
        elif ntype == "episode":
            reasons.append(f"significant event, {days}d old")
        if is_entity_checkin:
            reasons.append(f"entity with checkin=true, {days}d since touch")
        if imp >= 0.8:
            reasons.append(f"importance={imp:.2f}")
        if sal >= 0.7:
            reasons.append(f"salience={sal:.2f}")
        if bl >= 3:
            reasons.append(f"{bl} connections")

        results.append({
            "node_name": name,
            "node_path": str(path),
            "node_type": ntype,
            "score": round(score, 4),
            "reasons": reasons,
            "days_since_mention": days,
            "importance": imp,
            "salience": sal,
            "connection_count": bl,
        })

    results.sort(key=lambda r: -r["score"])
    return results[:top_n]


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------
def _in_quiet_hours(now: datetime, quiet: list[int]) -> bool:
    try:
        start, end = int(quiet[0]) % 24, int(quiet[1]) % 24
    except Exception:
        return False
    h = now.hour
    # Quiet window might wrap midnight (e.g. 22..9).
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end


def _active_pause(vault: Path) -> dict[str, Any] | None:
    """Read `_meta/proactive_pause.json`. Returns the parsed dict if an
    `until` timestamp is still in the future, else None."""
    import json
    path = vault / PAUSE_FILE_REL
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    until = data.get("until")
    if not until:
        return None
    try:
        until_ts = datetime.fromisoformat(str(until))
    except ValueError:
        return None
    if until_ts <= datetime.now():
        return None
    return {"until": until_ts}


def should_reach_out(
    candidates_list: list[dict[str, Any]],
    context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the top candidate, or None if any guard fires."""
    cfg = _pcfg(config)
    if not cfg.get("enabled"):
        return None
    if not candidates_list:
        return None

    min_score = float(cfg.get("min_score_threshold", DEFAULT_MIN_SCORE))
    daily_cap = int(cfg.get("daily_cap", DEFAULT_DAILY_CAP))
    min_gap = float(cfg.get("min_gap_hours", DEFAULT_MIN_GAP_HOURS))
    min_silence = float(cfg.get("min_silence_hours", DEFAULT_MIN_SILENCE_HOURS))
    quiet = cfg.get("quiet_hours") or DEFAULT_QUIET_HOURS
    now = context.get("now") or datetime.now()

    top = candidates_list[0]
    if top["score"] < min_score:
        return None
    if int(context.get("outreaches_today", 0)) >= daily_cap:
        return None
    if _in_quiet_hours(now, quiet):
        return None

    last = context.get("hours_since_last_outreach")
    if last is not None and float(last) < min_gap:
        return None

    silence = context.get("hours_since_last_user_activity")
    if silence is not None and float(silence) < min_silence:
        return None

    return top
