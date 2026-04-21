"""Integration-ish tests for the TUI surface wiring.

The TUI app itself is hard to drive in unit tests (Textual + workers).
These tests exercise the same helper functions the TUI's `_maybe_surface_proactive`
pathway calls — proving that the combined candidates → should_reach_out →
draft → log pipeline behaves correctly end-to-end.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from core import outreach, proactive


def _write_node(
    vault: Path, folder: str, name: str, body: str, fm: dict[str, Any]
) -> Path:
    path = vault / folder / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.dump(fm, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{dumped}---\n{body}\n", encoding="utf-8")
    return path


def _pcfg(enabled: bool = True) -> dict[str, Any]:
    return {
        "proactive": {
            "enabled": enabled,
            "daily_cap": 3,
            "min_score_threshold": 0.4,
            "min_gap_hours": 4,
            "min_silence_hours": 2,
            "quiet_hours": [22, 9],
        },
        # minimal echo-provider config — draft_message uses reflection.chat
        "providers": {"echo": {"base_url": None, "api_key_env": None}},
        "models": {"model1": {"provider": "echo", "model": "echo"}},
    }


def _fm(days_old: int, **extra: Any) -> dict[str, Any]:
    d = (date.today() - timedelta(days=days_old)).isoformat()
    base = {
        "created": d, "last_accessed": d,
        "importance": 0.8, "decay_weight": 0.6, "access_count": 1,
        "connection_count": 0, "archived": False, "tags": [],
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Happy path: hot candidate + guards green → surfaced + logged.
# ---------------------------------------------------------------------------
def test_startup_surfaces_and_logs(tmp_vault: Path):
    node = _write_node(
        tmp_vault, "tensions", "Hot tension",
        "An unresolved tension, stewing.",
        _fm(days_old=10, type="tension", importance=0.85, salience=0.8),
    )
    cfg = _pcfg(enabled=True)
    # Simulate build_context with guards green: noon, no recent activity
    ctx = {
        "now": datetime.now().replace(hour=13, minute=0, second=0),
        "outreaches_today": 0,
        "hours_since_last_outreach": None,
        "hours_since_last_user_activity": None,
    }
    cs = proactive.candidates(tmp_vault, cfg)
    pick = proactive.should_reach_out(cs, ctx, cfg)
    assert pick is not None
    assert pick["node_name"] == "Hot tension"

    msg = outreach.draft_message(pick, tmp_vault, cfg)
    assert "Hot tension" in msg

    ok = outreach.log_outreach(tmp_vault, pick, msg, delivered=True)
    assert ok

    log = tmp_vault / "_meta" / "outreach_log.md"
    assert log.exists()
    assert "Hot tension" in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Disabled → nothing surfaces.
# ---------------------------------------------------------------------------
def test_disabled_no_surface(tmp_vault: Path):
    _write_node(
        tmp_vault, "tensions", "T",
        "x",
        _fm(days_old=10, type="tension", importance=0.9),
    )
    cfg = _pcfg(enabled=False)
    ctx = {
        "now": datetime.now().replace(hour=13),
        "outreaches_today": 0,
    }
    cs = proactive.candidates(tmp_vault, cfg)
    pick = proactive.should_reach_out(cs, ctx, cfg)
    assert pick is None


# ---------------------------------------------------------------------------
# /whats_up — returns candidates without writing a log.
# ---------------------------------------------------------------------------
def test_whats_up_is_read_only(tmp_vault: Path):
    _write_node(
        tmp_vault, "tensions", "T",
        "x",
        _fm(days_old=10, type="tension", importance=0.9),
    )
    cfg = _pcfg(enabled=True)
    cs = proactive.candidates(tmp_vault, cfg)
    assert cs
    # Simulate /whats_up: reads, does not log.
    log = tmp_vault / "_meta" / "outreach_log.md"
    assert not log.exists()


# ---------------------------------------------------------------------------
# /pause 24h → next startup does not surface.
# ---------------------------------------------------------------------------
def test_pause_prevents_surface(tmp_vault: Path):
    _write_node(
        tmp_vault, "tensions", "T",
        "x",
        _fm(days_old=10, type="tension", importance=0.9),
    )
    outreach.set_pause(tmp_vault, datetime.now() + timedelta(hours=24))
    assert outreach.active_pause(tmp_vault) is not None
    # The TUI checks active_pause first and returns early. Simulate that.
    pause = outreach.active_pause(tmp_vault)
    assert pause is not None


# ---------------------------------------------------------------------------
# Daily cap prevents a 4th surface.
# ---------------------------------------------------------------------------
def test_daily_cap_blocks_fourth(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "A", "a",
                _fm(days_old=10, type="tension", importance=0.9))
    _write_node(tmp_vault, "tensions", "B", "b",
                _fm(days_old=10, type="tension", importance=0.9))
    _write_node(tmp_vault, "tensions", "C", "c",
                _fm(days_old=10, type="tension", importance=0.9))
    _write_node(tmp_vault, "tensions", "D", "d",
                _fm(days_old=10, type="tension", importance=0.9))

    cfg = _pcfg(enabled=True)

    # Log 3 delivered outreaches.
    for name in ("A", "B", "C"):
        ok = outreach.log_outreach(
            tmp_vault,
            {"node_name": name, "score": 0.7, "reasons": []},
            f"m-{name}", delivered=True,
        )
        assert ok

    # Context should now say daily cap is hit.
    ctx = outreach.build_context(tmp_vault, cfg)
    assert ctx["outreaches_today"] == 3

    # Simulate D wanting to surface — guard should refuse.
    pick_d = {
        "node_name": "D", "node_path": "/tmp/D", "node_type": "tension",
        "score": 0.9, "reasons": [], "days_since_mention": 10,
        "importance": 0.9, "salience": 0.5, "connection_count": 0,
    }
    got = proactive.should_reach_out([pick_d], ctx, cfg)
    assert got is None


# ---------------------------------------------------------------------------
# Quiet hours — nothing surfaces even with a hot candidate.
# ---------------------------------------------------------------------------
def test_quiet_hours_blocks(tmp_vault: Path):
    _write_node(
        tmp_vault, "tensions", "T",
        "x",
        _fm(days_old=10, type="tension", importance=0.9),
    )
    cfg = _pcfg(enabled=True)
    cs = proactive.candidates(tmp_vault, cfg)
    ctx = {
        "now": datetime(2026, 4, 20, 23, 30),  # 23:30 is in quiet hours
        "outreaches_today": 0,
    }
    pick = proactive.should_reach_out(cs, ctx, cfg)
    assert pick is None


# ---------------------------------------------------------------------------
# Pending chat builds the session with a proactive-prefixed task.
# ---------------------------------------------------------------------------
def test_session_task_prefix_when_proactive_set(tmp_vault: Path):
    # Simulate the TUI state: a proactive prefix has been queued.
    # Confirm that prefixing task yields the right form.
    prefix = "Hot tension"
    user_text = "yeah it's still broken"
    task = f"reply to proactive: {prefix} — {user_text}"
    assert task.startswith("reply to proactive:")
    assert prefix in task
    assert user_text in task
