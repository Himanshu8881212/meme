from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
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


def _base_fm(days_old: int = 0, **extra: Any) -> dict[str, Any]:
    d = (date.today() - timedelta(days=days_old)).isoformat()
    fm = {
        "created": d,
        "last_accessed": d,
        "access_count": 1,
        "importance": 0.6,
        "decay_weight": 0.6,
        "connection_count": 0,
        "archived": False,
        "tags": [],
    }
    fm.update(extra)
    return fm


def _cfg_proactive_on() -> dict[str, Any]:
    return {
        "proactive": {
            "enabled": True,
            "daily_cap": 3,
            "min_score_threshold": 0.4,   # lower so small test vaults trip it
            "min_gap_hours": 4,
            "min_silence_hours": 2,
            "quiet_hours": [22, 9],
        },
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_hot_tension_beats_stale_question(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "Hot tension",
                "Contradiction unresolved.",
                _base_fm(days_old=10, type="tension", importance=0.7, salience=0.7))
    _write_node(tmp_vault, "questions", "Stale question",
                "Old open question.",
                _base_fm(days_old=2, type="question", importance=0.4))

    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert cs, "expected at least one candidate"
    top = cs[0]
    assert top["node_name"] == "Hot tension"
    assert top["node_type"] == "tension"


def test_salience_beats_default(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "Heavy",
                "Heavy one.",
                _base_fm(days_old=8, type="tension", importance=0.6, salience=0.9))
    _write_node(tmp_vault, "tensions", "Light",
                "Light one.",
                _base_fm(days_old=8, type="tension", importance=0.6))

    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    top = cs[0]
    assert top["node_name"] == "Heavy"


def test_proactive_false_excludes(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "Muted",
                "Muted.",
                _base_fm(days_old=10, type="tension", importance=0.9, proactive=False))
    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert all(c["node_name"] != "Muted" for c in cs)


def test_pinned_not_surfaced_unless_checkin(tmp_vault: Path):
    _write_node(tmp_vault, "entities", "Mom",
                "Mom entity.",
                _base_fm(days_old=14, type="entity", importance=0.8, pin=True))
    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert all(c["node_name"] != "Mom" for c in cs)

    # Now with checkin=true she opts back in.
    _write_node(tmp_vault, "entities", "Mom",
                "Mom entity.",
                _base_fm(days_old=14, type="entity", importance=0.8,
                         pin=True, checkin=True))
    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert any(c["node_name"] == "Mom" for c in cs)


def test_recent_surface_excludes_24h(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "T",
                "x",
                _base_fm(days_old=10, type="tension", importance=0.8))
    # seed log with a recent entry for T
    log = tmp_vault / "_meta" / "outreach_log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    recent_ts = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
    log.write_text(
        f"# Outreach log\n\n- {recent_ts} | T | score=0.80 | delivered=true | reasons=-\n"
        f"    > hi\n",
        encoding="utf-8",
    )
    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert all(c["node_name"] != "T" for c in cs)


def test_recent_surface_72h_eligible(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "T2",
                "x",
                _base_fm(days_old=10, type="tension", importance=0.8))
    log = tmp_vault / "_meta" / "outreach_log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    old_ts = (datetime.now() - timedelta(hours=72)).isoformat(timespec="seconds")
    log.write_text(
        f"# Outreach log\n\n- {old_ts} | T2 | score=0.80 | delivered=true | reasons=-\n"
        f"    > hi\n",
        encoding="utf-8",
    )
    cs = proactive.candidates(tmp_vault, _cfg_proactive_on())
    assert any(c["node_name"] == "T2" for c in cs)


# ---------------------------------------------------------------------------
# should_reach_out guard clauses
# ---------------------------------------------------------------------------
def _mk_candidate(score: float = 0.8) -> dict[str, Any]:
    return {
        "node_name": "x", "node_path": "/x", "node_type": "tension",
        "score": score, "reasons": ["r"], "days_since_mention": 10,
        "importance": 0.8, "salience": 0.7, "connection_count": 2,
    }


def test_disabled_short_circuits():
    cfg = {"proactive": {"enabled": False}}
    assert proactive.should_reach_out([_mk_candidate()], {}, cfg) is None


def test_min_score_guard():
    cfg = _cfg_proactive_on()
    cfg["proactive"]["min_score_threshold"] = 0.9
    ctx = {"outreaches_today": 0}
    assert proactive.should_reach_out([_mk_candidate(0.5)], ctx, cfg) is None


def test_daily_cap_guard():
    cfg = _cfg_proactive_on()
    ctx = {"outreaches_today": 3}
    assert proactive.should_reach_out([_mk_candidate()], ctx, cfg) is None


def test_quiet_hours_guard():
    cfg = _cfg_proactive_on()
    night = datetime.now().replace(hour=23, minute=0, second=0)
    ctx = {"outreaches_today": 0, "now": night}
    assert proactive.should_reach_out([_mk_candidate()], ctx, cfg) is None


def test_quiet_hours_passes_at_noon():
    cfg = _cfg_proactive_on()
    noon = datetime.now().replace(hour=13, minute=0, second=0)
    ctx = {"outreaches_today": 0, "now": noon}
    assert proactive.should_reach_out([_mk_candidate()], ctx, cfg) is not None


def test_min_gap_guard():
    cfg = _cfg_proactive_on()
    ctx = {
        "outreaches_today": 0,
        "hours_since_last_outreach": 0.5,
        "now": datetime.now().replace(hour=13),
    }
    assert proactive.should_reach_out([_mk_candidate()], ctx, cfg) is None


def test_min_silence_guard():
    cfg = _cfg_proactive_on()
    ctx = {
        "outreaches_today": 0,
        "hours_since_last_user_activity": 0.5,
        "now": datetime.now().replace(hour=13),
    }
    assert proactive.should_reach_out([_mk_candidate()], ctx, cfg) is None


def test_all_guards_green_returns_top():
    cfg = _cfg_proactive_on()
    ctx = {
        "outreaches_today": 0,
        "hours_since_last_outreach": 24.0,
        "hours_since_last_user_activity": 10.0,
        "now": datetime.now().replace(hour=13),
    }
    got = proactive.should_reach_out([_mk_candidate(0.9)], ctx, cfg)
    assert got is not None
    assert got["node_name"] == "x"


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------
def test_parse_pause_offset():
    now = datetime(2026, 4, 20, 12, 0, 0)
    until = outreach.parse_pause_arg("24h", now=now)
    assert until == now + timedelta(hours=24)

    until = outreach.parse_pause_arg("30m", now=now)
    assert until == now + timedelta(minutes=30)


def test_parse_pause_until_date():
    now = datetime(2026, 4, 20)
    until = outreach.parse_pause_arg("until 2026-05-01", now=now)
    assert until == datetime(2026, 5, 1, 0, 0, 0)


def test_parse_pause_until_tomorrow():
    now = datetime(2026, 4, 20, 15, 0, 0)
    until = outreach.parse_pause_arg("until tomorrow", now=now)
    assert until == datetime(2026, 4, 21, 0, 0, 0)


def test_parse_pause_garbage():
    assert outreach.parse_pause_arg("nonsense") is None
    assert outreach.parse_pause_arg("") is None
