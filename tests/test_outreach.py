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


def _mk_cand(name: str, score: float = 0.8, path: str = "") -> dict[str, Any]:
    return {
        "node_name": name,
        "node_path": path,
        "node_type": "tension",
        "score": score,
        "reasons": ["unresolved tension, 10d old"],
        "days_since_mention": 10,
        "importance": 0.8, "salience": 0.7, "connection_count": 2,
    }


# ---------------------------------------------------------------------------
# Drafter (echo backend) produces output containing the node name.
# ---------------------------------------------------------------------------
def test_draft_message_contains_node_name(tmp_vault: Path, config: dict[str, Any]):
    node = _write_node(tmp_vault, "tensions", "Rest vs GraphQL",
                       "REST vs GraphQL tension unresolved.",
                       {"type": "tension", "importance": 0.7})
    cand = _mk_cand("Rest vs GraphQL", path=str(node))
    msg = outreach.draft_message(cand, tmp_vault, config)
    assert "Rest vs GraphQL" in msg
    assert msg.strip(), "drafter should return non-empty text"


# ---------------------------------------------------------------------------
# log_outreach writes the markdown log with the correct shape.
# ---------------------------------------------------------------------------
def test_log_outreach_creates_entry(tmp_vault: Path):
    cand = _mk_cand("Node A", score=0.72)
    ok = outreach.log_outreach(tmp_vault, cand, "hello about Node A", delivered=True)
    assert ok
    log = tmp_vault / "_meta" / "outreach_log.md"
    text = log.read_text(encoding="utf-8")
    assert "# Outreach log" in text
    assert "Node A" in text
    assert "score=0.72" in text
    assert "delivered=true" in text
    assert "hello about Node A" in text


def test_log_outreach_appends(tmp_vault: Path):
    outreach.log_outreach(tmp_vault, _mk_cand("A"), "first", True)
    outreach.log_outreach(tmp_vault, _mk_cand("B"), "second", True)
    log = tmp_vault / "_meta" / "outreach_log.md"
    text = log.read_text(encoding="utf-8")
    assert text.count("# Outreach log") == 1  # header only once
    assert "A" in text and "B" in text


def test_log_outreach_refuses_fourth(tmp_vault: Path):
    # Three deliveries today — fourth delivered=True must refuse.
    for i in range(3):
        outreach.log_outreach(tmp_vault, _mk_cand(f"N{i}"), f"m{i}", True)
    ok = outreach.log_outreach(tmp_vault, _mk_cand("N4"), "m4", True)
    assert ok is False
    # still fine to log delivered=False (audit-only)
    ok2 = outreach.log_outreach(tmp_vault, _mk_cand("N5"), "m5", False)
    assert ok2 is True


# ---------------------------------------------------------------------------
# build_context reads outreach_log correctly.
# ---------------------------------------------------------------------------
def test_build_context_counts_today(tmp_vault: Path, config: dict[str, Any]):
    outreach.log_outreach(tmp_vault, _mk_cand("A"), "a", True)
    outreach.log_outreach(tmp_vault, _mk_cand("B"), "b", True)
    ctx = outreach.build_context(tmp_vault, config)
    assert ctx["outreaches_today"] == 2
    assert ctx["hours_since_last_outreach"] is not None
    assert ctx["hours_since_last_outreach"] < 1.0


def test_build_context_empty(tmp_vault: Path, config: dict[str, Any]):
    ctx = outreach.build_context(tmp_vault, config)
    assert ctx["outreaches_today"] == 0
    assert ctx["hours_since_last_outreach"] is None


def test_build_context_old_entry_doesnt_count_today(tmp_vault: Path, config: dict[str, Any]):
    log = tmp_vault / "_meta" / "outreach_log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    old = (datetime.now() - timedelta(days=3)).isoformat(timespec="seconds")
    log.write_text(
        f"# Outreach log\n\n- {old} | X | score=0.70 | delivered=true | reasons=-\n"
        f"    > hi\n",
        encoding="utf-8",
    )
    ctx = outreach.build_context(tmp_vault, config)
    assert ctx["outreaches_today"] == 0
    # but hours_since_last_outreach should be set
    assert ctx["hours_since_last_outreach"] is not None
    assert ctx["hours_since_last_outreach"] > 48.0


# ---------------------------------------------------------------------------
# Pause helpers
# ---------------------------------------------------------------------------
def test_set_and_clear_pause(tmp_vault: Path):
    until = datetime.now() + timedelta(hours=5)
    outreach.set_pause(tmp_vault, until)
    got = outreach.active_pause(tmp_vault)
    assert got is not None
    assert abs((got - until).total_seconds()) < 2
    assert outreach.clear_pause(tmp_vault) is True
    assert outreach.active_pause(tmp_vault) is None


def test_active_pause_expired_is_none(tmp_vault: Path):
    past = datetime.now() - timedelta(hours=1)
    outreach.set_pause(tmp_vault, past)
    assert outreach.active_pause(tmp_vault) is None


# ---------------------------------------------------------------------------
# Node proactive flip (mute/unmute)
# ---------------------------------------------------------------------------
def test_set_node_proactive_false(tmp_vault: Path):
    _write_node(tmp_vault, "tensions", "Some Tension",
                "x",
                {"type": "tension", "tags": [], "importance": 0.7})
    ok = outreach.set_node_proactive(tmp_vault, "Some Tension", False)
    assert ok
    from utils import frontmatter
    fm, _ = frontmatter.read(tmp_vault / "tensions" / "Some Tension.md")
    assert fm["proactive"] is False


def test_set_node_proactive_missing(tmp_vault: Path):
    assert outreach.set_node_proactive(tmp_vault, "Nope", False) is False


# ---------------------------------------------------------------------------
# tail_log
# ---------------------------------------------------------------------------
def test_tail_log_returns_recent_first(tmp_vault: Path):
    # First 3 delivered (caps out the day), rest audit-only — all still logged.
    for i in range(5):
        delivered = i < 3
        outreach.log_outreach(tmp_vault, _mk_cand(f"N{i}"), f"m{i}", delivered)
    tail = outreach.tail_log(tmp_vault, n=3)
    assert len(tail) == 3
    # most recent first
    names = [t["node"] for t in tail]
    assert names == ["N4", "N3", "N2"]
