"""Persistent cron reminders — add / list / fire / reschedule / cancel."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core import cron, reflection


def _prep(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "_meta").mkdir(parents=True)
    return vault


def test_add_one_shot(tmp_path):
    vault = _prep(tmp_path)
    future = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    r = cron.add(vault, "call mom", once_at=future)
    assert r["ok"]
    entries = cron.load(vault)
    assert len(entries) == 1
    assert entries[0]["message"] == "call mom"
    assert entries[0]["active"] is True


def test_add_rejects_past_once_at(tmp_path):
    vault = _prep(tmp_path)
    past = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="seconds")
    r = cron.add(vault, "late", once_at=past)
    assert not r["ok"]


def test_add_cron_recurring(tmp_path):
    vault = _prep(tmp_path)
    r = cron.add(vault, "stretch", cron="0 9 * * 1-5")
    assert r["ok"]
    assert r["entry"]["next_fire"]


def test_add_rejects_bad_cron(tmp_path):
    vault = _prep(tmp_path)
    r = cron.add(vault, "nope", cron="not a cron")
    assert not r["ok"]


def test_add_requires_one_of(tmp_path):
    vault = _prep(tmp_path)
    r = cron.add(vault, "orphan")
    assert not r["ok"]


def test_due_picks_up_passed_entries(tmp_path):
    vault = _prep(tmp_path)
    future = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    cron.add(vault, "later", once_at=future)
    # Nothing due now.
    assert cron.due(vault) == []
    # Jump the clock forward two hours; same entry becomes due.
    future_now = datetime.now().astimezone() + timedelta(hours=2)
    assert len(cron.due(vault, now=future_now)) == 1


def test_mark_fired_one_shot_deactivates(tmp_path):
    vault = _prep(tmp_path)
    future = (datetime.now().astimezone() + timedelta(minutes=1)).isoformat(timespec="seconds")
    r = cron.add(vault, "x", once_at=future)
    cron.mark_fired(vault, r["entry"]["id"])
    e = cron.load(vault)[0]
    assert e["active"] is False
    assert e["last_fired"] is not None


def test_mark_fired_cron_reschedules(tmp_path):
    vault = _prep(tmp_path)
    r = cron.add(vault, "x", cron="0 * * * *")
    before = r["entry"]["next_fire"]
    cron.mark_fired(vault, r["entry"]["id"])
    e = cron.load(vault)[0]
    assert e["active"] is True
    assert e["next_fire"] is not None
    # The new next_fire must be at or after the original (could be equal
    # if we fire right on the hour mark, but usually strictly after).
    assert e["next_fire"] >= before


def test_remove(tmp_path):
    vault = _prep(tmp_path)
    r = cron.add(vault, "x", cron="0 * * * *")
    assert cron.remove(vault, r["entry"]["id"]) is True
    assert cron.load(vault) == []
    # Removing a missing id returns False.
    assert cron.remove(vault, "ghostid") is False


# ── dispatch tests ────────────────────────────────────────────────────────

def _cfg(vault: Path) -> dict:
    return {
        "vault_path": str(vault),
        "providers": {"echo": {"base_url": None, "api_key_env": None}},
        "models": {"model1": {"provider": "echo", "model": "echo"}},
    }


def _dispatch(vault: Path, name: str, args: dict):
    return reflection._model1_tool_dispatch(vault=vault, name=name, args=args, config=_cfg(vault))


def test_tool_schedule_list_cancel(tmp_path):
    vault = _prep(tmp_path)
    out = _dispatch(vault, "schedule_reminder", {"message": "stretch", "cron": "0 9 * * 1-5"})
    assert "scheduled" in out and "stretch" in out
    listing = _dispatch(vault, "list_reminders", {})
    assert "stretch" in listing
    rid = cron.load(vault)[0]["id"]
    cancel = _dispatch(vault, "cancel_reminder", {"id": rid})
    assert "removed" in cancel


def test_tool_schedule_rejects_both_missing(tmp_path):
    vault = _prep(tmp_path)
    out = _dispatch(vault, "schedule_reminder", {"message": "huh"})
    assert "error" in out.lower()


def test_tool_cancel_missing_id(tmp_path):
    vault = _prep(tmp_path)
    out = _dispatch(vault, "cancel_reminder", {"id": "nonexistent"})
    assert "no reminder" in out.lower()


def test_cron_tools_in_schema():
    names = {t["function"]["name"] for t in reflection.UTILITY_TOOL_SCHEMAS}
    assert {"schedule_reminder", "list_reminders", "cancel_reminder"} <= names
