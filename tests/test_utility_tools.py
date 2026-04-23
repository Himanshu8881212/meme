"""Utility tools — camera / voice self-control / clock / timer / web search.

Dispatch-level tests. Real camera / voice / network backends are stubbed
via core.runtime so the tests run hermetically.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import reflection, runtime


def _cfg(tmp_path: Path) -> dict:
    return {
        "vault_path": str(tmp_path / "vault"),
        "providers": {"echo": {"base_url": None, "api_key_env": None}},
        "models": {"model1": {"provider": "echo", "model": "echo"}},
    }


def _dispatch(name: str, args: dict, cfg: dict):
    return reflection._model1_tool_dispatch(
        vault=Path(cfg["vault_path"]), name=name, args=args, config=cfg,
    )


def test_current_time_contains_year_and_time(tmp_path):
    runtime.clear()
    out = _dispatch("current_time", {}, _cfg(tmp_path))
    assert "2026" in out or "2027" in out  # wall-clock date; loose check
    assert ":" in out  # hh:mm separator


def test_mute_self_without_backend(tmp_path):
    runtime.clear()
    out = _dispatch("mute_self", {}, _cfg(tmp_path))
    assert "no voice" in out.lower() or "silent" in out.lower()


def test_mute_unmute_with_backend(tmp_path):
    runtime.clear()
    voice = MagicMock(muted=False)
    runtime.set_context(voice=voice)
    out = _dispatch("mute_self", {}, _cfg(tmp_path))
    assert voice.muted is True
    assert "muted" in out.lower()
    out = _dispatch("unmute_self", {}, _cfg(tmp_path))
    assert voice.muted is False
    assert "unmuted" in out.lower()


def test_mute_auto_unmute_schedules(tmp_path):
    """Duration arg should schedule a timer and report the wait."""
    runtime.clear()
    voice = MagicMock(muted=False)
    runtime.set_context(voice=voice)
    with patch("threading.Timer") as timer_cls:
        out = _dispatch("mute_self", {"duration_seconds": 5}, _cfg(tmp_path))
    assert voice.muted is True
    timer_cls.assert_called_once()
    assert "5s" in out or "5 " in out


def test_set_timer_rejects_zero(tmp_path):
    runtime.clear()
    # Ensure vault dir exists for cron store
    (Path(tmp_path) / "vault" / "_meta").mkdir(parents=True, exist_ok=True)
    out = _dispatch("set_timer", {"seconds": 0, "message": "noop"}, _cfg(tmp_path))
    assert "positive" in out.lower() or "bad" in out.lower()


def test_set_timer_persists_and_returns_id(tmp_path):
    """set_timer routes through the cron store so the timer is cancellable."""
    from core import cron as _cron
    runtime.clear()
    vault = Path(tmp_path) / "vault"
    (vault / "_meta").mkdir(parents=True, exist_ok=True)
    out = _dispatch(
        "set_timer", {"seconds": 90, "message": "tea"}, _cfg(tmp_path),
    )
    assert "tea" in out
    assert "1m 30s" in out
    # Persistent entry exists and is cancellable by the returned id.
    entries = _cron.load(vault)
    assert len(entries) == 1
    rid = entries[0]["id"]
    assert rid in out
    cancel = _dispatch("cancel_reminder", {"id": rid}, _cfg(tmp_path))
    assert "removed" in cancel.lower()
    assert _cron.load(vault) == []


def test_capture_camera_no_backend(tmp_path):
    runtime.clear()
    out = _dispatch("capture_camera", {"question": "what's there?"}, _cfg(tmp_path))
    assert "unavailable" in out.lower() or "no video" in out.lower()


def test_capture_camera_vision_call(tmp_path):
    runtime.clear()
    video = MagicMock()
    video.start.return_value = (True, "ok")
    video.stop.return_value = ["data:image/jpeg;base64,FAKE="]
    runtime.set_context(video=video)

    with patch.object(reflection, "chat", return_value="A red apple on a desk.") as chat_mock:
        out = _dispatch(
            "capture_camera", {"question": "what do you see?"},
            _cfg(tmp_path),
        )
    assert out == "A red apple on a desk."
    assert chat_mock.call_count == 1
    sent_parts = chat_mock.call_args.kwargs["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in sent_parts)
    video.start.assert_called_once()
    video.stop.assert_called_once()


def test_capture_camera_open_failure(tmp_path):
    runtime.clear()
    video = MagicMock()
    video.start.return_value = (False, "no camera permission")
    runtime.set_context(video=video)
    out = _dispatch("capture_camera", {"question": "x"}, _cfg(tmp_path))
    assert "failed" in out.lower() and "permission" in out.lower()


def test_web_search_empty_query(tmp_path):
    runtime.clear()
    out = _dispatch("web_search", {"query": "   "}, _cfg(tmp_path))
    assert "empty" in out.lower()


def test_web_search_returns_formatted_hits(tmp_path, monkeypatch):
    runtime.clear()
    fake = [
        {"title": "Python 3.12 release notes", "href": "https://python.org/3.12",
         "body": "What's new in Python 3.12"},
        {"title": "Textual docs", "href": "https://textual.textualize.io/",
         "body": "Python TUI framework"},
    ]

    class _DDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, q, max_results=5):
            return fake[:max_results]

    import ddgs
    monkeypatch.setattr(ddgs, "DDGS", _DDGS)

    out = _dispatch("web_search", {"query": "python 3.12", "max_results": 2}, _cfg(tmp_path))
    assert "Python 3.12 release notes" in out
    assert "https://python.org/3.12" in out
    assert "Textual docs" in out


def test_utility_tools_in_schema():
    names = {t["function"]["name"] for t in reflection.UTILITY_TOOL_SCHEMAS}
    # The core home-assistant set must be present. More tools may be
    # added over time (cron, etc.), hence subset-check instead of eq.
    assert {
        "capture_camera", "mute_self", "unmute_self",
        "current_time", "set_timer", "web_search",
    } <= names


def test_model1_tool_schemas_include_utilities():
    schemas = reflection.model1_tool_schemas(config={"external_vault": {"path": None}})
    names = {t["function"]["name"] for t in schemas}
    for n in ("capture_camera", "mute_self", "current_time", "web_search"):
        assert n in names
    # obsidian tools excluded when external_vault.path is null
    assert not any(n.startswith("obsidian_") for n in names)
