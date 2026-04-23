"""Tool-use memory — Samantha learns which tool calls worked."""
from __future__ import annotations

from pathlib import Path

from core import tool_memory as tm


def _prep(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / "_meta").mkdir(parents=True)
    return v


def test_log_and_read_roundtrip(tmp_path):
    v = _prep(tmp_path)
    tm.log_call(v, "memory_search", {"query": "puppet"}, "=== Puppet.md ===\nstuff")
    tm.log_call(v, "obsidian_search", {"query": "grocery"}, "(no matches)")
    calls = tm.recent_calls(v)
    assert len(calls) == 2
    tools = [c["tool"] for c in calls]
    assert tools == ["memory_search", "obsidian_search"]


def test_classify_success_empty_error(tmp_path):
    v = _prep(tmp_path)
    tm.log_call(v, "t", {}, "normal content with details")
    tm.log_call(v, "t", {}, "(no matches)")
    tm.log_call(v, "t", {}, "error: something broke")
    tm.log_call(v, "t", {}, "(camera unavailable)")
    outcomes = [c["outcome"] for c in tm.recent_calls(v)]
    assert outcomes == ["success", "empty", "error", "error"]


def test_args_are_truncated(tmp_path):
    v = _prep(tmp_path)
    tm.log_call(v, "obsidian_create", {"body": "x" * 500}, "ok")
    c = tm.recent_calls(v)[0]
    assert len(c["args"]["body"]) <= 161  # 160 + ellipsis


def test_prompt_summary_has_recent_badges(tmp_path):
    v = _prep(tmp_path)
    tm.log_call(v, "obsidian_search", {"query": "grocery list"}, "(no matches)")
    tm.log_call(v, "obsidian_list", {"folder": "Groceries"}, "Groceries/list.md\n...")
    tm.log_call(v, "web_search", {"query": "lisbon weather"}, "error: no internet")
    out = tm.prompt_summary(v)
    # most-recent-first order
    assert out.index("web_search") < out.index("obsidian_list") < out.index("obsidian_search")
    assert "✓" in out  # obsidian_list succeeded
    assert "∅" in out  # obsidian_search empty
    assert "✗" in out  # web_search errored


def test_prompt_summary_empty_vault(tmp_path):
    v = _prep(tmp_path)
    assert tm.prompt_summary(v) == ""


def test_log_never_raises(tmp_path, monkeypatch):
    """Logging must never break a chat turn, even when IO blows up."""
    v = _prep(tmp_path)
    def _boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "open", _boom)
    # Must not raise.
    tm.log_call(v, "memory_search", {"q": "x"}, "result")


def test_prompt_summary_capped():
    """Hard cap on how many lines land in the prompt."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        v = Path(td) / "vault"
        (v / "_meta").mkdir(parents=True)
        for i in range(40):
            tm.log_call(v, "t", {"i": i}, "ok")
        out = tm.prompt_summary(v)
        # MAX_LINES_IN_PROMPT default is 12
        assert out.count("\n") < 15
