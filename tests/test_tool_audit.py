"""Tool-use pattern synthesis — promotes repeated patterns from the log
into permanent procedures/ notes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core import tool_memory as tm


def _seed(vault: Path) -> Path:
    for f in ("procedures", "_meta"):
        (vault / f).mkdir(parents=True, exist_ok=True)
    return vault


def _cfg(vault: Path) -> dict:
    return {
        "vault_path": str(vault),
        "providers": {"echo": {"base_url": None, "api_key_env": None}},
        "models": {
            "deep":   {"provider": "echo", "model": "echo"},
            "model1": {"provider": "echo", "model": "echo"},
        },
    }


def test_audit_skips_when_log_too_short(tmp_path):
    v = _seed(tmp_path / "vault")
    tm.log_call(v, "memory_search", {"query": "x"}, "ok")
    r = tm.audit_and_synthesize(v, _cfg(v))
    assert r["writes"] == []
    assert "need" in (r.get("skipped") or "").lower()


def test_audit_skips_when_model_sees_no_patterns(tmp_path):
    v = _seed(tmp_path / "vault")
    for i in range(20):
        tm.log_call(v, "memory_search", {"query": str(i)}, "ok")
    with patch("core.reflection.chat", return_value="(no patterns)"):
        r = tm.audit_and_synthesize(v, _cfg(v))
    assert r["writes"] == []
    assert r["skipped"]


def test_audit_applies_writes_from_model(tmp_path):
    v = _seed(tmp_path / "vault")
    for i in range(20):
        tm.log_call(v, "obsidian_search", {"query": "grocery list"}, "(no matches)")
    for i in range(3):
        tm.log_call(v, "obsidian_list", {"folder": "Groceries"}, "Groceries/list.md\n")
    fake_model_out = (
        '<<WRITE path="procedures/Looking up grocery list.md" action="create">>\n'
        "---\ntype: procedure\ntags: [tool-use, groceries]\n---\n"
        "When the user asks about the grocery list, use "
        "`obsidian_list(folder='Groceries')`. "
        "`obsidian_search('grocery list')` returned empty repeatedly.\n"
        "<<END>>\n"
    )
    with patch("core.reflection.chat", return_value=fake_model_out):
        r = tm.audit_and_synthesize(v, _cfg(v))
    assert len(r["writes"]) == 1
    w = r["writes"][0]
    assert w["action"] == "create"
    assert "Looking up grocery list" in w["path"]
    # The procedure note actually landed on disk:
    proc_dir = v / "procedures"
    notes = list(proc_dir.glob("*.md"))
    assert any("Looking up grocery list" in p.stem for p in notes)


def test_format_for_audit_includes_outcomes(tmp_path):
    v = _seed(tmp_path / "vault")
    tm.log_call(v, "memory_search", {"q": "alpha"}, "hits!")
    tm.log_call(v, "web_search", {"query": "beta"}, "error: offline")
    tm.log_call(v, "obsidian_search", {"query": "gamma"}, "(no matches)")
    out = tm.format_for_audit(v)
    assert "memory_search" in out and "success" in out
    assert "web_search" in out and "error" in out
    assert "obsidian_search" in out and "empty" in out
