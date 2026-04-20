"""Tests for Model 1 agentic multi-hop retrieval."""
from __future__ import annotations

import json as _json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import reflection
from utils import frontmatter


def _fake_resp(message_obj):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message_obj)]
    return resp


def _assistant(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls or []
    return m


def _tc(id_, name, args_json):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args_json
    return tc


def _write_node(vault: Path, folder: str, name: str, body: str) -> None:
    frontmatter.write(vault / folder / f"{name}.md", {
        "type": folder[:-1] if folder.endswith("s") else folder,
        "tags": [],
        "importance": 0.5,
        "decay_weight": 0.5,
    }, body)


def test_chat_with_tools_echo_fallback(seeded_vault: Path, config):
    """Echo backend short-circuits — no tool loop, falls back to plain chat."""
    text, log = reflection.chat_with_tools(
        role="model1", system="hi",
        messages=[{"role": "user", "content": "test"}],
        config=config, vault_path=seeded_vault,
    )
    assert "[ECHO]" in text
    assert log == []


def test_chat_with_tools_multihop_loop(tmp_vault: Path, config, monkeypatch):
    """The core test: Model 1 calls memory_search twice to chain two hops,
    then emits its final answer."""
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["model1"] = {"provider": "mistral", "model": "fake-model"}

    _write_node(tmp_vault, "entities", "Charles Dickens",
                "Charles Dickens wrote Our Mutual Friend. "
                "Married to Catherine Hogarth.")
    _write_node(tmp_vault, "entities", "Catherine Hogarth",
                "Catherine Hogarth, Belgian by citizenship, married Dickens.")

    # Three turns: tool → tool → final text
    turn1 = _assistant(
        content=None,
        tool_calls=[_tc("c1", "memory_search",
                        '{"query": "author of Our Mutual Friend"}')],
    )
    turn2 = _assistant(
        content=None,
        tool_calls=[_tc("c2", "memory_search",
                        '{"query": "Catherine Hogarth citizenship"}')],
    )
    turn3 = _assistant(content="Belgium.")

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=[
        _fake_resp(turn1), _fake_resp(turn2), _fake_resp(turn3),
    ])

    with patch.object(reflection, "_get_client", return_value=client):
        text, log = reflection.chat_with_tools(
            role="model1", system="",
            messages=[{"role": "user",
                       "content": "citizenship of spouse of author of Our Mutual Friend?"}],
            config=config, vault_path=tmp_vault,
            max_tokens=256, max_rounds=5,
        )

    assert "Belgium" in text
    assert len(log) == 2
    assert log[0]["tool"] == "memory_search"
    assert log[1]["tool"] == "memory_search"
    assert client.chat.completions.create.call_count == 3


def test_chat_with_tools_max_rounds_forces_answer(tmp_vault: Path, config, monkeypatch):
    """If the model keeps calling tools past max_rounds, we force a final
    non-tool answer. Must not loop forever."""
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["model1"] = {"provider": "mistral", "model": "fake-model"}

    _write_node(tmp_vault, "entities", "X", "body")

    # All turns are tool calls — should cap at max_rounds.
    tool_turn = _assistant(
        content=None,
        tool_calls=[_tc("c", "memory_search", '{"query": "x"}')],
    )
    forced = _assistant(content="Forced answer: don't know.")

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=[
        _fake_resp(tool_turn),  # round 1
        _fake_resp(tool_turn),  # round 2
        _fake_resp(forced),     # post-loop forced answer
    ])

    with patch.object(reflection, "_get_client", return_value=client):
        text, log = reflection.chat_with_tools(
            role="model1", system="",
            messages=[{"role": "user", "content": "loop?"}],
            config=config, vault_path=tmp_vault,
            max_rounds=2,
        )

    assert "Forced answer" in text
    assert len(log) == 2


def test_tool_dispatch_memory_search(seeded_vault: Path, config):
    result = reflection._model1_tool_dispatch(
        seeded_vault, "memory_search", {"query": "auth"}, config,
    )
    assert "AuthService" in result or "auth" in result.lower()


def test_tool_dispatch_memory_read(seeded_vault: Path, config):
    result = reflection._model1_tool_dispatch(
        seeded_vault, "memory_read", {"name": "AuthService"}, config,
    )
    assert "JWT Strategy" in result


def test_tool_dispatch_memory_find(seeded_vault: Path, config):
    result = reflection._model1_tool_dispatch(
        seeded_vault, "memory_find", {"query": "auth"}, config,
    )
    assert "AuthService" in result


def test_tool_dispatch_unknown_returns_error(seeded_vault: Path, config):
    result = reflection._model1_tool_dispatch(
        seeded_vault, "not_a_tool", {}, config,
    )
    assert "unknown tool" in result


def test_model1_tool_schemas_valid():
    """Structural test: schemas must match OpenAI tool-calling spec."""
    for s in reflection.MODEL1_TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn and "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
