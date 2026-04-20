"""Tests for the deterministic tool runner. Every tool must return exact
results — that's the whole point. If these pass, the reflection model can
trust them instead of guessing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core import reflection, tools
from utils import frontmatter


def test_list_nodes_by_tag(seeded_vault: Path):
    auth = tools.list_nodes_by_tag(seeded_vault, "auth")
    assert "AuthService" in auth
    assert "JWT Strategy" in auth
    assert "Login Bug" in auth
    assert "Mobile Team" not in auth


def test_count_nodes_by_tag(seeded_vault: Path):
    assert tools.count_nodes_by_tag(seeded_vault, "auth") == 3
    assert tools.count_nodes_by_tag(seeded_vault, "nonexistent") == 0


def test_tag_lookup_case_insensitive(seeded_vault: Path):
    # The seeded vault uses lowercase tags; query should match regardless.
    assert tools.list_nodes_by_tag(seeded_vault, "AUTH") == tools.list_nodes_by_tag(
        seeded_vault, "auth"
    )


def test_list_nodes_by_type(seeded_vault: Path):
    entities = tools.list_nodes_by_type(seeded_vault, "entity")
    assert "AuthService" in entities
    assert "Acme Corp" in entities
    assert "Mobile Team" in entities


def test_read_node_returns_markdown(seeded_vault: Path):
    content = tools.read_node(seeded_vault, "AuthService")
    # Frontmatter fence + body from the seeded fixture.
    assert content.startswith("---")
    assert "JWT Strategy" in content


def test_read_node_fuzzy_match(seeded_vault: Path):
    # Exact name works.
    assert "JWT Strategy" in tools.read_node(seeded_vault, "AuthService")
    # Substring should still resolve (case-insensitive).
    assert "JWT Strategy" in tools.read_node(seeded_vault, "authservice")


def test_read_node_missing_returns_message(seeded_vault: Path):
    result = tools.read_node(seeded_vault, "DefinitelyNotAThing")
    assert "no node" in result.lower()


def test_backlinks_to(seeded_vault: Path):
    bl = tools.backlinks_to(seeded_vault, "AuthService")
    assert "JWT Strategy" in bl
    assert "Login Bug" in bl
    assert "Redis Decision" in bl
    assert "Mobile Team" not in bl
    # A node never reports itself as a backlink.
    assert "AuthService" not in bl


def test_outbound_from(seeded_vault: Path):
    ob = tools.outbound_from(seeded_vault, "AuthService")
    assert set(ob) == {"JWT Strategy", "Login Bug", "Acme Corp"}


def test_outbound_from_nonexistent(seeded_vault: Path):
    assert tools.outbound_from(seeded_vault, "Missing") == []


def test_find_by_title_substring(seeded_vault: Path):
    hits = tools.find_by_title_substring(seeded_vault, "auth")
    assert "AuthService" in hits
    assert "Mobile Team" not in hits


def test_node_age_days(seeded_vault: Path):
    # Seeded with today's date → 0.
    assert tools.node_age_days(seeded_vault, "AuthService") == 0


def test_all_tags_with_counts(seeded_vault: Path):
    tag_counts = tools.all_tags_with_counts(seeded_vault)
    assert tag_counts["auth"] == 3  # AuthService, JWT Strategy, Login Bug
    assert tag_counts["backend"] >= 2
    # Returned sorted by frequency desc.
    counts = list(tag_counts.values())
    assert counts == sorted(counts, reverse=True)


def test_dispatch_unknown_tool_returns_error(seeded_vault: Path):
    result = tools.call(seeded_vault, "not_a_tool", {})
    assert "error" in result
    assert "unknown" in result["error"].lower()


def test_dispatch_bad_args_returns_error(seeded_vault: Path):
    result = tools.call(seeded_vault, "list_nodes_by_tag", {"wrong_arg": "x"})
    assert "error" in result


def test_dispatch_good_call(seeded_vault: Path):
    result = tools.call(seeded_vault, "count_nodes_by_tag", {"tag": "auth"})
    assert result == 3


# -------- Tool-calling integration test with a fake OpenAI client -----------


def _fake_response(message_obj):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message_obj)]
    return resp


def _assistant_message(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls or []
    return m


def _tool_call(id_, name, args_json):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args_json
    return tc


def test_deep_with_tools_loops_until_no_tool_calls(seeded_vault: Path, config, monkeypatch):
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["deep"] = {"provider": "mistral", "model": "fake-model"}

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=[
        _fake_response(_assistant_message(
            tool_calls=[_tool_call("c1", "count_nodes_by_tag", '{"tag": "auth"}')]
        )),
        _fake_response(_assistant_message(
            content='<<WRITE path="concepts/Fine.md" action="create">>\nbody\n<<END>>'
        )),
    ])

    with patch.object(reflection, "_get_client", return_value=client):
        output, log = reflection.deep_with_tools(
            vault_path=seeded_vault,
            vault_files=[],
            metrics={"total_nodes": 7},
            triggers=[],
            config=config,
            max_rounds=4,
        )

    assert len(log) == 1
    assert log[0]["tool"] == "count_nodes_by_tag"
    assert log[0]["result"] == 3
    assert "<<WRITE" in output
    assert client.chat.completions.create.call_count == 2


def test_deep_with_tools_falls_back_to_plain_deep_for_echo(seeded_vault: Path, config):
    # MEMORY_BACKEND=echo from the autouse fixture should short-circuit the
    # tool loop and use the plain deep() path.
    output, log = reflection.deep_with_tools(
        vault_path=seeded_vault,
        vault_files=[],
        metrics={"total_nodes": 7},
        triggers=[],
        config=config,
    )
    assert log == []
    assert "[ECHO]" in output
