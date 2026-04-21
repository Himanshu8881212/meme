"""RLM-lite tools — memory_list (metadata enumeration) + memory_summarize
(recursive sub-call to distill N nodes into one paragraph)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core import reflection
from utils import frontmatter


def _write(vault: Path, folder: str, name: str, body: str, ntype: str, tags=None):
    frontmatter.write(
        vault / folder / f"{name}.md",
        {"type": ntype, "importance": 0.6, "tags": tags or [],
         "connection_count": 0},
        body,
    )


def test_memory_list_filters_by_tag_and_type(tmp_vault: Path, config):
    _write(tmp_vault, "entities", "Puppet", "dog", "entity", ["pet", "family"])
    _write(tmp_vault, "entities", "Himanshu", "human", "entity", ["family"])
    _write(tmp_vault, "concepts", "Training", "teaching", "concept", ["pet"])

    out = reflection._model1_tool_dispatch(
        tmp_vault, "memory_list", {"tag": "pet"}, config,
    )
    assert "Puppet" in out
    assert "Training" in out
    assert "Himanshu" not in out

    out_t = reflection._model1_tool_dispatch(
        tmp_vault, "memory_list", {"type": "entity"}, config,
    )
    assert "Puppet" in out_t
    assert "Himanshu" in out_t
    assert "Training" not in out_t


def test_memory_list_empty_when_no_match(tmp_vault: Path, config):
    _write(tmp_vault, "entities", "Only", "body", "entity", ["foo"])
    out = reflection._model1_tool_dispatch(
        tmp_vault, "memory_list", {"tag": "nonexistent"}, config,
    )
    assert "no matching" in out.lower()


def test_memory_summarize_calls_sub_lm(tmp_vault: Path, config):
    _write(tmp_vault, "entities", "A", "alpha fact one", "entity")
    _write(tmp_vault, "entities", "B", "beta fact two", "entity")

    with patch.object(reflection, "chat", return_value="distilled paragraph") as mock:
        out = reflection._model1_tool_dispatch(
            tmp_vault, "memory_summarize",
            {"names": ["A", "B"], "query": "what are the facts?"},
            config,
        )
    assert out == "distilled paragraph"
    assert mock.call_count == 1
    # The sub-call should have been given both node bodies.
    call_kwargs = mock.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "alpha fact one" in user_msg
    assert "beta fact two" in user_msg


def test_memory_summarize_rejects_empty_names(tmp_vault: Path, config):
    out = reflection._model1_tool_dispatch(
        tmp_vault, "memory_summarize", {"names": [], "query": "x"}, config,
    )
    assert "no names" in out.lower()


def test_memory_list_and_summarize_registered_in_schemas():
    schema_names = {t["function"]["name"] for t in reflection.MODEL1_TOOL_SCHEMAS}
    assert "memory_list" in schema_names
    assert "memory_summarize" in schema_names
