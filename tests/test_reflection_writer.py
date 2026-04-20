from __future__ import annotations

from pathlib import Path

from core import reflection


VALID_BLOCK = """Intro prose that should be ignored.

<<WRITE path="concepts/New Concept.md" action="create">>
---
type: concept
tags: [x, y]
---
# New Concept
Body with [[Other]].
<<END>>

Trailing prose."""


def test_apply_writes_creates_file(tmp_vault: Path):
    results = reflection.apply_writes(VALID_BLOCK, tmp_vault)
    assert any(r["action"] == "create" for r in results)
    assert (tmp_vault / "concepts" / "New Concept.md").exists()


def test_apply_writes_rejects_path_escape(tmp_vault: Path):
    block = '<<WRITE path="../etc/Evil.md" action="create">>\nbody\n<<END>>'
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "rejected"


def test_apply_writes_rejects_bad_folder(tmp_vault: Path):
    block = '<<WRITE path="secrets/Key.md" action="create">>\nx\n<<END>>'
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "rejected"


def test_apply_writes_deletes(tmp_vault: Path):
    target = tmp_vault / "concepts" / "Gone.md"
    target.write_text("---\ntype: concept\n---\nbody\n", encoding="utf-8")
    block = '<<WRITE path="concepts/Gone.md" action="delete">>\n<<END>>'
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "deleted"
    assert not target.exists()


def test_apply_writes_updates(tmp_vault: Path):
    target = tmp_vault / "concepts" / "Update Me.md"
    target.write_text("old\n", encoding="utf-8")
    block = ('<<WRITE path="concepts/Update Me.md" action="update">>\n'
             '---\ntype: concept\n---\n# Updated\nbody\n<<END>>')
    reflection.apply_writes(block, tmp_vault)
    assert "Updated" in target.read_text(encoding="utf-8")


def test_multiple_blocks(tmp_vault: Path):
    blocks = (
        '<<WRITE path="concepts/A.md" action="create">>\na\n<<END>>\n\n'
        '<<WRITE path="concepts/B.md" action="create">>\nb\n<<END>>'
    )
    results = reflection.apply_writes(blocks, tmp_vault)
    assert len(results) == 2
    assert (tmp_vault / "concepts" / "A.md").exists()
    assert (tmp_vault / "concepts" / "B.md").exists()


def test_strip_thinking_removes_tags():
    text = "<think>internal reasoning</think>Final answer."
    assert reflection.strip_thinking(text) == "Final answer."


def test_strip_thinking_bracket_form():
    text = "[THINK]reasoning[/THINK]\nAnswer here."
    assert "reasoning" not in reflection.strip_thinking(text)
    assert "Answer here." in reflection.strip_thinking(text)


def test_apply_writes_strips_thinking_before_applying(tmp_vault: Path):
    model_output = (
        '<think>a <<WRITE path="entities/Evil.md" action="create">>\nbad\n<<END>></think>\n'
        '<<WRITE path="concepts/Good.md" action="create">>\ngood\n<<END>>'
    )
    results = reflection.apply_writes(model_output, tmp_vault)
    assert any(r["path"] == "concepts/Good.md" for r in results)
    assert not (tmp_vault / "entities" / "Evil.md").exists()


def test_normalize_content_string():
    assert reflection._normalize_content("hello") == "hello"


def test_normalize_content_none():
    assert reflection._normalize_content(None) == ""


def test_normalize_content_dict_list():
    content = [
        {"type": "thinking", "thinking": "reasoning"},
        {"type": "text", "text": "answer"},
    ]
    result = reflection._normalize_content(content)
    assert "<think>reasoning</think>" in result
    assert "answer" in result
    # After strip_thinking, only the answer remains.
    assert reflection.strip_thinking(result) == "answer"


def test_normalize_content_object_list():
    class Part:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    parts = [Part(type="text", text="hi")]
    assert "hi" in reflection._normalize_content(parts)
