"""Enforce the bright line between the internal vault and the external
Obsidian vault.

Writes on behalf of the user (obsidian_*) MUST NOT leak into Samantha's
own memory — no nodes, no flags, no reflection artefacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core import flagging, obsidian, reflection
from scheduler import session as session_mgr
from utils import indexer


def _snapshot_vault(vault: Path) -> set[str]:
    return {
        str(p.relative_to(vault))
        for p in vault.rglob("*.md")
        if p.is_file()
    }


def test_obsidian_output_is_plain_prose_not_flag_shaped(tmp_vault: Path, tmp_path: Path):
    ext = tmp_path / "obs"
    ext.mkdir()
    res = obsidian.create_note(ext, "Projects/Foo.md", "body here")
    assert res["ok"]
    # The preview the model would see back
    assert res["preview"].startswith("Created")
    for bad in ("[NOVEL", "[SALIENT", "[IDENTITY", "[HIGH-STAKES", "[CONTRADICTION"):
        assert bad not in res["preview"]
    assert flagging.extract(res["preview"]) == []


def test_obsidian_dispatch_returns_plain_prose(tmp_vault: Path, tmp_path: Path, config: dict[str, Any]):
    ext = tmp_path / "obs"
    ext.mkdir()
    config["external_vault"] = {"path": str(ext), "git_auto_commit": False}
    result = reflection._model1_tool_dispatch(
        tmp_vault, "obsidian_create",
        {"rel_path": "Foo.md", "body": "draft body"},
        config,
    )
    assert isinstance(result, str)
    assert "Created" in result
    assert flagging.extract(result) == []


def test_obsidian_create_does_not_touch_internal_vault(
    tmp_vault: Path, tmp_path: Path, config: dict[str, Any],
):
    ext = tmp_path / "obs"
    ext.mkdir()
    config["external_vault"] = {"path": str(ext), "git_auto_commit": False}

    before_nodes = _snapshot_vault(tmp_vault)
    before_idx = set(indexer.build(tmp_vault).keys())

    reflection._model1_tool_dispatch(
        tmp_vault, "obsidian_create",
        {"rel_path": "Projects/Draft.md", "body": "# Draft\nbody"},
        config,
    )
    reflection._model1_tool_dispatch(
        tmp_vault, "obsidian_update",
        {"rel_path": "Projects/Draft.md", "body": "appended", "mode": "append"},
        config,
    )

    after_nodes = _snapshot_vault(tmp_vault)
    after_idx = set(indexer.build(tmp_vault).keys())

    assert after_nodes == before_nodes, "internal vault gained files from obsidian write"
    assert after_idx == before_idx, "internal vault index grew from obsidian write"


def test_reflection_apply_writes_ignores_external_vault_paths(
    tmp_vault: Path, tmp_path: Path,
):
    ext = tmp_path / "obs"
    ext.mkdir()
    # A <<WRITE>> block whose folder isn't an allowed internal folder must
    # be rejected — the external vault is NOT an allowed folder.
    blob = (
        '<<WRITE path="Projects/FromReflection.md" action="create">>\n'
        "body\n<<END>>"
    )
    applied = reflection.apply_writes(blob, tmp_vault)
    assert len(applied) == 1
    assert applied[0]["action"] == "rejected"
    assert not (ext / "Projects" / "FromReflection.md").exists()


def test_external_vault_path_absent_from_reflection_logic():
    """External-vault touches in reflection.py must all flow through
    `core.obsidian` — no direct path handling, Git commands, or audit writes
    in the reflection module itself."""
    text = Path(reflection.__file__).read_text(encoding="utf-8")
    for banned in ("subprocess.run", "Path.expanduser", "audit_log", "git add"):
        assert banned not in text, f"{banned} leaked into reflection.py"
    # apply_writes() belongs to the internal vault only — it must not ever
    # route to core.obsidian.
    assert "apply_writes" in text
    for line in text.splitlines():
        if "apply_writes" in line and "obsidian" in line.lower():
            raise AssertionError(f"apply_writes sees obsidian: {line}")


def test_session_end_after_obsidian_write_creates_no_internal_node(
    tmp_vault: Path, tmp_path: Path, config: dict[str, Any], monkeypatch,
):
    ext = tmp_path / "obs"
    ext.mkdir()
    config["external_vault"] = {"path": str(ext), "git_auto_commit": False}
    # Seed _identity so _load_identity returns something sensible.
    (tmp_vault / "_identity").mkdir(exist_ok=True)
    (tmp_vault / "_identity" / "persona.md").write_text(
        "---\ntype: identity\n---\n\n# I am samantha\n", encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parent.parent
    meta = session_mgr.start(
        task="draft a note", tags=[], config=config, project_root=project_root,
    )
    assert "notebook" in meta["system_prompt"].lower()

    # Simulate the model writing to the external vault.
    reflection._model1_tool_dispatch(
        tmp_vault, "obsidian_create",
        {"rel_path": "Projects/Rate Limiter Design.md", "body": "# Rate limiter\nbody"},
        config,
    )

    before = _snapshot_vault(tmp_vault)

    # A transcript with nothing flag-shaped in it.
    transcript_body = (
        "## USER\ncan you jot down 'rate limiter design' in my notebook\n\n"
        "## ASSISTANT\nDone — created Projects/Rate Limiter Design.md (24 chars).\n"
    )
    result = session_mgr.end(
        session_output=transcript_body,
        session_meta=meta,
        config=config,
        project_root=project_root,
    )

    # Transcripts always get archived — that's fine, it's not an "entity" node.
    after = _snapshot_vault(tmp_vault)
    added = after - before
    # All new files should be inside _transcripts/ or _meta/ only.
    for rel in added:
        assert rel.startswith("_transcripts/") or rel.startswith("_meta/"), (
            f"unexpected internal node created after obsidian write: {rel}"
        )

    # No flags extracted from the ASSISTANT turn (because obsidian_create's
    # preview is plain prose).
    assert result["flags_found"] == 0


def test_truncate_obsidian_tool_outputs_caps_body():
    blob = (
        "## USER\nhi\n\n"
        "## TOOL obsidian_read\n"
        + ("x" * 2000)
        + "\n\n## ASSISTANT\nok\n"
    )
    out = session_mgr.truncate_obsidian_tool_outputs(blob)
    # original untouched blocks preserved; the tool block body is capped
    assert "## USER" in out and "## ASSISTANT" in out
    tool_block = out.split("## TOOL obsidian_read\n", 1)[1].split("## ASSISTANT", 1)[0]
    assert len(tool_block) < 700  # 500 + the "… [truncated]" marker


def test_obsidian_writes_do_not_create_flags_in_transcript_recovery(
    tmp_vault: Path, tmp_path: Path, config: dict[str, Any],
):
    """A transcript mentioning an obsidian write should not itself yield
    flag-shaped markers that would trigger memory writes."""
    body = (
        "## USER\ndraft a note about rate limiters\n\n"
        "## ASSISTANT\nI created Projects/Rate Limiter Design.md (183 chars) in your notebook.\n"
    )
    flags = flagging.extract(body)
    assert flags == []
