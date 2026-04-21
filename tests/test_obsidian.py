from __future__ import annotations

from pathlib import Path

from core import obsidian


def _ext(tmp_path: Path) -> Path:
    ext = tmp_path / "obs"
    ext.mkdir()
    return ext


def test_create_note_happy(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(ext, "Projects/Foo.md", "# Foo\nbody")
    assert res["ok"]
    assert res["action"] == "created"
    assert (ext / "Projects" / "Foo.md").exists()


def test_create_note_with_frontmatter(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(
        ext, "a.md", "body", frontmatter={"tags": ["x"]},
    )
    assert res["ok"]
    text = (ext / "a.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "tags:" in text


def test_create_auto_appends_md(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(ext, "NoExt", "b")
    assert res["ok"]
    assert (ext / "NoExt.md").exists()


def test_create_when_exists_rejects(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "b")
    res = obsidian.create_note(ext, "a.md", "b2")
    assert not res["ok"]
    assert "exists" in res["error"]


def test_update_replace(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "old")
    res = obsidian.update_note(ext, "a.md", "new", mode="replace")
    assert res["ok"]
    assert (ext / "a.md").read_text(encoding="utf-8").strip() == "new"


def test_update_append(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "one")
    res = obsidian.update_note(ext, "a.md", "two", mode="append")
    assert res["ok"]
    text = (ext / "a.md").read_text(encoding="utf-8")
    assert "one" in text and "two" in text
    assert text.index("one") < text.index("two")


def test_update_prepend(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "one")
    res = obsidian.update_note(ext, "a.md", "two", mode="prepend")
    assert res["ok"]
    text = (ext / "a.md").read_text(encoding="utf-8")
    assert text.index("two") < text.index("one")


def test_update_when_missing_rejects(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.update_note(ext, "nope.md", "x")
    assert not res["ok"]
    assert "missing" in res["error"]


def test_read_note(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "hello")
    text = obsidian.read_note(ext, "a.md")
    assert "hello" in text


def test_read_missing(tmp_path: Path):
    ext = _ext(tmp_path)
    text = obsidian.read_note(ext, "nope.md")
    assert "not found" in text.lower()


def test_list_notes(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.create_note(ext, "sub/b.md", "y")
    out = obsidian.list_notes(ext)
    assert "a.md" in out
    assert any(p.endswith("b.md") for p in out)


def test_list_notes_folder(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.create_note(ext, "sub/b.md", "y")
    out = obsidian.list_notes(ext, folder="sub")
    assert len(out) == 1
    assert "b.md" in out[0]


def test_list_excludes_trash(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.delete_note(ext, "a.md")
    assert obsidian.list_notes(ext) == []


def test_search_notes(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "alpha beta gamma")
    obsidian.create_note(ext, "b.md", "delta")
    hits = obsidian.search_notes(ext, "beta")
    assert len(hits) == 1
    assert hits[0]["path"] == "a.md"
    assert "beta" in hits[0]["snippet"]
    assert hits[0]["line_no"] >= 1


def test_add_wikilink(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "body")
    res = obsidian.add_wikilink(ext, "a.md", "Other Note")
    assert res["ok"]
    text = (ext / "a.md").read_text(encoding="utf-8")
    assert "[[Other Note]]" in text


def test_add_wikilink_with_label(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "body")
    res = obsidian.add_wikilink(ext, "a.md", "Other", label="display text")
    assert res["ok"]
    assert "[[Other|display text]]" in (ext / "a.md").read_text(encoding="utf-8")


def test_rename_note_updates_backlinks(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "Old.md", "body")
    obsidian.create_note(ext, "b.md", "link to [[Old]] here")
    obsidian.create_note(ext, "c.md", "link to [[Old|alias]] too")
    res = obsidian.rename_note(ext, "Old.md", "New.md")
    assert res["ok"]
    assert (ext / "New.md").exists()
    assert not (ext / "Old.md").exists()
    assert "[[New]]" in (ext / "b.md").read_text(encoding="utf-8")
    text_c = (ext / "c.md").read_text(encoding="utf-8")
    assert "[[New|alias]]" in text_c


def test_delete_soft_moves_to_trash(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "bye")
    res = obsidian.delete_note(ext, "a.md")
    assert res["ok"]
    assert not (ext / "a.md").exists()
    trashed = list((ext / obsidian.TRASH_DIR).glob("*"))
    assert len(trashed) == 1
    assert trashed[0].read_text(encoding="utf-8").strip() == "bye"


def test_path_escape_dotdot_rejected(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(ext, "../outside.md", "x")
    assert not res["ok"]
    assert "invalid" in res["error"]
    assert not (tmp_path / "outside.md").exists()


def test_path_escape_deep_dotdot_rejected(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(ext, "sub/../../outside.md", "x")
    assert not res["ok"]


def test_path_absolute_rejected(tmp_path: Path):
    ext = _ext(tmp_path)
    res = obsidian.create_note(ext, "/etc/passwd", "x")
    assert not res["ok"]


def test_path_escape_on_read(tmp_path: Path):
    ext = _ext(tmp_path)
    text = obsidian.read_note(ext, "../../etc/passwd")
    assert "invalid" in text.lower()


def test_audit_log_written_for_every_call(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.read_note(ext, "a.md")
    obsidian.list_notes(ext)
    obsidian.search_notes(ext, "x")
    obsidian.update_note(ext, "a.md", "y")
    obsidian.add_wikilink(ext, "a.md", "B")
    obsidian.rename_note(ext, "a.md", "b.md")
    obsidian.delete_note(ext, "b.md")

    audit = (ext / obsidian.AUDIT_FILE).read_text(encoding="utf-8")
    lines = [l for l in audit.splitlines() if l.strip()]
    assert len(lines) >= 8
    for expected in (
        "obsidian_create", "obsidian_read", "obsidian_list", "obsidian_search",
        "obsidian_update", "obsidian_link", "obsidian_rename", "obsidian_delete",
    ):
        assert expected in audit


def test_audit_log_on_failed_call(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "../bad.md", "x")
    audit = (ext / obsidian.AUDIT_FILE).read_text(encoding="utf-8")
    assert "obsidian_create" in audit
    assert "invalid" in audit.lower()


def test_resolve_vault_path_none():
    assert obsidian.resolve_vault_path(None) is None
    assert obsidian.resolve_vault_path({}) is None
    assert obsidian.resolve_vault_path({"external_vault": {"path": None}}) is None


def test_resolve_vault_path_expanduser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = obsidian.resolve_vault_path({"external_vault": {"path": "~/x"}})
    assert p is not None
    assert str(p).startswith(str(tmp_path.resolve()))


def test_schemas_filtered_when_disabled():
    from core import reflection
    schemas = reflection.model1_tool_schemas(config={})
    names = [s["function"]["name"] for s in schemas]
    assert not any(n.startswith("obsidian_") for n in names)


def test_schemas_present_when_enabled(tmp_path: Path):
    from core import reflection
    cfg = {"external_vault": {"path": str(tmp_path)}}
    schemas = reflection.model1_tool_schemas(config=cfg)
    names = [s["function"]["name"] for s in schemas]
    assert any(n.startswith("obsidian_") for n in names)
    assert "obsidian_create" in names
    assert "obsidian_update" in names
    assert "obsidian_read" in names
    assert "obsidian_search" in names
    assert "obsidian_list" in names
    assert "obsidian_link" in names


def test_git_auto_commit_creates_and_commits(tmp_path: Path):
    import shutil
    if shutil.which("git") is None:
        import pytest as _pt
        _pt.skip("git not available")
    ext = _ext(tmp_path)
    cfg = {"external_vault": {"path": str(ext), "git_auto_commit": True}}
    res = obsidian.create_note(ext, "a.md", "x", config=cfg)
    assert res["ok"]
    # External vault should now be a git repo with a samantha commit.
    import subprocess
    out = subprocess.run(
        ["git", "log", "--oneline"], cwd=str(ext), capture_output=True, text=True
    )
    assert out.returncode == 0
    assert "samantha:" in out.stdout
    assert "create" in out.stdout


def test_git_auto_commit_off_no_repo(tmp_path: Path):
    ext = _ext(tmp_path)
    cfg = {"external_vault": {"path": str(ext), "git_auto_commit": False}}
    res = obsidian.create_note(ext, "a.md", "x", config=cfg)
    assert res["ok"]
    assert not (ext / ".git").exists()


def test_git_head_none_when_not_repo(tmp_path: Path):
    ext = _ext(tmp_path)
    assert obsidian.git_head(ext) is None


def test_read_audit_tail(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.create_note(ext, "b.md", "y")
    tail = obsidian.read_audit_tail(ext, n=2)
    assert len(tail) == 2
    assert tail[0]["tool"] == "obsidian_create"
    assert tail[1]["tool"] == "obsidian_create"


def test_note_count_excludes_trash(tmp_path: Path):
    ext = _ext(tmp_path)
    obsidian.create_note(ext, "a.md", "x")
    obsidian.create_note(ext, "b.md", "y")
    obsidian.delete_note(ext, "a.md")
    assert obsidian.note_count(ext) == 1
