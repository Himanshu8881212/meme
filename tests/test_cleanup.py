"""Tests for the algorithmic cleanup phase that runs before deep reflection.
Catches obviously-trash nodes (empty, wrong location, missing frontmatter)
so the reasoning model doesn't waste tokens on them."""
from __future__ import annotations

from pathlib import Path

from core import monitor
from utils import frontmatter


def test_find_broken_catches_root_stub(tmp_vault: Path):
    """Obsidian auto-creates .md files at the vault root when you click an
    unresolved wikilink. Those have no type, no body worth speaking of, and
    live outside the allowed folders."""
    stub = tmp_vault / "Challenger Deep.md"
    stub.write_text("---\nlast_accessed: '2026-04-20'\n---\n", encoding="utf-8")

    broken = monitor.find_broken_nodes(tmp_vault)
    assert any("Challenger Deep" in b["path"] for b in broken)


def test_find_broken_catches_empty_body(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "entities" / "Ghost.md",
        {"type": "entity", "tags": [], "importance": 0.5},
        "   \n\n",
    )
    broken = monitor.find_broken_nodes(tmp_vault)
    assert any("Ghost" in b["path"] for b in broken)
    assert any(b["reason"] == "empty body" for b in broken)


def test_find_broken_catches_no_type(tmp_vault: Path):
    (tmp_vault / "entities" / "Typeless.md").write_text(
        "---\nimportance: 0.5\n---\nSome body text here.", encoding="utf-8"
    )
    broken = monitor.find_broken_nodes(tmp_vault)
    assert any(b["reason"] == "no type field" and "Typeless" in b["path"] for b in broken)


def test_find_broken_catches_no_frontmatter(tmp_vault: Path):
    (tmp_vault / "entities" / "NoFM.md").write_text("Just raw text.", encoding="utf-8")
    broken = monitor.find_broken_nodes(tmp_vault)
    assert any(b["reason"] == "no frontmatter" and "NoFM" in b["path"] for b in broken)


def test_find_broken_ignores_valid_nodes(seeded_vault: Path):
    broken = monitor.find_broken_nodes(seeded_vault)
    assert broken == []


def test_cleanup_actually_deletes(tmp_vault: Path):
    stub = tmp_vault / "Rogue.md"
    stub.write_text("---\n---\n", encoding="utf-8")
    assert stub.exists()
    removed = monitor.cleanup_broken(tmp_vault)
    assert any("Rogue" in r["path"] for r in removed)
    assert not stub.exists()


def test_cleanup_preserves_valid_nodes(seeded_vault: Path):
    before = set(p.name for p in seeded_vault.rglob("*.md"))
    monitor.cleanup_broken(seeded_vault)
    after = set(p.name for p in seeded_vault.rglob("*.md"))
    assert before == after


def test_find_orphans_catches_isolated_node(seeded_vault: Path):
    # Mobile Team is in the seeded vault with no inbound/outbound links.
    orphans = monitor.find_orphans(seeded_vault)
    assert "Mobile Team" in orphans


def test_find_orphans_excludes_nodes_with_outbound_only(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "concepts" / "Has Outbound.md",
        {"type": "concept", "tags": ["x"], "importance": 0.5},
        "Links to [[Some Target]] even if nothing links back.",
    )
    orphans = monitor.find_orphans(tmp_vault)
    assert "Has Outbound" not in orphans


def test_find_orphans_excludes_nodes_with_inbound_only(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "concepts" / "Referenced.md",
        {"type": "concept", "tags": ["x"], "importance": 0.5},
        "Some content with no outbound.",
    )
    frontmatter.write(
        tmp_vault / "concepts" / "References.md",
        {"type": "concept", "tags": ["x"], "importance": 0.5},
        "This links to [[Referenced]].",
    )
    orphans = monitor.find_orphans(tmp_vault)
    assert "Referenced" not in orphans


def test_find_orphans_ignores_transcripts(tmp_vault: Path):
    from scheduler import session as session_mgr
    session_mgr.archive_transcript(tmp_vault, "body", "a transcript task", [])
    orphans = monitor.find_orphans(tmp_vault)
    assert not any("transcript" in o.lower() for o in orphans)


def test_cleanup_is_safe_to_run_on_seeded_vault(seeded_vault: Path):
    removed = monitor.cleanup_broken(seeded_vault)
    assert removed == []
