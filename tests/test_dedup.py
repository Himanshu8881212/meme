"""De-dup auditor — flag same-type nodes that look like semantic twins."""
from __future__ import annotations

from pathlib import Path

from core import dedup
from utils import frontmatter


def _write(vault: Path, folder: str, name: str, body: str, ntype: str):
    frontmatter.write(
        vault / folder / f"{name}.md",
        {"type": ntype, "importance": 0.5, "connection_count": 0, "tags": []},
        body,
    )


def test_finds_title_duplicates(tmp_vault: Path):
    _write(tmp_vault, "entities", "Puppet the Dog",
           "A friendly black Labrador who lives with Himanshu.", "entity")
    _write(tmp_vault, "entities", "Puppet Dog",
           "The black lab named Puppet belonging to Himanshu.", "entity")
    _write(tmp_vault, "entities", "Ocelot",
           "A small wild cat native to South America.", "entity")

    cands = dedup.find_duplicate_candidates(tmp_vault)
    pairs = {tuple(sorted([c["a"], c["b"]])) for c in cands}
    assert ("Puppet Dog", "Puppet the Dog") in pairs
    # Ocelot shouldn't pair with the Puppet nodes.
    assert not any("Ocelot" in p for p in pairs)


def test_does_not_cross_types(tmp_vault: Path):
    _write(tmp_vault, "entities", "React Router",
           "React Router is a client-side routing library for React apps.", "entity")
    _write(tmp_vault, "concepts", "React Router",
           "React Router is a client-side routing library for React apps.", "concept")
    # Same name, same body, but different types — should NOT be flagged.
    # (Entities and concepts are allowed to co-exist with the same title.)
    cands = dedup.find_duplicate_candidates(tmp_vault)
    # Filter to just the React Router pair if it's there
    rr = [c for c in cands if {c["a"], c["b"]} == {"React Router"}]
    assert not rr


def test_body_overlap_without_title_overlap(tmp_vault: Path):
    _write(tmp_vault, "concepts", "Meditation practice",
           "Breath awareness, sitting upright, noticing thoughts arise, returning to breath, daily morning routine.",
           "concept")
    _write(tmp_vault, "concepts", "Mindfulness routine",
           "Breath awareness, sitting upright, noticing thoughts arise, returning to breath, daily morning routine.",
           "concept")
    cands = dedup.find_duplicate_candidates(tmp_vault)
    pairs = {tuple(sorted([c["a"], c["b"]])) for c in cands}
    assert ("Meditation practice", "Mindfulness routine") in pairs


def test_empty_vault_returns_empty(tmp_vault: Path):
    assert dedup.find_duplicate_candidates(tmp_vault) == []


def test_archived_nodes_skipped(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "entities" / "Dup A.md",
        {"type": "entity", "archived": True, "importance": 0.5,
         "connection_count": 0, "tags": []},
        "Same body tokens here everywhere alpha beta gamma delta.",
    )
    frontmatter.write(
        tmp_vault / "entities" / "Dup B.md",
        {"type": "entity", "archived": False, "importance": 0.5,
         "connection_count": 0, "tags": []},
        "Same body tokens here everywhere alpha beta gamma delta.",
    )
    # With one archived, no pair surfaces (we only audit live nodes).
    cands = dedup.find_duplicate_candidates(tmp_vault)
    assert cands == []
