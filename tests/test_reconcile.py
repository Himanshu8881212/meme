"""Tests for reflection.reconcile_tensions — mirrors resolutions from
tension nodes back into the affected entity bodies."""
from __future__ import annotations

from pathlib import Path

from core import reflection
from utils import frontmatter


def _tension(vault: Path, name: str, body: str) -> Path:
    path = vault / "tensions" / f"{name}.md"
    frontmatter.write(path, {"type": "tension", "tags": []}, body)
    return path


def _entity(vault: Path, name: str, body: str) -> Path:
    path = vault / "entities" / f"{name}.md"
    frontmatter.write(path, {"type": "entity", "tags": [], "importance": 0.5}, body)
    return path


def test_reconcile_appends_resolution_to_entity(tmp_vault: Path):
    _entity(tmp_vault, "Olga of Kiev", "Died in Kyiv.\nMarried to [[Igor of Kiev]].")
    _tension(tmp_vault, "Olga of Kiev death location discrepancy",
             "Two positions.\n\n## Resolution\nDied in Rodez (the later fact overrides).\n")

    results = reflection.reconcile_tensions(tmp_vault)
    assert len(results) == 1
    assert results[0]["entity"] == "Olga of Kiev"

    _, body = frontmatter.read(tmp_vault / "entities" / "Olga of Kiev.md")
    assert "Died in Kyiv" in body  # original preserved for history
    assert "Died in Rodez" in body  # resolved state appended
    assert "[[tensions/Olga of Kiev death location discrepancy]]" in body


def test_reconcile_is_idempotent(tmp_vault: Path):
    _entity(tmp_vault, "X", "Old body.")
    _tension(tmp_vault, "X discrepancy",
             "Stuff.\n\n## Resolution\nCurrent: Y\n")

    first = reflection.reconcile_tensions(tmp_vault)
    second = reflection.reconcile_tensions(tmp_vault)
    assert len(first) == 1
    assert len(second) == 0  # already applied

    _, body = frontmatter.read(tmp_vault / "entities" / "X.md")
    # The resolution block appears exactly once.
    assert body.count("Current: Y") == 1


def test_reconcile_skips_tension_without_resolution(tmp_vault: Path):
    _entity(tmp_vault, "Y", "body")
    _tension(tmp_vault, "Y conflict", "Just describes a conflict with no resolution section.")
    results = reflection.reconcile_tensions(tmp_vault)
    assert results == []


def test_reconcile_multiple_suffix_variants(tmp_vault: Path):
    """Tension titles can use any of several suffixes — all should map to the
    same entity stem."""
    _entity(tmp_vault, "Blair Walsh", "Plays as a placekicker.")
    _tension(tmp_vault, "Blair Walsh Sport",
             "Two sports mentioned.\n\n## Current state\nPlays rugby (latest).\n")

    results = reflection.reconcile_tensions(tmp_vault)
    assert any(r["entity"] == "Blair Walsh" for r in results)

    _, body = frontmatter.read(tmp_vault / "entities" / "Blair Walsh.md")
    assert "Plays rugby" in body


def test_reconcile_falls_back_to_substring_match(tmp_vault: Path):
    """If exact/stem match fails, substring match finds the entity."""
    _entity(tmp_vault, "Charles Dickens", "Wrote Our Mutual Friend.")
    _tension(tmp_vault, "Charles Dickens Authorship",
             "Body.\n\n## Resolution\nConfirmed author of Our Mutual Friend.\n")

    results = reflection.reconcile_tensions(tmp_vault)
    assert any(r["entity"] == "Charles Dickens" for r in results)


def test_reconcile_no_entity_found(tmp_vault: Path):
    """Tension with no matching entity is skipped gracefully."""
    _tension(tmp_vault, "Nonexistent Thing discrepancy",
             "Body.\n\n## Resolution\nSomething.\n")
    results = reflection.reconcile_tensions(tmp_vault)
    assert results == []


def test_apply_writes_calls_reconcile_by_default(tmp_vault: Path):
    """The apply_writes post-pass triggers reconciliation automatically."""
    _entity(tmp_vault, "Z", "old body")
    block = (
        '<<WRITE path="tensions/Z discrepancy.md" action="create">>\n'
        "---\ntype: tension\n---\n# Z discrepancy\n\n"
        "## Resolution\nZ is actually W now.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    # One creation + at least one reconciliation
    assert any(r.get("action") == "reconciled" for r in results)

    _, body = frontmatter.read(tmp_vault / "entities" / "Z.md")
    assert "Z is actually W now" in body


def test_apply_writes_reconcile_can_be_disabled(tmp_vault: Path):
    _entity(tmp_vault, "A", "old")
    block = (
        '<<WRITE path="tensions/A conflict.md" action="create">>\n'
        "---\ntype: tension\n---\n## Resolution\nA is now B.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault, reconcile=False)
    assert not any(r.get("action") == "reconciled" for r in results)

    _, body = frontmatter.read(tmp_vault / "entities" / "A.md")
    assert "A is now B" not in body
