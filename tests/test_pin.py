"""Pin flag — nodes marked `pin: true` never decay or archive."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from core import decay
from utils import frontmatter


def test_pinned_node_survives_extreme_age(tmp_vault: Path):
    """A node pinned a decade ago, never re-accessed, stays unarchived.
    Without pin, the 14-to-35-day half-life would have archived it long ago."""
    very_old = (date.today() - timedelta(days=3650)).isoformat()
    frontmatter.write(
        tmp_vault / "entities" / "Birthday.md",
        {
            "type": "entity",
            "created": very_old,
            "last_accessed": very_old,
            "access_count": 1,
            "importance": 0.7,
            "pin": True,
            "connection_count": 0,
            "tags": ["personal"],
        },
        "Himanshu's birthday is Feb 10.",
    )
    decay.run(tmp_vault, lambda_=0.02, archive_threshold=0.10)
    fm, _ = frontmatter.read(tmp_vault / "entities" / "Birthday.md")
    assert fm["archived"] is False
    # Decay weight should be anchored to importance, not crushed to near-zero.
    assert fm["decay_weight"] >= fm["importance"] - 1e-6


def test_unpinned_very_old_node_does_archive(tmp_vault: Path):
    very_old = (date.today() - timedelta(days=3650)).isoformat()
    frontmatter.write(
        tmp_vault / "entities" / "Trivia.md",
        {
            "type": "entity",
            "created": very_old,
            "last_accessed": very_old,
            "access_count": 1,
            "importance": 0.3,
            "connection_count": 0,
            "tags": ["misc"],
        },
        "Some forgettable trivia.",
    )
    decay.run(tmp_vault, lambda_=0.02, archive_threshold=0.10)
    fm, _ = frontmatter.read(tmp_vault / "entities" / "Trivia.md")
    assert fm["archived"] is True
