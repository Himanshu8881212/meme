from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

from core import decay
from utils import frontmatter


def test_strength_formula_matches_spec():
    s = decay.compute_strength(
        importance=0.9, connections=4, days_since_access=10, access_count=5, lambda_=0.05
    )
    expected = 0.9 * math.log1p(4) * math.exp(-0.05 * 10 / math.log1p(5))
    assert abs(s - expected) < 1e-9


def test_zero_days_leaves_decay_at_full_strength():
    s0 = decay.compute_strength(
        importance=0.5, connections=2, days_since_access=0, access_count=1, lambda_=0.05
    )
    s_future = decay.compute_strength(
        importance=0.5, connections=2, days_since_access=100, access_count=1, lambda_=0.05
    )
    assert s0 > s_future


def test_archive_threshold_flips_flag(tmp_vault: Path):
    old = (date.today() - timedelta(days=365 * 3)).isoformat()
    frontmatter.write(
        tmp_vault / "concepts" / "Ancient.md",
        {
            "type": "concept",
            "created": old,
            "last_accessed": old,
            "access_count": 1,
            "importance": 0.3,
            "decay_weight": 0.3,
            "connection_count": 1,
            "archived": False,
            "tags": ["legacy"],
        },
        "Ancient node nothing links to.",
    )
    result = decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)
    fm, _ = frontmatter.read(tmp_vault / "concepts" / "Ancient.md")
    assert result["processed"] == 1
    assert fm["archived"] is True


def test_fresh_heavily_linked_node_not_archived(seeded_vault: Path):
    decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    fm, _ = frontmatter.read(seeded_vault / "entities" / "AuthService.md")
    assert fm["archived"] is False
    assert fm["decay_weight"] > 0.1


def test_connection_count_recomputed_from_actual_backlinks(seeded_vault: Path):
    decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    fm, _ = frontmatter.read(seeded_vault / "entities" / "AuthService.md")
    # JWT Strategy, Login Bug, Redis Decision all point at AuthService.
    assert fm["connection_count"] == 3


def test_decay_does_not_delete_files(seeded_vault: Path):
    files_before = sorted(seeded_vault.rglob("*.md"))
    decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    files_after = sorted(seeded_vault.rglob("*.md"))
    assert files_before == files_after


def test_decay_idempotent_over_one_day(seeded_vault: Path):
    first = decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    second = decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    assert first["processed"] == second["processed"]


def test_invalid_last_accessed_does_not_crash(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "entities" / "Garbage.md",
        {"type": "entity", "last_accessed": "not-a-date", "importance": 0.5, "tags": []},
        "body",
    )
    result = decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)
    assert result["processed"] == 1
