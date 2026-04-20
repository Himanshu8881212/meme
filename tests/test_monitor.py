from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from core import monitor
from utils import frontmatter


def test_empty_vault_metrics(tmp_vault: Path, config):
    m = monitor.collect(tmp_vault)
    assert m["total_nodes"] == 0
    assert m["archived_ratio"] == 0.0
    triggers = monitor.check_thresholds(m, config)
    assert triggers == []


def test_hub_detection(seeded_vault: Path, config):
    m = monitor.collect(seeded_vault)
    hubs = dict(m["top_hubs"])
    assert hubs["AuthService"] == 3


def test_hub_triggers_when_over_limit(seeded_vault: Path, config):
    config["monitor"]["hub_backlink_limit"] = 2
    m = monitor.collect(seeded_vault)
    triggers = monitor.check_thresholds(m, config)
    assert any("hub_split:AuthService" in t for t in triggers)


def test_orphan_ratio(seeded_vault: Path, config):
    m = monitor.collect(seeded_vault)
    assert m["orphans"] >= 2  # Mobile Team + Old Stale Concept have no inbound
    assert m["orphan_ratio"] > 0.0


def test_tag_vocabulary_count(seeded_vault: Path):
    m = monitor.collect(seeded_vault)
    # backend, auth, strategy, incident, client, architecture, team, legacy
    assert m["tag_vocabulary"] >= 7


def test_tag_trigger_when_vocabulary_too_large(tmp_vault: Path, config):
    for i in range(60):
        frontmatter.write(
            tmp_vault / "concepts" / f"N{i}.md",
            {"type": "concept", "tags": [f"tag_{i}"], "importance": 0.5},
            "body",
        )
    config["monitor"]["tag_vocabulary_limit"] = 50
    m = monitor.collect(tmp_vault)
    triggers = monitor.check_thresholds(m, config)
    assert any("tag_consolidation" in t for t in triggers)


def test_stale_tension_trigger(tmp_vault: Path, config):
    old = (date.today() - timedelta(days=100)).isoformat()
    frontmatter.write(
        tmp_vault / "tensions" / "Old Tension.md",
        {"type": "tension", "created": old, "tags": ["design"], "importance": 0.6},
        "body",
    )
    config["monitor"]["tension_age_days_limit"] = 30
    m = monitor.collect(tmp_vault)
    triggers = monitor.check_thresholds(m, config)
    assert any("stale_tension" in t for t in triggers)


def test_type_counts_populated(seeded_vault: Path):
    m = monitor.collect(seeded_vault)
    counts = m["type_counts"]
    assert counts.get("entity", 0) >= 3
    assert counts.get("concept", 0) >= 2
    assert counts.get("decision", 0) >= 1
