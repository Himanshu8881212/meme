"""Qualitative scenarios. These are the 'does the memory actually work?' tests —
behavior-focused, not unit-focused. They exercise the system across multi-step
situations: accumulation, drift, forgetting, bridge discovery."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from core import decay, monitor, retrieval
from utils import frontmatter, indexer

pytestmark = pytest.mark.scenario


def _node(vault: Path, folder: str, name: str, body: str, **fm) -> Path:
    defaults = {
        "type": folder[:-1] if folder.endswith("s") else folder,
        "created": date.today().isoformat(),
        "last_accessed": date.today().isoformat(),
        "access_count": 1,
        "importance": 0.5,
        "decay_weight": 0.5,
        "connection_count": 0,
        "archived": False,
        "tags": [],
    }
    defaults.update(fm)
    path = vault / folder / f"{name}.md"
    frontmatter.write(path, defaults, body)
    return path


def test_scenario_contradiction_surfaces_in_retrieval(tmp_vault: Path, config):
    """A node asserts X; a later session retrieves the vault and
    can see BOTH the original claim and the contradicting episode."""
    _node(tmp_vault, "decisions", "Redis Decision",
          "We dropped Redis sessions. See [[AuthService]].",
          tags=["backend", "auth"], importance=0.8)
    _node(tmp_vault, "entities", "AuthService",
          "The auth service.", tags=["backend", "auth"], importance=0.9)
    _node(tmp_vault, "episodes", "Redis still active in staging",
          "Contradicts [[Redis Decision]] — staging is still using it.",
          tags=["backend", "auth", "incident"], importance=0.7)

    results = retrieval.retrieve(tmp_vault, "redis", ["backend", "auth"], config)
    names = [Path(p).stem for p, _ in results]

    assert "Redis Decision" in names
    assert "Redis still active in staging" in names


def test_scenario_repeated_access_resets_decay_clock(tmp_vault: Path):
    """A node accessed many times should decay more slowly than an equivalent
    untouched node. This is the 'rehearsal' behaviour humans exhibit."""
    old_date = (date.today() - timedelta(days=60)).isoformat()

    _node(tmp_vault, "concepts", "Often Accessed",
          "often referenced",
          last_accessed=old_date, access_count=50, importance=0.5, connection_count=3)
    _node(tmp_vault, "concepts", "Rarely Accessed",
          "rarely referenced",
          last_accessed=old_date, access_count=1, importance=0.5, connection_count=3)

    decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)

    fm_often, _ = frontmatter.read(tmp_vault / "concepts" / "Often Accessed.md")
    fm_rare, _ = frontmatter.read(tmp_vault / "concepts" / "Rarely Accessed.md")

    assert fm_often["decay_weight"] > fm_rare["decay_weight"], (
        "high-access node should decay slower"
    )


def test_scenario_hub_emerges_as_backlinks_accumulate(tmp_vault: Path, config):
    """Seeing a node referenced by many others should promote it as a hub in
    monitor output, eventually crossing the hub-split threshold."""
    _node(tmp_vault, "entities", "Central", "The central thing.",
          tags=["core"], importance=0.9)
    for i in range(8):
        _node(tmp_vault, "episodes", f"Event {i}",
              f"Event referencing [[Central]].",
              tags=["event"], importance=0.4)

    m = monitor.collect(tmp_vault)
    hubs = dict(m["top_hubs"])
    assert hubs["Central"] == 8

    config["monitor"]["hub_backlink_limit"] = 5
    triggers = monitor.check_thresholds(m, config)
    assert any("hub_split:Central" in t for t in triggers)


def test_scenario_orphan_detection_over_time(tmp_vault: Path):
    """Orphan nodes (no backlinks) should be detected by monitor so a deep
    reflection pass can consider linking or archiving them."""
    _node(tmp_vault, "entities", "Connected A", "Links to [[Connected B]].")
    _node(tmp_vault, "entities", "Connected B", "Links to [[Connected A]].")
    _node(tmp_vault, "concepts", "Forgotten", "No links in or out.")

    m = monitor.collect(tmp_vault)
    # Forgotten has no backlinks; it should be counted as an orphan.
    assert m["orphans"] >= 1


def test_scenario_decay_weight_distinguishes_old_vs_new(tmp_vault: Path):
    """After a decay pass, recent nodes should have higher weight than old ones,
    even if all other parameters are equal."""
    recent = date.today().isoformat()
    ancient = (date.today() - timedelta(days=400)).isoformat()

    _node(tmp_vault, "concepts", "Recent", "new thing",
          last_accessed=recent, access_count=1, connection_count=1, importance=0.6)
    _node(tmp_vault, "concepts", "Ancient", "old thing",
          last_accessed=ancient, access_count=1, connection_count=1, importance=0.6)

    decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)

    fm_r, _ = frontmatter.read(tmp_vault / "concepts" / "Recent.md")
    fm_a, _ = frontmatter.read(tmp_vault / "concepts" / "Ancient.md")
    assert fm_r["decay_weight"] > fm_a["decay_weight"]


def test_scenario_cross_cluster_bridge_wins_on_multi_tag_query(tmp_vault: Path, config):
    """A node tagged with BOTH clusters (auth + frontend) should outrank nodes
    tagged with only one when the query spans both."""
    _node(tmp_vault, "entities", "Backend Auth",
          "backend-only auth logic.", tags=["auth", "backend"])
    _node(tmp_vault, "entities", "Frontend UI",
          "frontend-only UI logic.", tags=["frontend", "ui"])
    _node(tmp_vault, "concepts", "Auth UI Bridge",
          "Connects backend auth with frontend UI flows.",
          tags=["auth", "frontend"])

    index = indexer.build(tmp_vault)
    weights = config["retrieval"]["weights"]
    query_tags = {"auth", "frontend"}

    scores = {
        name: retrieval.score_node(node, set(), query_tags, weights)
        for name, node in index.items()
    }
    # The bridge node has a perfect 2/2 tag overlap; the others have 1/2.
    assert scores["Auth UI Bridge"] >= scores["Backend Auth"]
    assert scores["Auth UI Bridge"] >= scores["Frontend UI"]


def test_scenario_archived_node_stays_retrievable_but_deprioritised(tmp_vault: Path, config):
    """Archived nodes aren't deleted — they can still be found, but score lower
    than an equivalent unarchived node. This is the 'quiet but not gone' test."""
    _node(tmp_vault, "concepts", "Active", "active node", tags=["x"], decay_weight=0.8, archived=False)
    _node(tmp_vault, "concepts", "Archived", "archived node", tags=["x"], decay_weight=0.8, archived=True)

    index = indexer.build(tmp_vault)
    weights = config["retrieval"]["weights"]
    s_active = retrieval.score_node(index["Active"], set(), {"x"}, weights)
    s_archived = retrieval.score_node(index["Archived"], set(), {"x"}, weights)
    assert s_active > s_archived
    # Both should still be present in the index.
    assert "Archived" in index


def test_scenario_empty_vault_session_does_not_crash(tmp_vault: Path, config):
    """The bootstrap problem — an empty vault should degrade gracefully."""
    results = retrieval.retrieve(tmp_vault, "anything at all", [], config)
    assert results == []

    m = monitor.collect(tmp_vault)
    assert m["total_nodes"] == 0
    assert monitor.check_thresholds(m, config) == []
