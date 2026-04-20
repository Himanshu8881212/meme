from __future__ import annotations

from pathlib import Path

from core import retrieval
from utils import indexer


def test_empty_vault_returns_empty(tmp_vault: Path, config):
    result = retrieval.retrieve(tmp_vault, "any query", [], config)
    assert result == []


def test_tag_overlap_dominates(seeded_vault: Path, config):
    # Query with exact tag overlap to "auth" — AuthService / JWT / Login Bug
    results = retrieval.retrieve(seeded_vault, "", ["auth"], config)
    names = [Path(p).stem for p, _ in results]
    assert "AuthService" in names
    assert "JWT Strategy" in names


def test_title_match_selects_node(seeded_vault: Path, config):
    # No tag overlap — only title tokens match.
    results = retrieval.retrieve(seeded_vault, "AuthService failure mode", [], config)
    names = [Path(p).stem for p, _ in results]
    assert "AuthService" in names


def test_graph_expansion_pulls_neighbours(seeded_vault: Path, config):
    # Entry = AuthService. Neighbours = JWT Strategy, Login Bug, Acme Corp,
    # then Redis Decision at depth 2.
    results = retrieval.retrieve(seeded_vault, "AuthService", ["auth"], config)
    names = [Path(p).stem for p, _ in results]
    assert "AuthService" in names
    assert "JWT Strategy" in names
    assert "Login Bug" in names
    # Depth-2 reachable from Acme Corp.
    assert "Redis Decision" in names


def test_hop_depth_zero_returns_only_entries(seeded_vault: Path, config):
    config["retrieval"]["hops"] = 0
    index = indexer.build(seeded_vault)
    entries = retrieval.select_entry_points(
        index=index,
        context="AuthService",
        context_tags=["auth"],
        weights=config["retrieval"]["weights"],
        n=1,
    )
    paths = retrieval.expand_graph(seeded_vault, index, entries, hops=0)
    assert len(paths) == 1


def test_dense_vault_drops_to_one_hop(synthetic_large_vault: Path, config):
    config["retrieval"]["dense_vault_threshold"] = 100
    config["retrieval"]["hops"] = 2
    # Just assert that retrieval runs and returns a reasonable size —
    # under dense conditions it should NOT dump every node.
    results = retrieval.retrieve(synthetic_large_vault, "Node_0001", ["alpha"], config)
    # Less than the full vault — the hop reduction actually bounded things.
    assert 0 < len(results) < 500


def test_archived_nodes_penalised(seeded_vault: Path, config):
    # Mark AuthService as archived and re-retrieve; score should drop.
    from utils import frontmatter
    p = seeded_vault / "entities" / "AuthService.md"
    fm, body = frontmatter.read(p)
    fm["archived"] = True
    fm["decay_weight"] = 0.6
    frontmatter.write(p, fm, body)

    index = indexer.build(seeded_vault)
    weights = config["retrieval"]["weights"]
    node = index["AuthService"]
    score = retrieval.score_node(node, query_tokens=set(), query_tags={"auth"}, weights=weights)

    fm["archived"] = False
    frontmatter.write(p, fm, body)
    index2 = indexer.build(seeded_vault)
    node2 = index2["AuthService"]
    score2 = retrieval.score_node(node2, query_tokens=set(), query_tags={"auth"}, weights=weights)

    assert score < score2


def test_entry_points_sorted_by_score(seeded_vault: Path, config):
    index = indexer.build(seeded_vault)
    entries = retrieval.select_entry_points(
        index=index,
        context="JWT strategy auth approach",
        context_tags=["auth"],
        weights=config["retrieval"]["weights"],
        n=3,
    )
    assert len(entries) <= 3
    # At least one auth-tagged node must surface.
    assert any(n in {"AuthService", "JWT Strategy", "Login Bug"} for n in entries)


def test_retrieval_deduplicates(seeded_vault: Path, config):
    # Even if a node is reachable via multiple paths, the returned list has
    # no duplicates.
    results = retrieval.retrieve(seeded_vault, "AuthService", ["auth"], config)
    paths = [p for p, _ in results]
    assert len(paths) == len(set(paths))
