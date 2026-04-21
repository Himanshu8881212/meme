"""Semantic retrieval — embeddings + hybrid RRF."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import embeddings, retrieval
from utils import frontmatter

pytestmark = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="sentence-transformers not installed",
)


def _write(vault: Path, folder: str, name: str, body: str, tags=None):
    frontmatter.write(
        vault / folder / f"{name}.md",
        {"type": folder.rstrip("s") if folder != "decisions" else "decision",
         "importance": 0.6, "tags": tags or [], "connection_count": 0},
        body,
    )


def test_build_index_roundtrip(tmp_vault: Path):
    _write(tmp_vault, "entities", "Puppet", "Himanshu's black Labrador dog.", ["pet"])
    _write(tmp_vault, "concepts", "Meditation", "Daily breath awareness practice.", ["routine"])
    result = embeddings.build_index(tmp_vault)
    assert result["indexed"] == 2
    assert result["re_encoded"] == 2

    loaded = embeddings.load_index(tmp_vault)
    assert loaded is not None
    names, vecs = loaded
    assert set(names) == {"Puppet", "Meditation"}
    assert vecs.shape == (2, 384)


def test_index_is_incremental(tmp_vault: Path):
    _write(tmp_vault, "entities", "A", "alpha content here.", ["t1"])
    _write(tmp_vault, "entities", "B", "beta content here.", ["t2"])
    first = embeddings.build_index(tmp_vault)
    assert first["re_encoded"] == 2

    # Running again should re-encode nothing.
    second = embeddings.build_index(tmp_vault)
    assert second["re_encoded"] == 0
    assert second["indexed"] == 2

    # Add a new node — only it gets encoded.
    _write(tmp_vault, "entities", "C", "gamma new content.", ["t3"])
    third = embeddings.build_index(tmp_vault)
    assert third["re_encoded"] == 1
    assert third["indexed"] == 3


def test_archived_nodes_excluded(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "entities" / "Live.md",
        {"type": "entity", "importance": 0.5, "tags": [], "connection_count": 0},
        "This node is live.",
    )
    frontmatter.write(
        tmp_vault / "entities" / "Gone.md",
        {"type": "entity", "archived": True, "importance": 0.5, "tags": [],
         "connection_count": 0},
        "This node is archived.",
    )
    embeddings.build_index(tmp_vault)
    names, _ = embeddings.load_index(tmp_vault)
    assert "Live" in names
    assert "Gone" not in names


def test_semantic_seeds_finds_paraphrase(tmp_vault: Path):
    """The whole point of embeddings: 'my dog' should find 'Puppet the Labrador'
    even though the query shares zero tokens with the node."""
    _write(tmp_vault, "entities", "Puppet the Labrador",
           "Male black lab, family pet, loves walks and treats.", ["animal"])
    _write(tmp_vault, "entities", "React Router",
           "Client-side routing library for single-page applications.", ["code"])
    _write(tmp_vault, "entities", "Mount Fuji",
           "Active stratovolcano in Japan, iconic summit for hikers.", ["geography"])
    embeddings.build_index(tmp_vault)

    seeds = embeddings.semantic_seeds(tmp_vault, "my dog", k=3)
    # Puppet the Labrador MUST be the top result — the only animal in the set.
    assert seeds[0][0] == "Puppet the Labrador"


def test_hybrid_retrieve_unions_both_lenses(tmp_vault: Path, config):
    """Verify that when retrieval.semantic is on, embedding seeds plus BM25
    seeds both contribute. A query sharing no tokens with the 'dog' node
    should still surface it via the semantic lens."""
    _write(tmp_vault, "entities", "Puppet the Labrador",
           "Male black lab, family pet, loves walks and treats.", ["animal"])
    _write(tmp_vault, "entities", "Alphabet Soup",
           "Random unrelated node about cooking letters in broth.", ["food"])
    embeddings.build_index(tmp_vault)

    config["retrieval"]["semantic"] = True
    config["retrieval"]["semantic_top_k"] = 2
    config["retrieval"]["entry_points"] = 2
    config["retrieval"]["hops"] = 0  # isolate to just the seed selection

    results = retrieval.retrieve(tmp_vault, "my dog", [], config)
    names = {Path(p).stem for p, _ in results}
    assert "Puppet the Labrador" in names, f"got: {names}"


def test_rrf_fuse_rewards_multi_lens_agreement():
    """Reciprocal-rank fusion: a node that appears in both rankings beats
    a node that tops only one. Pure algorithmic test — no embeddings needed."""
    lex = ["A", "B", "C", "D"]
    sem = ["C", "A", "E", "F"]
    fused = retrieval._rrf_fuse([lex, sem], n=4)
    # A: 1/(60+0) + 1/(60+1) = 0.0330, C: 1/(60+2) + 1/(60+0) = 0.0328 → A beats C
    # But both should outrank B (only in lex) and E (only in sem).
    assert fused[0] in ("A", "C")
    assert fused[1] in ("A", "C")
    assert fused[0] != fused[1]
    assert {"A", "C"} <= set(fused[:2])


def test_semantic_seeds_empty_when_no_index(tmp_vault: Path):
    seeds = embeddings.semantic_seeds(tmp_vault, "anything", k=5)
    assert seeds == []


def test_retrieve_works_with_semantic_off(tmp_vault: Path, config):
    """Flipping the flag off leaves behavior unchanged — no regression."""
    _write(tmp_vault, "entities", "A", "alpha beta gamma content.", ["t"])
    config["retrieval"]["semantic"] = False
    config["retrieval"]["entry_points"] = 1
    config["retrieval"]["hops"] = 0
    results = retrieval.retrieve(tmp_vault, "alpha", [], config)
    assert len(results) == 1
