"""BM25 body-content scoring — the fix for the 'answer is in a body
but not a title' failure mode that caused 90% of MemoryAgentBench failures."""
from __future__ import annotations

from pathlib import Path

from core import retrieval
from utils import frontmatter, indexer


def _node(vault: Path, folder: str, name: str, body: str, tags: list[str] | None = None) -> Path:
    path = vault / folder / f"{name}.md"
    frontmatter.write(path, {
        "type": folder[:-1] if folder.endswith("s") else folder,
        "tags": tags or [],
        "importance": 0.5,
        "decay_weight": 0.5,
    }, body)
    return path


def test_bm25_picks_node_whose_body_matches(tmp_vault: Path, config):
    """The core claim: a node whose title doesn't match the query but whose
    body contains the query terms should be retrieved when BM25 is enabled."""
    _node(tmp_vault, "entities", "Alpha",
          "Random content with no distinctive terms at all.")
    _node(tmp_vault, "entities", "Beta",
          "The spouse of the author Charles Dickens was Catherine Hogarth. "
          "Catherine was a citizen of Belgium.")
    _node(tmp_vault, "entities", "Gamma",
          "Yet more unrelated filler paragraphs about nothing in particular.")

    files = retrieval.retrieve(
        tmp_vault, "Charles Dickens spouse citizenship", [], config,
    )
    names = [Path(p).stem for p, _ in files]
    assert "Beta" in names, (
        f"Beta contains the answer in its body but wasn't retrieved. Got: {names}"
    )


def test_bm25_score_zero_without_query_tokens(tmp_vault: Path, config):
    _node(tmp_vault, "entities", "X", "any body here")
    index = indexer.build(tmp_vault)
    scores = retrieval._compute_bm25_scores(index, set())
    assert all(s == 0.0 for s in scores.values())


def test_bm25_handles_empty_bodies(tmp_vault: Path, config):
    _node(tmp_vault, "entities", "Empty", "")
    _node(tmp_vault, "entities", "Full", "has real content about dickens.")
    index = indexer.build(tmp_vault)
    scores = retrieval._compute_bm25_scores(index, {"dickens"})
    assert scores["Full"] > scores["Empty"]


def test_bm25_disabled_when_weight_zero(tmp_vault: Path, config):
    """Setting body_bm25 weight to 0 should skip the BM25 pass entirely.
    No crashes, no extra work, same behavior as before."""
    config["retrieval"]["weights"] = {
        "tag_overlap": 0.4,
        "keyword_in_title": 0.3,
        "body_bm25": 0.0,
        "decay_weight": 0.2,
        "recency": 0.1,
    }
    _node(tmp_vault, "entities", "Dickens's Spouse",
          "body is irrelevant because title carries the signal")
    files = retrieval.retrieve(tmp_vault, "dickens spouse", [], config)
    assert any("Dickens" in p for p, _ in files)


def test_existing_title_match_still_wins(tmp_vault: Path, config):
    """BM25 is additive — a node aligned on title AND body should be in
    the top results. We don't assert strict ordering because BM25 IDF
    over tiny corpora is sensitive to term frequency."""
    _node(tmp_vault, "entities", "Charles Dickens",
          "Charles Dickens wrote Our Mutual Friend and many other novels. "
          "Married Catherine Hogarth. Lived in London.")
    _node(tmp_vault, "entities", "Unrelated",
          "this paragraph just happens to mention charles dickens in passing")

    files = retrieval.retrieve(
        tmp_vault, "Charles Dickens Mutual Friend", [], config,
    )
    names = [Path(p).stem for p, _ in files]
    assert "Charles Dickens" in names[:2], (
        f"Charles Dickens should be in top 2. Got: {names}"
    )


def test_score_node_accepts_bm25_kwarg(tmp_vault: Path):
    """Regression: score_node signature must remain backward-compatible —
    old callers without the kwarg must still work."""
    _node(tmp_vault, "entities", "X", "body", tags=["t"])
    index = indexer.build(tmp_vault)
    node = index["X"]

    weights = {"tag_overlap": 0.5, "keyword_in_title": 0.3,
               "body_bm25": 0.1, "decay_weight": 0.05, "recency": 0.05}

    s_no_bm25 = retrieval.score_node(node, {"x"}, {"t"}, weights)
    s_with_bm25 = retrieval.score_node(node, {"x"}, {"t"}, weights, body_bm25=1.0)
    assert s_with_bm25 > s_no_bm25


def test_bm25_works_in_synthetic_large_vault(synthetic_large_vault: Path, config):
    """Performance sanity: BM25 over 500 nodes should not crash or take forever."""
    import time as _t
    start = _t.perf_counter()
    files = retrieval.retrieve(
        synthetic_large_vault, "Node_0100 random query", ["alpha"], config,
    )
    elapsed = _t.perf_counter() - start
    assert elapsed < 3.0, f"BM25 on 500 nodes took too long: {elapsed:.2f}s"
    assert len(files) > 0
