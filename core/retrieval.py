from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer, wikilinks

try:
    # BM25L handles small corpora and short documents better than BM25Okapi —
    # on our <10-node test vaults the Okapi variant produces IDF = 0 when a
    # term appears in ~half the docs. BM25L doesn't have that quirk.
    from rank_bm25 import BM25L as _BM25
except ImportError:
    _BM25 = None  # type: ignore[assignment]

try:
    from core import embeddings as _emb
except Exception:  # pragma: no cover
    _emb = None  # type: ignore[assignment]

WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if len(w) > 2}


def _recency_score(last_accessed: Any) -> float:
    if not last_accessed:
        return 0.0
    try:
        d = date.fromisoformat(str(last_accessed))
    except ValueError:
        return 0.0
    days = (date.today() - d).days
    if days <= 0:
        return 1.0
    return max(0.0, 1.0 - days / 90.0)


def score_node(
    node: dict[str, Any],
    query_tokens: set[str],
    query_tags: set[str],
    weights: dict[str, float],
    body_bm25: float = 0.0,
) -> float:
    node_tags = {str(t).lower() for t in node.get("tags", [])}
    title_tokens = _tokenize(Path(node["path"]).stem)

    tag_overlap = len(node_tags & query_tags) / max(len(query_tags), 1) if query_tags else 0.0
    title_hits = len(title_tokens & query_tokens) / max(len(title_tokens), 1) if title_tokens else 0.0
    decay = float(node.get("decay_weight", 0.0))
    recency = _recency_score(node.get("last_accessed"))

    if node.get("archived"):
        decay *= 0.3

    return (
        weights.get("tag_overlap", 0.0) * tag_overlap
        + weights.get("keyword_in_title", 0.0) * title_hits
        + weights.get("body_bm25", 0.0) * body_bm25
        + weights.get("decay_weight", 0.0) * decay
        + weights.get("recency", 0.0) * recency
    )


def _compute_bm25_scores(
    index: dict[str, dict[str, Any]], query_tokens: set[str],
) -> dict[str, float]:
    """Lexical scoring over node bodies. Normalized to [0, 1] by the max score
    in this batch, so it composes linearly with the other [0, 1] signals."""
    if _BM25 is None or not query_tokens:
        return {name: 0.0 for name in index}

    names: list[str] = []
    corpus: list[list[str]] = []
    for name, node in index.items():
        try:
            _, body = frontmatter.read(node["path"])
        except Exception:
            body = ""
        tokens = [w.lower() for w in WORD.findall(body) if len(w) > 2]
        names.append(name)
        corpus.append(tokens or ["_placeholder_"])

    try:
        bm25 = _BM25(corpus)
        raw = bm25.get_scores([q.lower() for q in query_tokens])
    except Exception:
        return {name: 0.0 for name in index}

    if len(raw) == 0:
        return {name: 0.0 for name in index}

    max_score = max(raw)
    if max_score <= 0:
        return {name: 0.0 for name in names}

    return {name: float(s) / max_score for name, s in zip(names, raw)}


def select_entry_points(
    index: dict[str, dict[str, Any]],
    context: str,
    context_tags: list[str],
    weights: dict[str, float],
    n: int,
) -> list[str]:
    query_tokens = _tokenize(context)
    query_tags = {t.lower() for t in context_tags}

    # BM25 is expensive — skip when its weight is 0.
    bm25_map: dict[str, float] = {}
    if weights.get("body_bm25", 0.0) > 0:
        bm25_map = _compute_bm25_scores(index, query_tokens)

    scored = [
        (
            score_node(
                node, query_tokens, query_tags, weights,
                body_bm25=bm25_map.get(name, 0.0),
            ),
            name,
        )
        for name, node in index.items()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for score, name in scored[:n] if score > 0]


def _rrf_fuse(
    rankings: list[list[str]],
    n: int,
    k: int = 60,
) -> list[str]:
    """Reciprocal-rank fusion: combine multiple ranked lists into one.

    A node's fused score is Σ 1/(k + rank_in_list_i). Whichever list a node
    appears in contributes — so a node strong under BM25 OR embeddings wins
    ties, and a node strong under BOTH lenses wins outright. k=60 is the
    standard tuning constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, name in enumerate(ranking):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)
    return [name for name, _ in fused[:n]]


def expand_graph(
    vault_path: str | Path,
    index: dict[str, dict[str, Any]],
    entry_names: list[str],
    hops: int,
) -> list[str]:
    seen: set[str] = set()
    frontier = list(entry_names)

    for _ in range(max(hops, 0) + 1):
        next_frontier: list[str] = []
        for name in frontier:
            if name in seen or name not in index:
                continue
            seen.add(name)
            _, body = frontmatter.read(index[name]["path"])
            for link in wikilinks.extract(body):
                if link not in seen and link in index:
                    next_frontier.append(link)
        frontier = next_frontier
        if not frontier:
            break

    return [index[name]["path"] for name in seen]


def retrieve(
    vault_path: str | Path,
    context: str,
    context_tags: list[str],
    config: dict[str, Any],
    include_transcripts: bool = False,
) -> list[tuple[str, str]]:
    index = indexer.build(vault_path)
    # Identity files are loaded separately into the system prompt — never retrieved.
    index = {n: v for n, v in index.items() if v.get("type") != "identity"}
    if not include_transcripts:
        index = {n: v for n, v in index.items() if v.get("type") != "transcript"}
    r = config["retrieval"]

    hops = r["hops"]
    if len(index) > r["dense_vault_threshold"]:
        hops = max(hops - 1, 1)

    # Lens 1: BM25 + tag + title + decay — current scoring path.
    lex_entries = select_entry_points(
        index=index,
        context=context,
        context_tags=context_tags,
        weights=r["weights"],
        n=r["entry_points"],
    )

    # Lens 2: semantic embeddings (opt-in via config). Runs only when the
    # sentence-transformers dep is installed and an index has been built.
    entries: list[str]
    semantic_on = bool(r.get("semantic")) and _emb is not None and _emb.is_available()
    if semantic_on:
        sem_top_k = int(r.get("semantic_top_k", r["entry_points"]))
        sem_pairs = _emb.semantic_seeds(vault_path, context, k=sem_top_k)
        # Filter out names not in current index (archived/transcript after build).
        sem_entries = [n for n, _ in sem_pairs if n in index]
        # Fuse via reciprocal-rank fusion — keeps each lens's best picks near
        # the top, rewards agreement between them.
        entries = _rrf_fuse(
            [lex_entries, sem_entries],
            n=r["entry_points"],
        )
    else:
        entries = lex_entries

    paths = expand_graph(vault_path, index, entries, hops)

    out: list[tuple[str, str]] = []
    for p in paths:
        out.append((p, Path(p).read_text(encoding="utf-8")))
    return out
