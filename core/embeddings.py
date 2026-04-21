"""Semantic retrieval layer — a parallel lens to BM25/tag scoring.

Produces the "semantic" seed set that complements lexical (BM25) and
graph (wikilinks). Each lens catches what the others miss:

    BM25        →  verbatim recall, rare terms
    embeddings  →  paraphrase, concept similarity
    wikilinks   →  multi-hop relationships

The three are combined via reciprocal-rank fusion in retrieval.py.

Local model (sentence-transformers/all-MiniLM-L6-v2, 384-dim), no API
calls. Index is a single .npz in _meta/ so the vault stays pure markdown.
Incremental — only re-encodes nodes whose body hash changed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from utils import frontmatter

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:  # pragma: no cover
    HAS_ST = False

_model: Any | None = None
_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
_dim: int = 384  # all-MiniLM-L6-v2 output dimension

INDEX_FILENAME = "embeddings.npz"


def is_available() -> bool:
    """Whether the semantic backend can be used. Callers gate on this."""
    return HAS_ST


def set_model(name: str) -> None:
    """Override the embedding model. Call before the first encode()."""
    global _model, _model_name
    _model = None
    _model_name = name


def _get_model() -> Any:
    global _model, _dim
    if _model is None:
        if not HAS_ST:
            raise RuntimeError(
                "sentence-transformers not installed — "
                "run: pip install sentence-transformers"
            )
        _model = SentenceTransformer(_model_name)
        # sentence-transformers 5+ renamed the method; fall back for older versions.
        for getter in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(_model, getter, None)
            if callable(fn):
                try:
                    _dim = int(fn())
                    break
                except Exception:
                    pass
    return _model


def encode(text: str) -> np.ndarray:
    """Single-text embedding, L2-normalized (so cosine = dot)."""
    m = _get_model()
    vec = m.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def encode_batch(texts: list[str]) -> np.ndarray:
    m = _get_model()
    vecs = m.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32,
    )
    return vecs.astype(np.float32)


# --- per-node text + hashing ------------------------------------------------

def _node_text(name: str, body: str, tags: list[Any]) -> str:
    """What we actually feed the encoder. Name + tags give semantic anchoring
    for short bodies; body is truncated so embedding cost stays bounded."""
    tag_line = "  ".join(str(t) for t in (tags or []))
    return f"{name}\n{tag_line}\n{body[:2000]}"


def _body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _index_path(vault: str | Path) -> Path:
    return Path(vault) / "_meta" / INDEX_FILENAME


def _eligible(fm: dict[str, Any], md_path: Path) -> bool:
    if md_path.name.startswith("."):
        return False
    parts = md_path.parts
    if any(p in parts for p in ("_transcripts", "_identity", "_archive")):
        return False
    if fm.get("type") in ("transcript", "identity"):
        return False
    if fm.get("archived"):
        return False
    return True


# --- index build / load / query ---------------------------------------------

def build_index(vault_path: str | Path, force: bool = False) -> dict[str, Any]:
    """(Re)build the semantic index. Lazy: only re-encodes nodes whose
    node-text hash has changed since the last build.

    Returns {"indexed": N, "re_encoded": M, "path": str}.
    """
    if not HAS_ST:
        return {"indexed": 0, "re_encoded": 0, "path": "(unavailable)"}

    vault = Path(vault_path)
    path = _index_path(vault)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Existing cache keyed by node name.
    cache: dict[str, tuple[str, np.ndarray]] = {}
    if path.exists() and not force:
        try:
            data = np.load(path, allow_pickle=True)
            old_names = data["names"]
            old_hashes = data["hashes"]
            old_vecs = data["vectors"]
            for i, n in enumerate(old_names):
                cache[str(n)] = (str(old_hashes[i]), old_vecs[i])
        except Exception:
            cache = {}

    names: list[str] = []
    hashes: list[str] = []
    vecs: list[np.ndarray] = []
    to_encode_idx: list[int] = []
    to_encode_texts: list[str] = []

    for md_path in vault.rglob("*.md"):
        try:
            fm, body = frontmatter.read(md_path)
        except Exception:
            continue
        if not _eligible(fm, md_path):
            continue
        name = md_path.stem
        text = _node_text(name, body, fm.get("tags") or [])
        h = _body_hash(text)
        names.append(name)
        hashes.append(h)
        cached = cache.get(name)
        if cached and cached[0] == h:
            vecs.append(cached[1])
        else:
            # Placeholder; fill after batch encode.
            vecs.append(None)  # type: ignore[arg-type]
            to_encode_idx.append(len(names) - 1)
            to_encode_texts.append(text)

    if to_encode_texts:
        new_vecs = encode_batch(to_encode_texts)
        for pos, new_vec in zip(to_encode_idx, new_vecs):
            vecs[pos] = new_vec

    if not names:
        # Empty vault → write an empty index (so load_index can distinguish
        # "never built" from "built but empty").
        np.savez(
            path,
            names=np.array([], dtype=object),
            hashes=np.array([], dtype=object),
            vectors=np.zeros((0, _dim), dtype=np.float32),
        )
        return {"indexed": 0, "re_encoded": 0, "path": str(path)}

    matrix = np.stack(vecs).astype(np.float32)
    np.savez(
        path,
        names=np.array(names, dtype=object),
        hashes=np.array(hashes, dtype=object),
        vectors=matrix,
    )
    return {
        "indexed": len(names),
        "re_encoded": len(to_encode_texts),
        "path": str(path),
    }


def load_index(vault_path: str | Path) -> tuple[list[str], np.ndarray] | None:
    """Return (names, vectors) or None if no index exists yet."""
    path = _index_path(vault_path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    names = [str(n) for n in data["names"]]
    vecs = data["vectors"]
    return names, vecs


def semantic_seeds(
    vault_path: str | Path, query: str, k: int = 10,
) -> list[tuple[str, float]]:
    """Return [(name, cosine_sim), ...] — top-k nodes by similarity.
    Empty list if the backend or index is unavailable."""
    if not HAS_ST or not query.strip():
        return []
    loaded = load_index(vault_path)
    if loaded is None:
        return []
    names, vecs = loaded
    if len(names) == 0:
        return []
    q = encode(query)
    # Vectors are already L2-normalized, so cosine = dot product.
    sims = vecs @ q
    order = np.argsort(-sims)[:k]
    return [(names[i], float(sims[i])) for i in order]
