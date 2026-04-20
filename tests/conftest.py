from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _force_echo(monkeypatch):
    """Every test uses the echo backend by default — never hits a real API."""
    monkeypatch.setenv("MEMORY_BACKEND", "echo")


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    for folder in ("entities", "concepts", "decisions", "episodes",
                   "tensions", "questions", "procedures", "_meta"):
        (vault / folder).mkdir(parents=True)
    return vault


@pytest.fixture
def config(tmp_vault: Path) -> dict[str, Any]:
    return {
        "vault_path": str(tmp_vault),
        "providers": {
            "echo": {"base_url": None, "api_key_env": None},
        },
        "models": {
            "model1":  {"provider": "echo", "model": "echo"},
            "routine": {"provider": "echo", "model": "echo"},
            "deep":    {"provider": "echo", "model": "echo"},
        },
        "retrieval": {
            "entry_points": 8,
            "hops": 2,
            "dense_vault_threshold": 1000,
            "weights": {
                "tag_overlap": 0.40,
                "keyword_in_title": 0.30,
                "decay_weight": 0.20,
                "recency": 0.10,
            },
        },
        "decay": {"lambda": 0.05, "archive_threshold": 0.10},
        "monitor": {
            "hub_backlink_limit": 5,
            "tag_vocabulary_limit": 50,
            "orphan_ratio_limit": 0.3,
            "archived_ratio_limit": 0.3,
            "tension_age_days_limit": 30,
        },
        "reflection": {"min_flags_for_reflection": 1},
        "session": {"max_context_files": 20, "history_window": 12},
        "mcp": {"server_name": "test", "default_search_limit": 8},
    }


def _write_node(vault: Path, folder: str, name: str, body: str, fm: dict[str, Any]) -> Path:
    path = vault / folder / f"{name}.md"
    import yaml
    dumped = yaml.dump(fm, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{dumped}---\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def seeded_vault(tmp_vault: Path) -> Path:
    """A small vault with a known link graph used by behavioural tests.

    Graph:
      AuthService  --> [[JWT Strategy]], [[Login Bug]], [[Acme Corp]]
      JWT Strategy --> [[AuthService]]
      Login Bug    --> [[AuthService]], [[JWT Strategy]]
      Acme Corp    --> [[Redis Decision]]
      Redis Decision --> [[AuthService]]
      Mobile Team  (orphan, no outbound/inbound)
    """
    today = date.today()
    recent = today.isoformat()
    old = (today - timedelta(days=120)).isoformat()

    base = {
        "created": recent,
        "last_accessed": recent,
        "access_count": 3,
        "importance": 0.6,
        "decay_weight": 0.6,
        "connection_count": 0,
        "surprise_flag": False,
        "archived": False,
    }

    _write_node(tmp_vault, "entities", "AuthService",
                "The auth service. Links to [[JWT Strategy]], [[Login Bug]], [[Acme Corp]].",
                {**base, "type": "entity", "tags": ["backend", "auth"], "importance": 0.9})
    _write_node(tmp_vault, "concepts", "JWT Strategy",
                "JWT approach. See [[AuthService]].",
                {**base, "type": "concept", "tags": ["auth", "strategy"]})
    _write_node(tmp_vault, "episodes", "Login Bug",
                "Production bug in March. Related: [[AuthService]], [[JWT Strategy]].",
                {**base, "type": "episode", "tags": ["incident", "auth"]})
    _write_node(tmp_vault, "entities", "Acme Corp",
                "Client. Drove [[Redis Decision]].",
                {**base, "type": "entity", "tags": ["client"]})
    _write_node(tmp_vault, "decisions", "Redis Decision",
                "Dropped Redis sessions for [[AuthService]].",
                {**base, "type": "decision", "tags": ["backend", "architecture"]})
    _write_node(tmp_vault, "entities", "Mobile Team",
                "Orphan node — no inbound or outbound wikilinks.",
                {**base, "type": "entity", "tags": ["team"]})
    _write_node(tmp_vault, "concepts", "Old Stale Concept",
                "Ancient node nothing links to.",
                {**base, "type": "concept", "tags": ["legacy"],
                 "last_accessed": old, "access_count": 1, "importance": 0.3})

    return tmp_vault


@pytest.fixture
def synthetic_large_vault(tmp_vault: Path) -> Path:
    """Generates 500 nodes with realistic cross-linking for scale tests."""
    import random
    random.seed(42)

    today = date.today().isoformat()
    names = [f"Node_{i:04d}" for i in range(500)]
    tag_pool = ["alpha", "beta", "gamma", "delta", "epsilon",
                "core", "aux", "infra", "domain", "meta"]

    for i, name in enumerate(names):
        folder = ("entities", "concepts", "decisions",
                  "episodes", "procedures")[i % 5]
        outbound = random.sample([n for j, n in enumerate(names) if j != i], k=random.randint(0, 5))
        body = f"{name} body. Links: " + " ".join(f"[[{t}]]" for t in outbound)
        fm = {
            "type": folder[:-1] if folder.endswith("s") else folder,
            "created": today,
            "last_accessed": today,
            "access_count": random.randint(1, 10),
            "importance": random.uniform(0.3, 0.9),
            "decay_weight": random.uniform(0.3, 0.9),
            "connection_count": len(outbound),
            "surprise_flag": False,
            "archived": False,
            "tags": random.sample(tag_pool, k=random.randint(1, 3)),
        }
        _write_node(tmp_vault, folder, name, body, fm)
    return tmp_vault
