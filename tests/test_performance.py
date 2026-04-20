"""Scale tests. Opt in with `pytest -m slow`."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from core import decay, monitor, retrieval
from utils import indexer

pytestmark = pytest.mark.slow


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def test_index_500_nodes_under_half_second(synthetic_large_vault: Path):
    _, t = _time(lambda: indexer.build(synthetic_large_vault))
    assert t < 0.5, f"index build took {t:.3f}s"


def test_backlink_count_500_under_one_second(synthetic_large_vault: Path):
    _, t = _time(lambda: indexer.backlink_counts(synthetic_large_vault))
    assert t < 1.0, f"backlink scan took {t:.3f}s"


def test_retrieval_500_under_half_second(synthetic_large_vault: Path, config):
    _, t = _time(lambda: retrieval.retrieve(
        synthetic_large_vault, "Node_0100 and some other words", ["alpha"], config,
    ))
    assert t < 0.5, f"retrieval took {t:.3f}s"


def test_decay_500_under_two_seconds(synthetic_large_vault: Path):
    _, t = _time(lambda: decay.run(
        synthetic_large_vault, lambda_=0.05, archive_threshold=0.10,
    ))
    assert t < 2.0, f"decay pass took {t:.3f}s"


def test_monitor_500_under_one_second(synthetic_large_vault: Path):
    _, t = _time(lambda: monitor.collect(synthetic_large_vault))
    assert t < 1.0, f"monitor collect took {t:.3f}s"


def test_retrieval_bounded_size_on_large_vault(synthetic_large_vault: Path, config):
    """Dynamic hop depth should prevent retrieval from returning half the vault."""
    config["retrieval"]["dense_vault_threshold"] = 200
    results = retrieval.retrieve(
        synthetic_large_vault, "Node_0100", ["alpha"], config,
    )
    assert len(results) < 200, f"retrieved too many: {len(results)}"
