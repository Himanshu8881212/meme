"""Tests for the verbatim-recall path: grep_vault + transcripts_by_date +
the Model 1 tools memory_grep and memory_by_date."""
from __future__ import annotations

from pathlib import Path

from core import reflection
from core.tools import grep_vault, transcripts_by_date
from scheduler import session as session_mgr
from utils import frontmatter


def _transcript(vault: Path, name: str, body: str) -> Path:
    """Write a fake transcript with the expected filename format."""
    path = vault / "_transcripts" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(path, {"type": "transcript", "immutable": True}, body)
    return path


def test_transcripts_by_date_single_day(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-03-143000-debugging", "USER: hi on march 3")
    _transcript(tmp_vault, "2026-03-04-090000-planning", "USER: march 4 talk")
    _transcript(tmp_vault, "2026-04-20-100000-other", "USER: other day")

    names = transcripts_by_date(tmp_vault, "2026-03-03")
    assert len(names) == 1
    assert "2026-03-03-143000-debugging" in names


def test_transcripts_by_date_range(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-01-100000-a", "USER: a")
    _transcript(tmp_vault, "2026-03-15-100000-b", "USER: b")
    _transcript(tmp_vault, "2026-03-31-100000-c", "USER: c")
    _transcript(tmp_vault, "2026-04-01-100000-d", "USER: d")

    names = transcripts_by_date(tmp_vault, "2026-03-10", "2026-03-31")
    assert len(names) == 2
    assert any("2026-03-15" in n for n in names)
    assert any("2026-03-31" in n for n in names)


def test_transcripts_by_date_empty_when_no_match(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-01-100000-a", "USER: a")
    assert transcripts_by_date(tmp_vault, "2026-05-01") == []


def test_transcripts_by_date_handles_bad_input(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-01-100000-a", "USER: a")
    assert transcripts_by_date(tmp_vault, "not-a-date") == []


def test_grep_vault_finds_in_transcripts(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-03-120000-checkout",
                "## USER\nLet's discuss the checkout flow bug.\n"
                "## ASSISTANT\nI suggest checking the tRPC config.\n")
    hits = grep_vault(tmp_vault, "tRPC")
    assert len(hits) == 1
    assert hits[0]["path"].endswith("2026-03-03-120000-checkout.md")
    assert "tRPC" in hits[0]["snippet"]


def test_grep_vault_finds_in_entities(tmp_vault: Path):
    frontmatter.write(
        tmp_vault / "entities" / "Auth.md",
        {"type": "entity", "tags": ["backend"]},
        "Auth is handled by JWT tokens.",
    )
    hits = grep_vault(tmp_vault, "JWT")
    assert len(hits) >= 1
    assert "entities/Auth.md" in hits[0]["path"]


def test_grep_vault_case_insensitive(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-03-120000-a", "FoObAr content")
    assert len(grep_vault(tmp_vault, "foobar")) == 1
    assert len(grep_vault(tmp_vault, "FOOBAR")) == 1


def test_grep_vault_respects_limit(tmp_vault: Path):
    body = "needle\n" * 50
    _transcript(tmp_vault, "2026-03-03-120000-big", body)
    hits = grep_vault(tmp_vault, "needle", limit=5)
    assert len(hits) == 5


def test_grep_vault_skips_archive(tmp_vault: Path):
    archive_dir = tmp_vault / "_archive" / "2026-03-01" / "entities"
    archive_dir.mkdir(parents=True)
    (archive_dir / "Old.md").write_text("---\ntype: entity\n---\nneedle", encoding="utf-8")
    hits = grep_vault(tmp_vault, "needle")
    assert not any("_archive" in h["path"] for h in hits)


def test_grep_vault_can_exclude_transcripts(tmp_vault: Path):
    _transcript(tmp_vault, "2026-03-03-120000-a", "needle in transcript")
    frontmatter.write(
        tmp_vault / "concepts" / "Concept.md",
        {"type": "concept"}, "needle in concept",
    )
    hits = grep_vault(tmp_vault, "needle", include_transcripts=False)
    assert len(hits) == 1
    assert "concepts/Concept.md" in hits[0]["path"]


def test_model1_tool_memory_grep(tmp_vault: Path, config):
    _transcript(tmp_vault, "2026-03-03-120000-x",
                "USER: the rate limiter should use token bucket")
    result = reflection._model1_tool_dispatch(
        tmp_vault, "memory_grep", {"phrase": "token bucket"}, config,
    )
    assert "token bucket" in result.lower()
    assert "2026-03-03" in result


def test_model1_tool_memory_by_date(tmp_vault: Path, config):
    _transcript(tmp_vault, "2026-03-03-120000-debugging", "x")
    _transcript(tmp_vault, "2026-04-20-090000-other", "y")
    result = reflection._model1_tool_dispatch(
        tmp_vault, "memory_by_date", {"start_date": "2026-03-03"}, config,
    )
    assert "2026-03-03" in result
    assert "2026-04-20" not in result


def test_verbatim_recall_round_trip(seeded_vault: Path, config):
    """End-to-end: write a transcript via session_mgr, recall it via grep."""
    config["vault_path"] = str(seeded_vault)
    name = session_mgr.archive_transcript(
        seeded_vault,
        transcript="USER: I just bought a designer handbag for $800\nASSISTANT: nice.",
        task="shopping chat",
        tags=["shopping"],
    )
    assert name
    hits = grep_vault(seeded_vault, "$800")
    assert len(hits) >= 1
    assert any(name in h["path"] for h in hits)
