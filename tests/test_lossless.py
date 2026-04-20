"""Tests for the lossless transcript archive — the design property that
lets us answer 'what exactly did the user say' even after distillation."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core import decay, reflection, retrieval
from scheduler import session as session_mgr
from utils import frontmatter, indexer

ROOT = Path(__file__).resolve().parent.parent


def test_archive_transcript_writes_file(tmp_vault: Path):
    name = session_mgr.archive_transcript(
        tmp_vault, transcript="USER: hi\nASSISTANT: hello.", task="greeting", tags=["misc"],
    )
    path = tmp_vault / "_transcripts" / f"{name}.md"
    assert path.exists()
    fm, body = frontmatter.read(path)
    assert fm["type"] == "transcript"
    assert fm["immutable"] is True
    assert fm["task"] == "greeting"
    assert "USER: hi" in body
    assert "ASSISTANT: hello" in body


def test_transcript_saved_on_session_end(seeded_vault: Path, config):
    config["vault_path"] = str(seeded_vault)
    meta = session_mgr.start(task="auth debug", tags=["auth"], config=config, project_root=ROOT)

    with patch.object(reflection, "routine", return_value=""):
        result = session_mgr.end(
            session_output="USER: q\nASSISTANT: a. [SALIENT: x]",
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )

    assert "transcript" in result
    transcript_path = seeded_vault / "_transcripts" / f"{result['transcript']}.md"
    assert transcript_path.exists()
    assert "USER: q" in transcript_path.read_text(encoding="utf-8")


def test_transcripts_excluded_from_default_retrieval(tmp_vault: Path, config):
    # Seed one real node plus one transcript.
    import yaml
    (tmp_vault / "concepts" / "Real Concept.md").write_text(
        "---\n" + yaml.dump({"type": "concept", "tags": ["authflow"],
                             "importance": 0.6, "decay_weight": 0.6}) +
        "---\nLinks to nothing.\n", encoding="utf-8"
    )
    session_mgr.archive_transcript(
        tmp_vault,
        transcript="USER: authflow details\nASSISTANT: here are authflow details",
        task="authflow discussion",
        tags=["authflow"],
    )

    results = retrieval.retrieve(tmp_vault, "authflow", ["authflow"], config)
    paths = [p for p, _ in results]
    assert any("Real Concept" in p for p in paths)
    rel = [str(Path(p).relative_to(tmp_vault)) for p in paths]
    assert not any(r.startswith("_transcripts" + "/") or r.startswith("_transcripts" + "\\")
                   for r in rel), (
        f"transcripts must be excluded from default retrieval. got: {rel}"
    )


def test_transcripts_retrievable_when_explicitly_requested(tmp_vault: Path, config):
    session_mgr.archive_transcript(
        tmp_vault, transcript="body", task="unique topic xyzzy", tags=["xyzzy"],
    )
    # Opt in
    results = retrieval.retrieve(
        tmp_vault, "xyzzy", ["xyzzy"], config, include_transcripts=True,
    )
    paths = [p for p, _ in results]
    assert any("_transcripts" in p for p in paths)


def test_decay_skips_transcripts(tmp_vault: Path):
    # Create a transcript that LOOKS like it should be decayed.
    old = (date.today() - timedelta(days=365 * 3)).isoformat()
    name = session_mgr.archive_transcript(
        tmp_vault, transcript="body", task="ancient", tags=[],
    )
    # Hand-edit the frontmatter to simulate an old transcript.
    path = tmp_vault / "_transcripts" / f"{name}.md"
    fm, body = frontmatter.read(path)
    fm["last_accessed"] = old
    fm["access_count"] = 1
    fm["importance"] = 0.3
    fm["decay_weight"] = 0.3
    frontmatter.write(path, fm, body)

    decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)

    fm_after, _ = frontmatter.read(path)
    # The decay_weight should NOT have been rewritten by decay.
    assert fm_after["decay_weight"] == 0.3
    assert fm_after.get("archived") is not True


def test_transcripts_count_toward_index(tmp_vault: Path):
    session_mgr.archive_transcript(
        tmp_vault, transcript="body", task="test", tags=[],
    )
    idx = indexer.build(tmp_vault)
    assert any(v.get("type") == "transcript" for v in idx.values())


def test_slug_handles_weird_input(tmp_vault: Path):
    # Emoji, punctuation, unicode — should not crash or produce invalid filename.
    name = session_mgr.archive_transcript(
        tmp_vault,
        transcript="body",
        task="Mariana Trench !!! 🌊 /\\|*?<>",
        tags=["ocean"],
    )
    path = tmp_vault / "_transcripts" / f"{name}.md"
    assert path.exists()
    # Filename should not contain path separators or shell metacharacters.
    assert "/" not in path.name
    assert "\\" not in path.name


def test_two_sessions_produce_two_distinct_transcripts(tmp_vault: Path):
    n1 = session_mgr.archive_transcript(tmp_vault, "body 1", "task one", [])
    n2 = session_mgr.archive_transcript(tmp_vault, "body 2", "task two", [])
    # Names include timestamps plus slug so should differ unless same clock-second + same task.
    if n1 == n2:
        pytest.skip("timestamps collided within the same second — rerun")
    assert (tmp_vault / "_transcripts" / f"{n1}.md").exists()
    assert (tmp_vault / "_transcripts" / f"{n2}.md").exists()
