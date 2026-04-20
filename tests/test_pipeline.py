from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core import decay, flagging, monitor, reflection, retrieval
from scheduler import session as session_mgr


ROOT = Path(__file__).resolve().parent.parent


def test_echo_backend_used(monkeypatch, config, tmp_vault: Path):
    result = reflection.chat(
        role="model1", system="hello", messages=[{"role": "user", "content": "test"}],
        config=config,
    )
    assert "[ECHO]" in result


def test_end_to_end_with_mocked_reflection(seeded_vault: Path, config):
    # Simulate a session. Start → chat happens via echo → end feeds a
    # synthetic reflection output (so we can assert vault writes).
    config["vault_path"] = str(seeded_vault)

    meta = session_mgr.start(
        task="investigate auth login bug",
        tags=["auth"],
        config=config,
        project_root=ROOT,
    )
    assert meta["retrieved_files"], "retrieval should find auth-tagged nodes"

    transcript = (
        "USER: I need to understand the auth bug.\n"
        "ASSISTANT: [NOVEL: new detail about JWT refresh tokens] "
        "[SALIENT: the bug happens only under load] "
        "[CONTRADICTION: earlier note says Redis was dropped, but staging still uses it]"
    )

    write_block = (
        '<<WRITE path="episodes/Auth load bug.md" action="create">>\n'
        '---\ntype: episode\ntags: [auth, incident]\nimportance: 0.7\n---\n'
        '# Auth load bug\nRelates to [[AuthService]].\n<<END>>'
    )

    with patch.object(reflection, "routine", return_value=write_block):
        result = session_mgr.end(
            session_output=transcript,
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )

    assert result["flags_found"] == 3
    assert result["reflection_run"] is True
    writes = result["writes"]
    assert any(w["action"] == "create" for w in writes)
    assert (seeded_vault / "episodes" / "Auth load bug.md").exists()


def test_access_count_bumped_on_retrieval(seeded_vault: Path, config):
    from utils import frontmatter
    config["vault_path"] = str(seeded_vault)

    path = seeded_vault / "entities" / "AuthService.md"
    fm_before, _ = frontmatter.read(path)
    before = fm_before["access_count"]

    meta = session_mgr.start(
        task="AuthService problem",
        tags=["auth"],
        config=config,
        project_root=ROOT,
    )
    with patch.object(reflection, "routine", return_value=""):
        session_mgr.end(
            session_output="[SALIENT: something]",
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )

    fm_after, _ = frontmatter.read(path)
    assert fm_after["access_count"] > before


def test_full_daily_cycle_keeps_vault_consistent(seeded_vault: Path, config):
    # Ingest → decay → monitor in sequence; vault should remain readable.
    transcript = "USER: hi. ASSISTANT: yes. [SALIENT: minor]"
    flags = flagging.extract(transcript)
    assert flags  # sanity

    fake_output = '<<WRITE path="episodes/Daily.md" action="create">>\n# Daily\n<<END>>'
    reflection.apply_writes(fake_output, seeded_vault)

    decay_result = decay.run(seeded_vault, lambda_=0.05, archive_threshold=0.10)
    metrics = monitor.collect(seeded_vault)

    assert decay_result["processed"] >= 7
    assert metrics["total_nodes"] >= 7
    assert all((seeded_vault / p).is_dir() for p in ("entities", "concepts", "decisions"))


def test_session_log_entry_appended(seeded_vault: Path, config):
    config["vault_path"] = str(seeded_vault)
    meta = session_mgr.start(task="x", tags=[], config=config, project_root=ROOT)
    with patch.object(reflection, "routine", return_value=""):
        session_mgr.end(
            session_output="[SALIENT: x]",
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )
    log = (seeded_vault / "_meta" / "session_log.md").read_text(encoding="utf-8")
    assert "task" in log
    assert '"x"' in log or "x" in log


def test_reflection_skipped_when_below_min_flags(seeded_vault: Path, config):
    config["vault_path"] = str(seeded_vault)
    config["reflection"]["min_flags_for_reflection"] = 5

    meta = session_mgr.start(task="no flags here", tags=[], config=config, project_root=ROOT)
    result = session_mgr.end(
        session_output="ASSISTANT: no flags at all.",
        session_meta=meta,
        config=config,
        project_root=ROOT,
    )
    assert result["reflection_run"] is False
    assert result["flags_found"] == 0
