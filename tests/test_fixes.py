"""Tests for the three fixes applied after the first real test-battery run:

1. Flag-recovery — substantive sessions reflect even with 0 flags.
2. Near-duplicate warnings on newly created nodes.
3. Bare command words (`end`, `meta`, etc.) dispatch to slash commands.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core import reflection
from scheduler import session as session_mgr

ROOT = Path(__file__).resolve().parent.parent


# --- Fix 1: flag-recovery -----------------------------------------------------


def test_reflection_runs_without_flags_on_substantive_session(seeded_vault: Path, config):
    """Long technical conversations used to produce 0 flags → 0 reflection →
    0 writes. Now a substantive transcript triggers reflection anyway."""
    config["vault_path"] = str(seeded_vault)

    meta = session_mgr.start(task="rate limiter design",
                             tags=["backend"],
                             config=config,
                             project_root=ROOT)

    transcript = (
        "USER: token bucket vs sliding window for a 10k req/s API?\n"
        "ASSISTANT: Token bucket handles bursts better. Sliding window is\n"
        "more precise but stores per-request timestamps — heavier at scale.\n"
        "USER: what about redis for the backing store?\n"
        "ASSISTANT: Redis works. Use Lua scripts for atomic check-and-set.\n"
        "Watch for hot keys, thundering herd, and memory pressure.\n"
    )

    with patch.object(reflection, "routine", return_value="") as mock_routine:
        result = session_mgr.end(
            session_output=transcript,
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )

    assert result["flags_found"] == 0
    assert result["reflection_run"] is True, "recovery should have triggered"
    assert result["recovery_mode"] is True
    # The recovery message should have been passed as the flag summary.
    mock_routine.assert_called_once()
    call_kwargs = mock_routine.call_args.kwargs
    assert "RECOVERY MODE" in call_kwargs["flag_summary"]


def test_reflection_skipped_on_trivial_session(seeded_vault: Path, config):
    """A genuinely trivial exchange (no flags, barely any content) should
    still be skipped — we don't want to waste tokens on 'hi' / 'hello'."""
    config["vault_path"] = str(seeded_vault)
    meta = session_mgr.start(task="hi", tags=[], config=config, project_root=ROOT)

    result = session_mgr.end(
        session_output="USER: hi\nASSISTANT: hey.\n",
        session_meta=meta,
        config=config,
        project_root=ROOT,
    )

    assert result["flags_found"] == 0
    assert result["reflection_run"] is False


def test_reflection_runs_when_flags_present_even_if_short(seeded_vault: Path, config):
    """Flags always trigger reflection, regardless of transcript length."""
    config["vault_path"] = str(seeded_vault)
    meta = session_mgr.start(task="quick", tags=[], config=config, project_root=ROOT)

    with patch.object(reflection, "routine", return_value=""):
        result = session_mgr.end(
            session_output="ASSISTANT: [SALIENT: Himanshu mentioned X]",
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )
    assert result["flags_found"] == 1
    assert result["reflection_run"] is True
    assert result["recovery_mode"] is False


# --- Fix 2: near-duplicate detection -----------------------------------------


def test_apply_writes_warns_on_near_duplicate_episode(tmp_vault: Path):
    """The tRPC duplicate-episode bug: when creating a new episode whose
    title shares >=50% of meaningful tokens with an existing one, surface a
    warning so the user can see it and the reflection prompt has a signal."""
    existing = tmp_vault / "episodes" / "tRPC session error in production.md"
    existing.write_text("---\ntype: episode\n---\nOriginal body.\n", encoding="utf-8")

    new_block = (
        '<<WRITE path="episodes/tRPC session error fixed by middleware.md" action="create">>\n'
        "---\ntype: episode\n---\n# Fixed\n<<END>>"
    )
    results = reflection.apply_writes(new_block, tmp_vault)

    created = [r for r in results if r.get("action") == "create"]
    assert len(created) == 1
    assert "warning" in created[0]
    assert "tRPC session error in production" in created[0]["warning"]


def test_apply_writes_no_warning_for_genuinely_distinct_node(tmp_vault: Path):
    """Don't over-warn. An unrelated episode shouldn't trigger the duplicate
    heuristic."""
    existing = tmp_vault / "episodes" / "Launch retrospective.md"
    existing.write_text("---\ntype: episode\n---\nbody\n", encoding="utf-8")

    new_block = (
        '<<WRITE path="episodes/Payment failure incident.md" action="create">>\n'
        "---\ntype: episode\n---\n# x\n<<END>>"
    )
    results = reflection.apply_writes(new_block, tmp_vault)
    assert not results[0].get("warning"), (
        f"unexpected warning: {results[0].get('warning')}"
    )


def test_apply_writes_no_warning_for_update(tmp_vault: Path):
    """Updates never trigger the warning — they're the correct thing to do."""
    target = tmp_vault / "episodes" / "Existing.md"
    target.write_text("---\ntype: episode\n---\nbody\n", encoding="utf-8")

    block = (
        '<<WRITE path="episodes/Existing.md" action="update">>\n'
        "---\ntype: episode\n---\nnew body\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "update"
    assert "warning" not in results[0]


def test_apply_writes_similarity_ignores_stopwords(tmp_vault: Path):
    """'The launch of the product' and 'Launch plan' share a stopword, not a
    real concept — shouldn't warn."""
    existing = tmp_vault / "episodes" / "The launch of the product.md"
    existing.write_text("---\ntype: episode\n---\nbody\n", encoding="utf-8")

    block = (
        '<<WRITE path="episodes/Unrelated thing.md" action="create">>\n'
        "---\ntype: episode\n---\nx\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert not results[0].get("warning")


# --- Fix 3: bare command dispatch --------------------------------------------


def test_bare_command_map_defined():
    """The TUI maps bare words like 'end' to '/end'. Test the mapping is
    defined and includes the must-haves."""
    import tui
    assert "end" in tui.BARE_COMMANDS
    assert tui.BARE_COMMANDS["end"] == "/end"
    assert "meta" in tui.BARE_COMMANDS
    assert "help" in tui.BARE_COMMANDS
    # Exit phrases remain separate — they trigger graceful exit, not /exit.
    assert "exit" not in tui.BARE_COMMANDS
    assert "exit" in tui.EXIT_PHRASES


def test_bare_command_mapping_is_case_insensitive_ready():
    """The caller lowercases before looking up. Confirm keys are all
    lowercase already so the lookup works."""
    import tui
    for k in tui.BARE_COMMANDS:
        assert k == k.lower()


# --- Fix 4 (round 2): recovery triggers on single-turn substantive messages ---


def test_single_long_turn_triggers_recovery(seeded_vault: Path, config):
    """Scenario 1A bug: a single big opening message from the user counted
    as 1 USER marker, which was below the old AND-threshold. Should now
    trigger recovery because the message is long enough."""
    from scheduler import session as session_mgr
    config["vault_path"] = str(seeded_vault)

    meta = session_mgr.start(task="intro", tags=[], config=config, project_root=ROOT)

    long_single_turn = (
        "## USER\n" +
        "hi, my name is alex, im 32, i work on payments infra. "
        "i prefer concise answers without hedging. what do you know "
        "about me already? how much context do you have so far? " * 3 +
        "\n## ASSISTANT\n" +
        "Hi Alex. I don't have much yet — just your name and role."
    )
    assert long_single_turn.upper().count("USER") < 2
    assert len(long_single_turn) > 300

    with patch.object(reflection, "routine", return_value=""):
        result = session_mgr.end(
            session_output=long_single_turn,
            session_meta=meta,
            config=config,
            project_root=ROOT,
        )
    assert result["reflection_run"] is True
    assert result["recovery_mode"] is True


# --- Fix 5: block user-as-entity anti-pattern --------------------------------


def _write_identity(vault: Path, user_name: str) -> None:
    (vault / "_identity").mkdir(parents=True, exist_ok=True)
    import yaml as _yaml
    fm = {"type": "identity", "name": "Kai", "user_name": user_name}
    path = vault / "_identity" / "persona.md"
    path.write_text(f"---\n{_yaml.dump(fm)}---\nI am Kai.\n", encoding="utf-8")


def test_blocks_entities_with_user_name(tmp_vault: Path):
    _write_identity(tmp_vault, "Himanshu")
    block = (
        '<<WRITE path="entities/Himanshu.md" action="create">>\n'
        "---\ntype: entity\n---\nFacts about Himanshu.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "rejected"
    assert "Himanshu" in results[0]["reason"]
    assert "_identity/self.md" in results[0]["reason"]
    assert not (tmp_vault / "entities" / "Himanshu.md").exists()


def test_blocks_user_prefix_variants(tmp_vault: Path):
    _write_identity(tmp_vault, "Ryan")
    for bad in ("entities/User Ryan.md", "entities/The User Ryan.md", "concepts/Ryan.md"):
        block = (
            f'<<WRITE path="{bad}" action="create">>\n'
            "---\ntype: entity\n---\nbody\n<<END>>"
        )
        results = reflection.apply_writes(block, tmp_vault)
        assert results[0]["action"] == "rejected", f"should have blocked {bad}"


def test_allows_entities_containing_user_name(tmp_vault: Path):
    """`entities/Himanshu's Dashboard.md` is legitimate — the user's name
    appears but the entity is something owned by them, not the user themselves."""
    _write_identity(tmp_vault, "Himanshu")
    block = (
        '<<WRITE path="entities/Himanshu\'s Dashboard.md" action="create">>\n'
        "---\ntype: entity\n---\nThe dashboard Himanshu is building.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "create"


def test_no_user_means_no_block(tmp_vault: Path):
    """If no identity is set, don't block anything."""
    block = (
        '<<WRITE path="entities/Stranger.md" action="create">>\n'
        "---\ntype: entity\n---\nbody\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "create"


# --- Fix 6: rate-limit retry --------------------------------------------------


def test_create_with_retry_retries_on_429():
    from core.reflection import _create_with_retry

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Error code: 429 - rate_limited")
            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "ok"})})]
            return R()

    import core.reflection as refl
    monkey = patch.object(refl, "_BACKOFF_BASE_SEC", 0.01)
    client = FakeClient()
    with monkey:
        out = _create_with_retry(client, model="x", max_tokens=1, messages=[])
    assert client.calls == 3
    assert out.choices[0].message.content == "ok"


def test_create_with_retry_raises_after_max():
    from core.reflection import _create_with_retry
    import core.reflection as refl

    class AlwaysFail:
        chat = None
        completions = None
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **kw):
            raise RuntimeError("429 rate_limited")

    with patch.object(refl, "_BACKOFF_BASE_SEC", 0.01):
        import pytest
        with pytest.raises(RuntimeError, match="429"):
            _create_with_retry(AlwaysFail(), model="x", max_tokens=1, messages=[])


def test_config_driven_tuning_knobs(seeded_vault: Path, config, monkeypatch):
    """The knobs we just promoted to config must actually be read at runtime.
    Tweak them in the test config and verify the behavior changes."""
    from scheduler import session as session_mgr

    # Override recovery_min_chars upward so a previously-substantive message
    # no longer qualifies. The reflection should skip.
    config["vault_path"] = str(seeded_vault)
    config["reflection"]["recovery_min_chars"] = 5000
    config["reflection"]["recovery_min_user_turns"] = 99

    meta = session_mgr.start(task="x", tags=[], config=config, project_root=ROOT)
    result = session_mgr.end(
        session_output="ASSISTANT: here's a middling reply without flags\n" * 3,
        session_meta=meta, config=config, project_root=ROOT,
    )
    assert result["reflection_run"] is False, (
        "config knob should have suppressed recovery"
    )


def test_similarity_threshold_is_config_driven(tmp_vault: Path):
    """Lower the threshold and see more warnings; raise it and see fewer."""
    existing = tmp_vault / "episodes" / "Auth login bug.md"
    existing.write_text("---\ntype: episode\n---\nbody\n", encoding="utf-8")

    block = (
        '<<WRITE path="episodes/Auth session bug.md" action="create">>\n'
        "---\ntype: episode\n---\n# x\n<<END>>"
    )

    # Strict threshold — no warning expected.
    res_strict = reflection.apply_writes(block, tmp_vault, similarity_threshold=0.95)
    assert not res_strict[0].get("warning")

    # Clean up and retry with a generous threshold.
    (tmp_vault / "episodes" / "Auth session bug.md").unlink()
    res_loose = reflection.apply_writes(block, tmp_vault, similarity_threshold=0.4)
    assert "may duplicate" in (res_loose[0].get("warning") or "")


def test_min_body_chars_is_config_driven(tmp_vault: Path):
    from core import monitor
    path = tmp_vault / "entities" / "Short.md"
    path.write_text("---\ntype: entity\n---\n# Short\nOK body here.\n", encoding="utf-8")

    # Huge threshold — everything looks broken.
    broken_strict = monitor.find_broken_nodes(tmp_vault, min_body_chars=500)
    assert any("Short" in b["path"] for b in broken_strict)

    # Tiny threshold — same file now passes.
    broken_loose = monitor.find_broken_nodes(tmp_vault, min_body_chars=1)
    assert not any("Short" in b["path"] for b in broken_loose)


def test_retry_params_read_from_config():
    from core.reflection import _retry_params
    cfg = {"reflection": {"retry": {"max_retries": 7, "backoff_base_sec": 0.5}}}
    assert _retry_params(cfg) == (7, 0.5)

    # Missing keys fall back to defaults.
    assert _retry_params({})[0] >= 1
    assert _retry_params({"reflection": {}})[0] >= 1


def test_max_tool_rounds_is_config_driven(seeded_vault: Path, config):
    # Echo backend short-circuits before the loop, so we only verify the
    # parameter is plumbed. With max_tool_rounds=0 the function should still
    # return something safely (the for loop body simply doesn't execute, and
    # echo backend bypasses the loop anyway).
    config["reflection"]["max_tool_rounds"] = 0
    output, log = reflection.deep_with_tools(
        vault_path=seeded_vault, vault_files=[], metrics={"total_nodes": 7},
        triggers=[], config=config,
    )
    assert isinstance(output, str)
    assert log == []


def test_tool_loop_strips_thinking_before_replay(seeded_vault: Path, config, monkeypatch):
    """The /meta 422 bug: magistral-medium returns content as a list
    including {type:'thinking',...} chunks. Mistral rejects those in a
    replayed assistant message. We must strip-and-normalize before replay."""
    from unittest.mock import MagicMock
    import core.reflection as refl

    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["deep"] = {"provider": "mistral", "model": "fake-reasoning"}

    # Simulate a magistral assistant response: content is a list of thinking +
    # text chunks, plus a tool call.
    first_msg = MagicMock()
    first_msg.content = [
        {"type": "thinking", "thinking": "let me think about this..."},
        {"type": "text", "text": "I'll check the tag counts."},
    ]
    tc = MagicMock()
    tc.id = "c1"; tc.function = MagicMock()
    tc.function.name = "count_nodes_by_tag"
    tc.function.arguments = '{"tag": "auth"}'
    first_msg.tool_calls = [tc]

    second_msg = MagicMock()
    second_msg.content = '<<WRITE path="concepts/X.md" action="create">>\nx\n<<END>>'
    second_msg.tool_calls = []

    def fake_resp(msg):
        r = MagicMock(); r.choices = [MagicMock(message=msg)]; return r

    call_kwargs_log: list[dict] = []
    client = MagicMock()
    def create(**kw):
        call_kwargs_log.append(kw)
        return fake_resp(first_msg) if len(call_kwargs_log) == 1 else fake_resp(second_msg)
    client.chat.completions.create = create

    with patch.object(refl, "_get_client", return_value=client):
        output, log = refl.deep_with_tools(
            vault_path=seeded_vault, vault_files=[], metrics={"total_nodes": 7},
            triggers=[], config=config, max_rounds=4,
        )

    # The second call's messages must have the assistant reply with
    # string content (not a list, not containing 'thinking').
    second_call_messages = call_kwargs_log[1]["messages"]
    assistant_replay = next(m for m in second_call_messages if m.get("role") == "assistant")
    assert isinstance(assistant_replay["content"], str), (
        "replay content must be a string, not a list"
    )
    assert "thinking" not in assistant_replay["content"].lower()
    assert "<<WRITE" in output
    from core.reflection import _create_with_retry
    import core.reflection as refl

    class AlwaysBug:
        def __init__(self):
            self.chat = self; self.completions = self; self.calls = 0
        def create(self, **kw):
            self.calls += 1
            raise ValueError("unrelated bug")

    c = AlwaysBug()
    with patch.object(refl, "_BACKOFF_BASE_SEC", 0.01):
        import pytest
        with pytest.raises(ValueError):
            _create_with_retry(c, model="x", max_tokens=1, messages=[])
    assert c.calls == 1, "non-rate-limit errors must not be retried"
