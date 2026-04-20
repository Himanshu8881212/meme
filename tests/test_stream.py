"""chat_stream — streaming chat used by the voice TUI for responsive TTS."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core import reflection


def _chunk(content):
    event = MagicMock()
    event.choices = [MagicMock(delta=MagicMock(content=content))]
    return event


def test_chat_stream_echo_yields_once(seeded_vault, config):
    """Echo backend emits a single blob (no upstream streaming)."""
    chunks = list(reflection.chat_stream(
        role="model1", system="hi",
        messages=[{"role": "user", "content": "x"}],
        config=config,
    ))
    assert len(chunks) == 1
    assert "[ECHO]" in chunks[0]


def test_chat_stream_yields_tokens(config, monkeypatch):
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["model1"] = {"provider": "mistral", "model": "fake-model"}

    # Fake the streaming iterator.
    events = [_chunk("Hello, "), _chunk("world"), _chunk("!")]
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=iter(events))

    with patch.object(reflection, "_get_client", return_value=client):
        out = list(reflection.chat_stream(
            role="model1", system="",
            messages=[{"role": "user", "content": "hi"}],
            config=config,
        ))
    assert out == ["Hello, ", "world", "!"]


def test_chat_stream_skips_none_content(config, monkeypatch):
    """OpenAI-compat endpoints sometimes send empty deltas (role-only, or
    final chunks with no content). We filter those out so the caller
    doesn't see spurious empty yields."""
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["model1"] = {"provider": "mistral", "model": "fake-model"}

    events = [_chunk("a"), _chunk(None), _chunk("b")]
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=iter(events))

    with patch.object(reflection, "_get_client", return_value=client):
        out = list(reflection.chat_stream(
            role="model1", system="",
            messages=[{"role": "user", "content": "x"}],
            config=config,
        ))
    assert out == ["a", "b"]


def test_sentence_splitter():
    """voice_tui chunks TTS at sentence boundaries to avoid mid-clause cuts."""
    from tui_common import split_sentence

    sent, rem = split_sentence("Hello world. How are")
    assert sent == "Hello world."
    assert rem == "How are"

    sent, rem = split_sentence("no terminator yet")
    assert sent == ""
    assert rem == "no terminator yet"

    # Multiple terminators — splits at first.
    sent, rem = split_sentence("First one! Then the second?")
    assert sent == "First one!"

    # Question mark.
    sent, _ = split_sentence("What? Really.")
    assert sent == "What?"

    # Handles closing quote / paren after terminator.
    sent, _ = split_sentence('She said "hi." Then left.')
    assert 'She said "hi."' == sent


def test_strip_meme_flags_preserves_paralinguistic():
    """ChatterBox Turbo tags like [laugh] must survive — they're speech
    instructions, not memory flags to strip."""
    from tui_common import strip_meme_flags

    out = strip_meme_flags("[laugh] Nice one. [NOVEL: Ryan likes jokes]")
    assert "[laugh]" in out
    assert "[NOVEL" not in out
    assert "Nice one" in out


def test_chat_stream_skips_magistral_thinking_chunks(config, monkeypatch):
    """Magistral streams reasoning in `thinking`-type content parts. Those
    must NEVER reach the voice TUI — otherwise TTS speaks the internal
    monologue. Regression test for the 'think think think' bug."""
    monkeypatch.delenv("MEMORY_BACKEND")
    config["providers"]["mistral"] = {"base_url": "https://x", "api_key_env": "FAKE"}
    config["models"]["model1"] = {"provider": "mistral", "model": "fake-magistral"}

    # Simulate magistral's structured stream: first thinking, then text.
    events = [
        _chunk([{"type": "thinking", "thinking": "let me reason..."}]),
        _chunk([{"type": "thinking", "thinking": " considering options..."}]),
        _chunk([{"type": "text", "text": "Hello,"}]),
        _chunk([{"type": "text", "text": " here is the answer."}]),
    ]
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=iter(events))

    with patch.object(reflection, "_get_client", return_value=client):
        out = list(reflection.chat_stream(
            role="model1", system="",
            messages=[{"role": "user", "content": "hi"}],
            config=config,
        ))

    joined = "".join(out)
    assert "let me reason" not in joined
    assert "considering options" not in joined
    assert "Hello," in joined
    assert "here is the answer." in joined


def test_sanitize_messages_drops_empty_assistants():
    """Mistral's 400 'Assistant message must have content or tool_calls'
    used to poison sessions when magistral streamed only reasoning. The
    API layer defensively filters those out."""
    from core.reflection import _sanitize_messages

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},          # should be dropped
        {"role": "user", "content": "you there?"},
        {"role": "assistant", "content": "   "},       # whitespace — dropped
        {"role": "user", "content": "hello??"},
    ]
    out = _sanitize_messages(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["user", "user", "user"]


def test_sanitize_messages_keeps_tool_calls():
    """Assistant messages with tool_calls but empty content ARE valid —
    they're how tool-calling models request a tool. Must keep them."""
    from core.reflection import _sanitize_messages

    msgs = [
        {"role": "user", "content": "what's the tag count?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "count_nodes_by_tag", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "3"},
    ]
    out = _sanitize_messages(msgs)
    assert len(out) == 3  # tool-call-bearing assistant kept


def test_extract_text_only_handles_all_shapes():
    """_extract_text_only is the gate that prevents reasoning leaks."""
    from core.reflection import _extract_text_only

    assert _extract_text_only("plain") == "plain"
    assert _extract_text_only(None) == ""
    # List of dict-style parts
    assert _extract_text_only([
        {"type": "thinking", "thinking": "DROP"},
        {"type": "text", "text": "keep"},
    ]) == "keep"
    # List of object-style parts
    class P:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    assert _extract_text_only([
        P(type="thinking", thinking="DROP"),
        P(type="text", text="keep"),
    ]) == "keep"


def test_strip_meme_flags_removes_all_meme_flags():
    from tui_common import strip_meme_flags

    cases = [
        "[NOVEL: x]", "[REPEAT: y]", "[CONTRADICTION: z]",
        "[SALIENT: a]", "[HIGH-STAKES: b]", "[ASSOCIATED: c]", "[IDENTITY: d]",
    ]
    text = " ".join(cases) + " keep this"
    out = strip_meme_flags(text)
    assert "keep this" in out
    for flag in cases:
        assert flag not in out
