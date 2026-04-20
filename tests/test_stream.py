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
