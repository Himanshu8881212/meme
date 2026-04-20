from __future__ import annotations

from core import flagging


def test_all_flag_types_recognised():
    text = (
        "[NOVEL: a] then [REPEAT: b] and [CONTRADICTION: c] with [SALIENT: d] "
        "plus [HIGH-STAKES: e] and [ASSOCIATED: f]."
    )
    flags = flagging.extract(text)
    kinds = {f["type"] for f in flags}
    assert kinds == {"NOVEL", "REPEAT", "CONTRADICTION", "SALIENT", "HIGH-STAKES", "ASSOCIATED"}


def test_case_insensitive():
    flags = flagging.extract("[novel: x] and [Salient: y]")
    kinds = [f["type"] for f in flags]
    assert kinds == ["NOVEL", "SALIENT"]


def test_payload_optional():
    flags = flagging.extract("Bare flag: [SALIENT] and payload [SALIENT: x].")
    assert len(flags) == 2
    assert flags[0]["payload"] == ""
    assert flags[1]["payload"] == "x"


def test_payload_trimmed():
    flags = flagging.extract("[NOVEL:   spacey   ]")
    assert flags[0]["payload"] == "spacey"


def test_context_window_included():
    # Context should include nearby text so the reflection pass can disambiguate.
    text = "Before text. [NOVEL: x] After text."
    flags = flagging.extract(text)
    assert "Before text" in flags[0]["context"]
    assert "After text" in flags[0]["context"]


def test_summarize_empty_and_nonempty():
    assert flagging.summarize([]) == "(no flags)"
    s = flagging.summarize([{"type": "NOVEL", "payload": "x", "context": "ctx"}])
    assert "NOVEL" in s and "ctx" in s


def test_non_flags_ignored():
    # Plain bracketed text that isn't a known flag type should not match.
    flags = flagging.extract("[TODO: buy milk] and [RANDOM: nope]")
    assert flags == []


def test_multiple_flags_preserve_order():
    text = "[NOVEL: first] then [SALIENT: second] finally [HIGH-STAKES: third]"
    flags = flagging.extract(text)
    assert [f["type"] for f in flags] == ["NOVEL", "SALIENT", "HIGH-STAKES"]
    assert [f["payload"] for f in flags] == ["first", "second", "third"]
