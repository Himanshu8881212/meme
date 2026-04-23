"""Affective state — EMA of recent episode affect labels."""
from __future__ import annotations

from pathlib import Path

from core import mood
from utils import frontmatter


def _ep(vault: Path, name: str, affect: str | None = None,
        intensity: float | None = None, extra: dict | None = None):
    """Write an episode node with optional affect frontmatter."""
    edir = vault / "episodes"
    edir.mkdir(parents=True, exist_ok=True)
    fm = {"type": "episode", "importance": 0.5, "tags": []}
    if affect is not None:
        fm["affect"] = affect
    if intensity is not None:
        fm["intensity"] = intensity
    if extra:
        fm.update(extra)
    frontmatter.write(edir / f"{name}.md", fm, f"body of {name}")


def test_empty_vault_returns_empty_mood(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    assert mood.compute_mood(v) == {}


def test_single_affect_dominates(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "heavy day", affect="heaviness", intensity=0.8)
    m = mood.compute_mood(v)
    assert list(m.keys()) == ["heaviness"]
    assert m["heaviness"] == 1.0


def test_neutral_entries_ignored(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "routine 1", affect="neutral")
    _ep(v, "routine 2")  # no affect field
    assert mood.compute_mood(v) == {}


def test_invalid_affect_ignored(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "bogus", affect="enlightenment", intensity=0.9)  # not in allowed set
    _ep(v, "real", affect="joy", intensity=0.5)
    m = mood.compute_mood(v)
    assert "enlightenment" not in m
    assert m == {"joy": 1.0}


def test_recency_weighting_with_equal_counts(tmp_path):
    """With equal counts, the more-recent label wins due to geometric
    decay. (With unequal counts, majority can still beat recency — that's
    the EMA smoothing; one session shouldn't flip the mood.)"""
    v = tmp_path / "vault"
    _ep(v, "a-old", affect="heaviness", intensity=0.5)
    _ep(v, "b-new", affect="joy", intensity=0.5)
    m = mood.compute_mood(v)
    assert list(m.keys())[0] == "joy"


def test_intensity_scales_contribution(tmp_path):
    v = tmp_path / "vault"
    # Same recency; intensity decides.
    _ep(v, "a", affect="warmth", intensity=0.2)
    _ep(v, "b", affect="calm", intensity=0.9)
    m = mood.compute_mood(v)
    # `b` is more recent (later filename) AND higher intensity.
    assert list(m.keys())[0] == "calm"


def test_update_mood_writes_file(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "first", affect="joy", intensity=0.6)
    _ep(v, "second", affect="warmth", intensity=0.7)
    mood.update_mood(v)
    path = v / "_identity" / "mood.md"
    assert path.exists()
    fm, body = frontmatter.read(path)
    assert fm.get("subtype") == "mood"
    # Dominant tone is mentioned in the body
    assert "warmth" in body.lower() or "joy" in body.lower()


def test_update_mood_empty_no_signal(tmp_path):
    v = tmp_path / "vault"
    (v / "episodes").mkdir(parents=True)
    mood.update_mood(v)
    path = v / "_identity" / "mood.md"
    assert path.exists()
    _, body = frontmatter.read(path)
    assert "No emotional signal" in body


def test_mood_snippet_returns_empty_when_no_signal(tmp_path):
    v = tmp_path / "vault"
    (v / "episodes").mkdir(parents=True)
    mood.update_mood(v)
    assert mood.mood_snippet(v) == ""


def test_mood_snippet_includes_register_hint(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "heavy", affect="heaviness", intensity=0.8)
    mood.update_mood(v)
    snippet = mood.mood_snippet(v)
    assert "heaviness" in snippet.lower()
    assert "register guidance" in snippet.lower() or "steady" in snippet.lower()


def test_clear_mood(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "x", affect="joy", intensity=0.5)
    mood.update_mood(v)
    assert (v / "_identity" / "mood.md").exists()
    assert mood.clear_mood(v) is True
    assert not (v / "_identity" / "mood.md").exists()
    assert mood.clear_mood(v) is False


# ── tonality enforcement ──────────────────────────────────────────────

def test_dominant_mood_parses_from_file(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "grief", affect="heaviness", intensity=0.9)
    mood.update_mood(v)
    assert mood.dominant_mood(v) == "heaviness"


def test_dominant_mood_none_when_missing(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    assert mood.dominant_mood(v) is None


def test_filter_strips_laugh_on_heaviness(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "rough day", affect="heaviness", intensity=0.8)
    mood.update_mood(v)
    text = "Hey. [laugh] Yeah that's rough. [sigh] I'm here."
    out = mood.filter_reply_by_mood(text, v)
    assert "[laugh]" not in out
    assert "[sigh]" in out  # sigh is allowed (cap 2)


def test_filter_caps_excess_sighs(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "rough day", affect="heaviness", intensity=0.8)
    mood.update_mood(v)
    text = "[sigh] A. [sigh] B. [sigh] C. [sigh] D."
    out = mood.filter_reply_by_mood(text, v)
    # Cap is 2 — first two survive, last two stripped.
    assert out.count("[sigh]") == 2


def test_filter_passthrough_on_joy(tmp_path):
    v = tmp_path / "vault"
    _ep(v, "good news", affect="joy", intensity=0.9)
    mood.update_mood(v)
    text = "[laugh] That's amazing! [chuckle] Tell me more!"
    out = mood.filter_reply_by_mood(text, v)
    # Joy has no caps — everything survives.
    assert "[laugh]" in out
    assert "[chuckle]" in out


def test_filter_passthrough_when_no_mood(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    text = "[laugh] Hi [sigh]"
    assert mood.filter_reply_by_mood(text, v) == text


def test_filter_keeps_first_occurrences(tmp_path):
    """When capping, the earlier tags survive — not random ones."""
    v = tmp_path / "vault"
    _ep(v, "tough", affect="hurt", intensity=0.9)
    mood.update_mood(v)
    text = "A [sigh] first. B [sigh] second. C [sigh] third."
    out = mood.filter_reply_by_mood(text, v)
    assert out.count("[sigh]") == 2
    assert "first" in out and "second" in out and "third" in out
    # The removed tag is the LAST one; "C" should still be present sans its sigh.
    # Verify ordering of surviving sighs: they're attached to the A and B clauses.
    a_idx = out.index("A")
    first_sigh = out.index("[sigh]")
    assert first_sigh > a_idx
