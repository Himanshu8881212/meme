from __future__ import annotations

from utils import wikilinks


def test_basic():
    assert wikilinks.extract("Hello [[Target]].") == ["Target"]


def test_alias_extracts_target():
    assert wikilinks.extract("See [[Target|the alias]].") == ["Target"]


def test_heading_strips_section():
    assert wikilinks.extract("[[Target#Section]]") == ["Target"]


def test_multiple_on_line():
    assert wikilinks.extract("[[A]] and [[B]] and [[C]]") == ["A", "B", "C"]


def test_empty_brackets_skipped():
    assert wikilinks.extract("[[]] and [[ ]] should be skipped, but [[Real]] counts") == ["Real"]


def test_unicode_names():
    assert wikilinks.extract("[[Café ☕]] and [[日本語]]") == ["Café ☕", "日本語"]


def test_noise_around_links_does_not_leak():
    text = "prefix [[ One ]] middle [[Two]] suffix."
    out = wikilinks.extract(text)
    assert "One" in out
    assert "Two" in out
    assert len(out) == 2


def test_plain_text_returns_nothing():
    assert wikilinks.extract("No links at all here. Just text and [one bracket].") == []
