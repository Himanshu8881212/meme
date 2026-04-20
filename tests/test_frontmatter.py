from __future__ import annotations

from pathlib import Path

import pytest

from utils import frontmatter


def test_roundtrip(tmp_path: Path):
    p = tmp_path / "a.md"
    original_fm = {"type": "concept", "tags": ["x", "y"], "importance": 0.7}
    frontmatter.write(p, original_fm, "# Body\n\nSome content with [[link]].")
    fm, body = frontmatter.read(p)
    assert fm["type"] == "concept"
    assert fm["tags"] == ["x", "y"]
    assert fm["importance"] == 0.7
    assert "[[link]]" in body


def test_missing_frontmatter_graceful(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# No frontmatter at all\n\nJust a body.", encoding="utf-8")
    fm, body = frontmatter.read(p)
    assert fm == {}
    assert "Just a body." in body


def test_partial_fence_graceful(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("---\ntype: concept\n(no closing fence)\n", encoding="utf-8")
    fm, body = frontmatter.read(p)
    # Accepts either empty-dict-with-raw-body or at-least-non-crashing.
    assert isinstance(fm, dict)
    assert isinstance(body, str)


def test_update_preserves_body(tmp_path: Path):
    p = tmp_path / "a.md"
    frontmatter.write(p, {"type": "entity", "tags": ["a"]}, "Keep this body text.")
    frontmatter.update(p, {"importance": 0.9, "tags": ["a", "b"]})
    fm, body = frontmatter.read(p)
    assert fm["importance"] == 0.9
    assert fm["tags"] == ["a", "b"]
    assert fm["type"] == "entity"
    assert "Keep this body text." in body


def test_unicode_roundtrip(tmp_path: Path):
    p = tmp_path / "u.md"
    body = "# Héllo 🌍\n\nLinks to [[Café ☕]].\n"
    frontmatter.write(p, {"type": "entity", "tags": ["café", "🌍"]}, body)
    fm, rt = frontmatter.read(p)
    assert "🌍" in rt
    assert "café" in fm["tags"]
