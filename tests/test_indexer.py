from __future__ import annotations

from pathlib import Path

from utils import indexer


def test_empty_vault(tmp_vault: Path):
    idx = indexer.build(tmp_vault)
    assert idx == {}


def test_build_includes_all_nodes(seeded_vault: Path):
    idx = indexer.build(seeded_vault)
    assert "AuthService" in idx
    assert "JWT Strategy" in idx
    assert "Mobile Team" in idx
    assert len(idx) == 7


def test_index_metadata_shape(seeded_vault: Path):
    idx = indexer.build(seeded_vault)
    node = idx["AuthService"]
    assert node["type"] == "entity"
    assert "auth" in node["tags"]
    assert isinstance(node["decay_weight"], float)
    assert isinstance(node["importance"], float)
    assert "path" in node


def test_backlink_counts(seeded_vault: Path):
    bl = indexer.backlink_counts(seeded_vault)
    # AuthService is referenced by JWT Strategy, Login Bug, Redis Decision → 3
    assert bl["AuthService"] == 3
    # JWT Strategy by AuthService and Login Bug → 2
    assert bl["JWT Strategy"] == 2
    # Mobile Team: no inbound
    assert bl.get("Mobile Team", 0) == 0


def test_index_skips_hidden_files(tmp_vault: Path):
    import yaml
    hidden = tmp_vault / "entities" / "_hidden.md"
    hidden.write_text("---\n" + yaml.dump({"type": "entity"}) + "---\nbody", encoding="utf-8")

    real = tmp_vault / "entities" / "Real.md"
    real.write_text("---\n" + yaml.dump({"type": "entity"}) + "---\nbody", encoding="utf-8")

    idx = indexer.build(tmp_vault)
    assert "Real" in idx
    assert "_hidden" not in idx


def test_build_tolerates_non_frontmatter_file(tmp_vault: Path):
    broken = tmp_vault / "entities" / "Broken.md"
    broken.write_text("not yaml at all", encoding="utf-8")
    idx = indexer.build(tmp_vault)
    # Should not crash; file either appears with defaults or is absent.
    assert isinstance(idx, dict)
