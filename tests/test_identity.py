"""Tests for the identity layer — the separation between 'who I am' and
'what I know'."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from core import decay, flagging, reflection, retrieval
from scheduler import session as session_mgr
from utils import frontmatter

ROOT = Path(__file__).resolve().parent.parent


def _write_identity(vault: Path, body: str, name: str = "June") -> Path:
    path = vault / "_identity" / "self.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(path, {
        "type": "identity",
        "name": name,
        "pronouns": "she/her",
        "persona_template": "june",
    }, body)
    return path


def test_identity_flag_type_recognised():
    flags = flagging.extract("User named me [IDENTITY: call me June from now on]")
    assert len(flags) == 1
    assert flags[0]["type"] == "IDENTITY"


def test_identity_injected_into_system_prompt(tmp_vault: Path, config):
    _write_identity(tmp_vault, "I am June.\nI help Himanshu remember things.")
    config["vault_path"] = str(tmp_vault)

    meta = session_mgr.start(task="hello", tags=[], config=config, project_root=ROOT)
    assert "I am June." in meta["system_prompt"]
    assert "I help Himanshu remember things." in meta["system_prompt"]


def test_missing_identity_emits_neutral_sentinel(tmp_vault: Path, config):
    # No _identity/self.md at all.
    config["vault_path"] = str(tmp_vault)
    meta = session_mgr.start(task="hi", tags=[], config=config, project_root=ROOT)
    assert "unnamed assistant" in meta["system_prompt"]


def test_identity_excluded_from_retrieval(tmp_vault: Path, config):
    _write_identity(tmp_vault, "I am Sage.")
    # Also drop a matching regular node.
    frontmatter.write(
        tmp_vault / "concepts" / "Sage Concept.md",
        {"type": "concept", "tags": ["philosophy"], "importance": 0.6},
        "about sagely wisdom",
    )
    results = retrieval.retrieve(tmp_vault, "sage", [], config)
    paths = [p for p, _ in results]
    rel = [str(Path(p).relative_to(tmp_vault)) for p in paths]
    assert not any(r.startswith("_identity/") or r.startswith("_identity\\") for r in rel)
    assert any("Sage Concept" in p for p in paths)


def test_identity_file_excluded_from_decay(tmp_vault: Path):
    _write_identity(tmp_vault, "I am Max.", name="Max")
    # Seed an extreme decay scenario to prove it wouldn't matter.
    fm, body = frontmatter.read(tmp_vault / "_identity" / "self.md")
    fm["last_accessed"] = "2020-01-01"
    fm["importance"] = 0.01
    fm["decay_weight"] = 0.01
    frontmatter.write(tmp_vault / "_identity" / "self.md", fm, body)

    decay.run(tmp_vault, lambda_=0.05, archive_threshold=0.10)

    fm_after, _ = frontmatter.read(tmp_vault / "_identity" / "self.md")
    # decay_weight should be untouched.
    assert fm_after["decay_weight"] == 0.01
    assert fm_after.get("archived") is not True


def test_apply_writes_allows_identity_folder(tmp_vault: Path):
    block = (
        '<<WRITE path="_identity/self.md" action="update">>\n'
        "---\ntype: identity\nname: Kai\n---\nI am Kai.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "update"
    fm, body = frontmatter.read(tmp_vault / "_identity" / "self.md")
    assert fm["name"] == "Kai"
    assert "I am Kai." in body


def test_persona_templates_all_parse_and_have_names():
    personas_dir = ROOT / "prompts" / "personas"
    found = list(personas_dir.glob("*.md"))
    assert len(found) >= 4
    for path in found:
        fm, body = frontmatter.read(path)
        assert "name" in fm, f"{path.name} missing name"
        assert "voice" in fm, f"{path.name} missing voice"
        assert len(body) > 100, f"{path.name} body too short to be a real persona"


def test_system_prompt_uses_first_person_markers():
    """The template itself must be written in first person —
    regressions here break everything downstream."""
    content = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    first_person = sum(content.count(m) for m in (" I ", "I'm ", "I ", " my ", "My ", "me.", "me,"))
    third_person = sum(content.count(m) for m in ("the model ", "This model", "the assistant"))
    assert first_person > third_person, (
        f"system prompt needs more first-person voice "
        f"(got {first_person} vs {third_person} third-person)"
    )


def test_identity_flag_extracted_alongside_others():
    text = "[NOVEL: something] and [IDENTITY: call me June] and [SALIENT: x]"
    flags = flagging.extract(text)
    kinds = [f["type"] for f in flags]
    assert kinds == ["NOVEL", "IDENTITY", "SALIENT"]


def test_persona_file_is_immutable(tmp_vault: Path):
    """The fix for the 'Kai's personality got wiped' bug: reflection must
    refuse to write to _identity/persona.md."""
    persona = tmp_vault / "_identity" / "persona.md"
    persona.parent.mkdir(parents=True, exist_ok=True)
    original = "I am Kai. I have opinions. I remember birthdays."
    frontmatter.write(persona, {"type": "identity", "name": "Kai"}, original)

    block = (
        '<<WRITE path="_identity/persona.md" action="update">>\n'
        "---\ntype: identity\nname: Kai\n---\nMuch less content.\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "rejected"
    assert "immutable" in results[0]["reason"]

    # File must still contain the original content.
    _, body = frontmatter.read(persona)
    assert "opinions" in body
    assert "birthdays" in body


def test_self_md_still_writable(tmp_vault: Path):
    """Protection on persona.md must NOT block writes to self.md."""
    self_path = tmp_vault / "_identity" / "self.md"
    self_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(self_path, {"type": "identity"}, "original self content")

    block = (
        '<<WRITE path="_identity/self.md" action="update">>\n'
        "---\ntype: identity\n---\nupdated self content\n<<END>>"
    )
    results = reflection.apply_writes(block, tmp_vault)
    assert results[0]["action"] == "update"
    _, body = frontmatter.read(self_path)
    assert "updated self content" in body


def test_load_identity_concatenates_persona_and_self(tmp_vault: Path, config):
    from scheduler.session import _load_identity
    persona = tmp_vault / "_identity" / "persona.md"
    self_path = tmp_vault / "_identity" / "self.md"
    persona.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(persona, {"type": "identity"}, "I am Kai. I have opinions.")
    frontmatter.write(self_path, {"type": "identity"},
                      "## Who Himanshu is to me\nHe's 28.")

    composed = _load_identity(tmp_vault)
    assert "I am Kai" in composed
    assert "Who Himanshu is to me" in composed
    assert "He's 28" in composed


def test_load_identity_works_with_only_persona(tmp_vault: Path):
    """If init just set persona and nothing's accumulated yet, we shouldn't
    crash or emit empty content."""
    from scheduler.session import _load_identity
    persona = tmp_vault / "_identity" / "persona.md"
    persona.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(persona, {"type": "identity"}, "I am Kai.")

    composed = _load_identity(tmp_vault)
    assert "I am Kai" in composed
