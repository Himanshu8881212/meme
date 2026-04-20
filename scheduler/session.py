from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core import flagging, reflection, retrieval
from utils import frontmatter

SESSIONS_DIR_NAME = "_meta"
SESSION_LOG = "session_log.md"
TRANSCRIPTS_DIR_NAME = "_transcripts"
SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    slug = SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "session"


def archive_transcript(vault: Path, transcript: str, task: str, tags: list[str]) -> str:
    """Write a session transcript verbatim under `_transcripts/`.

    Returns the transcript's name (without .md) suitable for a wikilink like
    `[[_transcripts/<name>|verbatim session]]`. Transcripts are immutable —
    decay never touches them, default retrieval hides them. They exist so the
    distilled graph can always be re-derived from the raw source.
    """
    ts = datetime.now()
    slug = _slugify(task or "session")[:40]
    name = f"{ts.strftime('%Y-%m-%d-%H%M%S')}-{slug}"
    path = vault / TRANSCRIPTS_DIR_NAME / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    fm = {
        "type": "transcript",
        "created": ts.isoformat(timespec="seconds"),
        "task": task,
        "tags": list(tags or []),
        "immutable": True,
    }
    body = f"# Session transcript — {ts.strftime('%Y-%m-%d %H:%M:%S')}\n\nTask: {task}\n\n{transcript}\n"
    frontmatter.write(path, fm, body)
    return name


def _load_system_prompt(prompts_dir: Path) -> str:
    return (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")


def _load_identity(vault: Path) -> str:
    """Compose identity from two files:
      _identity/persona.md — immutable personality (from template, set by init)
      _identity/self.md    — accumulating relationship + preferences

    Keeping them separate means reflection can never erode personality while
    still being able to learn about the user and record preferences.
    """
    persona_path = vault / "_identity" / "persona.md"
    self_path = vault / "_identity" / "self.md"

    parts: list[str] = []

    if persona_path.exists():
        _, body = frontmatter.read(persona_path)
        parts.append(body.strip())
    else:
        parts.append(
            "(No persona is set. I am an unnamed assistant. I will not "
            "invent a name or backstory — when the user gives me one, an "
            "`[IDENTITY: ...]` flag will persist it.)"
        )

    if self_path.exists():
        _, body = frontmatter.read(self_path)
        if body.strip():
            parts.append(body.strip())

    return "\n\n".join(parts)


def start(task: str, tags: list[str], config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    vault = Path(config["vault_path"])
    if not vault.is_absolute():
        vault = (project_root / vault).resolve()

    files = retrieval.retrieve(vault, task, tags, config)
    files = files[: config["session"]["max_context_files"]]

    prompts_dir = project_root / "prompts"
    system_prompt = _load_system_prompt(prompts_dir)

    context_block = "\n\n".join(
        f"=== {p} ===\n{c}" for p, c in files
    ) or "(my vault has nothing relevant to this query yet — bootstrap)"

    identity_block = _load_identity(vault)

    composed_system = (
        system_prompt
        .replace("{{IDENTITY}}", identity_block)
        .replace("{{RETRIEVED_CONTEXT}}", context_block)
    )

    return {
        "task": task,
        "tags": tags,
        "retrieved_files": [p for p, _ in files],
        "system_prompt": composed_system,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def _touch_retrieved_nodes(paths: list[str]) -> None:
    today = datetime.now().date().isoformat()
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        fm, body = frontmatter.read(path)
        fm["last_accessed"] = today
        fm["access_count"] = int(fm.get("access_count", 0)) + 1
        frontmatter.write(path, fm, body)


def end(
    session_output: str,
    session_meta: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    vault = Path(config["vault_path"])
    if not vault.is_absolute():
        vault = (project_root / vault).resolve()

    flags = flagging.extract(session_output)
    summary = flagging.summarize(flags)

    _touch_retrieved_nodes(session_meta.get("retrieved_files", []))

    transcript_name = archive_transcript(
        vault=vault,
        transcript=session_output,
        task=session_meta.get("task", ""),
        tags=session_meta.get("tags", []),
    )

    # A session is "substantive" if there's enough content to have durable
    # facts worth remembering even when Model 1 failed to flag them. Thresholds
    # are configurable — see config.reflection.recovery_*.
    r_cfg = config.get("reflection") or {}
    min_chars = int(r_cfg.get("recovery_min_chars", 150))
    min_user_turns = int(r_cfg.get("recovery_min_user_turns", 2))
    substantive = (
        len(session_output) > min_chars
        or session_output.upper().count("USER") >= min_user_turns
    )

    result: dict[str, Any] = {
        "flags_found": len(flags),
        "reflection_run": False,
        "reflection_output": None,
        "recovery_mode": False,
        "transcript": transcript_name,
    }

    min_flags = config["reflection"]["min_flags_for_reflection"]
    should_reflect = len(flags) >= min_flags or substantive
    if should_reflect:
        recovery = len(flags) < min_flags and substantive
        files = retrieval.retrieve(
            vault,
            context=summary if flags else session_meta.get("task", ""),
            context_tags=session_meta.get("tags", []),
            config=config,
        )
        transcript_ref = (
            f"\n\n(verbatim transcript archived at "
            f"`_transcripts/{transcript_name}.md` — link it from any episode "
            f"node you create via `[[_transcripts/{transcript_name}|verbatim session]]`)"
        )
        flag_block = summary if flags else (
            "(RECOVERY MODE: no in-session flags were emitted by Model 1. "
            "The transcript below is your only source. Read it carefully and "
            "extract durable facts as if you had been flagging in real time.\n\n"
            "Non-negotiable rules in recovery mode:\n"
            "1. ENTITY-FIRST. If the topic is a concept/technology/place/person, "
            "create a `concepts/` or `entities/` node — NEVER `episodes/<X> discussion.md`. "
            "Episodes are for specific events (meetings, incidents, decisions), "
            "not for 'we talked about X'.\n"
            "2. UPDATE BEFORE CREATE. Check the retrieved nodes first. If this "
            "topic already has a node, emit `action=\"update\"` with the full "
            "new body (preserving existing content). Don't create parallels.\n"
            "3. CONTRADICTION CHECK. If anything in the transcript evolves or "
            "conflicts with a retrieved node, update that node with a resolution "
            "section, or create a `tensions/` node linking both sides. Never "
            "leave parallel conflicting nodes in the vault.\n"
            "4. IDENTITY/PREFERENCE. If the user set a standing preference "
            "(even one as short as 'be more blunt' or 'tldr first'), update "
            "`_identity/self.md` under 'Standing preferences'. These are the "
            "rules they want me to hold across sessions.\n"
            "5. Silence is valid only when the transcript truly has no durable "
            "content. A 200+ char substantive exchange almost always does.)"
        )
        output = reflection.routine(
            flag_summary=flag_block,
            vault_files=files,
            session_notes=session_output[:8000] + transcript_ref,
            config=config,
        )
        writes = reflection.apply_writes(
            output, vault,
            similarity_threshold=float(r_cfg.get("duplicate_similarity_threshold", 0.5)),
        )
        result["reflection_run"] = True
        result["recovery_mode"] = recovery
        result["reflection_output"] = output
        result["writes"] = writes

    _log_session(vault, session_meta, flags, result)
    return result


def _log_session(
    vault: Path,
    meta: dict[str, Any],
    flags: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    log_path = vault / SESSIONS_DIR_NAME / SESSION_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "started_at": meta.get("started_at"),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "task": meta.get("task"),
        "tags": meta.get("tags"),
        "retrieved": len(meta.get("retrieved_files", [])),
        "flags": len(flags),
        "reflection_run": result["reflection_run"],
    }

    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Session Log\n\n"
    appended = existing + "\n```json\n" + json.dumps(entry, indent=2) + "\n```\n"
    log_path.write_text(appended, encoding="utf-8")
