from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core import decay, flagging, reflection, retrieval
from utils import frontmatter

SESSIONS_DIR_NAME = "_meta"
SESSION_LOG = "session_log.md"
TRANSCRIPTS_DIR_NAME = "_transcripts"
SLUG_RE = re.compile(r"[^a-z0-9]+")

# External-vault tool outputs can be arbitrarily large (a whole read_note
# dump). Truncate them when they make it into the archived transcript body.
_OBSIDIAN_TOOL_OUTPUT_LIMIT = 500
_OBSIDIAN_TOOL_BLOCK = re.compile(
    r"(## TOOL obsidian_\w+\n)(.+?)(?=\n##\s|\Z)",
    re.DOTALL,
)


def truncate_obsidian_tool_outputs(transcript: str) -> str:
    """Cap any `## TOOL obsidian_*` block in a transcript to 500 chars.

    No-op unless someone has explicitly inlined tool output; current front
    ends don't, but this keeps the separation guarantee honest.
    """
    def _sub(m: re.Match[str]) -> str:
        head, body = m.group(1), m.group(2)
        if len(body) <= _OBSIDIAN_TOOL_OUTPUT_LIMIT:
            return m.group(0)
        return head + body[:_OBSIDIAN_TOOL_OUTPUT_LIMIT] + "… [truncated]\n"
    return _OBSIDIAN_TOOL_BLOCK.sub(_sub, transcript)


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
    clean = truncate_obsidian_tool_outputs(transcript)
    body = f"# Session transcript — {ts.strftime('%Y-%m-%d %H:%M:%S')}\n\nTask: {task}\n\n{clean}\n"
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

    from core import obsidian as _ob
    ext_vault = _ob.resolve_vault_path(config)
    if ext_vault is not None:
        # Snapshot the current folder layout so the chat model knows where
        # things live without having to list the whole vault every turn.
        folders: list[str] = []
        try:
            for p in sorted(ext_vault.iterdir()):
                if p.is_dir() and not p.name.startswith(".") and p.name != "_trash":
                    folders.append(p.name)
        except Exception:
            folders = []
        folder_line = ", ".join(folders) if folders else "(empty — no folders yet)"
        composed_system += (
            "\n\n## I can read, write, and manage your notebook\n\n"
            f"Your Obsidian vault lives at `{ext_vault}`.\n"
            f"Current top-level folders: {folder_line}\n\n"
            "**READ when the question is about your stuff** — use "
            "`obsidian_search`, `obsidian_list`, `obsidian_read`. Triggers: "
            "'what's on my X list', 'what did I write about Y', 'do I have "
            "a note on Z', 'show me my reminders', 'find my notes on ...'.\n\n"
            "**WRITE when the user asks you to capture anything** — use "
            "`obsidian_create` (new), `obsidian_update` (edit existing — "
            "append/prepend/replace), `obsidian_link`. Always "
            "`obsidian_list` or `obsidian_search` FIRST so you don't "
            "duplicate a note that already exists.\n\n"
            "**MANAGE when the user asks you to reorganize** — use "
            "`obsidian_rename` (updates every incoming wikilink) and "
            "`obsidian_delete` (soft-delete to `_trash/`, reversible). "
            "Always confirm the exact target with a search/list first.\n\n"
            "### Searching well (this is where I usually fail)\n\n"
            "The user's question is NOT the query. Extract ONE or TWO key "
            "concept words and search for those. Examples:\n"
            "- 'what's on my grocery list?' → `obsidian_search(\"grocery\")`, "
            "not the whole question.\n"
            "- 'did I write anything about React Router?' → "
            "`obsidian_search(\"React Router\")`.\n"
            "- 'what's in my todo list?' → `obsidian_list(folder=\"Todo\")` "
            "is more complete than any phrase search.\n\n"
            "If the first search returns nothing, try a SHORTER query (just "
            "the root noun) or `obsidian_list` the likely folder. Only fall "
            "back to general-knowledge answers after I've actually looked.\n\n"
            "Notes in this vault are YOURS — my own memory only records "
            "that we had a conversation, not the notes' content. When I'm "
            "answering from a note I found, say so explicitly so the user "
            "can tell sourced-from-their-notes apart from my own guesses.\n"
        )

    # Inject recent tool-use history so the model learns across turns.
    try:
        from core import tool_memory as _tm
        summary = _tm.prompt_summary(vault)
    except Exception:
        summary = ""
    if summary:
        composed_system += (
            "\n\n## Recent tool use (learn from this)\n\n"
            "My last dozen tool calls and how they went. Use the pattern: "
            "repeat what worked (`✓`), avoid what returned nothing (`∅`), "
            "don't repeat errors (`✗`). If a query shape reliably returned "
            "empty results, try a shorter keyword or a different tool.\n\n"
            f"{summary}\n"
        )

    # Affective state — rolling mood from recent labeled episodes. Tells
    # the model what register to hold today: if recent sessions have
    # trended heavy, she shouldn't open with `[laugh]`.
    try:
        from core import mood as _mood
        mood_block = _mood.mood_snippet(vault)
    except Exception:
        mood_block = ""
    if mood_block:
        composed_system += f"\n\n{mood_block}\n"

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

    # Run decay after reflection so new/updated nodes keep their
    # freshness, while un-referenced nodes drift toward archive. Without
    # this call the archive stays empty and memory rot accumulates — the
    # vault grows forever, every node equally important.
    d_cfg = config.get("decay") or {}
    try:
        decayed = decay.run(
            vault_path=vault,
            lambda_=float(d_cfg.get("lambda", 0.02)),
            archive_threshold=float(d_cfg.get("archive_threshold", 0.10)),
        )
        result["decayed"] = decayed
    except Exception as exc:
        result["decay_error"] = f"{type(exc).__name__}: {exc}"

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
