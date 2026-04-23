"""Rule-based vault integrity scan + auto-repair.

Runs cheaply (no LLM calls) as part of /meta and the background meta
loop. This is AI memory — it heals itself. No human-review queue; every
issue this layer detects has an automatic repair path.

Catches the silent killers that the reflection pipeline has
historically produced:

  - placeholder rot (`<preserve existing body>`, `<incremented>` baked
    into real files because the writer blindly applied the model's
    template tokens) — stripped in place, deleted if nothing real remains
  - orphan `.md` files with empty stems (from obsidian_create("") bugs)
    — deleted
  - malformed frontmatter (non-numeric values where a number belongs)
    — coerced to safe defaults
  - invalid JSONL rows in the tool_memory log — moved to quarantine
  - tiny drive-by transcripts (< MIN_TRANSCRIPT_TURNS) — deleted

Paths of stripped/cleaned placeholder-rot nodes are also pushed into the
downstream deep-reflection pass via `placeholder_rot:` triggers, so the
model re-populates those bodies with real content in the same meta run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from utils import frontmatter, indexer

# Matches the template tokens reflection.md historically put in the
# model's mouth. `apply_writes` now rejects these at write time, but
# this scan finds the ones that leaked in before the guard existed.
PLACEHOLDER_RE = re.compile(
    r"<\s*(?:preserve existing body|preserve existing|increment existing|"
    r"existing body preserved|existing frontmatter preserved|"
    r"remainder of existing body preserved|incremented|preserved)\s*>|"
    r"\.\.\. *existing body preserved *\.\.\.",
    re.IGNORECASE,
)

_NUMERIC_FIELDS = ("access_count", "connection_count", "importance", "decay_weight")

# Transcripts with fewer total turns (USER + ASSISTANT headers combined)
# than this are "drive-by" exchanges — one-word tests, accidental
# submits, listen-mode false triggers. They clutter /history and give
# reflection nothing real to work with. Default 8 = 4 rounds each side.
MIN_TRANSCRIPT_TURNS = 8
_TURN_HDR = re.compile(r"^##\s+(USER|ASSISTANT)\b", re.MULTILINE | re.IGNORECASE)


def _count_turns(body: str) -> int:
    return len(_TURN_HDR.findall(body))


def _is_number(v: Any) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def scan(vault_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Walk the vault once, produce a bucketed list of issues. Read-only."""
    vault = Path(vault_path).resolve()
    issues: dict[str, list[dict[str, Any]]] = {
        "placeholder_rot": [],
        "orphan_md_files": [],
        "bad_frontmatter": [],
        "invalid_jsonl": [],
        "tiny_transcripts": [],
    }

    for path in vault.rglob("*.md"):
        if indexer._is_hidden(path, vault):
            continue
        # Files whose stem is empty (".md") are orphans — obsidian_create("")
        # used to produce these. Separate bucket from normal scan.
        if path.stem == "" or path.name == ".md":
            issues["orphan_md_files"].append({"path": str(path)})
            continue
        try:
            fm, body = frontmatter.read(path)
        except Exception as exc:
            issues["bad_frontmatter"].append({
                "path": str(path), "error": f"read failed: {exc}",
            })
            continue

        if PLACEHOLDER_RE.search(body):
            hits = PLACEHOLDER_RE.findall(body)
            issues["placeholder_rot"].append({
                "path": str(path),
                "hits": len(hits),
                "sample": hits[0] if hits else "",
            })

        bad_fields = [
            f for f in _NUMERIC_FIELDS
            if f in fm and not _is_number(fm[f])
        ]
        if bad_fields:
            issues["bad_frontmatter"].append({
                "path": str(path),
                "fields": {f: str(fm.get(f))[:40] for f in bad_fields},
            })

    # Scan transcripts for drive-by sessions.
    tdir = vault / "_transcripts"
    if tdir.exists():
        for p in tdir.glob("*.md"):
            try:
                _, body = frontmatter.read(p)
            except Exception:
                continue
            turns = _count_turns(body)
            if turns < MIN_TRANSCRIPT_TURNS:
                issues["tiny_transcripts"].append({
                    "path": str(p), "turns": turns,
                })

    # Scan tool_memory JSONL if present.
    tool_log = vault / "_meta" / "tool_log.jsonl"
    if tool_log.exists():
        for i, line in enumerate(tool_log.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues["invalid_jsonl"].append({
                    "file": "_meta/tool_log.jsonl", "line": i, "error": str(exc),
                })
                continue
            # Schema violation: `tool` field should be a tool identifier
            # (snake_case word), not a narrative sentence. Magistral used
            # to emit hallucinated tool calls whose "name" was the flag
            # text itself; we filter those at dispatch now, but old rows
            # remain. Treat any `tool` > 60 chars or containing spaces
            # as corrupt.
            t = obj.get("tool") or ""
            if not isinstance(t, str) or len(t) > 60 or " " in t:
                issues["invalid_jsonl"].append({
                    "file": "_meta/tool_log.jsonl", "line": i,
                    "error": f"bad tool field: {t[:40]!r}",
                })

    return issues


def repair(
    vault_path: str | Path,
    issues: dict[str, list[dict[str, Any]]],
    *,
    prune_tool_log: bool = True,
    remove_orphans: bool = True,
    patch_frontmatter: bool = True,
    delete_tiny_transcripts: bool = True,
) -> dict[str, Any]:
    """Apply safe repairs. Never rewrites node bodies — placeholder rot
    is reported only, because restoring the lost content requires human
    judgment or the reflection model."""
    vault = Path(vault_path).resolve()
    actions: dict[str, Any] = {
        "removed_orphans": [],
        "patched_frontmatter": [],
        "pruned_tool_log_rows": 0,
        "deleted_tiny_transcripts": [],
        "placeholder_rot_cleaned": [],
    }

    # 1. Orphan `.md` files — safe to delete.
    if remove_orphans:
        for o in issues.get("orphan_md_files", []):
            p = Path(o["path"])
            try:
                p.unlink()
                actions["removed_orphans"].append(o["path"])
            except Exception:
                pass

    # 2. Frontmatter numeric fields — coerce to sensible defaults.
    if patch_frontmatter:
        for bad in issues.get("bad_frontmatter", []):
            if "fields" not in bad:
                continue
            p = Path(bad["path"])
            try:
                fm, body = frontmatter.read(p)
            except Exception:
                continue
            changed = False
            defaults = {
                "access_count": 1, "connection_count": 0,
                "importance": 0.5, "decay_weight": 0.5,
            }
            for field in bad["fields"]:
                fm[field] = defaults.get(field, 0)
                changed = True
            if changed:
                try:
                    frontmatter.write(p, fm, body)
                    actions["patched_frontmatter"].append(bad["path"])
                except Exception:
                    pass

    # 3. Tool-log JSONL — prune corrupt rows into a quarantine file so
    # nothing is silently lost.
    if prune_tool_log and issues.get("invalid_jsonl"):
        tool_log = vault / "_meta" / "tool_log.jsonl"
        if tool_log.exists():
            bad_lines = {i["line"] for i in issues["invalid_jsonl"]}
            lines = tool_log.read_text(encoding="utf-8").splitlines()
            kept, dropped = [], []
            for n, line in enumerate(lines, 1):
                (dropped if n in bad_lines else kept).append(line)
            if dropped:
                quarantine = vault / "_meta" / "tool_log.quarantine.jsonl"
                with quarantine.open("a", encoding="utf-8") as f:
                    for line in dropped:
                        f.write(line + "\n")
                tool_log.write_text("\n".join(kept) + "\n", encoding="utf-8")
                actions["pruned_tool_log_rows"] = len(dropped)

    # 4. Drive-by transcripts (< MIN_TRANSCRIPT_TURNS) — safe to delete.
    # These are accidental submits, one-word tests, and listen-mode
    # false triggers. They clutter /history and contribute nothing
    # durable to memory. Session log entries are left untouched — the
    # record of "something happened" stays, the noise transcript goes.
    if delete_tiny_transcripts:
        for t in issues.get("tiny_transcripts", []):
            p = Path(t["path"])
            try:
                p.unlink()
                actions["deleted_tiny_transcripts"].append(t["path"])
            except Exception:
                pass

    # 5. Placeholder rot — strip the placeholder tokens in place. Any
    # surviving real content stays; deep reflection (the next stage of
    # /meta) re-populates the gaps because we pass these paths as
    # triggers. No human validation needed — this is AI memory, it
    # heals itself.
    for r in issues.get("placeholder_rot", []):
        p = Path(r["path"])
        try:
            fm, body = frontmatter.read(p)
        except Exception:
            continue
        cleaned = PLACEHOLDER_RE.sub("", body).strip()
        # If after stripping there's effectively nothing left, drop the
        # node entirely — it was nothing but rot.
        if len(cleaned) < 20:
            try:
                p.unlink()
                actions["placeholder_rot_cleaned"].append(
                    {"path": r["path"], "action": "deleted"}
                )
            except Exception:
                pass
            continue
        try:
            frontmatter.write(p, fm, cleaned + "\n")
            actions["placeholder_rot_cleaned"].append(
                {"path": r["path"], "action": "stripped"}
            )
        except Exception:
            pass

    return actions


def summary_line(issues: dict[str, list], actions: dict[str, Any]) -> str:
    return (
        f"integrity · rot-cleaned: {len(actions.get('placeholder_rot_cleaned', []))} · "
        f"orphans: {len(actions.get('removed_orphans', []))} · "
        f"fm-patched: {len(actions.get('patched_frontmatter', []))} · "
        f"log-pruned: {actions.get('pruned_tool_log_rows', 0)} · "
        f"tiny-tx-deleted: {len(actions.get('deleted_tiny_transcripts', []))}"
    )
