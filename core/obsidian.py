"""External Obsidian vault tools.

Every function takes the vault root as its first arg, stays inside it, and
appends a line to `.meme_audit.log` so the user can audit or `git revert`
anything Samantha does on their behalf.

This module is orthogonal to the internal vault. Writes here do NOT generate
memories, flags, or reflection artefacts.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from utils import frontmatter, wikilinks

AUDIT_FILE = ".meme_audit.log"
TRASH_DIR = "_trash"
_MAX_AUDIT_ARG_CHARS = 500


# ---------- path safety ----------------------------------------------------

def _safe_target(vault: Path, rel_path: str) -> Path | None:
    if not rel_path or rel_path != rel_path.strip():
        return None
    rp = rel_path.strip()
    if rp.startswith("/") or rp.startswith("\\"):
        return None
    # Hard-reject `..` anywhere in the path, even inside a component.
    parts = Path(rp).parts
    if any(p == ".." for p in parts):
        return None
    target = (vault / rp).resolve()
    try:
        target.relative_to(vault.resolve())
    except ValueError:
        return None
    return target


def _ensure_md(rel_path: str) -> str:
    return rel_path if rel_path.endswith(".md") else rel_path + ".md"


# ---------- audit + git ----------------------------------------------------

def _audit(vault: Path, tool: str, args: dict[str, Any], summary: str) -> None:
    try:
        vault.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": tool,
            "args": json.dumps(args, default=str)[:_MAX_AUDIT_ARG_CHARS],
            "result": summary[:_MAX_AUDIT_ARG_CHARS],
        }
        (vault / AUDIT_FILE).open("a", encoding="utf-8").write(
            json.dumps(line, ensure_ascii=False) + "\n"
        )
    except Exception:
        pass


def _git(vault: Path, *args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(vault),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def _is_git_repo(vault: Path) -> bool:
    code, _ = _git(vault, "rev-parse", "--is-inside-work-tree")
    return code == 0


def git_head(vault: Path) -> str | None:
    if not _is_git_repo(vault):
        return None
    code, out = _git(vault, "rev-parse", "--short", "HEAD")
    if code != 0:
        return None
    return out.splitlines()[0].strip() if out else None


def _auto_commit(vault: Path, config: dict[str, Any], tool: str, desc: str) -> None:
    cfg = (config or {}).get("external_vault") or {}
    if not cfg.get("git_auto_commit"):
        return
    if not _is_git_repo(vault):
        _git(vault, "init")
    _git(vault, "add", "-A")
    msg = f"samantha: {tool} {desc}"[:200]
    _git(vault, "commit", "-m", msg)


# ---------- helpers --------------------------------------------------------

def _ok(path: str, action: str, preview: str = "") -> dict[str, Any]:
    return {"ok": True, "path": path, "action": action, "preview": preview[:200]}


def _err(msg: str) -> dict[str, Any]:
    return {"ok": False, "error": msg}


def _list_md(vault: Path, folder: str | None) -> list[Path]:
    root = vault if not folder else (vault / folder)
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*.md"):
        rel = p.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] == TRASH_DIR:
            continue
        out.append(p)
    return out


# ---------- core operations ------------------------------------------------

def create_note(
    vault: Path,
    rel_path: str,
    body: str,
    *,
    frontmatter: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rel = _ensure_md(rel_path)
    target = _safe_target(vault, rel)
    if target is None:
        res = _err(f"invalid path: {rel_path}")
        _audit(vault, "obsidian_create", {"rel_path": rel_path}, res["error"])
        return res
    if target.exists():
        res = _err(f"already exists: {rel}")
        _audit(vault, "obsidian_create", {"rel_path": rel}, res["error"])
        return res
    target.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        from utils import frontmatter as _fm
        _fm.write(target, dict(frontmatter), body)
    else:
        target.write_text(body.rstrip() + "\n", encoding="utf-8")
    res = _ok(rel, "created", f"Created {rel} ({len(body)} chars).")
    _audit(vault, "obsidian_create", {"rel_path": rel, "body_chars": len(body)}, res["preview"])
    _auto_commit(vault, config or {}, "create", rel)
    return res


def update_note(
    vault: Path,
    rel_path: str,
    body: str,
    *,
    mode: Literal["replace", "append", "prepend"] = "replace",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rel = _ensure_md(rel_path)
    target = _safe_target(vault, rel)
    if target is None:
        res = _err(f"invalid path: {rel_path}")
        _audit(vault, "obsidian_update", {"rel_path": rel_path, "mode": mode}, res["error"])
        return res
    if not target.exists():
        res = _err(f"missing: {rel}")
        _audit(vault, "obsidian_update", {"rel_path": rel, "mode": mode}, res["error"])
        return res
    if mode == "replace":
        target.write_text(body.rstrip() + "\n", encoding="utf-8")
    elif mode == "append":
        existing = target.read_text(encoding="utf-8")
        sep = "" if existing.endswith("\n") else "\n"
        target.write_text(existing + sep + body.rstrip() + "\n", encoding="utf-8")
    elif mode == "prepend":
        existing = target.read_text(encoding="utf-8")
        target.write_text(body.rstrip() + "\n" + existing, encoding="utf-8")
    else:
        res = _err(f"bad mode: {mode}")
        _audit(vault, "obsidian_update", {"rel_path": rel, "mode": mode}, res["error"])
        return res
    res = _ok(rel, f"updated:{mode}", f"Updated {rel} ({mode}, {len(body)} chars).")
    _audit(vault, "obsidian_update", {"rel_path": rel, "mode": mode, "body_chars": len(body)}, res["preview"])
    _auto_commit(vault, config or {}, "update", f"{rel} ({mode})")
    return res


def read_note(vault: Path, rel_path: str) -> str:
    rel = _ensure_md(rel_path)
    target = _safe_target(vault, rel)
    if target is None:
        _audit(vault, "obsidian_read", {"rel_path": rel_path}, "invalid path")
        return f"(invalid path: {rel_path})"
    if not target.exists():
        _audit(vault, "obsidian_read", {"rel_path": rel}, "missing")
        return f"(not found: {rel})"
    text = target.read_text(encoding="utf-8")
    _audit(vault, "obsidian_read", {"rel_path": rel}, f"read {len(text)} chars")
    return text


def list_notes(
    vault: Path, folder: str | None = None, *, limit: int = 50,
) -> list[str]:
    if folder is not None:
        safe = _safe_target(vault, folder)
        if safe is None:
            _audit(vault, "obsidian_list", {"folder": folder}, "invalid folder")
            return []
    paths = _list_md(vault, folder)
    rels = sorted(str(p.relative_to(vault)) for p in paths)
    out = rels[:limit]
    _audit(vault, "obsidian_list", {"folder": folder, "limit": limit}, f"{len(out)} notes")
    return out


def search_notes(
    vault: Path, query: str, *, limit: int = 10,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        _audit(vault, "obsidian_search", {"query": query}, "empty query")
        return []
    needle = q.lower()
    hits: list[dict[str, Any]] = []
    for p in _list_md(vault, None):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(p.relative_to(vault))
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                hits.append({
                    "path": rel,
                    "line_no": i,
                    "snippet": line.strip()[:240],
                })
                if len(hits) >= limit:
                    _audit(vault, "obsidian_search", {"query": q, "limit": limit}, f"{len(hits)} hits")
                    return hits
    _audit(vault, "obsidian_search", {"query": q, "limit": limit}, f"{len(hits)} hits")
    return hits


def add_wikilink(
    vault: Path,
    rel_path: str,
    target: str,
    *,
    label: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rel = _ensure_md(rel_path)
    path = _safe_target(vault, rel)
    if path is None:
        res = _err(f"invalid path: {rel_path}")
        _audit(vault, "obsidian_link", {"rel_path": rel_path, "target": target}, res["error"])
        return res
    if not path.exists():
        res = _err(f"missing: {rel}")
        _audit(vault, "obsidian_link", {"rel_path": rel, "target": target}, res["error"])
        return res
    clean_target = (target or "").strip().removesuffix(".md")
    if not clean_target:
        res = _err("empty target")
        _audit(vault, "obsidian_link", {"rel_path": rel, "target": target}, res["error"])
        return res
    link = f"[[{clean_target}|{label}]]" if label else f"[[{clean_target}]]"
    text = path.read_text(encoding="utf-8")
    sep = "" if text.endswith("\n") else "\n"
    path.write_text(text + sep + link + "\n", encoding="utf-8")
    res = _ok(rel, "linked", f"Appended {link} to {rel}.")
    _audit(vault, "obsidian_link", {"rel_path": rel, "target": clean_target, "label": label}, res["preview"])
    _auto_commit(vault, config or {}, "link", f"{rel} -> {clean_target}")
    return res


def rename_note(
    vault: Path,
    old_rel: str,
    new_rel: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    old_r = _ensure_md(old_rel)
    new_r = _ensure_md(new_rel)
    old_p = _safe_target(vault, old_r)
    new_p = _safe_target(vault, new_r)
    if old_p is None or new_p is None:
        res = _err("invalid path")
        _audit(vault, "obsidian_rename", {"old": old_rel, "new": new_rel}, res["error"])
        return res
    if not old_p.exists():
        res = _err(f"missing: {old_r}")
        _audit(vault, "obsidian_rename", {"old": old_r, "new": new_r}, res["error"])
        return res
    if new_p.exists():
        res = _err(f"already exists: {new_r}")
        _audit(vault, "obsidian_rename", {"old": old_r, "new": new_r}, res["error"])
        return res
    new_p.parent.mkdir(parents=True, exist_ok=True)
    old_p.rename(new_p)

    # Rewrite incoming wikilinks across the vault. Match the old stem and the
    # old path-without-ext form; preserve any `|label` or `#anchor`.
    old_stem = Path(old_r).with_suffix("").as_posix()
    old_name = Path(old_r).stem
    new_stem = Path(new_r).with_suffix("").as_posix()
    new_name = Path(new_r).stem
    updated: list[str] = []
    for p in _list_md(vault, None):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        def _sub(m: re.Match[str]) -> str:
            target = m.group(1)
            rest = m.group(2) or ""
            base = target.split("#", 1)[0].strip()
            anchor = target[len(base):]
            if base == old_stem:
                return f"[[{new_stem}{anchor}{rest}]]"
            if base == old_name:
                return f"[[{new_name}{anchor}{rest}]]"
            return m.group(0)
        new_text = re.sub(r"\[\[([^\]\|]+)(\|[^\]]+)?\]\]", _sub, text)
        if new_text != text:
            p.write_text(new_text, encoding="utf-8")
            updated.append(str(p.relative_to(vault)))

    preview = f"Renamed {old_r} -> {new_r}. Updated {len(updated)} backlink(s)."
    res = _ok(new_r, "renamed", preview)
    res["updated_files"] = updated
    _audit(vault, "obsidian_rename", {"old": old_r, "new": new_r, "updated": len(updated)}, preview)
    _auto_commit(vault, config or {}, "rename", f"{old_r} -> {new_r}")
    return res


def delete_note(
    vault: Path,
    rel_path: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rel = _ensure_md(rel_path)
    target = _safe_target(vault, rel)
    if target is None:
        res = _err(f"invalid path: {rel_path}")
        _audit(vault, "obsidian_delete", {"rel_path": rel_path}, res["error"])
        return res
    if not target.exists():
        res = _err(f"missing: {rel}")
        _audit(vault, "obsidian_delete", {"rel_path": rel}, res["error"])
        return res
    trash = vault / TRASH_DIR
    trash.mkdir(parents=True, exist_ok=True)
    # Unique destination to avoid clobbering a prior soft-delete.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = trash / f"{stamp}-{Path(rel).name}"
    i = 1
    while dest.exists():
        dest = trash / f"{stamp}-{i}-{Path(rel).name}"
        i += 1
    target.rename(dest)
    res = _ok(rel, "deleted", f"Moved {rel} to {TRASH_DIR}/{dest.name}.")
    _audit(vault, "obsidian_delete", {"rel_path": rel, "trashed_as": dest.name}, res["preview"])
    _auto_commit(vault, config or {}, "delete", rel)
    return res


# ---------- config + audit helpers ----------------------------------------

def resolve_vault_path(config: dict[str, Any] | None) -> Path | None:
    cfg = (config or {}).get("external_vault") or {}
    raw = cfg.get("path")
    if not raw:
        return None
    import os
    return Path(os.path.expanduser(str(raw))).resolve()


def read_audit_tail(vault: Path, n: int = 10) -> list[dict[str, Any]]:
    path = vault / AUDIT_FILE
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            out.append({"raw": ln})
    return out


def note_count(vault: Path) -> int:
    return len(_list_md(vault, None))
