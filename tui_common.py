"""Shared Textual widgets + CSS used by both tui.py and voice_tui.py.

Keeps the visual language consistent so the two front ends look like
siblings of the same app, and avoids drift when either is tweaked.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, ListItem, Static

from utils import frontmatter

# --------------------------------------------------------------------------
# Theme — dark, purple accents. Shared CSS base that both apps extend.
# --------------------------------------------------------------------------
BASE_CSS = """
Screen { background: #0d1117; }

#main-layout { layout: horizontal; height: 1fr; }

#sidebar {
    width: 34; height: 100%;
    background: #161b22; border-right: solid #30363d;
    layout: vertical;
}

#app-header {
    width: 100%; height: auto;
    background: #161b22; border-bottom: solid #30363d;
    content-align: center middle; padding: 1 0;
}

#banner-label { width: 100%; color: #c9a0ff; text-align: center; }
#subtitle-label { width: 100%; color: #8b949e; text-align: center; text-style: italic; }

#sidebar-controls { height: auto; padding: 1; }

#new-chat-btn {
    width: 100%; height: 3; background: #238636; color: white;
    margin-bottom: 1;
}
#mute-btn { width: 100%; height: 3; background: #30363d; color: white; }
#mute-btn.--muted { background: #d29922; }

#transcript-list { height: 1fr; background: #161b22; scrollbar-size: 1 1; }

TranscriptItem {
    height: 3;
    background: #161b22;
    border-bottom: solid #1c2128;
    layout: horizontal;
    padding: 0 1;
}
TranscriptItem:hover { background: #1c2128; }
TranscriptItem.--highlight { border-left: solid #c9a0ff; background: #1c2128; }

.item-info  { width: 1fr; layout: vertical; }
.item-title { color: #c9d1d9; text-style: bold; height: 1; }
.item-date  { color: #8b949e; text-style: dim;  height: 1; }

.item-delete {
    width: 3; height: 1; min-width: 3;
    color: #8b949e; background: transparent; border: none;
    padding: 0; content-align: center middle;
    margin-top: 1;
}
.item-delete:hover { color: #f85149; background: #30363d; }

#chat-area { width: 1fr; height: 100%; layout: vertical; }

#chat-container {
    height: 1fr; background: #0d1117;
    padding: 1 2; overflow-y: auto; scrollbar-size: 1 1;
}

#input-area {
    height: auto; background: #161b22;
    border-top: solid #30363d; padding: 1 2; layout: vertical;
}

#input-row { height: 3; layout: horizontal; align: left middle; }

#user-input {
    width: 1fr; height: 3;
    background: #0d1117; border: round #30363d; color: #c9d1d9;
}
#user-input:focus { border: round #58a6ff; background: #0d1117; }

#status-bar {
    height: 1; background: transparent;
    color: #8b949e; text-align: center;
}

.user-message {
    width: 85%; color: #f0f6fc; background: #0d47a1;
    padding: 1 2; border-left: solid #58a6ff; margin-bottom: 1;
}
.ai-message {
    width: 85%; color: #c9d1d9; background: #1c2128;
    padding: 1 2; border-left: solid #c9a0ff; margin-bottom: 1;
}
.system-message {
    color: #8b949e; text-style: dim;
    margin-bottom: 1; padding: 0 2;
}
"""


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------
class ChatMessage(Static):
    """A single chat bubble — user / ai / system.

    AI and system messages render markdown so the model's `**bold**`,
    headers, code blocks, and lists appear formatted rather than literal.
    User messages stay plain text.
    """

    def __init__(self, content: str, kind: str = "user"):
        self.kind = kind
        super().__init__(self._render(content))
        self.add_class(f"{kind}-message")

    def _render(self, content: str):
        if not content:
            return ""
        if self.kind in ("ai", "system"):
            # Rich's Markdown handles partial / streaming text gracefully —
            # an open code fence mid-stream just renders as code-in-progress.
            return Markdown(content, code_theme="monokai", inline_code_theme="monokai")
        return content  # user messages: plain text

    def update_content(self, content: str) -> None:
        self.update(self._render(content))


class TranscriptItem(ListItem):
    """Sidebar item for a past session. Click to view. X to delete."""

    def __init__(self, name: str, stamp: str):
        super().__init__()
        self.transcript_name = name
        self.stamp = stamp

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="item-info"):
                parts = self.transcript_name.split("-", 5)
                pretty = parts[-1].replace("-", " ") if len(parts) == 6 else self.transcript_name
                yield Label(pretty[:28], classes="item-title")
                yield Label(self.stamp, classes="item-date")
            yield Button("×", id=f"del-{self.transcript_name}", classes="item-delete")


class StatusBar(Static):
    """Bottom status bar. `hotkey_hint` is the fixed left-hand section,
    `text` is the right-hand dynamic status."""

    text = reactive("")
    listening = reactive(False)
    hotkey_hint: str | None = None  # subclasses / owners set this

    def render(self) -> Text:
        t = Text(justify="center")
        if self.hotkey_hint:
            t.append(self.hotkey_hint, style="dim")
        if self.text:
            if self.hotkey_hint:
                t.append("   │   ", style="dim")
            style = "bold #3fb950" if self.listening else "bold #8b949e"
            t.append(self.text, style=style)
        return t


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def read_identity(vault: Path) -> tuple[str, str] | None:
    """Return (name, user_name) from _identity/persona.md, or None."""
    path = vault / "_identity" / "persona.md"
    if not path.exists():
        return None
    try:
        fm, _ = frontmatter.read(path)
        name = fm.get("name")
        user = fm.get("user_name") or ""
        return (str(name), str(user)) if name else None
    except Exception:
        return None


def transcript_entries(vault: Path, limit: int = 30) -> list[tuple[str, str]]:
    """Return [(transcript_name, pretty_stamp), ...] newest first."""
    tdir = vault / "_transcripts"
    if not tdir.exists():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(tdir.glob("*.md"), reverse=True)[:limit]:
        try:
            stamp = datetime.strptime(p.stem[:19], "%Y-%m-%d-%H%M%S").strftime("%b %d  %H:%M")
        except Exception:
            stamp = p.stem[:10]
        out.append((p.stem, stamp))
    return out


def parse_transcript(body: str) -> list[dict[str, str]]:
    """Reconstruct user/assistant turns from a stored transcript body."""
    chunks: list[dict[str, str]] = []
    role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if role and buf:
            chunks.append({"role": role, "content": "\n".join(buf).strip()})

    for line in body.splitlines():
        s = line.strip()
        if s.startswith("## USER"):
            flush(); role = "user"; buf = []
        elif s.startswith("## ASSISTANT"):
            flush(); role = "assistant"; buf = []
        else:
            buf.append(line)
    flush()
    return [c for c in chunks if c["content"]]


_MEME_FLAG_RE = re.compile(
    r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)(?::[^\]]*)?\]",
    re.IGNORECASE,
)


def strip_meme_flags(text: str) -> str:
    """Remove [NOVEL:], [SALIENT:] etc. — memory-system signals, not for display/TTS.
    Leaves ChatterBox paralinguistic tags ([laugh], [sigh], ...) alone."""
    return _MEME_FLAG_RE.sub("", text).strip()


# --- TTS cleaner ------------------------------------------------------------
# The TTS model speaks literally — `*bold*` becomes "asterisk bold asterisk".
# This cleaner strips every bit of markdown and code formatting but keeps
# ChatterBox paralinguistic tags ([laugh] etc.) intact.

_CODE_FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)(?:\n```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_MD_ITAL_STAR = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_BOLD_UND = re.compile(r"__([^_\n]+)__")
_MD_ITAL_UND = re.compile(r"(?<![A-Za-z_])_([^_\n]+)_(?![A-Za-z_])")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_NUMBERED = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_MD_HRULE = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_PIPE = re.compile(r"\s*\|\s*")
_EXTRA_WS = re.compile(r"[ \t]+")
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def clean_for_speech(text: str) -> str:
    """Strip markdown/code formatting so TTS doesn't speak symbols.
    Keeps ChatterBox paralinguistic tags ([laugh], [sigh], ...) intact."""
    if not text:
        return ""

    # 1. Fenced code blocks — replace with a brief narration.
    def _code_sub(m: re.Match) -> str:
        lang = (m.group(1) or "").strip()
        return f" (a code snippet in {lang}) " if lang else " (a code snippet) "
    out = _CODE_FENCE.sub(_code_sub, text)

    # 2. Strip inline-code backticks, keep the token.
    out = _INLINE_CODE.sub(r"\1", out)

    # 3. Markdown links → just the link text.
    out = _MD_LINK.sub(r"\1", out)

    # 4. Bold / italic markers.
    out = _MD_BOLD.sub(r"\1", out)
    out = _MD_BOLD_UND.sub(r"\1", out)
    out = _MD_ITAL_STAR.sub(r"\1", out)
    out = _MD_ITAL_UND.sub(r"\1", out)

    # 5. Headings, bullets, numbered lists, horizontal rules, blockquotes.
    out = _MD_HEADER.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = _MD_NUMBERED.sub("", out)
    out = _MD_HRULE.sub("", out)
    out = _MD_BLOCKQUOTE.sub("", out)

    # 6. Tables — drop pipes, keep cell text.
    out = _TABLE_PIPE.sub(" ", out)

    # 7. Meme flags out, paralinguistic tags stay.
    out = _MEME_FLAG_RE.sub("", out)
    # Also strip partial/unclosed flags that got cut by a mid-stream split —
    # without this, a sentence can end right inside `[NOVEL: some ` and the
    # opener gets spoken aloud.
    out = re.sub(
        r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)[^\]]*$",
        "",
        out,
        flags=re.IGNORECASE,
    )

    # 8. Stray markup chars that slipped through.
    out = out.replace("```", " ")

    # 9. Normalise whitespace.
    out = _EXTRA_WS.sub(" ", out)
    out = _EXTRA_NEWLINES.sub("\n\n", out)
    return out.strip()


_SENTENCE_END = re.compile(r"([.!?]+[\)\"']?)(\s+|$)")


def split_sentence(buffer: str) -> tuple[str, str]:
    """Return (completed_sentence, remainder). Empty sentence if none yet."""
    match = _SENTENCE_END.search(buffer)
    if not match:
        return "", buffer
    end = match.end()
    return buffer[:end].strip(), buffer[end:]


def log_error(vault: Path, source: str, exc: Exception) -> Path:
    """Append a timestamped error to vault/_meta/errors.log — a persistent
    record the user can `cat` outside the TUI (since Textual captures
    the mouse and ordinary terminal copy-paste doesn't work inside it).

    Returns the log path so the caller can surface it in a system message.
    """
    log_path = vault / "_meta" / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{stamp}] {source}\n{type(exc).__name__}: {exc}\n")
    return log_path
