"""Shared Textual widgets + CSS used by both tui.py and voice_tui.py.

Keeps the visual language consistent so the two front ends look like
siblings of the same app, and avoids drift when either is tweaked.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
    """A single chat bubble — user / ai / system."""

    def __init__(self, content: str, kind: str = "user"):
        super().__init__(content)
        self.kind = kind
        self.add_class(f"{kind}-message")

    def update_content(self, content: str) -> None:
        self.update(content)


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


_SENTENCE_END = re.compile(r"([.!?]+[\)\"']?)(\s+|$)")


def split_sentence(buffer: str) -> tuple[str, str]:
    """Return (completed_sentence, remainder). Empty sentence if none yet."""
    match = _SENTENCE_END.search(buffer)
    if not match:
        return "", buffer
    end = match.end()
    return buffer[:end].strip(), buffer[end:]
