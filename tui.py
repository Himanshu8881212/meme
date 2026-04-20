#!/usr/bin/env python3
"""meme — keyboard TUI.

Same theme as voice_tui.py, same memory underneath. Type to chat, or use
slash commands: /end, /meta, /monitor, /decay, /identity, /show, /index,
/context, /help. Esc or 'quit' to exit (auto-saves the session).

No voice stack — this front end is text-only. Cross-platform.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Button, Input, Label, ListView
from textual import work

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import decay, monitor, reflection, retrieval  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, frontmatter, indexer  # noqa: E402
from tui_common import (  # noqa: E402
    BASE_CSS, ChatMessage, StatusBar, TranscriptItem,
    read_identity, transcript_entries, parse_transcript, strip_meme_flags,
    log_error,
)

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VAULT = Path(CONFIG["vault_path"])
if not VAULT.is_absolute():
    VAULT = (ROOT / VAULT).resolve()

APP_NAME = "meme"
APP_TAGLINE = "a memory system with a point of view"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "q", ":q", ":wq"}

BARE_COMMANDS = {
    "end": "/end", "cancel": "/cancel",
    "meta": "/meta", "decay": "/decay", "monitor": "/monitor",
    "help": "/help", "context": "/context",
    "identity": "/identity", "whoami": "/identity",
    "index": "/index",
}

HELP = """How to use
  Just type. A session auto-starts on your first message.
  Slash commands (or bare words) for maintenance and inspection.

Session
  /end              end the session — distill flags into memory
  /cancel           discard the session without writing memory

Inspect
  /context          files retrieved for the current session
  /show <name>      print a vault node in full
  /identity         persona + relationship ('whoami' also works)

Maintenance
  /meta             deep reflection — merge, split, reconcile
  /decay            update decay weights across the vault
  /monitor          vault health metrics + triggers
  /index            vault stats + sample node names

Leave
  quit / exit / bye / Esc   saves the session on the way out
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class MemeTUI(App):
    CSS = BASE_CSS
    BINDINGS = [
        Binding("escape", "quit", "quit", show=True),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("ctrl+n", "new_session", "new session", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title = APP_NAME
        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, str]] = []
        self.transcript: list[str] = []
        self._chat_container: ScrollableContainer | None = None
        self._transcript_list: ListView | None = None
        self._current_ai: ChatMessage | None = None

    def compose(self) -> ComposeResult:
        yield Container(
            Label(APP_NAME, id="banner-label"),
            Label(APP_TAGLINE, id="subtitle-label"),
            id="app-header",
        )
        with Horizontal(id="main-layout"):
            with Vertical(id="sidebar"):
                with Container(id="sidebar-controls"):
                    yield Button("New session (Ctrl-N)", id="new-chat-btn", variant="success")
                yield ListView(id="transcript-list")
            with Vertical(id="chat-area"):
                yield ScrollableContainer(id="chat-container")
                with Vertical(id="input-area"):
                    with Horizontal(id="input-row"):
                        yield Input(
                            placeholder="Type to chat, or / for a command (/help for the list)",
                            id="user-input",
                        )
                    bar = StatusBar(id="status-bar")
                    bar.hotkey_hint = "Type send   /help   Ctrl-N new   Esc quit (auto-save)"
                    yield bar

    def on_mount(self) -> None:
        self._chat_container = self.query_one("#chat-container", ScrollableContainer)
        self._transcript_list = self.query_one("#transcript-list", ListView)
        self.query_one("#user-input", Input).focus()
        self._refresh_transcripts()

        persona = read_identity(VAULT)
        if persona:
            name, user = persona
            greet = f"{name} is here. {'Hi ' + user + '.' if user else 'Hi.'}"
        else:
            greet = "no persona set — run: python main.py init --persona june --user-name <name>"
        self._system_message(greet)

        m1 = (CONFIG.get("models") or {}).get("model1") or {}
        self._system_message(
            f"chat model: {m1.get('provider', '?')} / {m1.get('model', '?')}"
        )

    # ----- sidebar ----------------------------------------------------

    def _refresh_transcripts(self) -> None:
        self._transcript_list.clear()
        for name, stamp in transcript_entries(VAULT):
            self._transcript_list.append(TranscriptItem(name, stamp))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, TranscriptItem):
            return
        path = VAULT / "_transcripts" / f"{item.transcript_name}.md"
        if not path.exists():
            return
        self._chat_container.remove_children()
        self._system_message(f"viewing transcript — {item.transcript_name}")
        _, body = frontmatter.read(path)
        for chunk in parse_transcript(body):
            if chunk["role"] == "user":
                self._user_message(chunk["content"])
            else:
                self._ai_message(chunk["content"])

    # ----- buttons ----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = str(event.button.id)
        if bid == "new-chat-btn":
            self.action_new_session()
        elif bid.startswith("del-"):
            name = bid[len("del-"):]
            self._delete_transcript(name)
            event.stop()

    def _delete_transcript(self, name: str) -> None:
        path = VAULT / "_transcripts" / f"{name}.md"
        if path.exists():
            path.unlink()
            self._refresh_transcripts()
            self._system_message(f"deleted transcript — {name}")

    def action_new_session(self) -> None:
        if self.session and self.transcript:
            self._system_message("saving current session — reflecting...")
            self._save_then_reset()
        else:
            self._reset_for_new_session()

    @work(thread=True)
    def _save_then_reset(self) -> None:
        self._save_and_report()
        self.call_from_thread(self._reset_for_new_session)

    def _reset_for_new_session(self) -> None:
        self._chat_container.remove_children()
        self.session = None
        self.messages = []
        self.transcript = []
        self._refresh_transcripts()
        self._system_message("new session — type to start.")
        self.query_one("#user-input", Input).focus()

    # ----- input ------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        low = text.lower()

        if low in EXIT_PHRASES:
            self.action_quit()
            return

        if low in BARE_COMMANDS:
            text = BARE_COMMANDS[low]

        if text.startswith("/"):
            self._dispatch(text)
            return

        self._send(text)

    # ----- command dispatch -------------------------------------------

    def _dispatch(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        arg = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            self._system_message(f"unknown command: /{cmd}")
            return
        try:
            handler(arg)
        except Exception as exc:
            self._system_message(f"error: {exc}")

    def cmd_help(self, _: str) -> None:
        self._system_message(HELP)

    def cmd_end(self, _: str) -> None:
        self._end_session()

    def cmd_cancel(self, _: str) -> None:
        if not self.session:
            self._system_message("no active session.")
            return
        self.session = None
        self.messages = []
        self.transcript = []
        self._system_message("session discarded.")

    def cmd_context(self, _: str) -> None:
        if not self.session:
            self._system_message("no active session — nothing retrieved.")
            return
        files = self.session.get("retrieved_files", [])
        if not files:
            self._system_message("(nothing retrieved — bootstrap mode)")
            return
        lines = [str(Path(p).relative_to(VAULT)) for p in files]
        self._system_message("retrieved:\n  " + "\n  ".join(lines))

    def cmd_show(self, arg: str) -> None:
        if not arg.strip():
            self._system_message("usage: /show <node name>")
            return
        idx = indexer.build(VAULT)
        if arg not in idx:
            matches = [n for n in idx if arg.lower() in n.lower()]
            if not matches:
                self._system_message(f"no node matching '{arg}'")
                return
            arg = matches[0]
        text = Path(idx[arg]["path"]).read_text(encoding="utf-8")
        self._system_message(f"── {arg} ──\n{text}")

    def cmd_identity(self, _: str) -> None:
        path = VAULT / "_identity" / "persona.md"
        if not path.exists():
            self._system_message(
                "no identity set. run: python main.py init --persona june --user-name <name>"
            )
            return
        self._system_message(path.read_text(encoding="utf-8"))

    cmd_whoami = cmd_identity

    def cmd_index(self, _: str) -> None:
        idx = indexer.build(VAULT)
        sample = list(idx)[:15]
        msg = f"{len(idx)} nodes"
        if sample:
            msg += "\n  " + ", ".join(sample) + ("..." if len(idx) > 15 else "")
        self._system_message(msg)

    def cmd_decay(self, _: str) -> None:
        result = decay.run(
            vault_path=VAULT,
            lambda_=CONFIG["decay"]["lambda"],
            archive_threshold=CONFIG["decay"]["archive_threshold"],
        )
        self._system_message("decay: " + json.dumps(result))

    def cmd_monitor(self, _: str) -> None:
        metrics = monitor.collect(VAULT)
        triggers = monitor.check_thresholds(metrics, CONFIG)
        lines = [
            f"nodes: {metrics['total_nodes']}",
            f"archived: {metrics['archived']} ({metrics['archived_ratio']:.1%})",
            f"orphans: {metrics['orphans']} ({metrics['orphan_ratio']:.1%})",
            f"tags: {metrics['tag_vocabulary']}",
            f"avg decay weight: {metrics['avg_decay_weight']:.3f}",
        ]
        if metrics["top_hubs"]:
            lines.append("top hubs:")
            for name, count in metrics["top_hubs"][:5]:
                lines.append(f"  {name} ({count})")
        if triggers:
            lines.append("triggers:")
            for t in triggers:
                lines.append(f"  • {t}")
        self._system_message("\n".join(lines))

    def cmd_meta(self, _: str) -> None:
        """Deep reflection — runs in a worker so the UI stays responsive."""
        self._system_message("deep reflection running (this takes ~30-60s)...")
        self._run_meta()

    @work(thread=True)
    def _run_meta(self) -> None:
        try:
            min_body = int((CONFIG.get("monitor") or {}).get("min_body_chars", 20))
            removed = monitor.cleanup_broken(VAULT, min_body_chars=min_body)
            reconciled = reflection.reconcile_tensions(VAULT)
            orphans = monitor.find_orphans(VAULT)
            metrics = monitor.collect(VAULT)
            triggers = monitor.check_thresholds(metrics, CONFIG)
            if orphans:
                triggers.append(f"orphan_review: {', '.join(orphans[:10])}")
            sample_query = " ".join(n for n, _ in metrics["top_hubs"][:5])
            files = retrieval.retrieve(VAULT, sample_query, [], CONFIG)
            output, call_log = reflection.deep_with_tools(
                VAULT, files, metrics, triggers, CONFIG,
            )
            writes = reflection.apply_writes(
                output, VAULT,
                similarity_threshold=float(
                    (CONFIG.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
                ),
            )
            summary = (
                f"meta done · cleaned: {len(removed)} · reconciled: {len(reconciled)} · "
                f"orphans: {len(orphans)} · tool calls: {len(call_log)} · writes: {len(writes)}"
            )
            self.call_from_thread(self._system_message, summary)
            if writes:
                lines = ["writes:"] + [
                    f"  {w.get('action', '?')}  {w.get('path', '')}"
                    for w in writes[:15]
                ]
                self.call_from_thread(self._system_message, "\n".join(lines))
        except Exception as exc:
            self.call_from_thread(self._system_message, f"meta error: {exc}")

    # ----- chat -------------------------------------------------------

    def _send(self, user_text: str) -> None:
        self._user_message(user_text)
        self._start_ai_message()
        self._run_chat(user_text)

    @work(thread=True)
    def _run_chat(self, user_text: str) -> None:
        if self.session is None:
            try:
                self.session = session_mgr.start(
                    task=user_text, tags=[], config=CONFIG, project_root=ROOT,
                )
                self.messages = []
                self.transcript = []
                self.call_from_thread(
                    self._system_message,
                    f"session started · retrieved {len(self.session['retrieved_files'])} nodes",
                )
            except Exception as exc:
                self.call_from_thread(self._finish_ai_message, f"(start failed: {exc})")
                return

        self.messages.append({"role": "user", "content": user_text})
        self.transcript.append(f"## USER\n{user_text}")
        window = CONFIG["session"]["history_window"]

        agentic = (CONFIG.get("session") or {}).get("agentic_model1", True)
        try:
            if agentic:
                raw, _tool_log = reflection.chat_with_tools(
                    role="model1",
                    system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG,
                    vault_path=VAULT,
                    max_tokens=2048,
                )
            else:
                raw = reflection.chat(
                    role="model1",
                    system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG,
                    max_tokens=2048,
                )
            reply = reflection.strip_thinking(raw)
        except Exception as exc:
            log_path = log_error(VAULT, "chat", exc)
            reply = f"(model error: {exc}\n\nfull trace → {log_path})"

        if not reply.strip():
            reply = "…"
            self.call_from_thread(
                self._system_message,
                "(the model returned nothing — likely ran out of tokens mid-reasoning. "
                "Try again or ask a smaller question.)",
            )

        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append(f"## ASSISTANT\n{reply}")
        self.call_from_thread(self._finish_ai_message, reply)

    # ----- session lifecycle -----------------------------------------

    def _end_session(self) -> None:
        if not self.session:
            self._system_message("no active session.")
            return
        self._system_message("ending session — reflecting...")
        self._run_end()

    @work(thread=True)
    def _run_end(self) -> None:
        try:
            result = session_mgr.end(
                session_output="\n".join(self.transcript),
                session_meta=self.session,
                config=CONFIG, project_root=ROOT,
            )
            writes = result.get("writes") or []
            lines = [f"session ended · flags: {result['flags_found']} · "
                     f"reflection: {result['reflection_run']}"]
            if writes:
                lines.append("writes:")
                for w in writes[:10]:
                    lines.append(f"  {w.get('action','?')}  {w.get('path','')}")
            self.call_from_thread(self._system_message, "\n".join(lines))
            self.session = None
            self.messages = []
            self.transcript = []
            self.call_from_thread(self._refresh_transcripts)
        except Exception as exc:
            self.call_from_thread(self._system_message, f"end error: {exc}")

    def action_quit(self) -> None:
        """Exit — but run reflection visibly first if there's a session."""
        if self.session and self.transcript:
            self._system_message("saving session — extracting flags and reflecting...")
            self._save_then_exit()
        else:
            self.exit()

    @work(thread=True)
    def _save_then_exit(self) -> None:
        self._save_and_report()
        import time as _t
        _t.sleep(2.0)
        self.call_from_thread(self.exit)

    def _save_and_report(self) -> None:
        if not self.session or not self.transcript:
            return
        try:
            result = session_mgr.end(
                session_output="\n".join(self.transcript),
                session_meta=self.session,
                config=CONFIG, project_root=ROOT,
            )
        except Exception as exc:
            self.call_from_thread(self._system_message, f"save failed: {exc}")
            return
        flags = result.get("flags_found", 0)
        writes = result.get("writes") or []
        mode = " (recovery)" if result.get("recovery_mode") else ""
        lines = [f"session saved · flags: {flags} · writes: {len(writes)}{mode}"]
        for w in writes[:6]:
            lines.append(f"  {w.get('action', '?')}  {w.get('path', '')}")
        self.call_from_thread(self._system_message, "\n".join(lines))
        self.call_from_thread(self._refresh_transcripts)

    # ----- chat bubbles ----------------------------------------------

    def _user_message(self, content: str) -> None:
        self._chat_container.mount(ChatMessage(content, "user"))
        self._chat_container.scroll_end(animate=False)

    def _ai_message(self, content: str) -> None:
        self._chat_container.mount(ChatMessage(content, "ai"))
        self._chat_container.scroll_end(animate=False)

    def _system_message(self, content: str) -> None:
        self._chat_container.mount(ChatMessage(content, "system"))
        self._chat_container.scroll_end(animate=False)

    def _start_ai_message(self) -> None:
        self._current_ai = ChatMessage("thinking...", "ai")
        self._chat_container.mount(self._current_ai)
        self._chat_container.scroll_end(animate=False)

    def _finish_ai_message(self, content: str) -> None:
        if self._current_ai is not None:
            self._current_ai.update_content(content)
            self._current_ai = None
        self._chat_container.scroll_end(animate=False)


def main() -> None:
    MemeTUI().run()


if __name__ == "__main__":
    main()
