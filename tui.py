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

from core import decay, dedup, monitor, obsidian, outreach, proactive, reflection, retrieval  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, frontmatter, indexer  # noqa: E402
from tui_common import (  # noqa: E402
    BASE_CSS, ChatMessage, StatusBar, TranscriptItem,
    read_identity, transcript_entries, parse_transcript, strip_meme_flags,
    log_error, copy_to_clipboard,
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
    "index": "/index", "obsidian": "/obsidian",
    "outreach": "/outreach", "whats_up": "/whats_up",
    "pause": "/pause", "mute": "/mute", "unmute": "/unmute",
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

External notebook
  /obsidian         path, note count, last audit entry
  /obsidian recent  last 10 audit entries
  /obsidian diff    git diff HEAD~5 on the external vault

Proactive
  /whats_up         top 3 candidates w/ scores — read-only, no log
  /outreach         last 10 log entries
  /outreach status  enabled, paused-until, today's count
  /pause 24h        mute surfacing for 24h (also 30m, 2d, until ...)
  /pause off        clear the pause
  /mute <node>      set proactive: false on a node
  /unmute <node>    set proactive: true on a node

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
        Binding("ctrl+y", "copy_reply", "copy last reply", show=True),
        Binding("up", "history_prev", "prev input", show=False),
        Binding("down", "history_next", "next input", show=False),
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
        self._input_history: list[str] = []
        self._history_idx: int = 0
        self._last_reply: str = ""
        # True while a chat worker is running — blocks new submissions so
        # self.messages / self.transcript don't race.
        self._chat_busy: bool = False
        # Proactive: if the last startup surfaced an outreach, we tag the
        # next session start so retrieval + reflection know the context.
        self._proactive_prefix: str | None = None
        self._proactive_node_path: str | None = None
        self._pending_proactive: dict[str, Any] | None = None

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
                    bar.hotkey_hint = (
                        "Type send   ↑↓ history   Ctrl-Y copy reply   "
                        "Ctrl-N new   /help   Esc quit (auto-save)"
                    )
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

        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is not None:
            head = obsidian.git_head(ext) or "no git"
            self._system_message(
                f"external vault: {ext}  (notes: {obsidian.note_count(ext)}, git: {head})"
            )
        else:
            self._system_message("external vault: (disabled — set external_vault.path in config.yaml)")

        self._maybe_surface_proactive()

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
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
        self._history_idx = len(self._input_history)

        low = text.lower()

        if low in EXIT_PHRASES:
            self.action_quit()
            return

        if low in BARE_COMMANDS:
            text = BARE_COMMANDS[low]

        if text.startswith("/"):
            self._dispatch(text)
            return

        if self._chat_busy:
            self._system_message("(still thinking — hold on a moment.)")
            return
        self._send(text)

    # ----- history + clipboard ----------------------------------------

    def action_history_prev(self) -> None:
        inp = self.query_one("#user-input", Input)
        if self.focused is not inp or not self._input_history:
            return
        if self._history_idx > 0:
            self._history_idx -= 1
        inp.value = self._input_history[self._history_idx]
        inp.cursor_position = len(inp.value)

    def action_history_next(self) -> None:
        inp = self.query_one("#user-input", Input)
        if self.focused is not inp or not self._input_history:
            return
        if self._history_idx < len(self._input_history) - 1:
            self._history_idx += 1
            inp.value = self._input_history[self._history_idx]
        else:
            self._history_idx = len(self._input_history)
            inp.value = ""
        inp.cursor_position = len(inp.value)

    def action_copy_reply(self) -> None:
        if not self._last_reply:
            self._system_message("nothing to copy yet.")
            return
        if copy_to_clipboard(self._last_reply):
            self._system_message(f"copied last reply ({len(self._last_reply)} chars).")
        else:
            self._system_message("clipboard unavailable on this platform.")

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
        dups = dedup.find_duplicate_candidates(VAULT)
        lines = [
            f"nodes: {metrics['total_nodes']}",
            f"archived: {metrics['archived']} ({metrics['archived_ratio']:.1%})",
            f"orphans: {metrics['orphans']} ({metrics['orphan_ratio']:.1%})",
            f"tags: {metrics['tag_vocabulary']}",
            f"avg decay weight: {metrics['avg_decay_weight']:.3f}",
            f"duplicate candidates: {len(dups)}",
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

    def cmd_obsidian(self, arg: str) -> None:
        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is None:
            self._system_message("external vault is disabled (external_vault.path: null).")
            return
        sub = (arg or "").strip().lower()
        if sub == "recent":
            entries = obsidian.read_audit_tail(ext, n=10)
            if not entries:
                self._system_message("(no audit entries yet)")
                return
            lines = ["last audit entries:"]
            for e in entries:
                lines.append(f"  {e.get('ts','?')}  {e.get('tool','?')}  {e.get('result','')[:120]}")
            self._system_message("\n".join(lines))
            return
        if sub == "diff":
            import subprocess
            try:
                out = subprocess.run(
                    ["git", "diff", "HEAD~5"],
                    cwd=str(ext), capture_output=True, text=True, timeout=10,
                )
                text = (out.stdout or out.stderr or "(no diff)").strip()
            except Exception as exc:
                text = f"(git diff failed: {exc})"
            self._system_message(text[:4000] or "(no diff)")
            return
        # default summary
        tail = obsidian.read_audit_tail(ext, n=1)
        last = "-"
        if tail:
            e = tail[0]
            last = f"{e.get('ts','?')} {e.get('tool','?')} — {e.get('result','')[:80]}"
        head = obsidian.git_head(ext) or "(no git)"
        self._system_message(
            f"external vault: {ext}\n"
            f"notes: {obsidian.note_count(ext)}\n"
            f"head: {head}\n"
            f"last audit: {last}"
        )

    # ----- proactive outreach ----------------------------------------

    def _maybe_surface_proactive(self) -> None:
        pcfg = (CONFIG.get("proactive") or {})
        if not pcfg.get("enabled"):
            return
        pause = outreach.active_pause(VAULT)
        if pause is not None:
            return
        try:
            ctx = outreach.build_context(VAULT, CONFIG)
            cs = proactive.candidates(VAULT, CONFIG)
            pick = proactive.should_reach_out(cs, ctx, CONFIG)
        except Exception as exc:
            self._system_message(f"(proactive: error building candidates: {exc})")
            return
        if pick is None:
            return
        self._pending_proactive = pick
        self._system_message(f"📬 I've been thinking about {pick['node_name']}...")
        self._run_proactive_draft(pick)

    @work(thread=True)
    def _run_proactive_draft(self, pick: dict[str, Any]) -> None:
        try:
            msg = outreach.draft_message(pick, VAULT, CONFIG)
        except Exception as exc:
            self.call_from_thread(self._system_message, f"(outreach draft failed: {exc})")
            return
        try:
            outreach.log_outreach(VAULT, pick, msg, delivered=True)
        except Exception:
            pass
        self.call_from_thread(self._ai_message, msg)
        # Prime the next session so reflection knows the opener was proactive.
        self._proactive_prefix = pick["node_name"]
        self._proactive_node_path = pick.get("node_path")

    def cmd_whats_up(self, _: str) -> None:
        try:
            cs = proactive.candidates(VAULT, CONFIG)
        except Exception as exc:
            self._system_message(f"whats_up error: {exc}")
            return
        if not cs:
            self._system_message("(nothing on my mind — no candidates)")
            return
        lines = ["top candidates (read-only):"]
        for c in cs[:3]:
            reasons = "; ".join(c.get("reasons") or []) or "-"
            lines.append(
                f"  {c['score']:.2f}  [{c.get('node_type','?')}] {c['node_name']}"
                f"  — {reasons}"
            )
        self._system_message("\n".join(lines))

    def cmd_outreach(self, arg: str) -> None:
        sub = (arg or "").strip().lower()
        if sub == "status":
            pcfg = (CONFIG.get("proactive") or {})
            enabled = bool(pcfg.get("enabled"))
            pause = outreach.active_pause(VAULT)
            ctx = outreach.build_context(VAULT, CONFIG)
            lines = [
                f"enabled: {enabled}",
                f"paused until: {pause.isoformat(timespec='seconds') if pause else '-'}",
                f"delivered today: {ctx['outreaches_today']}  (cap={pcfg.get('daily_cap', 3)})",
            ]
            hs = ctx.get("hours_since_last_outreach")
            if hs is not None:
                lines.append(f"hours since last outreach: {hs:.1f}")
            self._system_message("\n".join(lines))
            return
        entries = outreach.tail_log(VAULT, n=10)
        if not entries:
            self._system_message("(outreach log is empty)")
            return
        lines = ["last outreaches:"]
        for e in entries:
            lines.append(
                f"  {e['ts'].isoformat(timespec='seconds')}  "
                f"score={e['score']:.2f}  delivered={e['delivered']}  {e['node']}"
            )
        self._system_message("\n".join(lines))

    def cmd_pause(self, arg: str) -> None:
        arg = (arg or "").strip()
        if arg.lower() == "off":
            if outreach.clear_pause(VAULT):
                self._system_message("proactive: unpaused.")
            else:
                self._system_message("no pause was active.")
            return
        until = outreach.parse_pause_arg(arg)
        if until is None:
            self._system_message(
                "usage: /pause 24h   |   /pause until 2026-05-01   |   /pause off"
            )
            return
        outreach.set_pause(VAULT, until)
        self._system_message(
            f"proactive: paused until {until.isoformat(timespec='seconds')}."
        )

    def cmd_mute(self, arg: str) -> None:
        name = (arg or "").strip()
        if not name:
            self._system_message("usage: /mute <node name>")
            return
        if outreach.set_node_proactive(VAULT, name, False):
            self._system_message(f"muted: {name}")
        else:
            self._system_message(f"no node matching '{name}'")

    def cmd_unmute(self, arg: str) -> None:
        name = (arg or "").strip()
        if not name:
            self._system_message("usage: /unmute <node name>")
            return
        if outreach.set_node_proactive(VAULT, name, True):
            self._system_message(f"unmuted: {name}")
        else:
            self._system_message(f"no node matching '{name}'")

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
            # Surface near-duplicate node pairs so the auditor can merge them.
            # This is how we keep the graph from rotting into parallel
            # synonymy nodes (the failure mode decay does NOT catch).
            dup_candidates = dedup.find_duplicate_candidates(VAULT)
            if dup_candidates:
                for c in dup_candidates[:10]:
                    triggers.append(
                        f"duplicate_merge_candidate: [{c['type']}] "
                        f"{c['a']} ↔ {c['b']} ({c['reason']})"
                    )
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
        self._chat_busy = True
        self._user_message(user_text)
        self._start_ai_message()
        self._run_chat(user_text)

    @work(thread=True)
    def _run_chat(self, user_text: str) -> None:
        if self.session is None:
            task = user_text
            if self._proactive_prefix:
                task = f"reply to proactive: {self._proactive_prefix} — {user_text}"
            try:
                self.session = session_mgr.start(
                    task=task, tags=[], config=CONFIG, project_root=ROOT,
                )
                # Inject the referenced proactive node into retrieved context
                # for the first turn so the model has grounding without a new
                # retrieval call.
                if self._proactive_node_path:
                    existing = list(self.session.get("retrieved_files") or [])
                    if self._proactive_node_path not in existing:
                        existing.insert(0, self._proactive_node_path)
                    self.session["retrieved_files"] = existing
                self._proactive_prefix = None
                self._proactive_node_path = None
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
        tool_log: list[dict[str, Any]] = []
        try:
            if agentic:
                raw, tool_log = reflection.chat_with_tools(
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
        for call in tool_log:
            tname = call.get("tool", "")
            if not str(tname).startswith("obsidian_"):
                continue
            args = call.get("args") or {}
            rel = args.get("rel_path") or args.get("folder") or args.get("query") or ""
            label = tname.replace("obsidian_", "")
            self.call_from_thread(self._system_message, f"📝 {label}: {rel}")
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
        self._last_reply = content
        self._chat_busy = False
        self._chat_container.scroll_end(animate=False)


def main() -> None:
    MemeTUI().run()


if __name__ == "__main__":
    main()
