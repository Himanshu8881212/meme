from __future__ import annotations

import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import decay, monitor, reflection, retrieval  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, indexer  # noqa: E402

env.load_dotenv(ROOT / ".env")

CONFIG_PATH = ROOT / "config" / "config.yaml"
HISTORY_FILE = ROOT / "vault" / "_meta" / ".tui_history"

HASHTAG = re.compile(r"#([\w\-]+)")
FLAG_RENDER = re.compile(
    r"\[(NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED)(?::\s*[^\]]+)?\]",
    re.IGNORECASE,
)

BANNER = r"""
  [bold cyan]meme[/bold cyan] — a memory system with a point of view
  {identity_line}
  [dim]vault[/dim] {vault}  [dim]·[/dim]  [dim]nodes[/dim] {nodes}
  [dim]chat[/dim]   {m1_provider} / {m1}
  [dim]write[/dim]  {mr_provider} / {mr}  (routine)
  [dim]audit[/dim]  {md_provider} / {md}  (deep)
"""

HELP = r"""
[bold]How to use[/bold]
  Just type. A session starts on your first message. The assistant
  remembers across sessions because it writes to a vault of markdown
  notes on disk — you can open that folder in Obsidian.

[bold]Session[/bold]
  [cyan]/end[/cyan]                  end the session — distill flags into memory
  [cyan]/cancel[/cyan]               discard the session without writing memory
  [cyan]/start <task>[/cyan]         manually start (auto-starts otherwise; optional [yellow]#tags[/yellow])

[bold]Inspect memory[/bold]
  [cyan]/context[/cyan]              files retrieved for the current session
  [cyan]/show <name>[/cyan]          print a vault node in full
  [cyan]/transcripts[/cyan]          list recent verbatim session archives
  [cyan]/transcript <name>[/cyan]    read a full transcript (lossless recall)
  [cyan]/identity[/cyan] / [cyan]/whoami[/cyan]   show current persona + relationship

[bold]Maintenance[/bold]
  [cyan]/meta[/cyan]                 deep reflection — merge, split, reconcile, consolidate
  [cyan]/decay[/cyan]                update decay weights across the vault
  [cyan]/monitor[/cyan]              vault health metrics + triggers
  [cyan]/index[/cyan]                vault stats and sample node names

[bold]Leave[/bold]
  [cyan]exit[/cyan] / [cyan]quit[/cyan] / [cyan]bye[/cyan] / Ctrl-D        saves the session on the way out

[dim]Anything that isn't a command gets sent to the assistant.
Inline flags like [yellow]\[NOVEL: ...][/yellow] are extracted on /end and turned into nodes.[/dim]
"""

# Bare-word exit phrases — recognised outside the slash-command grammar so the
# user doesn't have to remember `/exit`. Matched after strip() and lowercasing.
EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "q", ":q", ":wq"}

# Bare command words that map to slash commands. The user doesn't have to type
# the slash — typing `end` alone triggers `/end`.
BARE_COMMANDS = {
    "end": "/end",
    "cancel": "/cancel",
    "meta": "/meta",
    "decay": "/decay",
    "monitor": "/monitor",
    "help": "/help",
    "context": "/context",
    "identity": "/identity",
    "whoami": "/whoami",
    "transcripts": "/transcripts",
    "index": "/index",
}


class TUI:
    def __init__(self) -> None:
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.vault = Path(self.config["vault_path"])
        if not self.vault.is_absolute():
            self.vault = (ROOT / self.vault).resolve()

        self.console = Console()
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.prompt = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            style=Style.from_dict({"prompt": "ansicyan bold"}),
            multiline=False,
        )

        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, str]] = []
        self.transcript: list[str] = []

    def run(self) -> None:
        idx = indexer.build(self.vault)
        models = self.config["models"]

        # Read identity to show the assistant's name + the user's name.
        identity_line = "[dim](no persona — run[/dim] [cyan]python main.py init --persona june --user-name <name>[/cyan][dim])[/dim]"
        persona_path = self.vault / "_identity" / "persona.md"
        if persona_path.exists():
            try:
                import yaml as _y
                text = persona_path.read_text(encoding="utf-8")
                if text.startswith("---"):
                    fm = _y.safe_load(text.split("---", 2)[1]) or {}
                    name = fm.get("name", "unnamed")
                    user = fm.get("user_name", "")
                    identity_line = (
                        f"[bold]{name}[/bold] [dim]talking with[/dim] "
                        f"[bold]{user}[/bold]" if user
                        else f"[bold]{name}[/bold]"
                    )
            except Exception:
                pass

        from rich.text import Text as _Text
        from rich.console import Console as _C
        banner_text = BANNER.format(
            identity_line=identity_line,
            vault=self.vault,
            nodes=len(idx),
            m1=models["model1"]["model"],
            m1_provider=models["model1"]["provider"],
            mr=models["routine"]["model"],
            mr_provider=models["routine"]["provider"],
            md=models["deep"]["model"],
            md_provider=models["deep"]["provider"],
        ).strip()
        self.console.print(Panel(banner_text, border_style="cyan", expand=False))
        self.console.print(
            "[dim]Just start typing — a session auto-starts. "
            "Type [cyan]quit[/cyan] or Ctrl-D to leave, [cyan]/help[/cyan] for commands.[/dim]"
        )

        while True:
            try:
                prompt_label = "session> " if self.session else "(no session)> "
                line = self.prompt.prompt(prompt_label)
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self._graceful_exit()
                return

            line = line.strip()
            if not line:
                continue

            low = line.lower()
            if low in EXIT_PHRASES:
                self._graceful_exit()
                return

            if low in BARE_COMMANDS:
                line = BARE_COMMANDS[low]

            if line.startswith("/"):
                self._dispatch(line)
            else:
                if not self.session:
                    self._auto_start(line)
                    if not self.session:
                        continue
                self._chat(line)

    def _dispatch(self, line: str) -> None:
        parts = shlex.split(line, posix=True)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1:]

        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            self.console.print(f"[red]unknown command: /{cmd}[/red]")
            return
        try:
            handler(args)
        except (EOFError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self.console.print(f"[red]error:[/red] {exc}")

    def cmd_help(self, _: list[str]) -> None:
        self.console.print(HELP)

    def cmd_exit(self, _: list[str]) -> None:
        raise EOFError

    cmd_quit = cmd_exit

    def cmd_start(self, args: list[str]) -> None:
        if self.session:
            self.console.print("[yellow]a session is already active. /end or /cancel first.[/yellow]")
            return
        if not args:
            self.console.print("[red]usage: /start <task> [#tag1 #tag2][/red]")
            self.console.print("[dim]example: /start debugging the auth bug #auth #backend[/dim]")
            self.console.print("[dim]or just start typing — a session auto-starts from your first message.[/dim]")
            return
        raw = " ".join(args)
        tags = HASHTAG.findall(raw)
        task = HASHTAG.sub("", raw).strip()

        with self.console.status("[dim]retrieving memory...[/dim]"):
            meta = session_mgr.start(
                task=task,
                tags=tags,
                config=self.config,
                project_root=ROOT,
            )
        self.session = meta
        self.messages = []
        self.transcript = []

        n = len(meta["retrieved_files"])
        self.console.print(
            f"[green]session started.[/green] retrieved [bold]{n}[/bold] nodes "
            f"for task: [italic]{task}[/italic]"
            + (f"  tags: {', '.join(tags)}" if tags else "")
        )
        if n:
            self.console.print("[dim]use /context to see the retrieved files[/dim]")

    def cmd_cancel(self, _: list[str]) -> None:
        if not self.session:
            self.console.print("[yellow]no active session.[/yellow]")
            return
        self.session = None
        self.messages = []
        self.transcript = []
        self.console.print("[dim]session discarded.[/dim]")

    def cmd_end(self, args: list[str]) -> None:
        if not self.session:
            self.console.print("[yellow]no active session.[/yellow]")
            return

        output_text = "\n".join(self.transcript)
        with self.console.status("[dim]extracting flags, running reflection...[/dim]"):
            result = session_mgr.end(
                session_output=output_text,
                session_meta=self.session,
                config=self.config,
                project_root=ROOT,
            )

        recovery_note = " [yellow](recovery mode — no flags, reflected from transcript)[/yellow]" if result.get("recovery_mode") else ""
        self.console.print(
            f"[green]session ended.[/green] flags: [bold]{result['flags_found']}[/bold]  "
            f"reflection: [bold]{result['reflection_run']}[/bold]" + recovery_note
        )
        writes = result.get("writes") or []
        if writes:
            table = Table(title="Memory writes", show_lines=False)
            table.add_column("action", style="cyan")
            table.add_column("path")
            table.add_column("warning", style="yellow")
            for w in writes:
                table.add_row(w.get("action", "?"), w.get("path", ""), w.get("warning", "") or w.get("reason", ""))
            self.console.print(table)
        if "-v" in args and result.get("reflection_output"):
            self.console.print(Panel(result["reflection_output"], title="reflection output", border_style="dim"))

        self.session = None
        self.messages = []
        self.transcript = []

    def cmd_decay(self, _: list[str]) -> None:
        with self.console.status("[dim]running decay...[/dim]"):
            result = decay.run(
                vault_path=self.vault,
                lambda_=self.config["decay"]["lambda"],
                archive_threshold=self.config["decay"]["archive_threshold"],
            )
        self.console.print_json(data=result)

    def cmd_monitor(self, _: list[str]) -> None:
        metrics = monitor.collect(self.vault)
        triggers = monitor.check_thresholds(metrics, self.config)

        t = Table(title="Vault metrics", show_header=False, box=None)
        t.add_column(style="cyan")
        t.add_column()
        for key in ("total_nodes", "archived_ratio", "orphan_ratio", "tag_vocabulary", "avg_decay_weight"):
            v = metrics[key]
            t.add_row(key, f"{v:.3f}" if isinstance(v, float) else str(v))
        self.console.print(t)

        if metrics["top_hubs"]:
            hubs = Table(title="Top hubs (backlink count)", show_header=False, box=None)
            hubs.add_column(style="magenta")
            hubs.add_column(justify="right")
            for name, count in metrics["top_hubs"][:5]:
                hubs.add_row(name, str(count))
            self.console.print(hubs)

        if triggers:
            self.console.print(Panel("\n".join(f"- {t}" for t in triggers),
                                     title="monitor triggers", border_style="yellow"))
        else:
            self.console.print("[green]no monitor triggers.[/green]")

    def cmd_meta(self, args: list[str]) -> None:
        with self.console.status("[dim]phase 1: algorithmic cleanup + reconciliation...[/dim]"):
            min_body = int((self.config.get("monitor") or {}).get("min_body_chars", 20))
            removed = monitor.cleanup_broken(self.vault, min_body_chars=min_body)
            reconciled = reflection.reconcile_tensions(self.vault)
            orphans = monitor.find_orphans(self.vault)

        if removed:
            t = Table(title=f"Removed {len(removed)} broken nodes", show_lines=False)
            t.add_column("path", style="red")
            t.add_column("reason", style="dim")
            for item in removed:
                t.add_row(item["path"], item["reason"])
            self.console.print(t)
        else:
            self.console.print("[dim]phase 1: no broken nodes to remove.[/dim]")

        if reconciled:
            self.console.print(f"[dim]phase 1: reconciled {len(reconciled)} tension(s) into entity bodies.[/dim]")
        self.console.print(f"[dim]phase 1: {len(orphans)} orphan node(s) surfaced for model review.[/dim]")

        with self.console.status("[dim]phase 2: deep reflection with tools (magistral-medium)...[/dim]"):
            metrics = monitor.collect(self.vault)
            triggers = monitor.check_thresholds(metrics, self.config)
            if orphans:
                triggers.append(f"orphan_review: {', '.join(orphans[:10])}")
            sample_query = " ".join(n for n, _ in metrics["top_hubs"][:5])
            files = retrieval.retrieve(self.vault, sample_query, [], self.config)
            output, call_log = reflection.deep_with_tools(
                self.vault, files, metrics, triggers, self.config,
            )
            writes = reflection.apply_writes(
                output, self.vault,
                similarity_threshold=float(
                    (self.config.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
                ),
            )

        self.console.print(f"[green]deep reflection done.[/green] "
                           f"cleaned: [bold]{len(removed)}[/bold]  "
                           f"orphans flagged: [bold]{len(orphans)}[/bold]  "
                           f"tool calls: [bold]{len(call_log)}[/bold]  "
                           f"model writes: [bold]{len(writes)}[/bold]")
        if call_log:
            tc = Table(title="Tool calls", show_lines=False)
            tc.add_column("tool", style="magenta")
            tc.add_column("args", style="dim")
            for c in call_log[:20]:
                tc.add_row(c["tool"], str(c.get("args", ""))[:60])
            self.console.print(tc)
        if writes:
            table = Table(title="Meta writes", show_lines=False)
            table.add_column("action", style="cyan")
            table.add_column("path")
            for w in writes:
                table.add_row(w.get("action", "?"), w.get("path", ""))
            self.console.print(table)
        if "-v" in args:
            self.console.print(Panel(output, title="reflection output", border_style="dim"))

    def cmd_index(self, _: list[str]) -> None:
        idx = indexer.build(self.vault)
        sample = list(idx)[:15]
        self.console.print(f"[bold]{len(idx)}[/bold] nodes")
        if sample:
            self.console.print("[dim]" + ", ".join(sample) + ("..." if len(idx) > 15 else "") + "[/dim]")

    def cmd_context(self, _: list[str]) -> None:
        if not self.session:
            self.console.print("[yellow]no active session.[/yellow]")
            return
        files = self.session.get("retrieved_files", [])
        if not files:
            self.console.print("[dim](empty retrieval — bootstrap mode)[/dim]")
            return
        for p in files:
            self.console.print(f"  [cyan]{Path(p).relative_to(self.vault)}[/cyan]")

    def cmd_identity(self, _: list[str]) -> None:
        path = self.vault / "_identity" / "self.md"
        if not path.exists():
            self.console.print("[yellow]no identity set. run `python main.py init --persona june`[/yellow]")
            return
        self.console.print(Panel(Markdown(path.read_text(encoding="utf-8")),
                                 title="my identity", border_style="magenta"))

    cmd_whoami = cmd_identity

    def cmd_transcripts(self, _: list[str]) -> None:
        tdir = self.vault / "_transcripts"
        if not tdir.exists():
            self.console.print("[dim](no transcripts yet)[/dim]")
            return
        files = sorted(tdir.glob("*.md"), reverse=True)[:20]
        if not files:
            self.console.print("[dim](no transcripts yet)[/dim]")
            return
        for p in files:
            self.console.print(f"  [cyan]{p.stem}[/cyan]")
        self.console.print("[dim]use /transcript <name> to read one in full[/dim]")

    def cmd_transcript(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]usage: /transcript <name>[/red]  "
                               "[dim](see /transcripts for a list)[/dim]")
            return
        name = " ".join(args)
        tdir = self.vault / "_transcripts"
        path = tdir / f"{name}.md"
        if not path.exists():
            matches = list(tdir.glob(f"*{name}*.md"))
            if not matches:
                self.console.print(f"[red]no transcript matching '{name}'[/red]")
                return
            path = matches[0]
        self.console.print(Panel(Markdown(path.read_text(encoding="utf-8")),
                                 title=path.stem, border_style="cyan"))

    def cmd_show(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]usage: /show <node name>[/red]")
            return
        name = " ".join(args)
        idx = indexer.build(self.vault)
        if name not in idx:
            matches = [n for n in idx if name.lower() in n.lower()]
            if not matches:
                self.console.print(f"[red]no node matching '{name}'[/red]")
                return
            name = matches[0]
        text = Path(idx[name]["path"]).read_text(encoding="utf-8")
        self.console.print(Panel(Markdown(text), title=name, border_style="cyan"))

    def _graceful_exit(self) -> None:
        if self.session:
            self.console.print("[dim]saving session before exit...[/dim]")
            try:
                self.cmd_end([])
            except Exception as exc:
                self.console.print(f"[red]save failed:[/red] {exc}  [dim](session discarded)[/dim]")
        self.console.print("[dim]bye[/dim]")

    def _auto_start(self, first_message: str) -> None:
        tags = HASHTAG.findall(first_message)
        task = HASHTAG.sub("", first_message).strip() or "general chat"
        with self.console.status("[dim]starting session, retrieving memory...[/dim]"):
            self.session = session_mgr.start(
                task=task,
                tags=tags,
                config=self.config,
                project_root=ROOT,
            )
        self.messages = []
        self.transcript = []
        n = len(self.session["retrieved_files"])
        self.console.print(
            f"[dim green]→ session auto-started[/dim green]  "
            f"[dim](task: {task}{', tags: ' + ', '.join(tags) if tags else ''}, {n} nodes retrieved)[/dim]"
        )
        self.console.print("[dim]use /end when done to save what was learned, /help for commands[/dim]")

    def _chat(self, user_input: str) -> None:
        assert self.session is not None
        self.messages.append({"role": "user", "content": user_input})
        self.transcript.append(f"## USER\n{user_input}")
        window = self.config["session"]["history_window"]

        agentic = (self.config.get("session") or {}).get("agentic_model1", False)
        try:
            with self.console.status("[dim]Model 1 is thinking...[/dim]"):
                if agentic:
                    raw, tool_log = reflection.chat_with_tools(
                        role="model1",
                        system=self.session["system_prompt"],
                        messages=self.messages[-window:],
                        config=self.config,
                        vault_path=self.vault,
                        max_tokens=4096,
                    )
                    if tool_log:
                        tool_summary = "  ".join(
                            f"{t['tool']}({list(t['args'].values())[0][:30] if t['args'] else ''})"
                            for t in tool_log[:5]
                        )
                        self.console.print(f"[dim]↳ tools: {tool_summary}[/dim]")
                else:
                    raw = reflection.chat(
                        role="model1",
                        system=self.session["system_prompt"],
                        messages=self.messages[-window:],
                        config=self.config,
                        max_tokens=4096,
                    )
        except Exception as exc:
            self.console.print(f"[red]model error:[/red] {exc}")
            self.messages.pop()
            return

        cleaned = reflection.strip_thinking(raw)
        self.messages.append({"role": "assistant", "content": cleaned})
        self.transcript.append(f"## ASSISTANT\n{cleaned}")
        self.console.print(Panel(self._render_with_flags(cleaned),
                                 border_style="green",
                                 title=f"assistant — {datetime.now().strftime('%H:%M:%S')}"))

    @staticmethod
    def _render_with_flags(text: str) -> Text:
        rt = Text()
        last = 0
        for m in FLAG_RENDER.finditer(text):
            rt.append(text[last:m.start()])
            rt.append(m.group(0), style="bold yellow")
            last = m.end()
        rt.append(text[last:])
        return rt


def main() -> None:
    TUI().run()


if __name__ == "__main__":
    main()
