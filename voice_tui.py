#!/usr/bin/env python3
"""meme — voice-first TUI.

Hold OPTION to talk. Release to send. Or type text and press Enter.
Esc to quit (auto-saves the session into the vault).

The chat model is meme's Model 1 — whatever you set in config.yaml.
Switch between Mistral (cloud) and LM Studio (local) by flipping one line:

    models.model1.provider: mistral     # cloud (needs MISTRAL_API_KEY)
    models.model1.provider: lmstudio    # local (LM Studio running on :1234)

STT (parakeet-mlx) and TTS (mlx-audio ChatterBox) are always local.
"""
from __future__ import annotations

import os
import sys
import threading
import queue
import warnings
from datetime import datetime  # noqa: F401
from pathlib import Path
from typing import Any

import yaml

# --- Voice stack ------------------------------------------------------------
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
try:
    import sounddevice as sd
    import numpy as np
    import mlx.core as mx
    import parakeet_mlx
    from mlx_audio.tts.utils import load_model as load_tts
    from mlx_audio.tts.generate import load_audio
    from mlx_audio.tts.audio_player import AudioPlayer
    mx.set_default_device(mx.gpu)
    HAS_VOICE = True
except Exception:  # pragma: no cover — depends on host hardware
    HAS_VOICE = False
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False
sys.stderr = _stderr

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, ListItem, ListView, Static
from textual import work
from rich.text import Text

# --- meme core --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import reflection  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, frontmatter  # noqa: E402

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VAULT = Path(CONFIG["vault_path"])
if not VAULT.is_absolute():
    VAULT = (ROOT / VAULT).resolve()

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
REF_AUDIO_DEFAULT = str(ROOT / "reference.wav")

BANNER = r"""
 ███╗   ███╗███████╗███╗   ███╗███████╗
 ████╗ ████║██╔════╝████╗ ████║██╔════╝
 ██╔████╔██║█████╗  ██╔████╔██║█████╗
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝
"""

CSS = """
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
#sidebar-controls { height: auto; padding: 1; }
#new-chat-btn { width: 100%; height: 3; background: #238636; color: white; margin-bottom: 1; }
#mute-btn    { width: 100%; height: 3; background: #30363d; color: white; }
#mute-btn.--muted { background: #d29922; }
#transcript-list { height: 1fr; background: #161b22; scrollbar-size: 1 1; }
TranscriptItem {
    height: 3; background: #161b22; border-bottom: solid #1c2128;
    layout: horizontal; padding: 0 1;
}
TranscriptItem:hover { background: #1c2128; }
TranscriptItem.--highlight { border-left: solid #c9a0ff; background: #1c2128; }
.item-info  { width: 1fr; layout: vertical; }
.item-title { color: #c9d1d9; text-style: bold; height: 1; }
.item-date  { color: #8b949e; text-style: dim;  height: 1; }
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
#status-bar { height: 1; background: transparent; color: #8b949e; text-align: center; }
.user-message {
    width: 85%; color: #f0f6fc; background: #0d47a1;
    padding: 1 2; border-left: solid #58a6ff; margin-bottom: 1;
}
.ai-message {
    width: 85%; color: #c9d1d9; background: #1c2128;
    padding: 1 2; border-left: solid #c9a0ff; margin-bottom: 1;
}
.system-message { color: #8b949e; text-style: dim; margin-bottom: 1; padding: 0 2; }
"""


# ---------------------------------------------------------------------------
# Silence trimmer
# ---------------------------------------------------------------------------
def trim_silence(audio, threshold: float = 0.01):
    if len(audio) < SAMPLE_RATE * 0.3:
        return audio
    chunk = int(SAMPLE_RATE * 0.03)
    start = 0
    for i in range(0, len(audio) - chunk, chunk):
        if np.abs(audio[i:i + chunk]).mean() > threshold:
            start = max(0, i - chunk)
            break
    end = len(audio)
    for i in range(len(audio) - chunk, chunk, -chunk):
        if np.abs(audio[i:i + chunk]).mean() > threshold:
            end = min(len(audio), i + chunk * 2)
            break
    return audio[start:end]


# ---------------------------------------------------------------------------
# Voice backend
# ---------------------------------------------------------------------------
class VoiceBackend:
    def __init__(self, stt_model: str, tts_model: str, ref_audio: str):
        self.stt_model_id = stt_model
        self.tts_model_id = tts_model
        self.ref_audio_path = ref_audio
        self.stt = None
        self.tts = None
        self.player = None
        self.ref_audio = None
        self.tts_queue: "queue.Queue[str | None]" = queue.Queue()
        self.muted = False
        self.is_recording = False
        self.audio_chunks: list = []
        self.lock = threading.Lock()
        self.audio_stream = None

    def load(self, on_status=None) -> None:
        if not HAS_VOICE:
            return
        if on_status: on_status("loading STT...")
        self.stt = parakeet_mlx.from_pretrained(self.stt_model_id)
        dummy = mx.zeros((SAMPLE_RATE,), dtype=mx.float32)
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            t.add_audio(dummy)
        mx.eval(mx.array([0]))

        if on_status: on_status("loading TTS...")
        self.tts = load_tts(self.tts_model_id)
        self.player = AudioPlayer(sample_rate=self.tts.sample_rate)
        if os.path.exists(self.ref_audio_path):
            self.ref_audio = load_audio(self.ref_audio_path, self.tts.sample_rate)
            mx.eval(self.ref_audio)

        for res in self.tts.generate(text="Hi", ref_audio=self.ref_audio, verbose=False, stream=True):
            if hasattr(res, "audio") and res.audio is not None:
                mx.eval(res.audio); break

        threading.Thread(target=self._tts_worker, daemon=True).start()

        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            callback=self._audio_callback, blocksize=CHUNK_SIZE,
        )
        self.audio_stream.start()
        if on_status: on_status("ready")

    def shutdown(self) -> None:
        if self.audio_stream:
            try: self.audio_stream.stop()
            except Exception: pass
        self.tts_queue.put(None)

    def _audio_callback(self, indata, frames, time_info, status):
        with self.lock:
            if self.is_recording:
                self.audio_chunks.append(indata.copy())

    def start_recording(self) -> bool:
        if not HAS_VOICE or self.is_recording:
            return False
        with self.lock:
            self.is_recording = True
            self.audio_chunks = []
        return True

    def stop_recording(self) -> str | None:
        if not self.is_recording:
            return None
        with self.lock:
            self.is_recording = False
            if not self.audio_chunks:
                return None
            audio = np.concatenate(self.audio_chunks).flatten().astype(np.float32)
            self.audio_chunks = []
        audio = trim_silence(audio)
        if len(audio) < SAMPLE_RATE * 0.3:
            return None
        return self._transcribe(audio)

    def _transcribe(self, audio) -> str:
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            for i in range(0, len(audio), SAMPLE_RATE):
                chunk = audio[i:i + SAMPLE_RATE]
                if len(chunk) > 0:
                    t.add_audio(mx.array(chunk, dtype=mx.float32))
            if t.result:
                mx.eval(mx.array([0]))
                return t.result.text.strip()
        return ""

    def speak(self, text: str) -> None:
        if not HAS_VOICE or self.muted or not text or not text.strip():
            return
        self.tts_queue.put(text)

    def _tts_worker(self) -> None:
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                for res in self.tts.generate(
                    text=text, ref_audio=self.ref_audio, verbose=False, stream=True,
                ):
                    if hasattr(res, "audio") and res.audio is not None:
                        mx.eval(res.audio)
                        self.player.queue_audio(res.audio)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Textual widgets
# ---------------------------------------------------------------------------
class ChatMessage(Static):
    def __init__(self, content: str, kind: str = "user"):
        super().__init__(content)
        self.kind = kind
        self.add_class(f"{kind}-message")

    def update_content(self, content: str) -> None:
        self.update(content)


class TranscriptItem(ListItem):
    def __init__(self, name: str, stamp: str):
        super().__init__()
        self.transcript_name = name
        self.stamp = stamp

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="item-info"):
                parts = self.transcript_name.split("-", 5)
                pretty = parts[-1].replace("-", " ") if len(parts) == 6 else self.transcript_name
                yield Label(pretty[:30], classes="item-title")
                yield Label(self.stamp, classes="item-date")


class StatusBar(Static):
    text = reactive("")
    listening = reactive(False)

    def render(self) -> Text:
        t = Text(justify="center")
        if HAS_PYNPUT:
            t.append("OPTION", style="bold #58a6ff"); t.append(" talk   ", style="dim")
        t.append("Type", style="bold #58a6ff");    t.append(" send   ", style="dim")
        t.append("Esc", style="bold #58a6ff");     t.append(" quit (auto-save)", style="dim")
        if self.text:
            t.append("   │   ", style="dim")
            style = "bold #3fb950" if self.listening else "bold #8b949e"
            t.append(self.text, style=style)
        return t


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
class MemeVoiceTUI(App):
    CSS = CSS
    BINDINGS = [
        Binding("escape", "quit", "quit", show=True),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("ctrl+n", "new_session", "new session", show=True),
        Binding("ctrl+m", "toggle_mute", "mute", show=True),
    ]

    def __init__(self, voice: VoiceBackend | None = None) -> None:
        super().__init__()
        self.title = "meme — voice TUI"
        self.voice = voice
        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, str]] = []
        self.transcript: list[str] = []
        self._pynput_listener = None
        self._chat_container: ScrollableContainer | None = None
        self._transcript_list: ListView | None = None
        self._current_ai: ChatMessage | None = None

    def compose(self) -> ComposeResult:
        yield Container(Label(BANNER.strip(), id="banner-label"), id="app-header")
        with Horizontal(id="main-layout"):
            with Vertical(id="sidebar"):
                with Container(id="sidebar-controls"):
                    yield Button("New session (Ctrl-N)", id="new-chat-btn", variant="success")
                    yield Button("Mute audio (Ctrl-M)", id="mute-btn")
                yield ListView(id="transcript-list")
            with Vertical(id="chat-area"):
                yield ScrollableContainer(id="chat-container")
                with Vertical(id="input-area"):
                    with Horizontal(id="input-row"):
                        yield Input(placeholder="Type and press Enter — or hold OPTION to talk",
                                    id="user-input")
                    yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        self._chat_container = self.query_one("#chat-container", ScrollableContainer)
        self._transcript_list = self.query_one("#transcript-list", ListView)
        self.query_one("#user-input", Input).focus()
        self._refresh_transcripts()
        self._start_pynput()

        persona = _read_identity_name()
        if persona:
            name, user = persona
            greet = f"{name} is here. {'Hi ' + user + '.' if user else 'Hi.'}"
        else:
            greet = "no persona set — run: python main.py init --persona june --user-name <name>"
        self._system_message(greet)

        # Show which chat model is active — helps when switching local / API.
        m1 = (CONFIG.get("models") or {}).get("model1") or {}
        provider = m1.get("provider", "?")
        model = m1.get("model", "?")
        self._system_message(f"chat model: {provider} / {model}  ·  STT+TTS: local (Metal)")

        if not HAS_VOICE:
            self._system_message("voice stack not installed — text-only mode. "
                                 "install: pip install -r requirements-voice.txt")
        if not HAS_PYNPUT:
            self._system_message("pynput not installed — push-to-talk disabled.")

    def _refresh_transcripts(self) -> None:
        self._transcript_list.clear()
        tdir = VAULT / "_transcripts"
        if not tdir.exists():
            return
        files = sorted(tdir.glob("*.md"), reverse=True)[:30]
        for p in files:
            try:
                stamp = datetime.strptime(p.stem[:19], "%Y-%m-%d-%H%M%S").strftime("%b %d  %H:%M")
            except Exception:
                stamp = p.stem[:10]
            self._transcript_list.append(TranscriptItem(p.stem, stamp))

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
        for chunk in _parse_transcript(body):
            if chunk["role"] == "user":
                self._user_message(chunk["content"])
            else:
                self._ai_message(chunk["content"])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = str(event.button.id)
        if bid == "new-chat-btn":
            self.action_new_session()
        elif bid == "mute-btn":
            self.action_toggle_mute()

    def action_new_session(self) -> None:
        self._save_current_session()
        self._chat_container.remove_children()
        self.session = None
        self.messages = []
        self.transcript = []
        self._refresh_transcripts()
        self._system_message("new session — just type or hold OPTION to talk.")
        self.query_one("#user-input", Input).focus()

    def action_toggle_mute(self) -> None:
        if not self.voice:
            return
        self.voice.muted = not self.voice.muted
        btn = self.query_one("#mute-btn", Button)
        if self.voice.muted:
            btn.label = "Unmute audio (Ctrl-M)"
            btn.add_class("--muted")
            self._system_message("audio muted — text only.")
        else:
            btn.label = "Mute audio (Ctrl-M)"
            btn.remove_class("--muted")
            self._system_message("audio unmuted.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in ("quit", "exit", "bye", ":q"):
            self.action_quit()
            return
        self._send(text)

    def _start_pynput(self) -> None:
        if not HAS_PYNPUT:
            return

        def on_press(key):
            if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                if self.voice and self.voice.start_recording():
                    self.call_from_thread(self._on_listen_start)

        def on_release(key):
            if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                if self.voice:
                    self.call_from_thread(self._on_listen_stop)

        self._pynput_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._pynput_listener.start()

    def _on_listen_start(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.listening = True
        bar.text = "🎤 recording..."

    def _on_listen_stop(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.listening = False
        bar.text = "transcribing..."
        self._transcribe_and_send()

    @work(thread=True)
    def _transcribe_and_send(self) -> None:
        if not self.voice:
            return
        text = self.voice.stop_recording()
        if not text:
            self.call_from_thread(self._system_message, "(no speech)")
            self.call_from_thread(self._set_status, "")
            return
        self.call_from_thread(self._set_status, "")
        self.call_from_thread(self._send, text)

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

        try:
            reply = reflection.chat(
                role="model1",
                system=self.session["system_prompt"],
                messages=self.messages[-window:],
                config=CONFIG,
                max_tokens=1024,
            )
            reply = reflection.strip_thinking(reply)
        except Exception as exc:
            reply = f"(model error: {exc})"

        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append(f"## ASSISTANT\n{reply}")

        self.call_from_thread(self._finish_ai_message, reply)
        if self.voice:
            self.voice.speak(_clean_for_tts(reply))

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

    def _set_status(self, text: str) -> None:
        self.query_one("#status-bar", StatusBar).text = text

    def _save_current_session(self) -> None:
        if not self.session or not self.transcript:
            return
        try:
            session_mgr.end(
                session_output="\n".join(self.transcript),
                session_meta=self.session,
                config=CONFIG,
                project_root=ROOT,
            )
        except Exception:
            pass

    def action_quit(self) -> None:
        self._save_current_session()
        if self._pynput_listener:
            self._pynput_listener.stop()
        if self.voice:
            self.voice.shutdown()
        self.exit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_identity_name() -> tuple[str, str] | None:
    path = VAULT / "_identity" / "persona.md"
    if not path.exists():
        return None
    try:
        fm, _ = frontmatter.read(path)
        name = fm.get("name")
        user = fm.get("user_name") or ""
        return (str(name), str(user)) if name else None
    except Exception:
        return None


def _parse_transcript(body: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_role and current_lines:
            chunks.append({
                "role": current_role,
                "content": "\n".join(current_lines).strip(),
            })

    for line in body.splitlines():
        s = line.strip()
        if s.startswith("## USER"):
            flush(); current_role = "user"; current_lines = []
        elif s.startswith("## ASSISTANT"):
            flush(); current_role = "assistant"; current_lines = []
        else:
            current_lines.append(line)
    flush()
    return [c for c in chunks if c["content"]]


import re as _re  # noqa: E402

_FLAG_RE = _re.compile(
    r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)(?::[^\]]*)?\]",
    _re.IGNORECASE,
)


def _clean_for_tts(text: str) -> str:
    """Strip inline flags and markdown fences before sending to TTS."""
    out = _FLAG_RE.sub("", text)
    out = out.replace("```", "")
    return out.strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--tts-model", default="mlx-community/chatterbox-turbo-fp16")
    parser.add_argument("--ref-audio", default=REF_AUDIO_DEFAULT,
                        help="Reference wav for TTS voice cloning.")
    parser.add_argument("--no-voice", action="store_true",
                        help="Skip loading STT/TTS — text-only mode.")
    args = parser.parse_args()

    voice: VoiceBackend | None = None
    if HAS_VOICE and not args.no_voice:
        print("\n\033[2m[meme/voice] loading STT + TTS on Metal — first run downloads models.\033[0m\n")
        try:
            voice = VoiceBackend(args.stt_model, args.tts_model, args.ref_audio)
            voice.load(on_status=lambda s: print(f"\033[2m  {s}\033[0m"))
            print("\033[92m[meme/voice] ready\033[0m\n")
        except Exception as exc:
            print(f"\033[91m[meme/voice] voice load failed — {exc}\033[0m")
            print("\033[93m[meme/voice] starting in text-only mode\033[0m\n")
            voice = None
    elif not HAS_VOICE:
        print("\n\033[93m[meme/voice] voice stack not installed — text-only mode.\033[0m")
        print("\033[2m  install: pip install -r requirements-voice.txt\033[0m\n")

    MemeVoiceTUI(voice=voice).run()


if __name__ == "__main__":
    main()
