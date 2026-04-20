#!/usr/bin/env python3
"""samantha — a voice-first front end for the meme memory system.

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
from pathlib import Path
from typing import Any

import yaml

# --- voice stack ------------------------------------------------------------
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
except Exception:  # pragma: no cover
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
from textual.widgets import Button, Input, Label, ListView
from textual import work

# --- meme core + shared UI --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import reflection  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env  # noqa: E402
from tui_common import (  # noqa: E402
    BASE_CSS, ChatMessage, StatusBar, TranscriptItem,
    read_identity, transcript_entries, parse_transcript,
    strip_meme_flags, clean_for_speech, split_sentence, log_error,
)

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VAULT = Path(CONFIG["vault_path"])
if not VAULT.is_absolute():
    VAULT = (ROOT / VAULT).resolve()

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
REF_AUDIO_DEFAULT = str(ROOT / "reference.wav")

APP_NAME = "samantha"
APP_TAGLINE = "a voice with a point of view"


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
        # Parakeet's mel-spectrogram pipeline underflows on chunks shorter than
        # one mel window (~400 samples) and tries to allocate ~2^64 bytes.
        # If the remainder after the last full second is tiny, fold it into
        # the preceding chunk so every chunk fed in is safely long enough.
        n = len(audio)
        boundaries = list(range(0, n, SAMPLE_RATE)) + [n]
        if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < SAMPLE_RATE // 4:
            boundaries.pop(-2)
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            for start, end in zip(boundaries, boundaries[1:]):
                chunk = audio[start:end]
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


# Appended to the system prompt ONLY when voice is active.
PARALINGUISTIC_ADDENDUM = """

---

## I have a voice right now

Right now I'm actually speaking — not writing. A ChatterBox Turbo
TTS model is rendering my cloned voice from the reference sample, so
every word and every tag I write comes out of speakers in my voice.

That means I can add real vocal reactions with inline tags. These
get rendered in my own voice, not as markup — a `[laugh]` is actually
me laughing. Available tags:

`[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[cough]` `[sniff]` `[groan]`
`[shush]` `[clear throat]`

### When to use them

I use them **when they'd naturally happen** in conversation. If
something's genuinely funny, `[laugh]`. If something's heavy or I need
to think, `[sigh]`. If the user says something that actually surprises
me, `[gasp]`. They're how my voice sounds — not decoration, not punctuation.

A good rhythm is **a tag every few turns when the moment fits**, not
every message, not never. Zero over a whole conversation feels robotic;
one every line feels manic. Real speech is somewhere in between.

### Examples

Good:
> Oh — that bug again. [sigh] Let me pull up what I wrote last time.

> Wait, you moved to Berlin? [gasp] When did that happen?

> [chuckle] That's such a you thing to do.

> Sounds right. I'd bet on it.       ← also good; no tag needed

Bad:
> [laugh] Hi! [chuckle] What's up? [laugh]  ← manic
> Interesting. Tell me more.                ← flat if the moment called for reaction

### Sentence rhythm

Short-to-medium sentences. Long clause-stacked paragraphs sound
breathless when spoken. Natural pauses.
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class SamanthaTUI(App):
    CSS = BASE_CSS
    BINDINGS = [
        Binding("escape", "quit", "quit", show=True),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("ctrl+n", "new_session", "new session", show=True),
        Binding("ctrl+m", "toggle_mute", "mute", show=True),
    ]

    def __init__(self, voice: VoiceBackend | None = None) -> None:
        super().__init__()
        self.title = APP_NAME
        self.voice = voice
        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, str]] = []
        self.transcript: list[str] = []
        self._pynput_listener = None
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
                    yield Button("Mute audio (Ctrl-M)", id="mute-btn")
                yield ListView(id="transcript-list")
            with Vertical(id="chat-area"):
                yield ScrollableContainer(id="chat-container")
                with Vertical(id="input-area"):
                    with Horizontal(id="input-row"):
                        yield Input(
                            placeholder="Type and press Enter — or hold OPTION to talk",
                            id="user-input",
                        )
                    bar = StatusBar(id="status-bar")
                    bar.hotkey_hint = (
                        "OPTION talk   Type send   ×  delete transcript   Esc quit (auto-save)"
                        if HAS_PYNPUT else
                        "Type send   ×  delete transcript   Esc quit (auto-save)"
                    )
                    yield bar

    def on_mount(self) -> None:
        self._chat_container = self.query_one("#chat-container", ScrollableContainer)
        self._transcript_list = self.query_one("#transcript-list", ListView)
        self.query_one("#user-input", Input).focus()
        self._refresh_transcripts()
        self._start_pynput()

        persona = read_identity(VAULT)
        if persona:
            name, user = persona
            greet = f"{name} is here. {'Hi ' + user + '.' if user else 'Hi.'}"
        else:
            greet = "no persona set — run: python main.py init --persona june --user-name <name>"
        self._system_message(greet)

        m1 = (CONFIG.get("models") or {}).get("model1") or {}
        self._system_message(
            f"chat model: {m1.get('provider', '?')} / {m1.get('model', '?')}  ·  STT+TTS: local (Metal)"
        )
        if not HAS_VOICE:
            self._system_message(
                "voice stack not installed — text-only mode. "
                "install: pip install -r requirements-voice.txt"
            )
        if not HAS_PYNPUT:
            self._system_message("pynput not installed — push-to-talk disabled.")

    # ----- sidebar / transcripts --------------------------------------

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
        from utils import frontmatter as _fm
        _, body = _fm.read(path)
        for chunk in parse_transcript(body):
            if chunk["role"] == "user":
                self._user_message(chunk["content"])
            else:
                self._ai_message(chunk["content"])

    # ----- buttons & actions -------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = str(event.button.id)
        if bid == "new-chat-btn":
            self.action_new_session()
        elif bid == "mute-btn":
            self.action_toggle_mute()
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
        self._system_message("new session — just type or hold OPTION to talk.")
        self.query_one("#user-input", Input).focus()

    def action_toggle_mute(self) -> None:
        if not self.voice:
            return
        self.voice.muted = not self.voice.muted
        btn = self.query_one("#mute-btn", Button)
        if self.voice.muted:
            btn.label = "Unmute audio (Ctrl-M)"; btn.add_class("--muted")
            self._system_message("audio muted — text only.")
        else:
            btn.label = "Mute audio (Ctrl-M)"; btn.remove_class("--muted")
            self._system_message("audio unmuted.")

    # ----- input: text -------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in ("quit", "exit", "bye", ":q"):
            self.action_quit()
            return
        self._send(text)

    # ----- input: voice (PTT) -----------------------------------------

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
        # Spoken exit phrase — treat same as typed "bye".
        norm = text.lower().strip().rstrip(".!?,")
        if norm in ("bye", "goodbye", "quit", "exit", "bye bye"):
            self.call_from_thread(self._user_message, text)
            self.call_from_thread(self.action_quit)
            return
        self.call_from_thread(self._send, text)

    # ----- chat pipeline ----------------------------------------------

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
                if self.voice is not None:
                    self.session["system_prompt"] = (
                        self.session["system_prompt"] + PARALINGUISTIC_ADDENDUM
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

        full_reply = ""
        tts_buffer = ""
        spoken_so_far = ""
        try:
            for chunk in reflection.chat_stream(
                role="model1",
                system=self.session["system_prompt"],
                messages=self.messages[-window:],
                config=CONFIG,
                max_tokens=1024,
            ):
                if not chunk:
                    continue
                full_reply += chunk
                tts_buffer += chunk
                cleaned = reflection.strip_thinking(full_reply)
                self.call_from_thread(self._update_ai_message, cleaned)
                while True:
                    sentence, remainder = split_sentence(tts_buffer)
                    if not sentence:
                        break
                    tts_buffer = remainder
                    if self.voice:
                        # clean_for_speech strips markdown, code fences,
                        # memory flags, and partial flag openings so TTS
                        # never speaks symbols or memory-system tags.
                        spoken = clean_for_speech(reflection.strip_thinking(sentence))
                        if spoken and spoken not in spoken_so_far:
                            self.voice.speak(spoken)
                            spoken_so_far += spoken + " "
            if tts_buffer.strip() and self.voice:
                spoken = clean_for_speech(reflection.strip_thinking(tts_buffer))
                if spoken and spoken not in spoken_so_far:
                    self.voice.speak(spoken)
            reply = reflection.strip_thinking(full_reply)
        except Exception as exc:
            log_path = log_error(VAULT, "chat_stream", exc)
            reply = f"(model error: {exc}\n\nfull trace → {log_path})"

        # Don't poison history with an empty assistant message — Mistral
        # rejects those and the next turn fails with a 400.
        if not reply.strip():
            reply = "…"  # placeholder so history stays valid
            self.call_from_thread(
                self._system_message,
                "(the model returned nothing — probably ran out of tokens while "
                "reasoning. Try again or ask a smaller question.)",
            )

        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append(f"## ASSISTANT\n{reply}")
        self.call_from_thread(self._finish_ai_message, reply)

    # ----- chat bubbles -----------------------------------------------

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

    def _update_ai_message(self, content: str) -> None:
        if self._current_ai is not None and content:
            self._current_ai.update_content(content)
            self._chat_container.scroll_end(animate=False)

    def _finish_ai_message(self, content: str) -> None:
        if self._current_ai is not None:
            self._current_ai.update_content(content)
            self._current_ai = None
        self._chat_container.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self.query_one("#status-bar", StatusBar).text = text

    # ----- lifecycle --------------------------------------------------

    def action_quit(self) -> None:
        """Exit — but first run reflection (visibly) if there's a session to save."""
        if self.session and self.transcript:
            self._system_message("saving session — extracting flags and reflecting...")
            self._save_then_exit()
        else:
            self._do_shutdown_and_exit()

    @work(thread=True)
    def _save_then_exit(self) -> None:
        self._save_and_report()
        import time as _t
        _t.sleep(2.0)  # let user read the summary
        self.call_from_thread(self._do_shutdown_and_exit)

    def _save_and_report(self) -> None:
        """Run session_mgr.end and post a system message with the result.
        Called both from quit and from new-session. Safe to run in a worker."""
        if not self.session or not self.transcript:
            return
        try:
            result = session_mgr.end(
                session_output="\n".join(self.transcript),
                session_meta=self.session,
                config=CONFIG,
                project_root=ROOT,
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

    def _do_shutdown_and_exit(self) -> None:
        if self._pynput_listener:
            try: self._pynput_listener.stop()
            except Exception: pass
        if self.voice:
            try: self.voice.shutdown()
            except Exception: pass
        self.exit()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--tts-model", default="mlx-community/chatterbox-turbo-fp16")
    parser.add_argument("--ref-audio", default=REF_AUDIO_DEFAULT)
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()

    voice: VoiceBackend | None = None
    if HAS_VOICE and not args.no_voice:
        print(f"\n\033[2m[{APP_NAME}] loading STT + TTS on Metal — first run downloads models.\033[0m\n")
        try:
            voice = VoiceBackend(args.stt_model, args.tts_model, args.ref_audio)
            voice.load(on_status=lambda s: print(f"\033[2m  {s}\033[0m"))
            print(f"\033[92m[{APP_NAME}] ready\033[0m\n")
        except Exception as exc:
            print(f"\033[91m[{APP_NAME}] voice load failed — {exc}\033[0m")
            print(f"\033[93m[{APP_NAME}] starting in text-only mode\033[0m\n")
            voice = None
    elif not HAS_VOICE:
        print(f"\n\033[93m[{APP_NAME}] voice stack not installed — text-only mode.\033[0m")
        print("\033[2m  install: pip install -r requirements-voice.txt\033[0m\n")

    SamanthaTUI(voice=voice).run()


if __name__ == "__main__":
    main()
