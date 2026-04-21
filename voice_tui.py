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

import base64
import os
import sys
import threading
import time
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
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
sys.stderr = _stderr

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Button, Input, Label, ListView
from textual import work

# --- meme core + shared UI --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import obsidian, outreach, proactive, reflection  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env  # noqa: E402
from tui_common import (  # noqa: E402
    BASE_CSS, ChatMessage, StatusBar, TranscriptItem,
    read_identity, transcript_entries, parse_transcript,
    strip_meme_flags, clean_for_speech, split_sentence, log_error,
    copy_to_clipboard,
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


# ---------------------------------------------------------------------------
# Video backend — camera frames for "hold ⌘ to see" mode.
# ---------------------------------------------------------------------------
def probe_cameras(max_index: int = 4) -> list[dict]:
    """Try each camera index and return what actually answers.

    Needed on macOS because Continuity Camera puts an iPhone (if paired)
    at index 0 and demotes the MacBook's FaceTime HD to index 1 — so a
    naive `VideoCapture(0)` grabs the phone and leaves the laptop LED
    dark. Run this at startup so the user can pick the right index.
    """
    if not HAS_CV2:
        return []
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    out: list[dict] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if not cap or not cap.isOpened():
            if cap is not None:
                try: cap.release()
                except Exception: pass
            continue
        # Read a few frames so exposure / warm-up finishes before we grab
        # the resolution — AVFoundation returns 0x0 if we query too early.
        w = h = 0
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                break
            time.sleep(0.05)
        try: cap.release()
        except Exception: pass
        if w and h:
            out.append({"index": i, "width": w, "height": h})
    return out


class VideoBackend:
    """Captures camera frames on a background thread while push-to-see is
    held, then returns a handful of evenly-spaced base64-encoded JPEG frames
    suitable for Mistral's multimodal `image_url` content parts."""

    def __init__(self, camera_index: int = 0, target_width: int = 768,
                 capture_fps: int = 8):
        self.camera_index = camera_index
        self.target_width = target_width
        self.capture_interval = 1.0 / max(1, capture_fps)
        self.cap = None
        self.frames: list = []
        self.is_recording = False
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start_recording(self) -> tuple[bool, str]:
        """Try to open the camera and start the capture thread.
        Returns (ok, reason). `reason` is a human-readable hint on failure."""
        if not HAS_CV2:
            return False, "opencv-python not installed"
        if self.is_recording:
            return False, "already recording"
        # On macOS the default backend often silently fails; AVFoundation is
        # the one that actually talks to the camera kext.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap or not self.cap.isOpened():
            self.cap = None
            return False, (
                "camera open failed — check that YOUR TERMINAL app "
                "(not Python) has Camera permission in System Settings → "
                "Privacy & Security → Camera"
            )
        # Warm-up: AVFoundation returns frames immediately after open but
        # the first ~30 are BLACK (all zeros) while auto-exposure settles.
        # A naive `ret == True` check will happily accept those. Wait until
        # a frame actually has pixel variance, or give up after ~2s.
        warm_frame = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Cheap "is this actually an image?" check — a 32-sample
                # stride is plenty to distinguish real pixels from zeros.
                sample = frame[::32, ::32]
                if sample.mean() > 8.0 and sample.std() > 3.0:
                    warm_frame = frame
                    break
            time.sleep(0.03)
        if warm_frame is None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
            return False, (
                "camera opened but only returned black frames after 2s. "
                "Try pointing it at a brighter scene, or check that another "
                "app isn't already holding the camera."
            )
        with self.lock:
            self.is_recording = True
            self.frames = [warm_frame]  # seed with the first real frame
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True, "ok"

    def _capture_loop(self) -> None:
        while True:
            with self.lock:
                if not self.is_recording or self.cap is None:
                    break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Skip the occasional black frame AVFoundation drops in —
                # same pixel-variance check as warm-up.
                sample = frame[::32, ::32]
                if sample.mean() > 8.0 and sample.std() > 3.0:
                    with self.lock:
                        self.frames.append(frame)
            time.sleep(self.capture_interval)

    def stop_recording(self, n_frames: int = 3) -> list[str]:
        """Stop capture and return up to n_frames evenly-spaced frames as
        `data:image/jpeg;base64,...` URLs."""
        with self.lock:
            self.is_recording = False
            frames = self.frames
            self.frames = []
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if not frames:
            return []
        if len(frames) <= n_frames:
            sampled = frames
        else:
            step = len(frames) / n_frames
            sampled = [frames[int(i * step)] for i in range(n_frames)]
        out: list[str] = []
        for frame in sampled:
            h, w = frame.shape[:2]
            if w > self.target_width:
                scale = self.target_width / w
                frame = cv2.resize(frame, (self.target_width, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            out.append(f"data:image/jpeg;base64,{b64}")
        return out


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
        Binding("ctrl+y", "copy_reply", "copy last reply", show=True),
        Binding("up", "history_prev", "prev input", show=False),
        Binding("down", "history_next", "next input", show=False),
    ]

    def __init__(
        self,
        voice: VoiceBackend | None = None,
        video: VideoBackend | None = None,
    ) -> None:
        super().__init__()
        self.title = APP_NAME
        self.voice = voice
        self.video = video
        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[str] = []
        self._pynput_listener = None
        self._chat_container: ScrollableContainer | None = None
        self._transcript_list: ListView | None = None
        self._current_ai: ChatMessage | None = None
        self._input_history: list[str] = []
        self._history_idx: int = 0
        self._last_reply: str = ""
        # True while a chat worker is running — blocks concurrent sends.
        self._chat_busy: bool = False
        # Proactive outreach state (see tui.py).
        self._proactive_prefix: str | None = None
        self._proactive_node_path: str | None = None
        # Which PTT modifier is currently held: "audio" (Option), "vision"
        # (Command), or None. Set on press, cleared on release. Ensures
        # that releasing a different modifier doesn't stop a recording
        # started by the first one.
        self._ptt_mode: str | None = None
        # ⌘ debounce — the Cmd key fires a BAZILLION system shortcuts
        # (Cmd+V, Cmd+Tab, Cmd+Space). We refuse to start recording until
        # Cmd has been held *alone* for PTT_CMD_HOLD_MS with no other key.
        self._cmd_pending_since: float | None = None
        self._cmd_cancelled: bool = False

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
                    if HAS_PYNPUT:
                        bar.hotkey_hint = (
                            "⌥ talk   ⌘ see+talk   Type send   ↑↓ history   "
                            "Ctrl-Y copy reply   Esc quit (auto-save)"
                        ) if HAS_CV2 else (
                            "⌥ talk   Type send   ↑↓ history   "
                            "Ctrl-Y copy reply   Esc quit (auto-save)"
                        )
                    else:
                        bar.hotkey_hint = (
                            "Type send   ↑↓ history   Ctrl-Y copy reply   "
                            "Esc quit (auto-save)"
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
        if HAS_CV2 and self.video is not None:
            vm = (CONFIG.get("models") or {}).get("vision") or {}
            self._system_message(
                f"vision: hold ⌘ to see + speak · {vm.get('provider', '?')} / {vm.get('model', '?')}"
            )
        elif not HAS_CV2:
            self._system_message(
                "opencv-python not installed — hold-⌘ camera mode disabled. "
                "install: pip install opencv-python"
            )

        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is not None:
            head = obsidian.git_head(ext) or "no git"
            self._system_message(
                f"external vault: {ext}  (notes: {obsidian.note_count(ext)}, git: {head})"
            )
        else:
            self._system_message(
                "external vault: (disabled — set external_vault.path in config.yaml)"
            )

        self._maybe_surface_proactive()

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
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
        self._history_idx = len(self._input_history)
        if text.lower() in ("quit", "exit", "bye", ":q"):
            self.action_quit()
            return
        if text.startswith("/obsidian"):
            arg = text[len("/obsidian"):].strip()
            self._cmd_obsidian(arg)
            return
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lstrip("/").lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd == "whats_up":
                self._cmd_whats_up()
                return
            if cmd == "outreach":
                self._cmd_outreach(arg)
                return
            if cmd == "pause":
                self._cmd_pause(arg)
                return
            if cmd == "mute":
                self._cmd_mute(arg)
                return
            if cmd == "unmute":
                self._cmd_unmute(arg)
                return
        if self._chat_busy:
            self._system_message("(still thinking — hold on a moment.)")
            return
        self._send(text)

    # ----- proactive outreach --------------------------------------------
    def _maybe_surface_proactive(self) -> None:
        pcfg = (CONFIG.get("proactive") or {})
        if not pcfg.get("enabled"):
            return
        if outreach.active_pause(VAULT) is not None:
            return
        try:
            ctx = outreach.build_context(VAULT, CONFIG)
            cs = proactive.candidates(VAULT, CONFIG)
            pick = proactive.should_reach_out(cs, ctx, CONFIG)
        except Exception as exc:
            self._system_message(f"(proactive: {exc})")
            return
        if pick is None:
            return
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
        # Speak it — samantha's a voice agent, the nudge should be heard
        # not just read. Cleans markdown/flags before TTS so no symbols leak.
        if self.voice and not self.voice.muted:
            self.voice.speak(clean_for_speech(msg))
        self._proactive_prefix = pick["node_name"]
        self._proactive_node_path = pick.get("node_path")

    def _cmd_whats_up(self) -> None:
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

    def _cmd_outreach(self, arg: str) -> None:
        sub = (arg or "").strip().lower()
        if sub == "status":
            pcfg = (CONFIG.get("proactive") or {})
            pause = outreach.active_pause(VAULT)
            ctx = outreach.build_context(VAULT, CONFIG)
            lines = [
                f"enabled: {bool(pcfg.get('enabled'))}",
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

    def _cmd_pause(self, arg: str) -> None:
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
                "usage: /pause 24h  |  /pause until 2026-05-01  |  /pause off"
            )
            return
        outreach.set_pause(VAULT, until)
        self._system_message(
            f"proactive: paused until {until.isoformat(timespec='seconds')}."
        )

    def _cmd_mute(self, arg: str) -> None:
        name = (arg or "").strip()
        if not name:
            self._system_message("usage: /mute <node name>")
            return
        if outreach.set_node_proactive(VAULT, name, False):
            self._system_message(f"muted: {name}")
        else:
            self._system_message(f"no node matching '{name}'")

    def _cmd_unmute(self, arg: str) -> None:
        name = (arg or "").strip()
        if not name:
            self._system_message("usage: /unmute <node name>")
            return
        if outreach.set_node_proactive(VAULT, name, True):
            self._system_message(f"unmuted: {name}")
        else:
            self._system_message(f"no node matching '{name}'")

    def _cmd_obsidian(self, arg: str) -> None:
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

    # ----- input: voice (PTT) -----------------------------------------

    def _start_pynput(self) -> None:
        if not HAS_PYNPUT:
            return

        alt_keys = (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
        cmd_keys = (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
        # How long ⌘ must be held ALONE before we treat it as a vision-PTT
        # intent (rather than the start of a Cmd+V / Cmd+Tab shortcut).
        CMD_HOLD_SECONDS = 0.4

        def _cmd_timer_fired():
            # Timer thread. Start recording if the user is still pending
            # (hasn't released and hasn't pressed another key).
            if self._cmd_pending_since is None or self._cmd_cancelled:
                return
            if self._ptt_mode is not None:
                return
            self._cmd_pending_since = None
            if self.video is None:
                self.call_from_thread(
                    self._system_message,
                    "⌘ hold — vision disabled (cv2 missing or camera probe failed).",
                )
                return
            if not self.voice or not self.voice.start_recording():
                return
            ok, reason = self.video.start_recording()
            if ok:
                self._ptt_mode = "vision"
                self.call_from_thread(self._on_see_start)
            else:
                self._ptt_mode = "audio"
                self.call_from_thread(self._on_listen_start)
                self.call_from_thread(
                    self._system_message, f"⌘ held — camera failed · {reason}"
                )

        def on_press(key):
            # Any non-⌘ key press while ⌘ is pending cancels the intent —
            # this is what stops Cmd+V, Cmd+Tab, Cmd+Space from triggering
            # the camera.
            if self._cmd_pending_since is not None and key not in cmd_keys:
                self._cmd_cancelled = True

            if self._ptt_mode is not None:
                return
            if key in alt_keys:
                if self.voice and self.voice.start_recording():
                    self._ptt_mode = "audio"
                    self.call_from_thread(self._on_listen_start)
            elif key in cmd_keys:
                # Arm the debounce timer. Actual recording only starts if
                # ⌘ is still held alone after CMD_HOLD_SECONDS.
                import threading as _th
                self._cmd_pending_since = time.time()
                self._cmd_cancelled = False
                _th.Timer(CMD_HOLD_SECONDS, _cmd_timer_fired).start()

        def on_release(key):
            if key in cmd_keys:
                # Cancel any pending intent — user released before the
                # debounce window elapsed (i.e. this was a quick tap or
                # a shortcut, not a hold).
                self._cmd_pending_since = None
                self._cmd_cancelled = False
            if self._ptt_mode == "audio" and key in alt_keys:
                self._ptt_mode = None
                self.call_from_thread(self._on_listen_stop)
            elif self._ptt_mode == "vision" and key in cmd_keys:
                self._ptt_mode = None
                self.call_from_thread(self._on_see_stop)

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

    def _on_see_start(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.listening = True
        bar.text = "🎤📷 recording audio + video..."

    def _on_see_stop(self) -> None:
        bar = self.query_one("#status-bar", StatusBar)
        bar.listening = False
        bar.text = "transcribing + encoding frames..."
        self._transcribe_and_send_vision()

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

    @work(thread=True)
    def _transcribe_and_send_vision(self) -> None:
        if not self.voice or not self.video:
            return
        text = self.voice.stop_recording() or ""
        frames = self.video.stop_recording(n_frames=3)
        if not frames:
            self.call_from_thread(self._system_message, "(camera captured no frames)")
            self.call_from_thread(self._set_status, "")
            if not text:
                return
            self.call_from_thread(self._send, text)
            return
        self.call_from_thread(self._set_status, "")
        # Empty transcript + frames → treat as an implicit "describe this".
        prompt = text.strip() or "Describe what you see."
        self.call_from_thread(self._send, prompt, frames)

    # ----- chat pipeline ----------------------------------------------

    def _send(self, user_text: str, frames: list[str] | None = None) -> None:
        self._chat_busy = True
        label = user_text
        if frames:
            label = f"{user_text}  📷 [{len(frames)} frame{'s' if len(frames) != 1 else ''}]"
        self._user_message(label)
        self._start_ai_message()
        self._run_chat(user_text, frames)

    @work(thread=True)
    def _run_chat(self, user_text: str, frames: list[str] | None = None) -> None:
        if self.session is None:
            task = user_text
            if self._proactive_prefix:
                task = f"reply to proactive: {self._proactive_prefix} — {user_text}"
            try:
                self.session = session_mgr.start(
                    task=task, tags=[], config=CONFIG, project_root=ROOT,
                )
                if self._proactive_node_path:
                    existing = list(self.session.get("retrieved_files") or [])
                    if self._proactive_node_path not in existing:
                        existing.insert(0, self._proactive_node_path)
                    self.session["retrieved_files"] = existing
                self._proactive_prefix = None
                self._proactive_node_path = None
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

        if frames:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for url in frames:
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            self.messages.append({"role": "user", "content": content_parts})
            self.transcript.append(
                f"## USER\n{user_text}  [attached {len(frames)} camera frame(s)]"
            )
            # Use the vision role if the user configured one; fall back to model1.
            role = "vision" if "vision" in (CONFIG.get("models") or {}) else "model1"
        else:
            self.messages.append({"role": "user", "content": user_text})
            self.transcript.append(f"## USER\n{user_text}")
            role = "model1"
        window = CONFIG["session"]["history_window"]

        full_reply = ""
        tts_buffer = ""
        spoken_so_far = ""
        try:
            for chunk in reflection.chat_stream(
                role=role,
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

        # Images are costly to re-send every turn. Now that the model has
        # answered, collapse this turn's user message to text-only in history
        # (the model's reply already encodes what it saw).
        if frames and self.messages:
            last_user_idx = len(self.messages) - 1
            if isinstance(self.messages[last_user_idx].get("content"), list):
                self.messages[last_user_idx]["content"] = (
                    f"{user_text}  [had attached {len(frames)} camera frame(s)]"
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
        self._last_reply = content
        self._chat_busy = False
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
    parser.add_argument(
        "--camera-index", type=int, default=None,
        help="Which camera index to use for vision mode. If unset, probes "
             "all available cameras and picks the MacBook's built-in over "
             "Continuity-Camera-connected iPhones.",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="Probe available cameras, print their indices + resolutions, and exit.",
    )
    args = parser.parse_args()

    if args.list_cameras:
        cams = probe_cameras()
        if not cams:
            print("(no cameras detected — is opencv-python installed and the terminal granted camera access?)")
        else:
            for c in cams:
                print(f"  index {c['index']}: {c['width']}x{c['height']}")
        return

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

    video: VideoBackend | None = None
    if HAS_CV2:
        idx = args.camera_index
        cams = probe_cameras()
        if idx is None:
            if not cams:
                idx = 0  # will fail at press-time with a clear message
            elif len(cams) == 1:
                idx = cams[0]["index"]
            else:
                # Multiple cameras → prefer the MacBook's built-in. Continuity
                # Camera (iPhone) shows up at the LOWER index and reports much
                # higher resolutions (e.g. 1920×1440, 4032×3024). The built-in
                # FaceTime HD is 1280×720 or 1920×1080 — pick the HIGHER-index
                # one whose resolution is ≤ 1920×1080, which is almost always
                # the MacBook.
                candidates = [c for c in cams if c["width"] <= 1920 and c["height"] <= 1080]
                picked = max(candidates, key=lambda c: c["index"]) if candidates else cams[-1]
                idx = picked["index"]
                print(
                    f"\033[2m[{APP_NAME}] cameras: " +
                    ", ".join(f"idx {c['index']} ({c['width']}x{c['height']})" for c in cams) +
                    f"  →  picking index {idx}. Override with --camera-index N.\033[0m"
                )
        video = VideoBackend(camera_index=idx)

    SamanthaTUI(voice=voice, video=video).run()


if __name__ == "__main__":
    main()
