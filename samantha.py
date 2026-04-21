#!/usr/bin/env python3
"""samantha — minimal voice + memory chat TUI.

Hold ⌥ to talk. Hold ⌘ for camera + talk. Type to chat.
Slash commands: /help shows the full list.

Replaces the over-engineered tui.py / voice_tui.py with a single-pane
chat following Textual's idiomatic Header + VerticalScroll + Input + Footer.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# ── optional voice / vision backends ──────────────────────────────────────
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
except Exception:
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

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

# ── project ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import cron as cron_mod, decay, dedup, monitor, obsidian, outreach, proactive, reflection, retrieval, runtime  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, frontmatter, indexer  # noqa: E402

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VAULT = Path(CONFIG["vault_path"])
if not VAULT.is_absolute():
    VAULT = (ROOT / VAULT).resolve()

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
REF_AUDIO_DEFAULT = str(ROOT / "reference.wav")


# ── tiny helpers (kept local so this file is self-contained) ──────────────
def read_identity(vault: Path) -> tuple[str, str] | None:
    path = vault / "_identity" / "persona.md"
    if not path.exists():
        return None
    try:
        fm, _ = frontmatter.read(path)
        return (str(fm.get("name") or "samantha"), str(fm.get("user_name") or ""))
    except Exception:
        return None


def copy_to_clipboard(text: str) -> bool:
    cmds = {"darwin": ["pbcopy"], "linux": ["wl-copy"], "win32": ["clip"]}
    cmd = cmds.get(sys.platform) or (["xclip", "-selection", "clipboard"] if sys.platform.startswith("linux") else None)
    if cmd is None:
        return False
    try:
        subprocess.run(cmd, input=text.encode(), check=True)
        return True
    except Exception:
        return False


# markdown / flag strippers for TTS (kept tiny — full version in tui_common)
import re as _re
_FLAG_RE = _re.compile(
    r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)(?::[^\]]*)?\]",
    _re.IGNORECASE,
)
_MD_PATTERNS = [
    (_re.compile(r"```[a-zA-Z0-9_+\-]*\n.*?(?:\n```|\Z)", _re.DOTALL), " code snippet "),
    (_re.compile(r"`([^`\n]+)`"), r"\1"),
    (_re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (_re.compile(r"\*\*([^*\n]+)\*\*"), r"\1"),
    (_re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"\1"),
    (_re.compile(r"__([^_\n]+)__"), r"\1"),
    (_re.compile(r"^\s*#{1,6}\s+", _re.MULTILINE), ""),
    (_re.compile(r"^\s*[-*+]\s+", _re.MULTILINE), ""),
]
_SENTENCE_END = _re.compile(r"([.!?]+[\)\"']?)(\s+|$)")


def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat, repl in _MD_PATTERNS:
        out = pat.sub(repl, out)
    out = _FLAG_RE.sub("", out)
    out = _re.sub(
        r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)[^\]]*$",
        "", out, flags=_re.IGNORECASE,
    )
    return _re.sub(r"\s+", " ", out).strip()


def split_sentence(buf: str) -> tuple[str, str]:
    m = _SENTENCE_END.search(buf)
    if not m:
        return "", buf
    return buf[:m.end()].strip(), buf[m.end():]


# ── chat bubble + row widgets ─────────────────────────────────────────────
def _fmt_time(ts: datetime | None = None) -> str:
    ts = ts or datetime.now()
    return ts.strftime("%I:%M %p").lstrip("0").lower()


class Bubble(Static):
    """A single chat bubble — user, ai, or system. Renders a sender-line
    header + timestamp on top and markdown (or plain text) body below."""

    def __init__(self, content: str, kind: str = "user", sender: str = "") -> None:
        self.kind = kind
        self.sender = sender
        self._stamp = _fmt_time()
        self._raw = content  # kept so click-to-copy yields clean source text
        super().__init__(self._renderable(content))
        self.add_class(kind)

    def _renderable(self, content: str):
        if self.kind == "system":
            return content
        header = Text(f"{self.sender}  ·  {self._stamp}", style="bold #8b949e")
        body = (
            Markdown(content, code_theme="monokai", inline_code_theme="monokai")
            if self.kind == "ai"
            else Text(content)
        )
        return Group(header, Text(""), body)

    def set_content(self, content: str) -> None:
        self._raw = content
        self.update(self._renderable(content))

    def on_click(self) -> None:
        # Click any bubble → clipboard. Quiet success signal via a one-line
        # system row; stays out of the way.
        if not self._raw:
            return
        if copy_to_clipboard(self._raw):
            self.app.notify(
                f"copied {self.kind} message ({len(self._raw)} chars)",
                timeout=2,
            )


class Row(Horizontal):
    """Horizontal wrapper that aligns the bubble left/right/center by role."""

    def __init__(self, bubble: Bubble, kind: str) -> None:
        super().__init__()
        self.add_class(f"row-{kind}")
        self.bubble = bubble

    def compose(self) -> ComposeResult:
        yield self.bubble


# ── voice backend (optional, same engine as before, trimmed) ──────────────
# VAD is an opt-in import — only loaded when `/listen on` flips it on.
try:
    from silero_vad import load_silero_vad as _load_vad
    from silero_vad import VADIterator as _VADIterator
    HAS_VAD = True
except Exception:
    HAS_VAD = False


class VoiceBackend:
    def __init__(self, stt_model: str, tts_model: str, ref_audio: str):
        self.stt_model_id, self.tts_model_id, self.ref_audio_path = stt_model, tts_model, ref_audio
        self.stt = self.tts = self.player = self.ref_audio = None
        self.tts_queue: queue.Queue[str | None] = queue.Queue()
        self.muted = False
        self.is_recording = False
        self.audio_chunks: list = []
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()   # serialize all mx.eval() calls
        self.audio_stream = None
        # ── VAD / hands-free listening ──────────────────────────────────
        # Flipped on by start_listening(); the audio callback feeds frames
        # into a bounded deque which a worker thread drains through silero.
        self.listen_mode: bool = False
        self._vad_model = None
        self._vad_iter = None
        self._vad_frames: queue.Queue = queue.Queue(maxsize=200)
        self._vad_thread: threading.Thread | None = None
        self._on_speech = None  # callback(text: str) when an utterance ends
        # TTS gate — set while _tts_worker is mid-generation so we don't
        # feed Samantha's own voice into VAD as if the user were speaking.
        self._tts_busy = threading.Event()
        # Cooldown after TTS finishes; tail audio lingers in the player.
        self._tts_cooldown_until: float = 0.0

    def load(self, on_status=None) -> None:
        if not HAS_VOICE:
            return
        if on_status: on_status("loading STT...")
        self.stt = parakeet_mlx.from_pretrained(self.stt_model_id)
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            t.add_audio(mx.zeros((SAMPLE_RATE,), dtype=mx.float32))
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
            samplerate=SAMPLE_RATE, channels=1, callback=self._audio_cb, blocksize=CHUNK_SIZE,
        )
        self.audio_stream.start()
        if on_status: on_status("ready")

    def shutdown(self) -> None:
        self.stop_listening()
        if self.audio_stream:
            try: self.audio_stream.stop()
            except Exception: pass
        self.tts_queue.put(None)

    # ── VAD / continuous listening ──────────────────────────────────────
    def start_listening(self, on_speech) -> tuple[bool, str]:
        """Turn on hands-free mode. `on_speech(text)` is invoked on the VAD
        worker thread whenever a full utterance has been transcribed."""
        if not HAS_VAD:
            return False, "silero-vad not installed — pip install silero-vad"
        if not HAS_VOICE:
            return False, "voice stack unavailable"
        if self.listen_mode:
            return True, "already listening"
        if self._vad_model is None:
            try:
                self._vad_model = _load_vad()
            except Exception as exc:
                return False, f"VAD model load failed: {exc}"
        # Fresh iterator per start (internal state carries across).
        self._vad_iter = _VADIterator(
            self._vad_model,
            threshold=0.5,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=800,
        )
        # Drain any stale frames captured while listen_mode was off.
        while not self._vad_frames.empty():
            try: self._vad_frames.get_nowait()
            except queue.Empty: break
        self._on_speech = on_speech
        self.listen_mode = True
        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()
        return True, "listening"

    def stop_listening(self) -> None:
        if not self.listen_mode:
            return
        self.listen_mode = False
        t = self._vad_thread
        self._vad_thread = None
        if t:
            try: t.join(timeout=1.5)
            except Exception: pass

    def _tts_gate(self) -> bool:
        """True while we must NOT accept VAD-driven speech. Closes the
        echo loop: Samantha's voice plays → mic picks it up → VAD fires →
        she replies to herself.

        The key fix for the "last-chunk leak" is that while audio is STILL
        playing (buffered_samples > 0) we *extend* the cooldown — so
        even after the player's buffer drains, we wait out the room
        echo + speaker latency before opening the gate.
        """
        playing = False
        try:
            if self.player is not None and self.player.buffered_samples > 0:
                playing = True
        except Exception:
            pass
        if playing:
            self._tts_cooldown_until = max(
                self._tts_cooldown_until, time.time() + 3.0,
            )
            return True
        if self._tts_busy.is_set():
            return True
        if not self.tts_queue.empty():
            return True
        if time.time() < self._tts_cooldown_until:
            return True
        return False

    def listen_status(self) -> dict:
        """Snapshot of the VAD / echo-guard state — for /listen status."""
        try:
            buffered = int(self.player.buffered_samples) if self.player else 0
        except Exception:
            buffered = -1
        return {
            "listen_mode": self.listen_mode,
            "muted": self.muted,
            "tts_busy": self._tts_busy.is_set(),
            "tts_queue_size": self.tts_queue.qsize(),
            "buffered_samples": buffered,
            "cooldown_remaining_ms": max(
                0, int((self._tts_cooldown_until - time.time()) * 1000),
            ),
            "gate_closed": self._tts_gate(),
        }

    def _vad_worker(self) -> None:
        """Drain VAD frames, detect utterances, transcribe each on end-of-speech.

        Three defences against the echo loop:
          (a) Audio callback already drops frames while gate is closed;
              so the queue only carries "real" frames.
          (b) On every gate→open transition we reset silero's internal
              state AND discard any frames that snuck in during the race.
          (c) RMS gate: if the utterance's average energy is below
              `min_user_rms`, assume residual room echo and drop it —
              real user speech is 3-10× louder than typical echo tail.
        """
        import numpy as _np
        speech_buf: list = []
        in_speech = False
        gate_was_closed = True
        min_user_rms = 0.015  # empirical floor — below = echo/background
        min_utterance_confidence_rms = 0.020

        def _reset_vad():
            if hasattr(self._vad_iter, "reset_states"):
                try: self._vad_iter.reset_states()
                except Exception: pass

        while self.listen_mode:
            if self._tts_gate():
                if not gate_was_closed:
                    gate_was_closed = True
                    speech_buf.clear()
                    in_speech = False
                    _reset_vad()
                time.sleep(0.03)
                continue

            # Gate is open — flush any frames that raced in from the callback
            # right at the transition before we can trust them.
            if gate_was_closed:
                gate_was_closed = False
                drained = 0
                while not self._vad_frames.empty() and drained < 64:
                    try: self._vad_frames.get_nowait()
                    except queue.Empty: break
                    drained += 1
                _reset_vad()
                speech_buf.clear()
                in_speech = False
                continue

            try:
                frame = self._vad_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            if len(frame) != 512:
                continue

            # RMS check — kill any frame whose energy is below the noise floor
            # BEFORE feeding silero. Even if VAD would've fired on room-
            # shaped residual, we catch it here.
            rms = float(_np.sqrt(_np.mean(frame.astype(_np.float32) ** 2)))

            try:
                import torch
                tensor = torch.from_numpy(frame.astype(_np.float32))
                result = self._vad_iter(tensor)
            except Exception:
                result = None

            if in_speech:
                speech_buf.append(frame)

            if result:
                if "start" in result and not in_speech:
                    # Extra skepticism on utterance starts — a quiet start
                    # is almost always echo. Real speech starts louder.
                    if rms < min_user_rms:
                        continue
                    in_speech = True
                    speech_buf.append(frame)
                elif "end" in result and in_speech:
                    in_speech = False
                    audio = _np.concatenate(speech_buf).flatten().astype(_np.float32)
                    speech_buf = []
                    if len(audio) < SAMPLE_RATE * 0.3:
                        continue
                    # Whole-utterance RMS gate — a too-quiet utterance is
                    # echo tail, not a real user turn.
                    utt_rms = float(_np.sqrt(_np.mean(audio ** 2)))
                    if utt_rms < min_utterance_confidence_rms:
                        continue
                    try:
                        text = self._transcribe(audio)
                    except Exception:
                        text = ""
                    if text and self._on_speech:
                        try:
                            self._on_speech(text)
                        except Exception:
                            pass

    def _audio_cb(self, indata, frames, time_info, status):
        with self.lock:
            if self.is_recording:
                self.audio_chunks.append(indata.copy())
        # Listen mode — ONLY buffer frames when the gate is open. If we
        # queue echo frames while Samantha is talking and drain them later,
        # VAD detects "speech" right after she finishes and loops. This is
        # the hands-free echo-loop bug in one line.
        if self.listen_mode and not self._tts_gate():
            try:
                self._vad_frames.put_nowait(indata.copy().flatten())
            except queue.Full:
                pass

    def start_recording(self) -> bool:
        if not HAS_VOICE or self.is_recording:
            return False
        with self.lock:
            self.is_recording, self.audio_chunks = True, []
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
        audio = self._trim_silence(audio)
        if len(audio) < SAMPLE_RATE * 0.3:
            return None
        return self._transcribe(audio)

    @staticmethod
    def _trim_silence(audio, threshold=0.01):
        if len(audio) < SAMPLE_RATE * 0.3:
            return audio
        chunk = int(SAMPLE_RATE * 0.03)
        start, end = 0, len(audio)
        for i in range(0, len(audio) - chunk, chunk):
            if np.abs(audio[i:i+chunk]).mean() > threshold:
                start = max(0, i - chunk); break
        for i in range(len(audio) - chunk, chunk, -chunk):
            if np.abs(audio[i:i+chunk]).mean() > threshold:
                end = min(len(audio), i + chunk*2); break
        return audio[start:end]

    def _transcribe(self, audio) -> str:
        n = len(audio)
        boundaries = list(range(0, n, SAMPLE_RATE)) + [n]
        if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < SAMPLE_RATE // 4:
            boundaries.pop(-2)
        with self.gpu_lock:
            _prev = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
                    for s, e in zip(boundaries, boundaries[1:]):
                        chunk = audio[s:e]
                        if len(chunk) > 0:
                            t.add_audio(mx.array(chunk, dtype=mx.float32))
                    if t.result:
                        mx.eval(mx.array([0]))
                        return t.result.text.strip()
            finally:
                sys.stderr.close()
                sys.stderr = _prev
        return ""

    def speak(self, text: str) -> None:
        if not HAS_VOICE or self.muted or not text or not text.strip():
            return
        # Set the busy flag NOW — not when _tts_worker picks the item up.
        # Otherwise VAD has a small window to mistake our own TTS for user
        # speech between queue.put() and the worker starting generation.
        self._tts_busy.set()
        self.tts_queue.put(text)

    def _tts_worker(self) -> None:
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            _prev = sys.stderr
            sys.stderr = open(os.devnull, "w")
            self._tts_busy.set()
            try:
                for res in self.tts.generate(text=text, ref_audio=self.ref_audio, verbose=False, stream=True):
                    if hasattr(res, "audio") and res.audio is not None:
                        with self.gpu_lock:
                            mx.eval(res.audio)
                        self.player.queue_audio(res.audio)
            except Exception as exc:
                # Previously this was `pass` — which made TTS failures
                # invisible. Log them so the user can `cat` the file after
                # a silent voice issue.
                try:
                    log_dir = VAULT / "_meta"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with (log_dir / "tts_errors.log").open("a", encoding="utf-8") as f:
                        f.write(
                            f"[{datetime.now().isoformat(timespec='seconds')}] "
                            f"text={text[:120]!r} err={type(exc).__name__}: {exc}\n"
                        )
                except Exception:
                    pass
            finally:
                sys.stderr.close()
                sys.stderr = _prev
                self._tts_busy.clear()
                # Post-generation cooldown — `buffered_samples` doesn't
                # see the OS + hardware audio buffer (~150ms on macOS),
                # so we pad beyond what the player reports. 3s is
                # conservative but eliminates the "tail leaks as user
                # speech" class of bug in practice.
                self._tts_cooldown_until = time.time() + 3.0


class VideoBackend:
    def __init__(self, idx: int = 0):
        self.idx = idx
        self.cap = None
        self.frames: list = []
        self.is_recording = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def start(self) -> tuple[bool, str]:
        if not HAS_CV2:
            return False, "cv2 not installed"
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.idx, backend)
        if not self.cap or not self.cap.isOpened():
            return False, "camera open failed (grant Camera permission to the terminal app)"
        deadline = time.time() + 2.0
        warm = None
        while time.time() < deadline:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                s = frame[::32, ::32]
                if s.mean() > 8.0 and s.std() > 3.0:
                    warm = frame; break
            time.sleep(0.03)
        if warm is None:
            try: self.cap.release()
            except Exception: pass
            return False, "camera returned only black frames"
        with self.lock:
            self.is_recording, self.frames = True, [warm]
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True, "ok"

    def _loop(self) -> None:
        while True:
            with self.lock:
                if not self.is_recording or self.cap is None:
                    break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                s = frame[::32, ::32]
                if s.mean() > 8.0 and s.std() > 3.0:
                    with self.lock:
                        self.frames.append(frame)
            time.sleep(0.1)

    def stop(self, n: int = 3) -> list[str]:
        with self.lock:
            self.is_recording = False
            frames = self.frames
            self.frames = []
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if not frames:
            return []
        sampled = frames if len(frames) <= n else [frames[int(i * len(frames) / n)] for i in range(n)]
        out = []
        for f in sampled:
            h, w = f.shape[:2]
            if w > 768:
                f = cv2.resize(f, (768, int(h * 768 / w)))
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
        return out


# ── app ───────────────────────────────────────────────────────────────────
EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "q", ":q"}

HELP = """
**Commands**

- `/listen on` — hands-free voice (VAD, no button)  ·  `/listen off` — back to push-to-talk
- `/edit` — tweak your last message and resend (also Ctrl+E)
- `/copy` — copy last reply (also Ctrl+Y)
- `/history` — past conversations  ·  `/load <name>` / `/delete <name>`
- `/status` — model, voice, memory, external vault
- `/obsidian` — external notebook (`/obsidian recent`, `/obsidian diff`)
- `/whats_up` — what's on my mind  ·  `/pause 24h` / `/pause off`
- `/outreach` — recent proactive nudges  ·  `/mute <name>` / `/unmute <name>`
- `/show <name>` — print a vault node  ·  `/context` — what was retrieved
- `/meta` · `/decay` · `/monitor` · `/index` · `/identity` — maintenance

**Mouse** — click any bubble to copy its text.

**Keys** — ↑/↓ history  ·  Ctrl-E edit  ·  Ctrl-Y copy  ·  Ctrl-M mute  ·  Esc quit
"""


class Samantha(App):

    CSS = """
    Screen { background: #0b0e14; }

    Header { background: #11131a; color: #a78bfa; text-style: bold; }
    Footer { background: #11131a; color: #6b7280; }

    VerticalScroll#chat {
        padding: 1 3;
        background: #0b0e14;
        scrollbar-size: 1 1;
        scrollbar-color: #30363d #0b0e14;
    }

    /* Row wrappers — align bubbles left/right/center */
    Row        { height: auto; margin-top: 1; }
    Row.row-user   { align: right top; }
    Row.row-ai     { align: left  top; }
    Row.row-system { align: center top; margin-top: 0; }

    /* Bubbles */
    Bubble { height: auto; padding: 1 2; }

    Bubble.user {
        width: 70%;
        background: #1e3a8a;
        color: #f3f4f6;
        border: round #3b82f6;
    }

    Bubble.ai {
        width: 80%;
        background: #1a1a2e;
        color: #e5e7eb;
        border: round #8b5cf6;
    }

    /* Subtle hover hint — any bubble is clickable to copy. */
    Bubble.user:hover { border: round #93c5fd; }
    Bubble.ai:hover   { border: round #c4b5fd; }

    Bubble.system {
        width: 90%;
        color: #6b7280;
        background: transparent;
        text-style: italic;
        padding: 0 2;
    }

    Input {
        background: #11131a;
        color: #e5e7eb;
        border: round #30363d;
        margin: 1 2 0 2;
        height: 3;
        padding: 0 1;
    }
    Input:focus { border: round #8b5cf6; }
    """

    BINDINGS = [
        Binding("escape", "quit", "quit", show=True),
        Binding("ctrl+c", "quit", show=False),
        Binding("ctrl+y", "copy_reply", "copy"),
        Binding("ctrl+e", "edit_last", "edit last"),
        Binding("ctrl+m", "toggle_mute", "mute"),
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
        # Ctrl+N still works — just not advertised in the footer anymore.
        Binding("ctrl+n", "new_session", show=False),
    ]

    TITLE = "samantha"
    SUB_TITLE = "a memory with a voice"

    def __init__(self, voice: VoiceBackend | None = None, video: VideoBackend | None = None) -> None:
        super().__init__()
        self.voice = voice
        self.video = video
        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[str] = []
        self._chat: VerticalScroll | None = None
        self._input: Input | None = None
        self._current_ai: Bubble | None = None
        self._input_history: list[str] = []
        self._history_idx: int = 0
        self._last_reply: str = ""
        self._chat_busy: bool = False
        self._ptt_mode: str | None = None
        self._cmd_pending: float | None = None
        self._cmd_cancelled: bool = False
        self._proactive_prefix: str | None = None
        self._proactive_node_path: str | None = None
        self._listener = None
        # Sender names on each bubble's header line — populated in on_mount
        # from _identity/persona.md so labels match the live persona.
        self._user_name: str = "you"
        self._ai_name: str = "samantha"

    # ── compose / mount ─────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat")
        yield Input(placeholder="type your message  ·  /help for commands  ·  Esc to quit", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._chat = self.query_one("#chat", VerticalScroll)
        self._input = self.query_one("#input", Input)
        self._input.focus()

        # Publish live handles so the new utility tools (capture_camera,
        # mute_self, set_timer, …) can reach into the process without
        # being plumbed through the reflection dispatch signature.
        runtime.set_context(
            app=self, voice=self.voice, video=self.video,
            vault=VAULT, config=CONFIG,
        )

        persona = read_identity(VAULT)
        if persona:
            name, user = persona
            self._ai_name = name.lower()
            if user:
                self._user_name = user.lower()
            self.title = name
            self.sub_title = f"with {user}" if user else "a memory with a voice"
            greet = f"Hi{(' ' + user) if user else ''}."
            self._ai(greet)
            # Samantha is a voice agent — when she arrives, she should actually
            # arrive. Speak the greeting out loud if the TTS stack is alive
            # and not muted.
            if self.voice and not self.voice.muted:
                self.voice.speak(greet)
        else:
            self._sys(
                "No persona set. Run `python main.py init --persona samantha --user-name <name>` "
                "then relaunch."
            )

        self._start_pynput()
        self._maybe_surface_proactive()
        self._start_cron()

    # ── input routing ────────────────────────────────────────────────────
    @on(Input.Submitted)
    async def on_submit(self, event: Input.Submitted) -> None:
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
        if text.startswith("/"):
            self._dispatch(text)
            return
        if self._chat_busy:
            self._sys("(still thinking — hold on a moment.)")
            return
        self._send(text)

    # ── chat pipeline ────────────────────────────────────────────────────
    def _send(self, user_text: str, frames: list[str] | None = None) -> None:
        self._chat_busy = True
        label = user_text + (f"  📷 [{len(frames)}]" if frames else "")
        self._user(label)
        self._start_ai()
        self._run_chat(user_text, frames)

    @work(thread=True)
    def _run_chat(self, user_text: str, frames: list[str] | None = None) -> None:
        if self.session is None:
            try:
                task = user_text
                if self._proactive_prefix:
                    task = f"reply to proactive: {self._proactive_prefix} — {user_text}"
                self.session = session_mgr.start(
                    task=task, tags=[], config=CONFIG, project_root=ROOT,
                )
                if self._proactive_node_path:
                    existing = self.session.setdefault("retrieved_files", [])
                    if self._proactive_node_path not in existing:
                        existing.insert(0, self._proactive_node_path)
                self._proactive_prefix = None
                self._proactive_node_path = None
                self.messages = []
                self.transcript = []
                # Session start is now silent — we were leaking the lifecycle
                # concept into every conversation. The chat should feel rolling.
            except Exception as exc:
                self.call_from_thread(self._finish_ai, f"(start failed: {exc})")
                self._chat_busy = False
                return

        if frames:
            parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for url in frames:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            self.messages.append({"role": "user", "content": parts})
            self.transcript.append(f"## USER\n{user_text}  [attached {len(frames)} camera frame(s)]")
            role = "vision" if "vision" in (CONFIG.get("models") or {}) else "model1"
        else:
            self.messages.append({"role": "user", "content": user_text})
            self.transcript.append(f"## USER\n{user_text}")
            role = "model1"

        window = CONFIG["session"]["history_window"]
        agentic = bool((CONFIG.get("session") or {}).get("agentic_model1", True)) and not frames
        reply = ""
        try:
            if agentic:
                reply, call_log = reflection.chat_with_tools(
                    role=role,
                    system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG, vault_path=VAULT, max_tokens=2048,
                )
                # surface obsidian writes
                for c in call_log or []:
                    name = c.get("tool", "")
                    if name.startswith("obsidian_"):
                        args = c.get("args") or {}
                        path = args.get("rel_path") or args.get("old_rel") or ""
                        self.call_from_thread(self._sys, f"📝 {name}: {path}")
            else:
                # streaming path for both text-only and vision turns
                full = ""
                tts_buf = ""
                spoken = ""
                for chunk in reflection.chat_stream(
                    role=role,
                    system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG, max_tokens=1536,
                ):
                    if not chunk:
                        continue
                    full += chunk
                    tts_buf += chunk
                    cleaned = reflection.strip_thinking(full)
                    self.call_from_thread(self._update_ai, cleaned)
                    if self.voice and not self.voice.muted:
                        while True:
                            sent, rem = split_sentence(tts_buf)
                            if not sent:
                                break
                            tts_buf = rem
                            s = clean_for_speech(reflection.strip_thinking(sent))
                            if s and s not in spoken:
                                self.voice.speak(s)
                                spoken += s + " "
                if tts_buf.strip() and self.voice and not self.voice.muted:
                    s = clean_for_speech(reflection.strip_thinking(tts_buf))
                    if s and s not in spoken:
                        self.voice.speak(s)
                reply = reflection.strip_thinking(full)
        except Exception as exc:
            reply = f"(model error: {exc})"

        if not reply.strip():
            reply = "…"

        # Agentic turns: speak the final reply since streaming didn't happen.
        if agentic and self.voice and not self.voice.muted and reply != "…":
            self.voice.speak(clean_for_speech(reply))

        # Collapse multimodal user message to text-only after the turn lands.
        if frames and self.messages:
            idx = len(self.messages) - 1
            if isinstance(self.messages[idx].get("content"), list):
                self.messages[idx]["content"] = f"{user_text}  [had attached {len(frames)} camera frame(s)]"

        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append(f"## ASSISTANT\n{reply}")
        self.call_from_thread(self._finish_ai, reply)
        self._chat_busy = False

    # ── bubble helpers ──────────────────────────────────────────────────
    def _mount_row(self, bubble: Bubble, kind: str) -> None:
        self._chat.mount(Row(bubble, kind))
        self._chat.scroll_end(animate=False)

    def _user(self, text: str) -> None:
        self._mount_row(Bubble(text, "user", sender=self._user_name), "user")

    def _ai(self, text: str) -> None:
        self._mount_row(Bubble(text, "ai", sender=self._ai_name), "ai")

    def _sys(self, text: str) -> None:
        # System messages have no sender header — they're ambient status lines.
        self._mount_row(Bubble(text, "system"), "system")

    def _start_ai(self) -> None:
        self._current_ai = Bubble("thinking…", "ai", sender=self._ai_name)
        self._mount_row(self._current_ai, "ai")

    def _update_ai(self, text: str) -> None:
        if self._current_ai is not None and text:
            self._current_ai.set_content(text)
            self._chat.scroll_end(animate=False)

    def _finish_ai(self, text: str) -> None:
        if self._current_ai is not None:
            self._current_ai.set_content(text)
            self._current_ai = None
        self._last_reply = text
        self._chat.scroll_end(animate=False)

    # ── actions (keyboard bindings) ─────────────────────────────────────
    def action_history_prev(self) -> None:
        if self.focused is not self._input or not self._input_history:
            return
        if self._history_idx > 0:
            self._history_idx -= 1
        self._input.value = self._input_history[self._history_idx]
        self._input.cursor_position = len(self._input.value)

    def action_history_next(self) -> None:
        if self.focused is not self._input or not self._input_history:
            return
        if self._history_idx < len(self._input_history) - 1:
            self._history_idx += 1
            self._input.value = self._input_history[self._history_idx]
        else:
            self._history_idx = len(self._input_history)
            self._input.value = ""
        self._input.cursor_position = len(self._input.value)

    def action_copy_reply(self) -> None:
        if not self._last_reply:
            self._sys("nothing to copy yet.")
            return
        self._sys(f"copied last reply ({len(self._last_reply)} chars)."
                  if copy_to_clipboard(self._last_reply) else "clipboard unavailable.")

    def action_edit_last(self) -> None:
        """Drop the last user→assistant exchange and put the prompt back
        in the input for editing. Like ChatGPT's pencil — rewinds one turn."""
        if self._chat_busy:
            self._sys("(still thinking — wait for the reply, then edit.)")
            return

        # Find the most recent user message in the model history.
        last_user_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            self._sys("no prior message to edit.")
            return

        # Extract the editable text (handle multimodal parts).
        content = self.messages[last_user_idx].get("content")
        if isinstance(content, list):
            text = next((p.get("text", "") for p in content if p.get("type") == "text"), "")
        else:
            text = str(content or "")

        # Drop the user msg + everything after it from history.
        self.messages = self.messages[:last_user_idx]

        # Drop the tail of the transcript back through (and including) the
        # most recent USER block.
        while self.transcript and not self.transcript[-1].startswith("## USER"):
            self.transcript.pop()
        if self.transcript and self.transcript[-1].startswith("## USER"):
            self.transcript.pop()

        # Remove trailing bubble rows back through the last user row.
        rows = list(self._chat.query(Row))
        for row in reversed(rows):
            is_user = "row-user" in row.classes
            row.remove()
            if is_user:
                break

        self._input.value = text
        self._input.cursor_position = len(text)
        self._input.focus()
        self._sys("editing — tweak and press Enter to resend.")

    def action_toggle_mute(self) -> None:
        if not self.voice:
            return
        self.voice.muted = not self.voice.muted
        self._sys("audio muted." if self.voice.muted else "audio unmuted.")

    def action_new_session(self) -> None:
        # Silent reset — save in the background if there's something to save,
        # then clear the view. No "saving..." / "session saved" chatter.
        if self.session and self.transcript:
            self._save_then_reset()
        else:
            self._reset()

    @work(thread=True)
    def _save_then_reset(self) -> None:
        self._save_and_report()
        self.call_from_thread(self._reset)

    def _reset(self) -> None:
        self._chat.remove_children()
        self.session = None
        self.messages = []
        self.transcript = []
        self._input.focus()

    def action_quit(self) -> None:
        # Double-tap to force-quit: if we're already saving, a second Esc
        # bails out immediately so the user is never trapped.
        if getattr(self, "_quitting", False):
            print("\n[samantha] force-quit (save may be incomplete)")
            self.exit()
            return
        if self.session and self.transcript:
            self._quitting = True
            self._sys("💾 saving + reflecting (Esc again to hard-quit)…")
            self._save_then_exit()
        else:
            self.exit()

    @work(thread=True)
    def _save_then_exit(self) -> None:
        self._save_and_report()
        # Let the user read the status line before the window closes.
        time.sleep(1.5)
        self.call_from_thread(self.exit)

    def _save_and_report(self) -> None:
        if not self.session or not self.transcript:
            print("\n[samantha] no active session — nothing to save.")
            return
        try:
            result = session_mgr.end(
                session_output="\n".join(self.transcript),
                session_meta=self.session, config=CONFIG, project_root=ROOT,
            )
            flags = result.get("flags_found", 0)
            writes = len(result.get("writes") or [])
            reflected = bool(result.get("reflection_run"))
            transcript_name = result.get("transcript", "")
            refl_str = "ran" if reflected else "skipped (short session)"
            # Show ALWAYS — previously we only surfaced on flags>0 OR writes>0,
            # which made short chats look like they weren't saving at all.
            msg = (
                f"✓ saved — transcript: {transcript_name}\n"
                f"  flags: {flags}  writes: {writes}  reflection: {refl_str}"
            )
            self.call_from_thread(self._sys, msg)
            # Also mirror to stdout so the line survives the TUI tearing down.
            print(f"\n[samantha] {msg}")
        except Exception as exc:
            err = f"save failed: {type(exc).__name__}: {exc}"
            try:
                self.call_from_thread(self._sys, err)
            except Exception:
                pass
            print(f"\n[samantha] {err}")

    # ── slash commands ──────────────────────────────────────────────────
    def _dispatch(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        arg = parts[1] if len(parts) > 1 else ""
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            self._sys(f"unknown: /{cmd}")
            return
        try:
            fn(arg)
        except Exception as exc:
            self._sys(f"error: {exc}")

    def cmd_help(self, _: str) -> None:
        self._sys(HELP)

    def cmd_status(self, _: str) -> None:
        m1 = (CONFIG.get("models") or {}).get("model1") or {}
        lines = [f"**model:** {m1.get('provider','?')} / {m1.get('model','?')}"]
        lines.append("**voice:** " + ("on" if self.voice else "off")
                     + ("  (muted)" if (self.voice and self.voice.muted) else ""))
        lines.append("**vision:** " + ("on" if self.video else "off"))
        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is not None:
            lines.append(f"**external vault:** `{ext}` ({obsidian.note_count(ext)} notes, {obsidian.git_head(ext) or 'no git'})")
        else:
            lines.append("**external vault:** off")
        idx = indexer.build(VAULT)
        lines.append(f"**memory:** {len(idx)} nodes")
        self._sys("\n".join(lines))

    def cmd_end(self, _: str) -> None:
        if not self.session:
            self._sys("no active session.")
            return
        self._sys("ending session — reflecting...")
        self._save_then_reset()

    def cmd_cancel(self, _: str) -> None:
        if not self.session:
            self._sys("no active session.")
            return
        self.session = self.messages = self.transcript = None
        self.session = None
        self.messages, self.transcript = [], []
        self._sys("session discarded.")

    def cmd_copy(self, _: str) -> None:
        self.action_copy_reply()

    def cmd_edit(self, _: str) -> None:
        self.action_edit_last()

    def cmd_context(self, _: str) -> None:
        if not self.session:
            self._sys("no active session.")
            return
        files = self.session.get("retrieved_files", [])
        if not files:
            self._sys("(nothing retrieved — bootstrap mode)")
            return
        self._sys("retrieved:\n  " + "\n  ".join(
            str(Path(p).relative_to(VAULT)) for p in files
        ))

    def cmd_show(self, arg: str) -> None:
        if not arg.strip():
            self._sys("usage: /show <node>")
            return
        idx = indexer.build(VAULT)
        if arg not in idx:
            matches = [n for n in idx if arg.lower() in n.lower()]
            if not matches:
                self._sys(f"no node matching '{arg}'")
                return
            arg = matches[0]
        self._sys(f"── {arg} ──\n" + Path(idx[arg]["path"]).read_text(encoding="utf-8"))

    def cmd_identity(self, _: str) -> None:
        p = VAULT / "_identity" / "persona.md"
        self._sys(p.read_text(encoding="utf-8") if p.exists() else "no persona set.")

    cmd_whoami = cmd_identity

    def cmd_index(self, _: str) -> None:
        idx = indexer.build(VAULT)
        sample = list(idx)[:15]
        self._sys(f"{len(idx)} nodes" + (
            "\n  " + ", ".join(sample) + ("..." if len(idx) > 15 else "") if sample else ""
        ))

    def cmd_decay(self, _: str) -> None:
        r = decay.run(
            vault_path=VAULT, lambda_=CONFIG["decay"]["lambda"],
            archive_threshold=CONFIG["decay"]["archive_threshold"],
        )
        self._sys("decay: " + json.dumps(r))

    def cmd_monitor(self, _: str) -> None:
        m = monitor.collect(VAULT)
        triggers = monitor.check_thresholds(m, CONFIG)
        dups = dedup.find_duplicate_candidates(VAULT)
        lines = [
            f"nodes: {m['total_nodes']}  archived: {m['archived']}",
            f"orphans: {m['orphans']}  tags: {m['tag_vocabulary']}",
            f"duplicate candidates: {len(dups)}",
        ]
        if triggers:
            lines.append("triggers:")
            lines.extend(f"  • {t}" for t in triggers)
        self._sys("\n".join(lines))

    def cmd_meta(self, _: str) -> None:
        self._sys("deep reflection running (30-60s)...")
        self._run_meta()

    @work(thread=True)
    def _run_meta(self) -> None:
        try:
            removed = monitor.cleanup_broken(
                VAULT, min_body_chars=int((CONFIG.get("monitor") or {}).get("min_body_chars", 20)),
            )
            reconciled = reflection.reconcile_tensions(VAULT)
            orphans = monitor.find_orphans(VAULT)
            m = monitor.collect(VAULT)
            triggers = monitor.check_thresholds(m, CONFIG)
            if orphans:
                triggers.append(f"orphan_review: {', '.join(orphans[:10])}")
            sample_q = " ".join(n for n, _ in m["top_hubs"][:5])
            files = retrieval.retrieve(VAULT, sample_q, [], CONFIG)
            output, call_log = reflection.deep_with_tools(VAULT, files, m, triggers, CONFIG)
            writes = reflection.apply_writes(
                output, VAULT,
                similarity_threshold=float((CONFIG.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)),
            )
            self.call_from_thread(
                self._sys,
                f"meta done · cleaned: {len(removed)} · reconciled: {len(reconciled)} · "
                f"orphans: {len(orphans)} · tool calls: {len(call_log)} · writes: {len(writes)}",
            )
        except Exception as exc:
            self.call_from_thread(self._sys, f"meta error: {exc}")

    def cmd_history(self, _: str) -> None:
        tdir = VAULT / "_transcripts"
        if not tdir.exists():
            self._sys("(no transcripts yet)")
            return
        lines = []
        for p in sorted(tdir.glob("*.md"), reverse=True)[:15]:
            try:
                stamp = datetime.strptime(p.stem[:19], "%Y-%m-%d-%H%M%S").strftime("%b %d %H:%M")
            except Exception:
                stamp = p.stem[:10]
            lines.append(f"  {stamp}  {p.stem}")
        self._sys("recent transcripts:\n" + "\n".join(lines))

    def cmd_load(self, arg: str) -> None:
        if not arg.strip():
            self._sys("usage: /load <name>")
            return
        p = VAULT / "_transcripts" / f"{arg}.md"
        if not p.exists():
            matches = [x for x in (VAULT / "_transcripts").glob(f"*{arg}*.md")]
            if not matches:
                self._sys(f"no transcript matching '{arg}'")
                return
            p = matches[0]
        self._chat.remove_children()
        self._sys(f"viewing transcript — {p.stem}")
        _, body = frontmatter.read(p)
        role, buf = None, []
        def flush():
            if role and buf:
                text = "\n".join(buf).strip()
                if role == "user":
                    self._user(text)
                else:
                    self._ai(text)
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("## USER"):
                flush(); role, buf = "user", []
            elif s.startswith("## ASSISTANT"):
                flush(); role, buf = "assistant", []
            else:
                buf.append(line)
        flush()

    def cmd_delete(self, arg: str) -> None:
        if not arg.strip():
            self._sys("usage: /delete <name>")
            return
        p = VAULT / "_transcripts" / f"{arg}.md"
        if p.exists():
            p.unlink()
            self._sys(f"deleted {arg}")
        else:
            self._sys(f"no transcript '{arg}'")

    # ── obsidian ────────────────────────────────────────────────────────
    def cmd_obsidian(self, arg: str) -> None:
        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is None:
            self._sys("external vault disabled (set external_vault.path in config.yaml).")
            return
        sub = arg.strip().lower()
        if sub == "recent":
            tail = obsidian.read_audit_tail(ext, 10)
            self._sys("audit:\n" + (tail or "(empty)"))
        elif sub == "diff":
            try:
                out = subprocess.run(
                    ["git", "-C", str(ext), "diff", "HEAD~5"],
                    capture_output=True, text=True, timeout=5,
                )
                self._sys(out.stdout or "(no diff)")
            except Exception as exc:
                self._sys(f"diff failed: {exc}")
        else:
            head = obsidian.git_head(ext) or "no git"
            tail = (obsidian.read_audit_tail(ext, 1) or "(none)").strip()
            self._sys(f"external vault: {ext}\nnotes: {obsidian.note_count(ext)}  git: {head}\nlast: {tail}")

    # ── proactive ───────────────────────────────────────────────────────
    def cmd_whats_up(self, _: str) -> None:
        try:
            cs = proactive.candidates(VAULT, CONFIG)
        except Exception as exc:
            self._sys(f"whats_up error: {exc}")
            return
        if not cs:
            self._sys("(nothing on my mind)")
            return
        lines = ["top candidates (read-only):"]
        for c in cs[:3]:
            reasons = "; ".join(c.get("reasons") or []) or "-"
            lines.append(f"  {c['score']:.2f}  [{c.get('node_type','?')}] {c['node_name']} — {reasons}")
        self._sys("\n".join(lines))

    def cmd_pause(self, arg: str) -> None:
        a = arg.strip().lower()
        if a in ("off", "resume", "clear", ""):
            outreach.clear_pause(VAULT)
            self._sys("proactive: unpaused." if a else "usage: /pause <24h|until date|off>")
            return
        until = outreach.parse_pause_spec(a)
        if until is None:
            self._sys("couldn't parse — try /pause 24h  or  /pause until 2026-05-01")
            return
        outreach.set_pause(VAULT, until)
        self._sys(f"proactive: paused until {until.isoformat(timespec='seconds')}.")

    def cmd_outreach(self, arg: str) -> None:
        if arg.strip().lower() == "status":
            pcfg = (CONFIG.get("proactive") or {})
            ctx = outreach.build_context(VAULT, CONFIG)
            paused = outreach.active_pause(VAULT)
            state = f"paused until {paused}" if paused else "active" if pcfg.get("enabled") else "disabled"
            self._sys(
                f"state: {state}  ·  today: {ctx.get('outreaches_today', 0)}/{pcfg.get('daily_cap', 3)}"
            )
            return
        tail = outreach.tail_log(VAULT, 10)
        self._sys("recent outreach:\n" + (tail or "(none)"))

    def cmd_listen(self, arg: str) -> None:
        """`/listen on` → hands-free. `/listen off` → back to push-to-talk.
        `/listen status` → show VAD + echo-guard state."""
        mode = arg.strip().lower()
        if not self.voice:
            self._sys("voice backend unavailable.")
            return
        if mode == "status":
            s = self.voice.listen_status()
            gate = "🔒 GATED (ignoring mic)" if s["gate_closed"] else "🎤 OPEN (listening)"
            self._sys(
                f"**VAD status**\n\n"
                f"- mode: {'on' if s['listen_mode'] else 'off'}\n"
                f"- gate: {gate}\n"
                f"- muted: {s['muted']}\n"
                f"- tts_busy: {s['tts_busy']}\n"
                f"- queue: {s['tts_queue_size']}\n"
                f"- buffered_samples (speaker still playing): {s['buffered_samples']}\n"
                f"- cooldown left: {s['cooldown_remaining_ms']} ms"
            )
            return
        if mode in ("off", "stop", "pause"):
            self.voice.stop_listening()
            self._sys("🔇 hands-free off. hold ⌥ to talk.")
            return
        if mode in ("", "on", "start", "resume"):
            ok, reason = self.voice.start_listening(self._on_vad_speech)
            self._sys(f"🎤 {reason}" if ok else f"listen failed: {reason}")
            return
        self._sys("usage: /listen [on|off|status]")

    def _on_vad_speech(self, text: str) -> None:
        """Invoked on the VAD thread when a full utterance is transcribed."""
        norm = text.lower().strip().rstrip(".!?,")
        if norm in ("bye", "goodbye", "quit", "exit", "bye bye"):
            self.call_from_thread(self._user, text)
            self.call_from_thread(self.action_quit)
            return
        # Drop utterances captured while a reply is still streaming — VAD
        # shouldn't have fired (TTS gate) but background chatter might slip
        # through. Simpler to ignore than to queue.
        if self._chat_busy:
            return
        # Bounce back to the UI thread — _send mounts widgets.
        self.call_from_thread(self._send, text)

    def cmd_mute(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._sys("usage: /mute <node>")
            return
        self._sys(f"muted {name}" if outreach.set_node_proactive(VAULT, name, False) else f"no node '{name}'")

    def cmd_unmute(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._sys("usage: /unmute <node>")
            return
        self._sys(f"unmuted {name}" if outreach.set_node_proactive(VAULT, name, True) else f"no node '{name}'")

    # ── scheduled reminders (cron) ──────────────────────────────────────
    @work(thread=True)
    def _start_cron(self) -> None:
        """Poll vault/_meta/schedule.json every 30s and fire due entries.
        Daemon-ish worker — lives for the TUI's session; the schedule
        itself is persistent across launches."""
        # Quick first sweep so any missed-while-closed one-shots fire now.
        self._cron_tick()
        while True:
            time.sleep(30)
            try:
                self._cron_tick()
            except Exception as exc:
                # Never crash the worker; just log and keep ticking.
                try:
                    (VAULT / "_meta").mkdir(parents=True, exist_ok=True)
                    with (VAULT / "_meta" / "cron_errors.log").open("a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {type(exc).__name__}: {exc}\n")
                except Exception:
                    pass

    def _cron_tick(self) -> None:
        due = cron_mod.due(VAULT)
        for entry in due:
            msg = entry.get("message") or ""
            label = f"⏰ reminder: {msg}"
            # System bubble in the chat + spoken if audio is on.
            self.call_from_thread(self._sys, label)
            if self.voice and not self.voice.muted and msg:
                self.voice.speak(msg)
            cron_mod.mark_fired(VAULT, entry["id"])

    # ── proactive surface on launch ─────────────────────────────────────
    def _maybe_surface_proactive(self) -> None:
        pcfg = (CONFIG.get("proactive") or {})
        if not pcfg.get("enabled"):
            return
        try:
            cs = proactive.candidates(VAULT, CONFIG)
            ctx = outreach.build_context(VAULT, CONFIG)
            pick = proactive.should_reach_out(cs, ctx, CONFIG)
        except Exception as exc:
            self._sys(f"(proactive: {exc})")
            return
        if not pick:
            return
        self._run_proactive(pick)

    @work(thread=True)
    def _run_proactive(self, pick: dict[str, Any]) -> None:
        try:
            msg = outreach.draft_message(pick, VAULT, CONFIG)
        except Exception as exc:
            self.call_from_thread(self._sys, f"(outreach draft failed: {exc})")
            return
        try:
            outreach.log_outreach(VAULT, pick, msg, delivered=True)
        except Exception:
            pass
        self.call_from_thread(self._ai, msg)
        if self.voice and not self.voice.muted:
            self.voice.speak(clean_for_speech(msg))
        self._proactive_prefix = pick["node_name"]
        self._proactive_node_path = pick.get("node_path")

    # ── PTT (voice + vision) ────────────────────────────────────────────
    def _start_pynput(self) -> None:
        if not HAS_PYNPUT or not self.voice:
            return
        alt_keys = (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
        cmd_keys = (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
        HOLD = 0.4

        def _cmd_fire():
            if self._cmd_pending is None or self._cmd_cancelled or self._ptt_mode is not None:
                return
            self._cmd_pending = None
            if self.video is None:
                self.call_from_thread(self._sys, "⌘ hold — vision disabled.")
                return
            if not self.voice.start_recording():
                return
            ok, reason = self.video.start()
            if ok:
                self._ptt_mode = "vision"
                self.call_from_thread(self._sys, "🎤📷 recording...")
            else:
                self._ptt_mode = "audio"
                self.call_from_thread(self._sys, f"🎤 audio-only · {reason}")

        def on_press(key):
            if self._cmd_pending is not None and key not in cmd_keys:
                self._cmd_cancelled = True
            if self._ptt_mode is not None:
                return
            if key in alt_keys:
                if self.voice.start_recording():
                    self._ptt_mode = "audio"
                    self.call_from_thread(self._sys, "🎤 recording...")
            elif key in cmd_keys:
                self._cmd_pending = time.time()
                self._cmd_cancelled = False
                threading.Timer(HOLD, _cmd_fire).start()

        def on_release(key):
            if key in cmd_keys:
                self._cmd_pending = None
                self._cmd_cancelled = False
            if self._ptt_mode == "audio" and key in alt_keys:
                self._ptt_mode = None
                self._handle_audio_release()
            elif self._ptt_mode == "vision" and key in cmd_keys:
                self._ptt_mode = None
                self._handle_vision_release()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    @work(thread=True)
    def _handle_audio_release(self) -> None:
        text = self.voice.stop_recording() if self.voice else None
        if not text:
            self.call_from_thread(self._sys, "(no speech)")
            return
        norm = text.lower().strip().rstrip(".!?,")
        if norm in ("bye", "goodbye", "quit", "exit", "bye bye"):
            self.call_from_thread(self._user, text)
            self.call_from_thread(self.action_quit)
            return
        self.call_from_thread(self._send, text)

    @work(thread=True)
    def _handle_vision_release(self) -> None:
        text = (self.voice.stop_recording() if self.voice else "") or ""
        frames = self.video.stop(3) if self.video else []
        if not frames:
            self.call_from_thread(self._sys, "(no frames captured)")
            if not text:
                return
            self.call_from_thread(self._send, text)
            return
        prompt = text.strip() or "Describe what you see."
        self.call_from_thread(self._send, prompt, frames)


# ── entrypoint ────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stt-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    p.add_argument("--tts-model", default="mlx-community/chatterbox-turbo-fp16")
    p.add_argument("--ref-audio", default=REF_AUDIO_DEFAULT)
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--camera-index", type=int, default=None)
    args = p.parse_args()

    voice = None
    if HAS_VOICE and not args.no_voice:
        print(f"\n\033[2m[samantha] loading voice stack on Metal...\033[0m\n")
        try:
            voice = VoiceBackend(args.stt_model, args.tts_model, args.ref_audio)
            voice.load(on_status=lambda s: print(f"\033[2m  {s}\033[0m"))
            print(f"\033[92m[samantha] ready\033[0m\n")
        except Exception as exc:
            print(f"\033[91m[samantha] voice load failed — {exc}\033[0m")
            voice = None

    video = None
    if HAS_CV2 and voice is not None:
        idx = args.camera_index if args.camera_index is not None else 0
        video = VideoBackend(idx)

    try:
        Samantha(voice=voice, video=video).run()
    finally:
        if voice:
            voice.shutdown()


if __name__ == "__main__":
    main()
