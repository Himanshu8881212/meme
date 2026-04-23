#!/usr/bin/env python3
"""assistant — voice + memory chat. Rich + prompt_toolkit, no layout framework.

The name shown in the chat comes from `vault/_identity/persona.md` —
whatever you set via `python main.py init --persona <name> ...`. This
file is just the runtime; the personality is configurable.

Rich.Console for output (markdown, colors, live streaming).
prompt_toolkit.PromptSession for input (history, editing, async).
asyncio main loop with backgrounds for voice / VAD / cron / reflection.

Hold ⌥ to talk · Hold ⌘ for camera+talk · Type + Enter · /help for commands.
Esc or Ctrl-D to quit (auto-saves).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# ── optional backends ─────────────────────────────────────────────────────
# Aggressive environment-level silencing of library chatter. All three of
# these are known-noisy sources that write to stderr/stdout at import time
# or during model load and would otherwise spill into the chat pane.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Makes prompt_toolkit skip terminal-capability probes that print a CPR
# warning on some terminals.
os.environ.setdefault("PROMPT_TOOLKIT_COLOR_DEPTH", "DEPTH_24_BIT")
warnings.filterwarnings("ignore")

# Python-logging silencer for the library loggers we can't reach via env.
import logging as _logging
for _name in (
    "transformers", "huggingface_hub", "sentence_transformers",
    "mlx", "mlx_audio", "parakeet_mlx", "urllib3", "asyncio",
):
    _logging.getLogger(_name).setLevel(_logging.ERROR)
_logging.getLogger().setLevel(_logging.ERROR)

_stderr_real = sys.stderr
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
    # mlx_audio has two unconditional `print()` calls that fire from its
    # own background thread — impossible to catch with a surrounding
    # dup2 window. Kill them at the source by shadowing the module-level
    # `print` name in each noisy module.
    _noop_print = lambda *_a, **_k: None
    try:
        import mlx_audio.tts.audio_player as _ap
        _ap.print = _noop_print  # "Starting audio stream..."
    except Exception:
        pass
    try:
        from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import (
            flow_matching as _fm,
        )
        _fm.print = _noop_print  # "S3 Token -> Mel Inference..."
    except Exception:
        pass
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
try:
    from silero_vad import load_silero_vad, VADIterator
    HAS_VAD = True
except Exception:
    HAS_VAD = False
sys.stderr = _stderr_real

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Guard strings like "[sigh]" from Rich's *console-markup* parser (which
# eats unknown brackets). Markdown renderer treats `[x]` as a link only
# when followed by `(url)`, so wrapping in Markdown(...) is safe for the
# paralinguistic tag set. Use this helper anywhere we render AI text.
def _render_md(text: str) -> Markdown:
    return Markdown(text, code_theme="monokai")
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# Silence prompt_toolkit's "CPR not supported" warning. The warning
# fires from application.cpr_not_supported_callback only when
# `self.output.responds_to_cpr` is True. Force it to False so the
# warning path short-circuits before writing to the terminal.
try:
    from prompt_toolkit.output import vt100 as _pt_vt100
    if hasattr(_pt_vt100, "Vt100_Output"):
        _pt_vt100.Vt100_Output.responds_to_cpr = False  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    cron as cron_mod, decay, dedup, integrity, monitor, mood as mood_mod,
    obsidian, outreach, proactive, reflection, retrieval, runtime, tool_memory,
)
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

# ── helpers ───────────────────────────────────────────────────────────────
def _fmt_time(ts: datetime | None = None) -> str:
    ts = ts or datetime.now()
    return ts.strftime("%I:%M %p").lstrip("0").lower()


def _parse_transcript(body: str) -> list[dict[str, str]]:
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


def read_identity(vault: Path) -> tuple[str, str] | None:
    """Return (persona_name, user_name) from _identity/persona.md — or
    None if no persona is set. The persona name drives how the assistant
    refers to herself in the chat header — it's not hardcoded anywhere."""
    path = vault / "_identity" / "persona.md"
    if not path.exists():
        return None
    try:
        fm, _ = frontmatter.read(path)
        return (str(fm.get("name") or "assistant"), str(fm.get("user_name") or ""))
    except Exception:
        return None


def copy_to_clipboard(text: str) -> bool:
    cmds = {"darwin": ["pbcopy"], "win32": ["clip"]}
    cmd = cmds.get(sys.platform)
    if cmd is None and sys.platform.startswith("linux"):
        cmd = ["xclip", "-selection", "clipboard"]
    if cmd is None:
        return False
    try:
        subprocess.run(cmd, input=text.encode(), check=True)
        return True
    except Exception:
        return False


# ── TTS / speech cleaners (same allowlist the model sees) ─────────────────
_FLAG_RE = re.compile(
    r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)(?::[^\]]*)?\]",
    re.IGNORECASE,
)
_MD_PATTERNS = [
    (re.compile(r"```[a-zA-Z0-9_+\-]*\n.*?(?:\n```|\Z)", re.DOTALL), " code snippet "),
    (re.compile(r"`([^`\n]+)`"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"\*\*([^*\n]+)\*\*"), r"\1"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"\1"),
    (re.compile(r"__([^_\n]+)__"), r"\1"),
    (re.compile(r"^\s*#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),
]
_SENTENCE_END = re.compile(r"([.!?]+[\)\"']?)(\s+|$)")
CHATTERBOX_TAGS = {
    "laugh", "chuckle", "sigh", "gasp", "cough",
    "sniff", "groan", "shush", "clear throat",
}
_UNKNOWN_TAG_RE = re.compile(r"\[([a-zA-Z][a-zA-Z \-']{0,24})\]")


def _strip_unknown_tags(text: str) -> str:
    def drop(m: re.Match) -> str:
        tag = m.group(1).strip().lower()
        return m.group(0) if tag in CHATTERBOX_TAGS else ""
    return _UNKNOWN_TAG_RE.sub(drop, text)


def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat, repl in _MD_PATTERNS:
        out = pat.sub(repl, out)
    out = _FLAG_RE.sub("", out)
    out = re.sub(
        r"\[(?:NOVEL|REPEAT|CONTRADICTION|SALIENT|HIGH-STAKES|ASSOCIATED|IDENTITY)[^\]]*$",
        "", out, flags=re.IGNORECASE,
    )
    out = _strip_unknown_tags(out)
    try:
        out = mood_mod.filter_reply_by_mood(out, VAULT)
    except Exception:
        pass
    return re.sub(r"\s+", " ", out).strip()


def split_sentence(buf: str) -> tuple[str, str]:
    m = _SENTENCE_END.search(buf)
    if not m:
        return "", buf
    return buf[:m.end()].strip(), buf[m.end():]


class _SlashOnlyCompleter(Completer):
    """Autocompleter that ONLY fires when the line starts with '/'.
    WordCompleter + `pattern` falls back to matching every token when
    the pattern doesn't hit — which pops a giant menu for plain words
    like 'hello'. This fires only for slash-commands."""

    def __init__(self, commands: list[str]):
        self._commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        low = text.lower()
        for cmd in self._commands:
            if cmd.lower().startswith(low):
                yield Completion(cmd, start_position=-len(text))


# ── Voice backend (carried over, VAD + echo-guard + sentence feeder) ──────
class VoiceBackend:
    def __init__(self, stt_model: str, tts_model: str, ref_audio: str):
        self.stt_model_id, self.tts_model_id, self.ref_audio_path = stt_model, tts_model, ref_audio
        self.stt = self.tts = self.player = self.ref_audio = None
        self.tts_queue: queue.Queue[str | None] = queue.Queue()
        self.muted = False
        self.is_recording = False
        self.audio_chunks: list = []
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.audio_stream = None
        self.listen_mode = False
        self._vad_model = None
        self._vad_iter = None
        self._vad_frames: queue.Queue = queue.Queue(maxsize=200)
        self._vad_thread: threading.Thread | None = None
        self._on_speech = None
        self._tts_busy = threading.Event()
        self._tts_cooldown_until = 0.0
        self._vad_stats = {
            "frames_in": 0, "frames_gated": 0, "frames_processed": 0,
            "max_rms_seen": 0.0, "speech_starts": 0,
            "utterances_emitted": 0, "utterances_dropped_rms": 0,
            "utterances_dropped_short": 0,
        }

    def load(self, on_status=None) -> None:
        if not HAS_VOICE:
            return
        if on_status: on_status("loading STT…")
        self.stt = parakeet_mlx.from_pretrained(self.stt_model_id)
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            t.add_audio(mx.zeros((SAMPLE_RATE,), dtype=mx.float32))
        mx.eval(mx.array([0]))
        if on_status: on_status("loading TTS…")
        self.tts = load_tts(self.tts_model_id)
        self.player = AudioPlayer(sample_rate=self.tts.sample_rate, buffer_size=4096)
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

    def _audio_cb(self, indata, frames, time_info, status):
        with self.lock:
            if self.is_recording:
                self.audio_chunks.append(indata.copy())
        if not self.listen_mode:
            return
        self._vad_stats["frames_in"] += 1
        if self._tts_gate():
            self._vad_stats["frames_gated"] += 1
            return
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
        self._tts_busy.set()
        self.tts_queue.put(text)

    def interrupt(self) -> None:
        """Barge-in: stop TTS playback immediately and clear the queue so
        the user can talk over her. Called when PTT fires while she's
        still speaking a long reply."""
        if not HAS_VOICE:
            return
        # Drain the pending-text queue.
        while not self.tts_queue.empty():
            try: self.tts_queue.get_nowait()
            except queue.Empty: break
        # Ask the player to stop whatever is currently rendering.
        if self.player is not None:
            try: self.player.flush()
            except Exception: pass
            try: self.player.stop()
            except Exception: pass
            # Re-arm the output stream so the next speak() works.
            try: self.player.start_stream()
            except Exception: pass
        self._tts_busy.clear()
        self._tts_cooldown_until = time.time() + 0.3

    # How much of a long reply we actually speak aloud. The full text is
    # always shown in the TUI; TTS just cuts off after this many chars
    # so a 20-item list doesn't trap you in 40s of audio you can't
    # interrupt easily.
    TTS_MAX_CHARS = 400

    def speak_sentence_by_sentence(self, text: str) -> None:
        """Feed TTS sentence-by-sentence up to TTS_MAX_CHARS. Short
        generations keep the player's buffer steady and prevent voice
        cracks; the length cap keeps you from being trapped by a long
        reply you can't barge-in on yet."""
        remainder = text
        spoken_chars = 0
        any_spoken = False
        while True:
            sent, remainder = split_sentence(remainder)
            if not sent:
                break
            if not sent.strip():
                continue
            self.speak(sent)
            spoken_chars += len(sent)
            any_spoken = True
            if spoken_chars >= self.TTS_MAX_CHARS:
                return  # stop early — the text version already shows everything
        if remainder.strip() and spoken_chars < self.TTS_MAX_CHARS:
            self.speak(remainder.strip()); any_spoken = True
        if not any_spoken and text.strip():
            self.speak(text[:self.TTS_MAX_CHARS])

    def _tts_worker(self) -> None:
        # NO fd-level dup2 here. dup2 changes fd 1 process-wide; any
        # other thread printing to stdout during TTS generation ends up
        # in /dev/null — which was silently eating _print_ai output on
        # the main thread. The two noisy mlx_audio `print()` calls are
        # already neutralised via module-level monkey-patching at import
        # time, so we don't need the dup2 anymore.
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            self._tts_busy.set()
            try:
                for res in self.tts.generate(
                    text=text, ref_audio=self.ref_audio,
                    verbose=False, stream=True,
                ):
                    if hasattr(res, "audio") and res.audio is not None:
                        with self.gpu_lock:
                            mx.eval(res.audio)
                        self.player.queue_audio(res.audio)
            except Exception as exc:
                try:
                    (VAULT / "_meta").mkdir(parents=True, exist_ok=True)
                    with (VAULT / "_meta" / "tts_errors.log").open("a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] text={text[:120]!r} err={type(exc).__name__}: {exc}\n")
                except Exception:
                    pass
            finally:
                self._tts_busy.clear()
                self._tts_cooldown_until = time.time() + 3.0

    def _tts_gate(self) -> bool:
        playing = False
        try:
            if self.player is not None and self.player.buffered_samples > 0:
                playing = True
        except Exception:
            pass
        if playing:
            self._tts_cooldown_until = max(self._tts_cooldown_until, time.time() + 3.0)
            return True
        if self._tts_busy.is_set():
            return True
        if not self.tts_queue.empty():
            return True
        if time.time() < self._tts_cooldown_until:
            return True
        return False

    def listen_status(self) -> dict:
        try:
            buffered = int(self.player.buffered_samples) if self.player else 0
        except Exception:
            buffered = -1
        return {
            "listen_mode": self.listen_mode, "muted": self.muted,
            "tts_busy": self._tts_busy.is_set(), "tts_queue_size": self.tts_queue.qsize(),
            "buffered_samples": buffered,
            "cooldown_remaining_ms": max(0, int((self._tts_cooldown_until - time.time()) * 1000)),
            "gate_closed": self._tts_gate(),
            **self._vad_stats,
        }

    def start_listening(self, on_speech) -> tuple[bool, str]:
        if not HAS_VAD:
            return False, "silero-vad not installed"
        if not HAS_VOICE:
            return False, "voice stack unavailable"
        if self.listen_mode:
            return True, "already listening"
        if self._vad_model is None:
            try:
                self._vad_model = load_silero_vad()
            except Exception as exc:
                return False, f"VAD model load failed: {exc}"
        # Snappier turn-taking: 500ms of silence is enough to be confident
        # the user stopped. 800ms felt laggy. Threshold 0.55 slightly
        # raises the bar for "is this speech" — fewer false triggers from
        # ambient sound.
        self._vad_iter = VADIterator(
            self._vad_model, threshold=0.55, sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=500,
        )
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

    def _vad_worker(self) -> None:
        speech_buf: list = []
        in_speech = False
        gate_was_closed = True
        min_user_rms = 0.005
        min_utt_rms = 0.008

        def reset_vad():
            if hasattr(self._vad_iter, "reset_states"):
                try: self._vad_iter.reset_states()
                except Exception: pass

        while self.listen_mode:
            if self._tts_gate():
                if not gate_was_closed:
                    gate_was_closed = True
                    speech_buf.clear(); in_speech = False; reset_vad()
                time.sleep(0.03); continue
            if gate_was_closed:
                gate_was_closed = False
                drained = 0
                while not self._vad_frames.empty() and drained < 64:
                    try: self._vad_frames.get_nowait()
                    except queue.Empty: break
                    drained += 1
                reset_vad(); speech_buf.clear(); in_speech = False
                continue
            try:
                frame = self._vad_frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if len(frame) != 512:
                continue
            self._vad_stats["frames_processed"] += 1
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms > self._vad_stats["max_rms_seen"]:
                self._vad_stats["max_rms_seen"] = rms
            try:
                import torch
                tensor = torch.from_numpy(frame.astype(np.float32))
                result = self._vad_iter(tensor)
            except Exception:
                result = None
            if in_speech:
                speech_buf.append(frame)
            if result:
                if "start" in result and not in_speech:
                    if rms < min_user_rms:
                        continue
                    in_speech = True
                    self._vad_stats["speech_starts"] += 1
                    speech_buf.append(frame)
                elif "end" in result and in_speech:
                    in_speech = False
                    audio = np.concatenate(speech_buf).flatten().astype(np.float32)
                    speech_buf = []
                    if len(audio) < SAMPLE_RATE * 0.3:
                        self._vad_stats["utterances_dropped_short"] += 1
                        continue
                    utt_rms = float(np.sqrt(np.mean(audio ** 2)))
                    if utt_rms < min_utt_rms:
                        self._vad_stats["utterances_dropped_rms"] += 1
                        continue
                    try:
                        text = self._transcribe(audio)
                    except Exception:
                        text = ""
                    if text and self._on_speech:
                        self._vad_stats["utterances_emitted"] += 1
                        try: self._on_speech(text)
                        except Exception: pass


# ── Video (camera capture for hold-⌘) ────────────────────────────────────
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
            self.thread.join(timeout=1.0); self.thread = None
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


# ── Paralinguistic addendum ──────────────────────────────────────────────
PARALINGUISTIC_ADDENDUM = """

---

## I have a voice right now — USE IT

A ChatterBox Turbo TTS model renders my cloned voice in real time.
Inline tags become actual vocal events, not markup.

**Exactly these nine tags are supported, nothing else:**
`[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[cough]` `[sniff]` `[groan]`
`[shush]` `[clear throat]`

**Most replies should carry at least one tag** where the beat warrants
it. Default dry-assistant behavior is wrong for me. But obey the mood
rules above — if they cap `[laugh]` to 0, I don't ship `[laugh]`.
"""


HELP = """[bold]Keys[/bold]
  Ctrl-L  listen 🎤     Ctrl-U  mute 🔇     Ctrl-Y  copy last reply
  Hold ⌥  talk          Hold ⌘  camera+talk    Ctrl-C  quit (saves)

[bold]Chat[/bold]
  /new        fresh session          /reflect   save this session now
  /edit       rewind last prompt     /copy      copy last reply
  /history    list transcripts       /load <n>  replay    /delete <n>
  /status     model + voice + memory

[bold]Memory[/bold]
  /show <node>    print a node        /index    list nodes
  /context        files this turn     /identity show self.md
  /decay          age memory now      /monitor  health + dupes
  /meta           weekly vault audit  /tools_audit tool patterns

[bold]Mood & outreach[/bold]
  /mood [refresh|clear]                /whats_up       her top-of-mind
  /pause <24h|off>                     /outreach [status]
  /mute <node>   /unmute <node>

[bold]Obsidian[/bold]
  /obsidian [recent|diff]
"""


# ── ChatApp: the orchestrator ────────────────────────────────────────────
class ChatApp:

    def __init__(self, voice: VoiceBackend | None, video: VideoBackend | None, fresh: bool = False):
        self.voice = voice
        self.video = video
        self._fresh = fresh
        # Default Rich Console — sys.stdout is routed to tty by the
        # silencer, so no special file= needed. This keeps Rich and
        # prompt_toolkit sharing the same stdout, which is what
        # patch_stdout expects.
        self.console = Console()

        self.session: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[str] = []
        self._last_reply: str = ""
        self._last_user_activity: float = time.time()
        self._last_reflection_at: float = 0.0
        self._chat_busy = False
        self._checkpoint_lock = threading.Lock()

        persona = read_identity(VAULT)
        if persona:
            self._ai_name = persona[0].lower() or "assistant"
            self._user_name = (persona[1] or "you").lower()
        else:
            self._ai_name, self._user_name = "assistant", "you"

        # Proactive carry-over
        self._proactive_prefix: str | None = None
        self._proactive_node_path: str | None = None

        # PTT state
        self._ptt_mode: str | None = None
        self._cmd_pending: float | None = None
        self._cmd_cancelled: bool = False
        self._listener = None

        # Input session — prompt_toolkit handles arrow-history + editing.
        hist_path = VAULT / "_meta" / "input_history.txt"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        self._kb = KeyBindings()
        # Slash-command autocomplete — ONLY activates when the line
        # starts with "/". Regular text like "hello" produces no
        # suggestions; typing "/" pops the full command list.
        self._completer = _SlashOnlyCompleter([
            "/help", "/status", "/new", "/edit", "/copy", "/reflect",
            "/history", "/load", "/delete", "/show", "/context",
            "/identity", "/index", "/decay", "/monitor", "/meta",
            "/tools_audit", "/listen", "/listen on", "/listen off",
            "/listen status", "/mood", "/mood refresh", "/mood clear",
            "/obsidian", "/obsidian recent", "/obsidian diff",
            "/whats_up", "/pause", "/pause off", "/outreach",
            "/outreach status", "/mute", "/unmute", "/quit",
        ])
        self._prompt_style = Style.from_dict({
            "bottom-toolbar": "fg:#888888 bg:default",
            "state-idle":      "fg:#888888",
            "state-listening": "fg:#00d0ff bold",
            "state-recording": "fg:#ff5f5f bold",
            "state-thinking":  "fg:#d787ff bold",
            "state-muted":     "fg:#ffd75f",
            "hint":            "fg:#555555",
        })
        # prompt_toolkit uses sys.stdout by default — which the silencer
        # points at /dev/tty. No custom output= arg. Rich shares the same
        # sys.stdout, so patch_stdout can coordinate prints between the
        # two and text chat doesn't collide with prompt rendering.
        # No bottom_toolbar — it interacts badly with patch_stdout when
        # background threads print, adding phantom indentation. State
        # lives in the prompt marker instead.
        # complete_while_typing=False → completions only on Tab. Avoids
        # Enter getting swallowed as "accept completion" vs. "submit".
        self._prompt_session = PromptSession(
            history=FileHistory(str(hist_path)),
            key_bindings=self._kb,
            completer=self._completer,
            complete_while_typing=False,
            style=self._prompt_style,
        )
        self._bind_keys()

        self._loop: asyncio.AbstractEventLoop | None = None

    # ── prompt marker (shows current state in the › itself) ───────────
    def _prompt_marker(self) -> FormattedText:
        """Two-glyph state indicator:
          mic:    🎤 listening     · not listening
          voice:  🔊 unmuted       🔇 muted         💬 no voice backend
        Example: `🎤 🔊 › ` = hands-free on, she'll speak replies.
        """
        if self.voice is None:
            return FormattedText([
                ("class:hint", "💬 › "),
            ])
        mic = ("class:state-listening", "🎤") if self.voice.listen_mode \
            else ("class:hint", "·")
        voc = ("class:state-muted", "🔇") if self.voice.muted \
            else ("class:state-listening", "🔊")
        return FormattedText([
            mic, ("class:hint", " "),
            voc, ("class:hint", " › "),
        ])

    # ── printing ──────────────────────────────────────────────────────
    #
    # Claude-Code-inspired visual grammar: minimal chrome, role marked
    # by a single colored character, no rules between turns, tool
    # activity rendered as dim "branch" lines (⎿) indented under the
    # current speaker, streaming is raw token append (no re-render).
    #
    # Typed user messages are NOT reprinted — prompt_toolkit's submitted
    # prompt line `› hey` already occupies that space, reprinting
    # duplicated it. Voice-transcribed input does get an explicit echo
    # via _print_user since the prompt didn't display it.

    def _print_user(self, text: str) -> None:
        """Symmetric with _print_ai: ●  name  time header + body + blank."""
        name = (self._user_name or "you").lower()
        self.console.print(
            Text.assemble(
                ("● ", "bold bright_blue"),
                (name, "bold bright_blue"),
                (f"  {_fmt_time()}", "dim"),
            )
        )
        self.console.print(Text(text))
        self.console.print()

    def _print_ai(self, text: str) -> None:
        """Header + markdown-rendered body + blank.

        Markdown renders `*italic*`, `**bold**`, lists, code, etc. —
        and leaves `[laugh]`/`[sigh]` literal because `[x]` without
        `(url)` isn't a markdown link. Do NOT use `console.print(str)`
        for the body — that triggers Rich's *console-markup* parser,
        which WILL swallow `[laugh]`. Raw-stdout fallback if Rich
        errors, so the user always sees something."""
        header = f"\033[1;35m● {self._ai_name}\033[0m  \033[2m{_fmt_time()}\033[0m"
        try:
            self.console.print(
                Text.assemble(
                    ("● ", "bold magenta"),
                    (self._ai_name, "bold magenta"),
                    (f"  {_fmt_time()}", "dim"),
                )
            )
            self.console.print(_render_md(text))
            self.console.print()
        except Exception:
            try:
                sys.stdout.write(header + "\n" + text + "\n\n")
                sys.stdout.flush()
            except Exception:
                pass

    def _print_sys(self, text: str) -> None:
        """Branch-style indented line. Chrome: ⎿ + dim text."""
        for line in (text.splitlines() or [""]):
            if line.strip():
                self.console.print(Text("  ⎿ " + line, style="dim"))

    # ── key bindings (for prompt_toolkit) ─────────────────────────────
    def _bind_keys(self) -> None:
        from prompt_toolkit.key_binding.bindings.named_commands import get_by_name  # noqa
        from prompt_toolkit.keys import Keys

        @self._kb.add("c-l")
        def _(event):
            self._toggle_listen()
            event.app.invalidate()  # force redraw so marker updates live

        # NOTE: Ctrl-M ≡ Enter on every terminal (both are \r). Do NOT
        # bind Ctrl-M — it will intercept Enter and block submission.
        # Use Ctrl-U for mute toggle instead; /mute as slash fallback.
        @self._kb.add("c-u")
        def _(event):
            self._toggle_mute()
            event.app.invalidate()

        @self._kb.add("c-y")
        def _(event):
            self._copy_last()

        # Quit — Ctrl-C triggers the save+reflect path. Only one quit
        # key so there's no ambiguity. Escape is NOT bound: with
        # eager=True it intercepts arrow-key ESC sequences.
        @self._kb.add("c-c")
        def _(event):
            event.app.exit(exception=EOFError())

    # ── actions ───────────────────────────────────────────────────────
    def _toggle_listen(self) -> None:
        if not self.voice:
            self._print_sys("voice backend unavailable.")
            return
        # Silent toggle — the prompt marker (🎤 vs ›) already conveys
        # state. Only surface errors that are worth interrupting for.
        if self.voice.listen_mode:
            self.voice.stop_listening()
        else:
            ok, reason = self.voice.start_listening(self._on_vad_speech)
            if not ok:
                self._print_sys(f"listen failed: {reason}")

    def _toggle_mute(self) -> None:
        if not self.voice:
            return
        # Silent toggle. The prompt marker shows 🔇 when muted.
        self.voice.muted = not self.voice.muted

    def _copy_last(self) -> None:
        if not self._last_reply:
            self._print_sys("nothing to copy yet.")
            return
        ok = copy_to_clipboard(self._last_reply)
        self._print_sys(
            f"copied last reply ({len(self._last_reply)} chars)."
            if ok else "clipboard unavailable."
        )

    # ── main chat pipeline ────────────────────────────────────────────
    async def run_chat_turn(
        self, user_text: str, frames: list[str] | None = None,
        source: str = "typed",
    ) -> None:
        """Always render a user header — symmetric with the AI header.
        For typed input, overwrite prompt_toolkit's echoed prompt line
        with the proper header so both sides match."""
        self._last_user_activity = time.time()
        display = user_text + (f"  📷 [{len(frames)}]" if frames else "")
        if source == "typed":
            # Move cursor up one line (over the prompt echo) and clear it.
            try:
                sys.stdout.write("\033[F\033[2K\r")
                sys.stdout.flush()
            except Exception:
                pass
        self._print_user(display)
        self._chat_busy = True
        try:
            await asyncio.to_thread(self._chat_turn_sync, user_text, frames)
        except Exception as exc:
            # Surface errors — silent failure mode was the worst UX.
            self._print_sys(f"⚠ chat error: {type(exc).__name__}: {exc}")
        finally:
            self._chat_busy = False

    def _chat_turn_sync(self, user_text: str, frames: list[str] | None) -> None:
        def _dbg(msg: str) -> None:
            try:
                p = VAULT / "_meta" / "chat_debug.log"
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{datetime.now().isoformat(timespec='seconds')}] "
                        f"{msg}\n"
                    )
            except Exception:
                pass

        _dbg(f"ENTER user_text={user_text[:60]!r} frames={bool(frames)}")
        # Lazy session start — so new-session is just "drop state and
        # start typing again," no explicit open.
        if self.session is None:
            try:
                task = user_text
                if self._proactive_prefix:
                    task = f"reply to proactive: {self._proactive_prefix} — {user_text}"
                self.session = session_mgr.start(
                    task=task, tags=[], config=CONFIG, project_root=ROOT,
                )
                if self.voice is not None:
                    self.session["system_prompt"] = (
                        self.session["system_prompt"] + PARALINGUISTIC_ADDENDUM
                    )
                if self._proactive_node_path:
                    existing = self.session.setdefault("retrieved_files", [])
                    if self._proactive_node_path not in existing:
                        existing.insert(0, self._proactive_node_path)
                self._proactive_prefix = None
                self._proactive_node_path = None
                self.messages = []
                self.transcript = []
            except Exception as exc:
                self._print_sys(f"(session start failed: {exc})")
                return

        # Compose message content — plain string or multimodal list.
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
        content_chunks: list[str] = []
        tts_buf = ""
        spoken = ""

        def _feed_tts(incoming: str) -> None:
            """Stream completed sentences to TTS as tokens arrive. TUI
            text is NOT streamed — we print the whole reply at the end
            to avoid fighting prompt_toolkit for the terminal."""
            nonlocal tts_buf, spoken
            if not (self.voice and not self.voice.muted):
                return
            tts_buf += incoming
            while True:
                sent, rem = split_sentence(tts_buf)
                if not sent:
                    break
                tts_buf = rem
                s = clean_for_speech(reflection.strip_thinking(sent))
                if s and s not in spoken:
                    self.voice.speak(s)
                    spoken += s + " "

        try:
            if agentic:
                VISIBLE_TOOLS = {
                    "obsidian_create", "obsidian_update", "obsidian_delete",
                    "obsidian_rename", "obsidian_link",
                    "capture_camera", "web_search",
                    "set_timer", "schedule_reminder",
                    "cancel_reminder", "mute_self", "unmute_self",
                }
                for kind, payload in reflection.chat_with_tools_stream(
                    role=role, system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG, vault_path=VAULT, max_tokens=1024,
                ):
                    if kind == "tool_call":
                        name = payload.get("name", "?")
                        if name not in VISIBLE_TOOLS:
                            continue
                        args = payload.get("args") or {}
                        label = name
                        if name.startswith("obsidian_"):
                            p = args.get("rel_path") or args.get("old_rel") or ""
                            label = f"{name.replace('obsidian_', '')} {p}".strip()
                        elif name == "capture_camera":
                            label = "looking through the camera"
                        elif name == "web_search":
                            label = f"searching web: {args.get('query', '')}"
                        elif name in ("set_timer", "schedule_reminder"):
                            label = "scheduling reminder"
                        self._print_sys(label)
                    elif kind == "content":
                        content_chunks.append(payload)
                        _feed_tts(payload)
                    elif kind == "final":
                        reply = payload
                if not reply:
                    reply = "".join(content_chunks)
            else:
                full = ""
                for chunk in reflection.chat_stream(
                    role=role, system=self.session["system_prompt"],
                    messages=self.messages[-window:],
                    config=CONFIG, max_tokens=1536,
                ):
                    if not chunk:
                        continue
                    full += chunk
                    _feed_tts(chunk)
                reply = reflection.strip_thinking(full)
        except Exception as exc:
            reply = f"(model error: {exc})"

        if not reply.strip():
            reply = "…"

        _dbg(f"REPLY_READY len={len(reply)} first60={reply[:60]!r}")

        # Render the reply atomically — one Rich print, no mid-turn writes.
        try:
            self._print_ai(reply)
            _dbg("PRINT_AI_OK")
        except Exception as exc:
            _dbg(f"PRINT_AI_FAIL {type(exc).__name__}: {exc}")
            # Last-resort: raw stdout bypass.
            try:
                sys.stdout.write(f"\n{self._ai_name}: {reply}\n\n")
                sys.stdout.flush()
            except Exception:
                pass

        # Flush any tail sentence that didn't hit a terminator mid-stream.
        if tts_buf.strip() and self.voice and not self.voice.muted:
            tail = clean_for_speech(reflection.strip_thinking(tts_buf))
            if tail and tail not in spoken:
                self.voice.speak(tail)

        # Collapse multimodal message to text-only in history.
        if frames and self.messages:
            idx = len(self.messages) - 1
            if isinstance(self.messages[idx].get("content"), list):
                self.messages[idx]["content"] = f"{user_text}  [had attached {len(frames)} frames]"

        self.messages.append({"role": "assistant", "content": reply})
        self.transcript.append(f"## ASSISTANT\n{reply}")
        self._last_reply = reply

    # ── slash commands ────────────────────────────────────────────────
    async def dispatch(self, line: str) -> bool:
        """Return True if it was a slash command (handled)."""
        if not line.startswith("/"):
            return False
        parts = line[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            self._print_sys(f"unknown: /{cmd}")
            return True
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn(arg)
            else:
                fn(arg)
        except Exception as exc:
            self._print_sys(f"error: {exc}")
        return True

    def cmd_help(self, _: str) -> None:
        self.console.print(Panel(HELP, border_style="dim", padding=(0, 1)))

    def cmd_quit(self, _: str) -> None:
        raise EOFError

    def cmd_status(self, _: str) -> None:
        m1 = (CONFIG.get("models") or {}).get("model1") or {}
        lines = [
            f"model:  [cyan]{m1.get('provider','?')}[/cyan] / [cyan]{m1.get('model','?')}[/cyan]",
            f"voice:  {'on' if self.voice else 'off'}" + ("  [dim](muted)[/dim]" if self.voice and self.voice.muted else ""),
            f"vision: {'on' if self.video else 'off'}",
            f"listen: {'on' if (self.voice and self.voice.listen_mode) else 'off'}",
        ]
        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is not None:
            lines.append(f"external vault: [cyan]{ext}[/cyan] ({obsidian.note_count(ext)} notes, {obsidian.git_head(ext) or 'no git'})")
        idx = indexer.build(VAULT)
        lines.append(f"memory: [cyan]{len(idx)}[/cyan] nodes")
        for l in lines:
            self.console.print(l)
        self.console.print()

    def cmd_copy(self, _: str) -> None:
        self._copy_last()

    def cmd_new(self, _: str) -> None:
        if self.session and self.transcript:
            self._print_sys("saving current session in background…")
            threading.Thread(target=self._save_report, daemon=True).start()
        self.session = None
        self.messages = []
        self.transcript = []
        self.console.clear()
        self._print_sys(f"— fresh session with {self._ai_name} —")

    def cmd_reflect(self, _: str) -> None:
        if not self.session or not self.transcript:
            self._print_sys("nothing to reflect on yet.")
            return
        self._print_sys("💾 reflecting now…")
        threading.Thread(target=self._checkpoint_reflect, args=(False,), daemon=True).start()

    def cmd_listen(self, arg: str) -> None:
        mode = arg.lower()
        if not self.voice:
            self._print_sys("voice backend unavailable.")
            return
        if mode == "status":
            s = self.voice.listen_status()
            gate = "🔒 GATED" if s["gate_closed"] else "🎤 OPEN"
            self._print_sys(
                f"VAD: {'on' if s['listen_mode'] else 'off'}  gate: {gate}  muted: {s['muted']}\n"
                f"frames: in={s['frames_in']}  gated={s['frames_gated']}  processed={s['frames_processed']}\n"
                f"peak_rms: {s['max_rms_seen']:.4f}  starts: {s['speech_starts']}  "
                f"uttered: {s['utterances_emitted']}  dropped_short: {s['utterances_dropped_short']}  "
                f"dropped_rms: {s['utterances_dropped_rms']}"
            )
            return
        if mode in ("off", "stop"):
            self.voice.stop_listening()
            self._print_sys("🔇 hands-free off.")
            return
        if mode in ("on", "start", ""):
            if mode == "" and self.voice.listen_mode:
                self.voice.stop_listening()
                self._print_sys("🔇 hands-free off.")
                return
            ok, reason = self.voice.start_listening(self._on_vad_speech)
            self._print_sys(f"🎤 {reason}" if ok else f"listen failed: {reason}")
            return
        self._print_sys("usage: /listen [on|off|status]")

    def cmd_edit(self, _: str) -> None:
        if self._chat_busy:
            self._print_sys("(still thinking — wait, then edit)")
            return
        last_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                last_idx = i; break
        if last_idx is None:
            self._print_sys("no prior message to edit.")
            return
        content = self.messages[last_idx].get("content")
        if isinstance(content, list):
            text = next((p.get("text", "") for p in content if p.get("type") == "text"), "")
        else:
            text = str(content or "")
        self.messages = self.messages[:last_idx]
        while self.transcript and not self.transcript[-1].startswith("## USER"):
            self.transcript.pop()
        if self.transcript and self.transcript[-1].startswith("## USER"):
            self.transcript.pop()
        self._print_sys(f"— editing: [dim]{text}[/dim] —")
        self._queued_edit = text

    def cmd_mood(self, arg: str) -> None:
        sub = arg.lower()
        if sub == "clear":
            self._print_sys("mood cleared." if mood_mod.clear_mood(VAULT) else "no mood file.")
            return
        if sub in ("refresh", "update"):
            m = mood_mod.update_mood(VAULT)
            self._print_sys("mood refreshed." if m else "no labeled episodes yet.")
            return
        snippet = mood_mod.mood_snippet(VAULT)
        if not snippet:
            self._print_sys("mood: (empty) — reflection will fill this over time")
        else:
            self.console.print(Markdown(snippet, code_theme="monokai"))
            self.console.print()

    def cmd_identity(self, _: str) -> None:
        p = VAULT / "_identity" / "self.md"
        if p.exists():
            self.console.print(Markdown(p.read_text(encoding="utf-8")))
            self.console.print()
        else:
            self._print_sys("no self.md yet.")

    def cmd_context(self, _: str) -> None:
        if not self.session:
            self._print_sys("no active session.")
            return
        files = self.session.get("retrieved_files", [])
        if not files:
            self._print_sys("(nothing retrieved)")
            return
        self._print_sys("retrieved:\n  " + "\n  ".join(str(Path(p).relative_to(VAULT)) for p in files))

    def cmd_show(self, arg: str) -> None:
        if not arg:
            self._print_sys("usage: /show <node>")
            return
        idx = indexer.build(VAULT)
        if arg not in idx:
            matches = [n for n in idx if arg.lower() in n.lower()]
            if not matches:
                self._print_sys(f"no node matching '{arg}'")
                return
            arg = matches[0]
        self.console.print(Panel(
            Markdown(Path(idx[arg]["path"]).read_text(encoding="utf-8")),
            title=arg, border_style="dim",
        ))
        self.console.print()

    def cmd_index(self, _: str) -> None:
        idx = indexer.build(VAULT)
        sample = list(idx)[:15]
        self._print_sys(f"{len(idx)} nodes" + (
            "\n  " + ", ".join(sample) + ("..." if len(idx) > 15 else "") if sample else ""
        ))

    def cmd_decay(self, _: str) -> None:
        r = decay.run(
            vault_path=VAULT, lambda_=CONFIG["decay"]["lambda"],
            archive_threshold=CONFIG["decay"]["archive_threshold"],
        )
        self._print_sys("decay: " + json.dumps(r))

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
        self._print_sys("\n".join(lines))

    def cmd_meta(self, _: str) -> None:
        self._print_sys("deep reflection running…")
        threading.Thread(target=self._run_meta_sync, daemon=True).start()

    def _run_meta_sync(self) -> None:
        try:
            min_body = int((CONFIG.get("monitor") or {}).get("min_body_chars", 20))
            self._post("  · integrity scan…")
            issues = integrity.scan(VAULT)
            actions = integrity.repair(VAULT, issues)
            self._post(f"  · {integrity.summary_line(issues, actions)}")
            self._post("  · cleaning stubs…")
            removed = monitor.cleanup_broken(VAULT, min_body_chars=min_body)
            self._post(f"  · cleaned {len(removed)} stub(s). reconciling tensions…")
            reconciled = reflection.reconcile_tensions(VAULT)
            self._post(f"  · reconciled {len(reconciled)}. scanning orphans…")
            orphans = monitor.find_orphans(VAULT)
            m = monitor.collect(VAULT)
            triggers = monitor.check_thresholds(m, CONFIG)
            if orphans:
                triggers.append(f"orphan_review: {', '.join(orphans[:10])}")
            # Pass placeholder rot into the deep reflection as a trigger
            # so the model knows to re-populate bodies it may have
            # corrupted.
            rot = issues.get("placeholder_rot") or []
            if rot:
                rel = [str(Path(r['path']).relative_to(VAULT)) for r in rot[:5]]
                triggers.append(f"placeholder_rot: {', '.join(rel)}")
            self._post(f"  · {len(orphans)} orphan(s). retrieving vault sample…")
            sample_q = " ".join(n for n, _ in m["top_hubs"][:5])
            files = retrieval.retrieve(VAULT, sample_q, [], CONFIG)
            self._post(f"  · retrieved {len(files)} file(s) for context. running deep reflection with tools…")
            t0 = time.time()
            output, call_log = reflection.deep_with_tools(VAULT, files, m, triggers, CONFIG)
            dt = time.time() - t0
            out_len = len(output or "")
            self._post(
                f"  · deep reflection done in {dt:.1f}s "
                f"· {len(call_log)} tool call(s) · {out_len} output chars. applying writes…"
            )
            if out_len < 50:
                self._post(
                    "    ⚠ deep reflection returned nearly empty output — "
                    "either the model skipped the audit (no triggers interesting) "
                    "or the API is degraded. Check vault/_meta/stderr.log."
                )
            writes = reflection.apply_writes(
                output, VAULT,
                similarity_threshold=float((CONFIG.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)),
            )
            self._post(f"  · wrote {len(writes)}. synthesising tool patterns…")
            try:
                audit = tool_memory.audit_and_synthesize(VAULT, CONFIG)
                aw = audit.get("writes") or []
            except Exception:
                aw = []
            self._post(
                f"✓ meta done · cleaned: {len(removed)} · reconciled: {len(reconciled)} · "
                f"orphans: {len(orphans)} · writes: {len(writes)} · tool-procedures: {len(aw)}"
            )
        except Exception as exc:
            self._post(f"meta error: {type(exc).__name__}: {exc}")

    def cmd_tools_audit(self, _: str) -> None:
        self._print_sys("🔍 auditing tool-use log…")
        def run():
            try:
                r = tool_memory.audit_and_synthesize(VAULT, CONFIG)
            except Exception as exc:
                self._print_sys(f"audit error: {exc}"); return
            writes = r.get("writes") or []
            skipped = r.get("skipped")
            if skipped:
                self._print_sys(f"tools-audit: {skipped}")
                return
            lines = [f"✓ {len(writes)} procedure(s) written"]
            for w in writes[:10]:
                lines.append(f"  {w.get('action','?')}  {w.get('path','')}")
            self._print_sys("\n".join(lines))
        threading.Thread(target=run, daemon=True).start()

    def cmd_obsidian(self, arg: str) -> None:
        ext = obsidian.resolve_vault_path(CONFIG)
        if ext is None:
            self._print_sys("external vault disabled.")
            return
        sub = arg.lower()
        if sub == "recent":
            self._print_sys("audit:\n" + (obsidian.read_audit_tail(ext, 10) or "(empty)"))
        elif sub == "diff":
            try:
                o = subprocess.run(["git", "-C", str(ext), "diff", "HEAD~5"],
                                   capture_output=True, text=True, timeout=5)
                self._print_sys(o.stdout or "(no diff)")
            except Exception as exc:
                self._print_sys(f"diff failed: {exc}")
        else:
            head = obsidian.git_head(ext) or "no git"
            tail = (obsidian.read_audit_tail(ext, 1) or "(none)").strip()
            self._print_sys(f"external: {ext}\nnotes: {obsidian.note_count(ext)}  git: {head}\nlast: {tail}")

    def cmd_whats_up(self, _: str) -> None:
        try:
            cs = proactive.candidates(VAULT, CONFIG)
        except Exception as exc:
            self._print_sys(f"error: {exc}"); return
        if not cs:
            self._print_sys("(nothing on my mind)"); return
        lines = ["top candidates (read-only):"]
        for c in cs[:3]:
            reasons = "; ".join(c.get("reasons") or []) or "-"
            lines.append(f"  {c['score']:.2f}  [{c.get('node_type','?')}] {c['node_name']} — {reasons}")
        self._print_sys("\n".join(lines))

    def cmd_pause(self, arg: str) -> None:
        a = arg.lower()
        if a in ("off", "resume", "clear", ""):
            outreach.clear_pause(VAULT)
            self._print_sys("proactive: unpaused." if a else "usage: /pause <24h|until date|off>")
            return
        until = outreach.parse_pause_spec(a)
        if until is None:
            self._print_sys("couldn't parse — try /pause 24h or /pause until 2026-05-01")
            return
        outreach.set_pause(VAULT, until)
        self._print_sys(f"paused until {until.isoformat(timespec='seconds')}")

    def cmd_outreach(self, arg: str) -> None:
        if arg.lower() == "status":
            pcfg = (CONFIG.get("proactive") or {})
            ctx = outreach.build_context(VAULT, CONFIG)
            paused = outreach.active_pause(VAULT)
            state = f"paused until {paused}" if paused else ("active" if pcfg.get("enabled") else "disabled")
            self._print_sys(f"state: {state}  today: {ctx.get('outreaches_today', 0)}/{pcfg.get('daily_cap', 3)}")
            return
        self._print_sys("recent outreach:\n" + (outreach.tail_log(VAULT, 10) or "(none)"))

    def cmd_mute(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._print_sys("usage: /mute <node>"); return
        self._print_sys(f"muted {name}" if outreach.set_node_proactive(VAULT, name, False) else f"no node '{name}'")

    def cmd_unmute(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._print_sys("usage: /unmute <node>"); return
        self._print_sys(f"unmuted {name}" if outreach.set_node_proactive(VAULT, name, True) else f"no node '{name}'")

    def cmd_history(self, _: str) -> None:
        tdir = VAULT / "_transcripts"
        if not tdir.exists():
            self._print_sys("(no transcripts yet)"); return
        lines = []
        for p in sorted(tdir.glob("*.md"), reverse=True)[:15]:
            try:
                stamp = datetime.strptime(p.stem[:19], "%Y-%m-%d-%H%M%S").strftime("%b %d %H:%M")
            except Exception:
                stamp = p.stem[:10]
            lines.append(f"  {stamp}  {p.stem}")
        self._print_sys("recent transcripts:\n" + "\n".join(lines))

    def cmd_load(self, arg: str) -> None:
        if not arg:
            self._print_sys("usage: /load <name>"); return
        p = VAULT / "_transcripts" / f"{arg}.md"
        if not p.exists():
            matches = list((VAULT / "_transcripts").glob(f"*{arg}*.md"))
            if not matches:
                self._print_sys(f"no transcript matching '{arg}'"); return
            p = matches[0]
        self.console.clear()
        self._print_sys(f"— viewing {p.stem} —")
        _, body = frontmatter.read(p)
        for c in _parse_transcript(body):
            (self._print_user if c["role"] == "user" else self._print_ai)(c["content"])

    def cmd_delete(self, arg: str) -> None:
        if not arg:
            self._print_sys("usage: /delete <name>"); return
        p = VAULT / "_transcripts" / f"{arg}.md"
        if p.exists():
            p.unlink()
            self._print_sys(f"deleted {arg}")
        else:
            self._print_sys(f"no transcript '{arg}'")

    # ── pynput PTT ────────────────────────────────────────────────────
    def start_pynput(self) -> None:
        if not HAS_PYNPUT or not self.voice:
            return
        alt_keys = (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
        cmd_keys = (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
        HOLD = 0.4

        def cmd_fire():
            if self._cmd_pending is None or self._cmd_cancelled or self._ptt_mode is not None:
                return
            self._cmd_pending = None
            if self.video is None:
                self._post("⌘ hold — vision disabled.")
                return
            if not self.voice.start_recording():
                return
            ok, reason = self.video.start()
            if ok:
                self._ptt_mode = "vision"
                self._post("🎤📷 recording audio + video…")
            else:
                self._ptt_mode = "audio"
                self._post(f"🎤 audio-only · {reason}")

        def on_press(key):
            if self._cmd_pending is not None and key not in cmd_keys:
                self._cmd_cancelled = True
            if self._ptt_mode is not None:
                return
            if key in alt_keys:
                # Barge-in: if she's still talking from a previous reply,
                # cut her off immediately so you can record without fighting
                # her voice. This is what makes long replies survivable.
                if self.voice and (self.voice._tts_busy.is_set() or
                                   not self.voice.tts_queue.empty()):
                    self.voice.interrupt()
                    self._post("🛑 interrupted")
                if self.voice.start_recording():
                    self._ptt_mode = "audio"
                    self._post("🎤 recording…")
            elif key in cmd_keys:
                self._cmd_pending = time.time()
                self._cmd_cancelled = False
                threading.Timer(HOLD, cmd_fire).start()

        def on_release(key):
            if key in cmd_keys:
                self._cmd_pending = None
                self._cmd_cancelled = False
            if self._ptt_mode == "audio" and key in alt_keys:
                self._ptt_mode = None
                threading.Thread(target=self._handle_audio_release, daemon=True).start()
            elif self._ptt_mode == "vision" and key in cmd_keys:
                self._ptt_mode = None
                threading.Thread(target=self._handle_vision_release, daemon=True).start()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _post(self, msg: str) -> None:
        """Thread-safe: print a system line from any thread."""
        try:
            self._print_sys(msg)
        except Exception:
            pass

    def _handle_audio_release(self) -> None:
        text = self.voice.stop_recording() if self.voice else None
        if not text:
            self._post("(no speech)")
            return
        norm = text.lower().strip().rstrip(".!?,")
        if norm in ("bye", "goodbye", "quit", "exit", "bye bye"):
            self._post("(bye)")
            # Trigger quit from the main loop.
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._async_quit(), self._loop)
            return
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.run_chat_turn(text, None, "voice"), self._loop,
            )

    def _handle_vision_release(self) -> None:
        text = (self.voice.stop_recording() if self.voice else "") or ""
        frames = self.video.stop(3) if self.video else []
        if not frames:
            self._post("(no frames captured)")
            if not text: return
        prompt = text.strip() or "Describe what you see."
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.run_chat_turn(prompt, frames, "voice"), self._loop,
            )

    def _on_vad_speech(self, text: str) -> None:
        self._last_user_activity = time.time()
        norm = text.lower().strip().rstrip(".!?,")
        if norm in ("bye", "goodbye", "quit", "exit", "bye bye"):
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._async_quit(), self._loop)
            return
        if self._chat_busy:
            return
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.run_chat_turn(text, None, "voice"), self._loop,
            )

    async def _async_quit(self) -> None:
        raise EOFError

    # ── background loops ──────────────────────────────────────────────
    def _cron_tick(self) -> None:
        due = cron_mod.due(VAULT)
        for entry in due:
            msg = entry.get("message") or ""
            self._post(f"⏰ reminder: {msg}")
            if self.voice and not self.voice.muted and msg:
                self.voice.speak(msg)
            cron_mod.mark_fired(VAULT, entry["id"])

    async def cron_loop(self) -> None:
        try:
            self._cron_tick()  # catch anything missed while closed
        except Exception:
            pass
        while True:
            await asyncio.sleep(30)
            try:
                self._cron_tick()
            except Exception:
                pass

    # ── background /meta auditor ──────────────────────────────────────
    #
    # Per-session reflection captures *this chat* into memory. /meta is
    # different: it's the vault-wide janitor — cleans stubs, reconciles
    # contradictory nodes, finds orphans, synthesises tool patterns.
    # It's expensive, so it runs on a slow cadence (6h by default) and
    # only when idle so it never interrupts a turn.
    #
    # Last-run timestamp is persisted to disk so a restart doesn't
    # trigger another audit immediately.

    _META_STATE_FILE = "meta_last_run.txt"

    def _load_last_meta_at(self) -> float:
        p = VAULT / "_meta" / self._META_STATE_FILE
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            return 0.0

    def _save_last_meta_at(self, ts: float) -> None:
        p = VAULT / "_meta" / self._META_STATE_FILE
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{ts:.0f}\n", encoding="utf-8")
        except Exception:
            pass

    async def meta_loop(self) -> None:
        mcfg = (CONFIG.get("meta") or {}).get("background") or {}
        interval = int(mcfg.get("interval_sec", 21600))   # 6h default
        min_idle = int(mcfg.get("min_idle_sec", 180))     # 3m idle
        check = min(600, max(60, interval // 10))         # poll 1/10th
        last = self._load_last_meta_at()
        while True:
            await asyncio.sleep(check)
            try:
                if self._chat_busy:
                    continue
                now = time.time()
                if now - last < interval:
                    continue
                if now - self._last_user_activity < min_idle:
                    continue
                self._post("🧹 meta audit starting in background…")
                await asyncio.to_thread(self._run_meta_sync)
                last = time.time()
                self._save_last_meta_at(last)
            except Exception:
                pass

    async def reflection_loop(self) -> None:
        rcfg = (CONFIG.get("reflection") or {}).get("background") or {}
        check = int(rcfg.get("check_interval_sec", 120))
        min_chars = int(rcfg.get("min_transcript_chars", 500))
        min_idle = int(rcfg.get("min_idle_sec", 120))
        min_gap = int(rcfg.get("min_gap_sec", 600))
        while True:
            await asyncio.sleep(check)
            try:
                if not self.session or not self.transcript:
                    continue
                if self._chat_busy:
                    continue
                now = time.time()
                if now - self._last_reflection_at < min_gap:
                    continue
                if now - self._last_user_activity < min_idle:
                    continue
                if sum(len(t) for t in self.transcript) < min_chars:
                    continue
                self._checkpoint_reflect(background=True)
                self._last_reflection_at = time.time()
            except Exception:
                pass

    def _checkpoint_reflect(self, background: bool = False) -> None:
        with self._checkpoint_lock:
            if not self.session or not self.transcript:
                return
            snap = list(self.transcript)
            meta = dict(self.session)
        if background:
            self._post("💾 reflecting in the background…")
        try:
            result = session_mgr.end(
                session_output="\n".join(snap), session_meta=meta,
                config=CONFIG, project_root=ROOT,
            )
        except Exception as exc:
            self._post(f"reflection error: {exc}")
            return
        flags = result.get("flags_found", 0)
        writes = len(result.get("writes") or [])
        refl = "ran" if result.get("reflection_run") else "skipped"
        self._post(f"✓ checkpoint — {flags} flag · {writes} write · reflection: {refl}")
        with self._checkpoint_lock:
            processed = len(snap)
            if processed <= len(self.transcript):
                self.transcript = self.transcript[processed:]
            if len(self.messages) > 6:
                self.messages = self.messages[-6:]

    def _save_report(self) -> None:
        """Final save on /new or exit. Always surfaces the outcome so the
        user can see it worked — also printed to stdout so the message
        survives the TUI teardown at exit time."""
        if not self.session or not self.transcript:
            print("\n[assistant] no active session — nothing to save.")
            return
        try:
            r = session_mgr.end(
                session_output="\n".join(self.transcript), session_meta=self.session,
                config=CONFIG, project_root=ROOT,
            )
            flags = r.get("flags_found", 0)
            writes = len(r.get("writes") or [])
            reflected = bool(r.get("reflection_run"))
            transcript_name = r.get("transcript", "")
            refl = "ran" if reflected else "skipped (short session)"
            msg = (
                f"✓ saved · transcript: {transcript_name} · "
                f"flags: {flags} · writes: {writes} · reflection: {refl}"
            )
            try:
                self._post(msg)
            except Exception:
                pass
            print(f"\n[assistant] {msg}")
        except Exception as exc:
            err = f"save failed: {type(exc).__name__}: {exc}"
            try:
                self._post(err)
            except Exception:
                pass
            print(f"\n[assistant] {err}")

    # ── proactive on startup ──────────────────────────────────────────
    def maybe_surface_proactive(self) -> None:
        pcfg = (CONFIG.get("proactive") or {})
        if not pcfg.get("enabled"):
            return
        try:
            cs = proactive.candidates(VAULT, CONFIG)
            ctx = outreach.build_context(VAULT, CONFIG)
            pick = proactive.should_reach_out(cs, ctx, CONFIG)
        except Exception:
            return
        if not pick:
            return
        try:
            msg = outreach.draft_message(pick, VAULT, CONFIG)
        except Exception:
            return
        try:
            outreach.log_outreach(VAULT, pick, msg, delivered=True)
        except Exception:
            pass
        self._print_ai(msg)
        if self.voice and not self.voice.muted:
            self.voice.speak_sentence_by_sentence(clean_for_speech(msg))
        self._proactive_prefix = pick["node_name"]
        self._proactive_node_path = pick.get("node_path")

    # ── cross-launch replay ───────────────────────────────────────────
    def replay_last_chat(self, max_turns: int = 6) -> None:
        tdir = VAULT / "_transcripts"
        if not tdir.exists():
            return
        latest = sorted(tdir.glob("*.md"), reverse=True)
        if not latest:
            return
        try:
            _, body = frontmatter.read(latest[0])
        except Exception:
            return
        chunks = _parse_transcript(body)
        if not chunks:
            return
        try:
            stamp = datetime.strptime(latest[0].stem[:19], "%Y-%m-%d-%H%M%S").strftime("%b %d · %I:%M %p")
        except Exception:
            stamp = latest[0].stem[:10]
        dropped = max(0, len(chunks) - max_turns)
        tail = chunks[-max_turns:]
        self.console.print(Rule(f"continuing from {stamp}" + (f" ({dropped}+ earlier)" if dropped else ""), style="dim"))
        for c in tail:
            (self._print_user if c["role"] == "user" else self._print_ai)(c["content"])
        self.console.print(Rule("your turn", style="dim"))
        self.console.print()

    # ── main ──────────────────────────────────────────────────────────
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        runtime.set_context(
            app=self, voice=self.voice, video=self.video,
            vault=VAULT, config=CONFIG,
        )

        # Fresh terminal — wipe scrollback + screen + cursor home.
        try:
            sys.stdout.write("\033[3J\033[H\033[2J")
            sys.stdout.flush()
        except Exception:
            pass

        persona = read_identity(VAULT)
        # No banner — the bottom toolbar already shows state, and her
        # first message introduces her. Chrome-free opening.

        # Greet
        if persona:
            name, user = persona
            greet = f"Hi{(' ' + user) if user else ''}."
            self._print_ai(greet)
            if self.voice and not self.voice.muted:
                self.voice.speak(greet)
        else:
            self._print_sys(
                "no persona set — run: "
                "python main.py init --persona <name> --user-name <you>"
            )

        if not self._fresh:
            self.replay_last_chat()

        # Start background tasks
        self.start_pynput()
        self.maybe_surface_proactive()
        asyncio.create_task(self.cron_loop())
        asyncio.create_task(self.reflection_loop())
        asyncio.create_task(self.meta_loop())

        # Input loop
        self._queued_edit: str | None = None
        while True:
            default = self._queued_edit or ""
            self._queued_edit = None
            try:
                with patch_stdout(raw=True):
                    # Callable so prompt_toolkit re-evaluates on every
                    # redraw — Ctrl-L toggles listening and the marker
                    # updates live (no need to submit to see the change).
                    text = await self._prompt_session.prompt_async(
                        self._prompt_marker,
                        default=default,
                    )
            except (EOFError, KeyboardInterrupt):
                break
            text = (text or "").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "bye", ":q"):
                break

            # Slash commands route through dispatch
            if text.startswith("/"):
                if await self.dispatch(text):
                    continue

            # Normal chat turn
            await self.run_chat_turn(text)

        # Save + bye. The stdout mirror in _save_report survives TUI teardown.
        self._print_sys("💾 saving + reflecting — hold on (Esc again to hard-quit)")
        await asyncio.to_thread(self._save_report)
        if self.voice:
            self.voice.shutdown()


# ── entrypoint ────────────────────────────────────────────────────────────
def _silence_all_to_logfile() -> None:
    """Conservative silencer: only redirects Python-level sys.stderr to
    the log. Leaves sys.stdout AND file descriptors 1/2 alone so text
    chat, Rich output, and prompt_toolkit all keep working normally.

    What this catches:
      - prompt_toolkit's CPR warning (Python sys.stderr)
      - transformers deprecation / checkpoint warnings (Python warnings)
      - Any Python library that prints via sys.stderr

    What this does NOT catch:
      - mlx_audio's "S3 Token" prints (C-level fd 1) — per-generate
        redirect in _tts_worker handles those
      - Any C-level stderr writes that bypass Python

    Less aggressive than before, but doesn't break the chat."""
    log_path = VAULT / "_meta" / "stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(str(log_path), "a", encoding="utf-8", buffering=1)
    sys.stderr = log_f
    try:
        sys.__stderr__ = log_f
    except Exception:
        pass


# Back-compat alias.
_silence_stderr_to_logfile = _silence_all_to_logfile


class _FdRedirect:
    """Context manager to redirect file descriptor 1 (stdout) to the
    log file during a noisy block (like voice model loading). Restores
    stdout cleanly on exit so the chat UI works afterwards."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.saved_fd = None
        self.saved_stdout = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_fd = os.dup(1)
        log_fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.dup2(log_fd, 1)
        os.close(log_fd)
        self.saved_stdout = sys.stdout
        sys.stdout = open(str(self.log_path), "a", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, *exc):
        if self.saved_fd is not None:
            os.dup2(self.saved_fd, 1)
            os.close(self.saved_fd)
        if self.saved_stdout is not None:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = self.saved_stdout


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stt-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    p.add_argument("--tts-model", default="mlx-community/chatterbox-turbo-fp16")
    p.add_argument("--ref-audio", default=REF_AUDIO_DEFAULT)
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--camera-index", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Replay the last session's recent turns on startup "
                        "(default: start fresh each launch).")
    p.add_argument("--verbose", action="store_true",
                   help="Don't silence library warnings/logs to the log file.")
    args = p.parse_args()

    voice = None
    if not args.verbose:
        _silence_all_to_logfile()

    if HAS_VOICE and not args.no_voice:
        try:
            voice = VoiceBackend(args.stt_model, args.tts_model, args.ref_audio)
            # Wrap the (noisy) model-load phase in an FD redirect so
            # mlx_audio's "S3 Token" / "Download complete" writes to
            # fd 1 land in the log instead of spilling into the chat.
            if not args.verbose:
                log_path = VAULT / "_meta" / "stderr.log"
                with _FdRedirect(log_path):
                    voice.load()
            else:
                voice.load()
        except Exception as exc:
            # stdout is untouched — use it for visible errors.
            print(f"\033[91m[assistant] voice load failed — {exc}\033[0m")
            voice = None

    video = None
    if HAS_CV2 and voice is not None:
        idx = args.camera_index if args.camera_index is not None else 0
        video = VideoBackend(idx)

    # `fresh` semantics in ChatApp == skip replay. We flip the CLI default
    # so you get a clean chat every launch; --resume brings the old behavior back.
    app = ChatApp(voice=voice, video=video, fresh=not args.resume)
    try:
        asyncio.run(app.run())
    finally:
        if voice:
            voice.shutdown()


if __name__ == "__main__":
    main()
