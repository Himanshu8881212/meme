"""Echo-guard: the VAD gate must stay closed while Samantha's TTS is
playing AND for a cooldown after the speaker goes quiet.

The bug we're guarding against: the last chunk of her own voice leaking
back into the mic, triggering a fake user turn, and sending her into a
self-reply loop.
"""
from __future__ import annotations

import importlib
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def voice():
    """Construct a VoiceBackend instance with HAS_VOICE/HAS_VAD faked off
    so __init__ doesn't try to load any model. We only need the gate
    logic, which is pure Python."""
    from importlib import import_module
    mod = import_module("samantha")
    # Bypass __init__ entirely — we just want the gate methods.
    backend = mod.VoiceBackend.__new__(mod.VoiceBackend)
    # Minimum state needed for _tts_gate + listen_status.
    import queue as _q
    import threading as _th
    backend.tts_queue = _q.Queue()
    backend._tts_busy = _th.Event()
    backend._tts_cooldown_until = 0.0
    backend.player = MagicMock()
    backend.player.buffered_samples = 0
    backend.listen_mode = False
    backend.muted = False
    backend._vad_stats = {
        "frames_in": 0, "frames_gated": 0, "frames_processed": 0,
        "max_rms_seen": 0.0, "speech_starts": 0,
        "utterances_emitted": 0, "utterances_dropped_rms": 0,
        "utterances_dropped_short": 0,
    }
    return backend


def test_gate_closed_while_tts_busy(voice):
    voice._tts_busy.set()
    assert voice._tts_gate() is True


def test_gate_closed_while_queue_has_items(voice):
    voice.tts_queue.put("pending")
    assert voice._tts_gate() is True


def test_gate_closed_while_player_buffered(voice):
    voice.player.buffered_samples = 4096  # speaker still playing
    assert voice._tts_gate() is True


def test_gate_extends_cooldown_while_playing(voice):
    """While buffered_samples > 0, each gate check should push the
    cooldown further into the future. That's the fix for the
    'last chunk leaks' bug."""
    voice.player.buffered_samples = 2048
    voice._tts_cooldown_until = 0.0
    voice._tts_gate()
    c1 = voice._tts_cooldown_until
    assert c1 > time.time() + 2.5  # pushed ≥3s out (padded for OS buffer)
    # Advance time a tiny bit — a second check should re-extend.
    time.sleep(0.05)
    voice._tts_gate()
    c2 = voice._tts_cooldown_until
    assert c2 >= c1


def test_gate_closed_during_cooldown_even_after_player_quiet(voice):
    """Player drains → cooldown window still keeps the gate closed."""
    voice.player.buffered_samples = 0
    voice._tts_busy.clear()
    # Cooldown set 1 second in the future.
    voice._tts_cooldown_until = time.time() + 1.0
    assert voice._tts_gate() is True


def test_gate_opens_when_everything_quiet(voice):
    voice.player.buffered_samples = 0
    voice._tts_busy.clear()
    voice._tts_cooldown_until = time.time() - 1.0  # past
    assert voice._tts_gate() is False


def test_status_snapshot_reflects_state(voice):
    voice.player.buffered_samples = 512
    voice._tts_busy.set()
    voice.listen_mode = True
    s = voice.listen_status()
    assert s["listen_mode"] is True
    assert s["tts_busy"] is True
    assert s["buffered_samples"] == 512
    assert s["gate_closed"] is True
