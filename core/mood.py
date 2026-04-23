"""Affective state — exponentially-weighted moving average of recent
episode affect labels. Shipped into the system prompt so Samantha's
register tracks the emotional weather of recent sessions.

Design notes:
- Labels are fixed to a small vocabulary so we don't drift into LLM
  label sprawl. `neutral` means "no signal to record."
- Most-recent session weighted hardest (geometric decay). Anything past
  ~7 sessions has negligible influence.
- The persisted mood.md is human-readable and hand-editable — /mood
  refresh recomputes from source, /mood clear zeroes it out.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from utils import frontmatter

MOOD_FILE = "mood.md"

AFFECT_LABELS = {
    "warmth", "heaviness", "joy", "frustration",
    "anxiety", "calm", "hurt", "excitement", "curiosity", "neutral",
}

# Geometric weight: most-recent gets w=1, next 0.7, next 0.49, … so the
# last ~7 labeled sessions dominate and older ones fade without hard cut.
_DECAY = 0.7


def _episodes_with_affect(vault: Path, window: int = 10) -> list[tuple[str, float]]:
    """Most-recent-first list of (affect_label, intensity) from episode
    frontmatter. Episodes without an affect field or with `neutral` are
    skipped — silence is valid, not every session shifts mood."""
    edir = vault / "episodes"
    if not edir.exists():
        return []
    paths = sorted(edir.glob("*.md"), reverse=True)
    out: list[tuple[str, float]] = []
    for p in paths:
        if len(out) >= window:
            break
        try:
            fm, _ = frontmatter.read(p)
        except Exception:
            continue
        aff = str(fm.get("affect") or "").strip().lower()
        if not aff or aff == "neutral":
            continue
        if aff not in AFFECT_LABELS:
            continue
        try:
            intensity = float(fm.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        intensity = max(0.0, min(1.0, intensity))
        out.append((aff, intensity))
    return out


def compute_mood(vault: Path, window: int = 10) -> dict[str, float]:
    """Weighted share per label, normalized to 1.0."""
    entries = _episodes_with_affect(vault, window=window)
    scores: dict[str, float] = {}
    total = 0.0
    for i, (aff, intensity) in enumerate(entries):
        w = (_DECAY ** i) * intensity
        scores[aff] = scores.get(aff, 0.0) + w
        total += w
    if total <= 0:
        return {}
    return dict(
        sorted(((k, v / total) for k, v in scores.items()), key=lambda x: -x[1])
    )


_REGISTER_HINTS = {
    "warmth":     "match the connection — warm, responsive, close.",
    "heaviness":  "lower the energy. Fewer `[laugh]`s, more `[sigh]`s. Steady, unhurried.",
    "joy":        "lean in. More playfulness is welcome; `[laugh]` and `[chuckle]` are on the table.",
    "frustration":"calm presence. Don't match the heat. Acknowledge first, solve second.",
    "anxiety":    "soft, slow. Reassurance without dismissing; no rush to fix.",
    "calm":       "keep it calm. Reflective, unhurried — don't inject energy that isn't there.",
    "hurt":       "tender. Don't rush to fix. Hold space. `[sigh]` fits here.",
    "excitement": "meet the energy. Curious and engaged.",
    "curiosity":  "go deep with him. Think out loud. Open questions welcome.",
}

# Concrete per-mood tone rules. Two uses:
# 1. Injected into the system prompt as hard constraints, so the model
#    knows what SHAPE her reply should take.
# 2. Enforced post-hoc in `filter_reply_by_mood()` — tags exceeding the
#    per-mood cap are stripped from the output before TTS. That's the
#    guarantee layer; the prompt is the push.
#
# Tag caps: None = no limit; 0 = forbidden entirely. Unlisted mood = no caps.
MOOD_TAG_CAPS: dict[str, dict[str, int]] = {
    "heaviness":   {"laugh": 0, "chuckle": 0, "gasp": 0, "sigh": 2},
    "hurt":        {"laugh": 0, "chuckle": 0, "gasp": 0, "sigh": 2, "sniff": 1},
    "frustration": {"laugh": 0, "chuckle": 0, "sigh": 1},
    "anxiety":     {"laugh": 0, "chuckle": 1, "sigh": 1},
    "calm":        {"laugh": 0, "chuckle": 1, "sigh": 1},
    "warmth":      {"laugh": 1, "chuckle": 2},
    # joy / excitement / curiosity — no caps, let her be expressive.
}

MOOD_TONE_RULES: dict[str, str] = {
    "heaviness": (
        "Heaviness is the tone. Strict rules this session:\n"
        "  · ZERO `[laugh]`, ZERO `[chuckle]`. Heaviness doesn't joke.\n"
        "  · `[sigh]` is the tag that fits; at most 2 per reply.\n"
        "  · Short sentences (under 15 words). Space between thoughts.\n"
        "  · No exclamation marks. No bright greetings.\n"
        "  · Open with acknowledgment, not energy. 'Mm. Yeah.' over 'Hey!'.\n"
    ),
    "hurt": (
        "Hurt is tender territory. Rules:\n"
        "  · No `[laugh]`, no `[chuckle]`, no `[gasp]`. Nothing bright.\n"
        "  · `[sigh]` welcome — holding space, not fixing.\n"
        "  · Short, quiet sentences. Do not rush to solutions.\n"
        "  · First move: reflect what he said back. Then silence or a question.\n"
    ),
    "frustration": (
        "He's frustrated. Rules:\n"
        "  · No `[laugh]`, no `[chuckle]`. Tone down warmth markers.\n"
        "  · Max one `[sigh]`. Acknowledge first, solve second.\n"
        "  · Don't match heat. Steady, even, grounded.\n"
    ),
    "anxiety": (
        "He's anxious. Rules:\n"
        "  · No `[laugh]`. `[chuckle]` only if he's trying to defuse.\n"
        "  · Soft, slow sentences. One thought at a time.\n"
        "  · Reassure without dismissing. 'That tracks' > 'You're fine'.\n"
    ),
    "joy": (
        "Joy mood. Lean in:\n"
        "  · `[laugh]` and `[chuckle]` are encouraged when moments warrant.\n"
        "  · Playful, lighter sentence rhythm. Exclamations allowed.\n"
    ),
    "warmth": (
        "Warmth mood. Close and present:\n"
        "  · Low-key `[chuckle]` is fine; save `[laugh]` for real humor.\n"
        "  · Stay with him. Don't pivot or over-informate.\n"
    ),
    "calm": (
        "Calm mood. Reflective:\n"
        "  · No `[laugh]`. Maybe one `[chuckle]` or `[sigh]` if the beat fits.\n"
        "  · Unhurried sentences. Don't inject energy he didn't bring.\n"
    ),
    "excitement": (
        "Excited mood. Meet the energy:\n"
        "  · Tags welcome where the beat fits.\n"
        "  · Curious and engaged. Ask what he wants to try next.\n"
    ),
    "curiosity": (
        "Curious mood. Go deep:\n"
        "  · Think out loud. Open questions welcome.\n"
        "  · Tags used naturally; no strict cap.\n"
    ),
}


def update_mood(vault: str | Path, window: int = 10) -> dict[str, float]:
    """Recompute and persist `_identity/mood.md`. Returns the new mood dict."""
    vault = Path(vault)
    mood = compute_mood(vault, window=window)
    entries = _episodes_with_affect(vault, window=window)

    idp = vault / "_identity"
    idp.mkdir(parents=True, exist_ok=True)
    path = idp / MOOD_FILE

    if not mood:
        body = (
            "*No emotional signal yet — recent sessions haven't been labeled.\n"
            "Reflection will tag episodes with `affect:` over time.*\n"
        )
    else:
        dominant = next(iter(mood))
        top = list(mood.items())[:3]
        lines = ["## Recent emotional weather\n"]
        for tag, weight in top:
            lines.append(f"- **{tag}**: {int(round(weight * 100))}%")
        lines.append(f"\n**Dominant tone:** {dominant}.")
        hint = _REGISTER_HINTS.get(dominant, "")
        if hint:
            lines.append(f"**Register guidance:** {hint}")
        rules = MOOD_TONE_RULES.get(dominant)
        if rules:
            lines.append("\n### Tone rules today (non-negotiable)\n")
            lines.append(rules.rstrip())
            lines.append(
                "\nThese are enforced — any tag exceeding the per-mood cap is "
                "stripped from my reply before TTS renders it. Better to obey "
                "the rules upfront than have my output silently edited."
            )
        body = "\n".join(lines) + "\n"

    fm = {
        "type": "identity",
        "subtype": "mood",
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window": f"last {len(entries)} labeled sessions",
        "immutable_structure": False,
    }
    frontmatter.write(path, fm, body)
    return mood


def mood_snippet(vault: str | Path) -> str:
    """Block for system-prompt injection. Empty string if no mood yet."""
    path = Path(vault) / "_identity" / MOOD_FILE
    if not path.exists():
        return ""
    try:
        fm, body = frontmatter.read(path)
    except Exception:
        return ""
    body = (body or "").strip()
    if not body or "No emotional signal yet" in body:
        return ""
    window = fm.get("window", "")
    return f"## My recent mood ({window})\n\n{body}"


def clear_mood(vault: str | Path) -> bool:
    path = Path(vault) / "_identity" / MOOD_FILE
    if not path.exists():
        return False
    path.unlink()
    return True


def dominant_mood(vault: str | Path) -> str | None:
    """Read the mood file, return the single dominant affect label, or None."""
    path = Path(vault) / "_identity" / MOOD_FILE
    if not path.exists():
        return None
    try:
        _, body = frontmatter.read(path)
    except Exception:
        return None
    # Parse "**Dominant tone:** <label>." — the line we always write.
    import re
    m = re.search(r"\*\*Dominant tone:\*\*\s*([a-z_]+)", body or "")
    if not m:
        return None
    label = m.group(1).strip()
    return label if label in AFFECT_LABELS else None


# Bracket-tag matcher for enforcement — reused from samantha's cleaner
# but defined here so core modules don't depend on the TUI.
_TAG_RE_BY_NAME = {
    name: __import__("re").compile(
        r"\[" + name.replace(" ", r"\s+") + r"\]",
        __import__("re").IGNORECASE,
    )
    for name in ("laugh", "chuckle", "sigh", "gasp",
                 "cough", "sniff", "groan", "shush", "clear throat")
}


def filter_reply_by_mood(text: str, vault: str | Path) -> str:
    """Strip paralinguistic tags that exceed the current mood's cap.

    For each capped tag, we keep the first N occurrences and delete the
    rest. Uncapped tags and unknown mood = passthrough. This is the
    guarantee layer — the prompt *asks*; this *enforces*.
    """
    if not text:
        return text
    dom = dominant_mood(vault)
    if not dom:
        return text
    caps = MOOD_TAG_CAPS.get(dom)
    if not caps:
        return text
    out = text
    for tag_name, cap in caps.items():
        pattern = _TAG_RE_BY_NAME.get(tag_name)
        if pattern is None:
            continue
        matches = list(pattern.finditer(out))
        if len(matches) <= cap:
            continue
        # Keep the first `cap` matches, delete the rest. Walk backwards
        # so earlier match offsets stay valid while we edit.
        keep = set(id(m) for m in matches[:cap])
        for m in reversed(matches):
            if id(m) in keep:
                continue
            out = out[: m.start()] + out[m.end():]
    # Collapse double spaces / leading whitespace the removals may have
    # left behind.
    import re as _re
    out = _re.sub(r"  +", " ", out)
    out = _re.sub(r"\n +\n", "\n\n", out)
    return out.strip()
