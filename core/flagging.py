from __future__ import annotations

import re

FLAG_TYPES = ("NOVEL", "REPEAT", "CONTRADICTION", "SALIENT", "HIGH-STAKES", "ASSOCIATED", "IDENTITY")
FLAG_RE = re.compile(
    r"\[(" + "|".join(FLAG_TYPES) + r")(?::\s*([^\]]+))?\]",
    re.IGNORECASE,
)


def extract(text: str) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for match in FLAG_RE.finditer(text):
        kind = match.group(1).upper()
        payload = (match.group(2) or "").strip()
        start = max(match.start() - 120, 0)
        end = min(match.end() + 120, len(text))
        context = text[start:end].strip()
        flags.append({"type": kind, "payload": payload, "context": context})
    return flags


def summarize(flags: list[dict[str, str]]) -> str:
    if not flags:
        return "(no flags)"
    lines = []
    for f in flags:
        head = f["type"]
        if f["payload"]:
            head = f"{head}: {f['payload']}"
        lines.append(f"- [{head}]\n  {f['context']}")
    return "\n".join(lines)
