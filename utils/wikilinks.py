from __future__ import annotations

import re

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def extract(text: str) -> list[str]:
    out = []
    for match in WIKILINK.findall(text):
        target = match.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            out.append(target)
    return out
