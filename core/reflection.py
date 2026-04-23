from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

# Mistral's free tier aggressively rate-limits. Retry with exponential backoff
# on 429s so a session doesn't die mid-flight. 3 tries at 2/4/8 seconds.
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 2.0


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate_limited" in msg or "rate limit" in msg


def _create_with_retry(
    client: Any,
    *,
    max_retries: int | None = None,
    backoff_base_sec: float | None = None,
    **kwargs: Any,
) -> Any:
    retries = max_retries if max_retries is not None else _MAX_RETRIES
    backoff = backoff_base_sec if backoff_base_sec is not None else _BACKOFF_BASE_SEC
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limited(exc) or attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _retry_params(config: dict[str, Any]) -> tuple[int, float]:
    r = (config.get("reflection") or {}).get("retry") or {}
    return (
        int(r.get("max_retries", _MAX_RETRIES)),
        float(r.get("backoff_base_sec", _BACKOFF_BASE_SEC)),
    )

from core import tools as vault_tools

# Tools exposed to Model 1 during a regular chat turn. These let the model
# iteratively walk its memory graph across multiple hops — "find the author
# of X" → "find that author's spouse" → "find their citizenship" — instead
# of one-shot retrieval that can only surface one hop of context.
#
# Obsidian tools (below) are ONLY included when `external_vault.path` is set;
# see model1_tool_schemas().
MEMORY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Ranked top-K search over my memory (BM25 + embeddings + graph). "
                "Returns the most relevant nodes as concatenated markdown. "
                "Best for SPECIFIC questions — 'who is X', 'when did Y happen', "
                "chaining facts ('the author of X', then 'their spouse'). "
                "**Do NOT use for 'all of X' or 'everything about Y' questions** "
                "— top-K will silently drop the long tail. For those, use "
                "memory_list followed by memory_summarize instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Topic, name, or concept to search for."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": (
                "Fetch the full body of a specific node by name. Use after "
                "memory_search surfaces a node name you need to read in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Exact or partial node name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_find",
            "description": (
                "List node names whose titles contain a substring. Use when you "
                "know roughly what the node is called but not its exact name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_grep",
            "description": (
                "Find the exact phrase across all of memory — distilled nodes "
                "AND verbatim transcripts. Use for 'what did I say about X' or "
                "'find the line where Y came up'. Returns up to 10 snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string",
                               "description": "Exact phrase to search (case-insensitive)."},
                },
                "required": ["phrase"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_by_date",
            "description": (
                "List transcripts from a date range. Use for 'what did we talk "
                "about on March 3rd' or 'show me last week's sessions'. Returns "
                "transcript names; pair with memory_read to get the full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string",
                                   "description": "YYYY-MM-DD (inclusive)."},
                    "end_date": {"type": "string",
                                 "description": "YYYY-MM-DD (inclusive). Omit for single day."},
                },
                "required": ["start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": (
                "**Use this for aggregate/completeness questions** — anything "
                "like 'everything about X', 'summarize all Y', 'all my Z', "
                "'health history of ...', 'over the past year ...'. Returns "
                "the FULL list of matching node names cheaply (metadata only, "
                "no bodies), so you can then call memory_summarize on them. "
                "This is the RIGHT tool when top-N ranked retrieval would "
                "miss important items. Prefer this over memory_search whenever "
                "the user's question implies 'all' or 'entire'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string",
                            "description": "Filter by tag (case-insensitive)."},
                    "type": {"type": "string",
                             "description": "entity|concept|decision|episode|tension|question|procedure"},
                    "limit": {"type": "integer",
                              "description": "Max results (default 100)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_summarize",
            "description": (
                "Second half of the aggregate-query pattern: feed ≤30 node "
                "names + a focus question, get back ONE distilled paragraph. "
                "A separate sub-call reads every body and condenses them, "
                "so this is how you answer 'summarize all X' without "
                "dumping 30 full nodes into my context. **Workflow:** first "
                "call memory_list → then pass those names here → use the "
                "returned paragraph in your reply. Don't skip this step for "
                "'everything about' / 'full history of' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Node names to summarize (from memory_list).",
                    },
                    "query": {
                        "type": "string",
                        "description": "What the summary should focus on.",
                    },
                },
                "required": ["names", "query"],
            },
        },
    },
]

# Home-assistant-style capabilities: eyes, voice self-control, clock,
# timers, web search. Always registered — each gracefully reports back if
# its runtime backend isn't loaded in the current process.
UTILITY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "capture_camera",
            "description": (
                "Turn the camera on NOW, capture a short burst of frames, "
                "analyze them with a vision model, then close the camera. "
                "Use when the user asks about something physically present — "
                "'what is this', 'describe what you see', 'look at my desk', "
                "'what's on the screen'. Returns a paragraph describing "
                "the scene. Do NOT use just because vision MIGHT help — only "
                "when the user's question clearly needs eyes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to describe / look for.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mute_self",
            "description": (
                "Stop speaking out loud (TTS off). Use when the user says "
                "'stay quiet', 'shush', 'stop talking', 'be silent for N "
                "minutes'. Text replies still appear in chat — only the "
                "audio is suppressed. Optionally auto-unmute after N seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Auto-unmute after this many seconds. Omit for indefinite.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unmute_self",
            "description": "Resume speaking out loud (turn TTS back on).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": (
                "Current wall-clock time + date in the user's local timezone. "
                "Use whenever the user asks for the time, today's date, day "
                "of the week, or whenever a response's correctness depends "
                "on 'now'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": (
                "Schedule a proactive reminder. After `seconds` pass, Samantha "
                "will post a system line AND speak the message aloud. Use for "
                "'remind me in 10 minutes to X', 'set a 5 minute timer', "
                "'wake me in an hour'. The timer is PERSISTED — it shows up "
                "in `list_reminders` and is cancellable via `cancel_reminder` "
                "by its id. Pass the returned id along to the user if they "
                "might want to cancel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "How long to wait before firing."},
                    "message": {"type": "string", "description": "What to say when the timer fires."},
                },
                "required": ["seconds", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": (
                "Schedule a reminder at a specific time or on a recurring "
                "cron schedule. Unlike set_timer (relative seconds), this "
                "survives TUI restarts — it's persisted in the vault and "
                "fires whenever Samantha is running. Provide EITHER `cron` "
                "(standard 5-field) or `once_at` (ISO datetime), not both.\n"
                "Cron examples:\n"
                "  '0 9 * * 1-5'  → every weekday at 9:00\n"
                "  '0 18 * * 1'   → every Monday at 18:00\n"
                "  '0 8 1 * *'    → 8 AM on the 1st of every month\n"
                "  '*/15 * * * *' → every 15 minutes\n"
                "Once_at example: '2026-04-25T15:30'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "What to say when it fires."},
                    "cron": {"type": "string", "description": "Cron expression (recurring)."},
                    "once_at": {"type": "string", "description": "ISO datetime (one-shot)."},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": (
                "List every scheduled reminder in the vault (id, message, "
                "cron/once_at, next_fire). Use when the user asks what's "
                "scheduled, or to find an id before cancelling."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": (
                "Remove a scheduled reminder (or a timer set via `set_timer`) "
                "by its id. If the user asks to cancel but you don't have the "
                "id, call `list_reminders` first to find it — match on the "
                "message text the user references."
            ),
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Free web search via DuckDuckGo. Use for current info beyond "
                "my knowledge cutoff — news, prices, live facts, recent "
                "events. Returns up to `max_results` hits as title / url / "
                "snippet triples. Summarize the findings in your reply; "
                "don't dump the raw URLs at the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "description": "Default 5, max 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

OBSIDIAN_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "obsidian_create",
            "description": (
                "Create a new note in the user's external Obsidian vault. Use "
                "this when the user asks you to capture a thought, draft "
                "something, organize an idea, or create a reference note. This "
                "writes to THEIR notebook, not yours — you will not remember "
                "this action in your own memory. Always call obsidian_list or "
                "obsidian_search first to check for existing notes on the same "
                "topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string",
                                 "description": "Vault-relative path, e.g. 'Projects/Foo.md'."},
                    "body": {"type": "string",
                             "description": "Full markdown body."},
                    "frontmatter": {"type": "object",
                                    "description": "Optional YAML frontmatter."},
                },
                "required": ["rel_path", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_update",
            "description": (
                "Update an existing note in the user's Obsidian vault. Writes "
                "to THEIR notebook, not your memory. Use mode=append to add to "
                "the end (safest for ongoing notes), prepend for the top, or "
                "replace to overwrite entirely."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                    "body": {"type": "string"},
                    "mode": {"type": "string",
                             "enum": ["replace", "append", "prepend"],
                             "description": "Default: replace."},
                },
                "required": ["rel_path", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_read",
            "description": (
                "Read a note from the user's Obsidian vault. Use this when the "
                "user asks 'what did I write about X' or when you need context "
                "from their existing notes before adding to them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                },
                "required": ["rel_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_search",
            "description": (
                "Case-insensitive phrase search across the user's Obsidian "
                "vault. Returns matching lines with paths. Use before creating "
                "a new note to avoid duplicating an existing one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_list",
            "description": (
                "List note paths in the user's Obsidian vault, optionally "
                "filtered to a folder. Use to orient yourself before writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string",
                               "description": "Vault-relative folder, or omit for root."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_link",
            "description": (
                "Append a wikilink to an existing note in the user's Obsidian "
                "vault. Writes to THEIR notebook, not your memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                    "target": {"type": "string",
                               "description": "Wikilink target (without [[ ]])."},
                    "label": {"type": "string",
                              "description": "Optional display label."},
                },
                "required": ["rel_path", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_rename",
            "description": (
                "Rename / move a note in the user's Obsidian vault. Updates "
                "every incoming `[[wikilink]]` across the vault so nothing "
                "breaks. Use when the user asks to rename or move a note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_rel": {"type": "string", "description": "Current relative path (.md optional)."},
                    "new_rel": {"type": "string", "description": "New relative path (.md optional)."},
                },
                "required": ["old_rel", "new_rel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obsidian_delete",
            "description": (
                "Soft-delete a note from the user's Obsidian vault — moves it "
                "to `_trash/` (reversible). Use when the user explicitly asks "
                "to delete / remove / throw out a note. Always confirm the "
                "exact note via `obsidian_list` or `obsidian_search` first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {"type": "string"},
                },
                "required": ["rel_path"],
            },
        },
    },
]


def model1_tool_schemas(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Memory tools + utility tools always; obsidian_* only when
    external_vault.path is set."""
    from core import obsidian as _ob
    vault = _ob.resolve_vault_path(config)
    base = list(MEMORY_TOOL_SCHEMAS) + list(UTILITY_TOOL_SCHEMAS)
    if vault is None:
        return base
    return base + list(OBSIDIAN_TOOL_SCHEMAS)


# Back-compat alias — some callers (and older tests) still reference the old
# static name. The default list matches the old behaviour: memory tools only.
MODEL1_TOOL_SCHEMAS = MEMORY_TOOL_SCHEMAS

ALLOWED_FOLDERS = {
    "entities", "concepts", "decisions", "episodes",
    "tensions", "questions", "procedures", "_meta", "_identity",
}

WRITE_BLOCK = re.compile(
    # Tolerant to single-vs-double angle brackets — magistral sometimes emits
    # `action="..."">` instead of `...">>`. Accept either on both sides.
    r'<<WRITE\s+path="([^"]+)"\s+action="(create|update|delete)">{1,2}\s*'
    r"(.*?)"
    r"\s*<<END>{1,2}",
    re.DOTALL,
)

_TITLE_SPLIT = re.compile(r"[\s_\-]+")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "about", "this", "that",
    "not", "are", "was", "were", "been", "have", "has", "had", "can", "could",
    "should", "would", "will", "all", "any", "some", "one", "two",
}


def _title_tokens(stem: str) -> set[str]:
    return {
        t.lower()
        for t in _TITLE_SPLIT.split(stem)
        if len(t) > 2 and t.lower() not in _STOPWORDS
    }


def _find_similar(
    folder: Path, new_name: str, threshold: float = 0.5,
) -> list[tuple[str, float]]:
    if not folder.exists():
        return []
    new_tokens = _title_tokens(new_name)
    if not new_tokens:
        return []
    similar: list[tuple[str, float]] = []
    for path in folder.glob("*.md"):
        if path.stem == new_name:
            continue
        other = _title_tokens(path.stem)
        if not other:
            continue
        overlap = len(new_tokens & other) / max(len(new_tokens), len(other))
        if overlap >= threshold:
            similar.append((path.stem, overlap))
    similar.sort(key=lambda x: -x[1])
    return similar

THINK_BLOCK = re.compile(
    r"(?:<think>.*?</think>|\[THINK\].*?\[/THINK\]|\[THINKING\].*?\[/THINKING\])",
    re.DOTALL | re.IGNORECASE,
)

# Magistral sometimes echoes tool invocations as literal text in its
# content stream (e.g. `obsidian_list{"folder": "Drafts"}`) instead of
# firing them through the tool-call channel. These leaks bypass the
# tool-dispatch pipeline and spray raw invocation syntax into the
# user-visible reply. Strip them before anything sees the text.
# Pattern matches: word chars + `{` + any chars up to the matching `}`.
# We keep this conservative (requires `{` immediately after name) so
# ordinary prose like `set_timer(5)` or `obsidian_list in the UI`
# isn't affected.
_TOOL_CALL_LEAK = re.compile(
    r"\b[a-z][a-z0-9_]{2,}\s*\{[^{}\n]{0,500}\}",
    re.IGNORECASE,
)


def strip_tool_call_leaks(text: str) -> str:
    if not text:
        return text
    return _TOOL_CALL_LEAK.sub("", text)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Global override. Set MEMORY_BACKEND=echo to force the offline backend on
# every call — useful for testing plumbing without API credits.
BACKEND_OVERRIDE_ENV = "MEMORY_BACKEND"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _format_context(vault_files: list[tuple[str, str]]) -> str:
    if not vault_files:
        return "(vault is empty or no relevant nodes retrieved)"
    return "\n\n".join(f"=== {path} ===\n{content}" for path, content in vault_files)


def strip_thinking(text: str) -> str:
    cleaned = THINK_BLOCK.sub("", text)
    cleaned = strip_tool_call_leaks(cleaned)
    return cleaned.strip()


def _extract_text_only(content: Any) -> str:
    """Return only the user-visible text from a streaming content chunk.

    Magistral returns deltas whose `content` is a list of parts — some are
    `text`, some are `thinking`/`reasoning`. For streaming to TTS we want
    ONLY the text, never the internal reasoning trace. Thinking chunks are
    silently dropped here so the voice TUI can't accidentally speak them.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text") or "")
            # thinking / reasoning chunks: skip entirely
        else:
            kind = getattr(item, "type", None)
            if kind == "text":
                parts.append(getattr(item, "text", "") or "")
    return "".join(parts)


def _normalize_content(content: Any) -> str:
    """Flatten a chat-completion content field to a plain string.

    Magistral (and other reasoning models) return content as a list of parts:
      [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
    Wrap thinking parts in <think> tags so strip_thinking() can remove them
    when the caller wants the final answer only.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            kind = item.get("type")
            if kind == "text":
                parts.append(item.get("text") or "")
            elif kind in ("thinking", "reasoning"):
                think = item.get("thinking") or item.get("reasoning") or item.get("text") or ""
                parts.append(f"<think>{think}</think>")
            else:
                parts.append(item.get("text") or "")
            continue
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
            continue
        think = getattr(item, "thinking", None) or getattr(item, "reasoning", None)
        if think is not None:
            parts.append(f"<think>{think}</think>")
    return "".join(parts)


def _resolve(role: str, config: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    override = os.environ.get(BACKEND_OVERRIDE_ENV)
    spec = config["models"][role]
    provider_name = override or spec["provider"]
    provider = config["providers"][provider_name]
    return provider_name, provider, spec["model"]


_clients: dict[str, Any] = {}


def _get_client(provider_name: str, provider: dict[str, Any]) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed: `pip install openai`")
    if provider_name in _clients:
        return _clients[provider_name]
    api_key = "not-needed"
    # api_key_env may hold either an env var name (preferred) or a literal key.
    # If os.environ has a match, prefer it; otherwise fall back to the literal.
    raw = (provider.get("api_key_env") or "").strip()
    if raw:
        api_key = os.environ.get(raw, raw)
    client = OpenAI(base_url=provider["base_url"], api_key=api_key)
    _clients[provider_name] = client
    return client


def _echo(system: str, messages: list[dict[str, str]]) -> str:
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "(no user message)",
    )
    return (
        f"[ECHO] offline backend active. "
        f"system prompt chars: {len(system)}. "
        f"last user message: {last_user[:200]}"
    )


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mistral (and strict OpenAI-compat providers) reject assistant
    messages with no content AND no tool_calls. This can happen if a prior
    turn's stream produced only reasoning tokens (all dropped by text-only
    extraction) — the empty reply then poisons subsequent turns with 400s.

    Drop those empty assistant messages defensively at the API boundary.
    Handles string content *and* multimodal list content (text+image parts).
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            raw = m.get("content")
            if raw is None:
                is_empty = True
            elif isinstance(raw, list):
                is_empty = not raw
            else:
                is_empty = not str(raw).strip()
            if is_empty and not m.get("tool_calls"):
                continue
        out.append(m)
    return out


def _invoke(role: str, system: str, messages: list[dict[str, str]],
            config: dict[str, Any], max_tokens: int) -> str:
    provider_name, provider, model = _resolve(role, config)
    if provider_name == "echo":
        return _echo(system, messages)

    client = _get_client(provider_name, provider)
    retries, backoff = _retry_params(config)
    safe_messages = _sanitize_messages(list(messages))
    resp = _create_with_retry(
        client,
        max_retries=retries,
        backoff_base_sec=backoff,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, *safe_messages],
    )
    return _normalize_content(resp.choices[0].message.content)


def chat(role: str, system: str, messages: list[dict[str, str]],
         config: dict[str, Any], max_tokens: int = 4096) -> str:
    return _invoke(role, system, messages, config, max_tokens)


def chat_stream(
    role: str,
    system: str,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    max_tokens: int = 4096,
):
    """Yield response text chunks as they arrive from the provider.

    Used by voice_tui.py so TTS can start speaking before the full answer
    is generated. On the echo backend, yields the whole response at once.
    """
    provider_name, provider, model = _resolve(role, config)
    if provider_name == "echo":
        yield _echo(system, messages)
        return

    client = _get_client(provider_name, provider)
    retries, backoff = _retry_params(config)

    safe_messages = _sanitize_messages(list(messages))
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *safe_messages],
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content is None:
                    continue
                # Text-only extraction: thinking/reasoning chunks are silently
                # dropped so the caller (e.g. voice TTS) never speaks them.
                text = _extract_text_only(content)
                if text:
                    yield text
            return
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limited(exc) or attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    if last_exc is not None:
        raise last_exc


def _model1_tool_dispatch(
    vault: Path, name: str, args: dict[str, Any], config: dict[str, Any],
) -> str:
    """Execute a tool call requested by Model 1 during a chat turn."""
    # Local import to avoid circular dep — retrieval imports reflection indirectly.
    from core import retrieval as _r

    if name == "memory_search":
        query = str(args.get("query", ""))
        files = _r.retrieve(vault, query, [], config, include_transcripts=True)
        files = files[:6]
        if not files:
            return "(no matching memory found)"
        blocks = []
        for p, c in files:
            rel = Path(p).relative_to(vault)
            blocks.append(f"=== {rel} ===\n{c[:2000]}")
        return "\n\n".join(blocks)

    if name == "memory_read":
        return vault_tools.read_node(vault, str(args.get("name", "")))

    if name == "memory_find":
        matches = vault_tools.find_by_title_substring(vault, str(args.get("query", "")))
        return "\n".join(matches) if matches else "(no matches)"

    if name == "memory_grep":
        hits = vault_tools.grep_vault(vault, str(args.get("phrase", "")))
        if not hits:
            return "(no matches)"
        return "\n".join(f"{h['path']}:{h['line_no']}: {h['snippet']}" for h in hits)

    if name == "memory_by_date":
        names = vault_tools.transcripts_by_date(
            vault,
            str(args.get("start_date", "")),
            args.get("end_date") or None,
        )
        return "\n".join(names) if names else "(no transcripts in range)"

    if name == "memory_list":
        from utils import indexer as _idx
        tag = (args.get("tag") or "").lower().strip() or None
        ntype = (args.get("type") or "").strip() or None
        limit = int(args.get("limit") or 100)
        idx = _idx.build(vault)
        rows: list[str] = []
        for nm, meta in idx.items():
            if ntype and meta.get("type") != ntype:
                continue
            tags = [str(t).lower() for t in (meta.get("tags") or [])]
            if tag and tag not in tags:
                continue
            imp = meta.get("importance", "?")
            tag_str = ",".join(tags) if tags else "-"
            rows.append(
                f"  {nm}  (type={meta.get('type', '?')}, tags={tag_str}, importance={imp})"
            )
            if len(rows) >= limit:
                break
        if not rows:
            return "(no matching nodes)"
        return f"{len(rows)} node(s):\n" + "\n".join(rows)

    if name.startswith("obsidian_"):
        from core import obsidian as _ob
        ext = _ob.resolve_vault_path(config)
        if ext is None:
            return "(external Obsidian vault is not configured)"
        try:
            if name == "obsidian_create":
                r = _ob.create_note(
                    ext,
                    str(args.get("rel_path", "")),
                    str(args.get("body", "")),
                    frontmatter=args.get("frontmatter") or None,
                    config=config,
                )
            elif name == "obsidian_update":
                r = _ob.update_note(
                    ext,
                    str(args.get("rel_path", "")),
                    str(args.get("body", "")),
                    mode=str(args.get("mode", "replace")),
                    config=config,
                )
            elif name == "obsidian_read":
                return _ob.read_note(ext, str(args.get("rel_path", "")))
            elif name == "obsidian_search":
                hits = _ob.search_notes(ext, str(args.get("query", "")))
                if not hits:
                    return "(no matches)"
                return "\n".join(
                    f"{h['path']}:{h['line_no']}: {h['snippet']}" for h in hits
                )
            elif name == "obsidian_list":
                folder = args.get("folder")
                out = _ob.list_notes(ext, folder if folder else None)
                return "\n".join(out) if out else "(no notes)"
            elif name == "obsidian_link":
                r = _ob.add_wikilink(
                    ext,
                    str(args.get("rel_path", "")),
                    str(args.get("target", "")),
                    label=args.get("label") or None,
                    config=config,
                )
            elif name == "obsidian_rename":
                r = _ob.rename_note(
                    ext,
                    str(args.get("old_rel", "")),
                    str(args.get("new_rel", "")),
                    config=config,
                )
            elif name == "obsidian_delete":
                r = _ob.delete_note(
                    ext,
                    str(args.get("rel_path", "")),
                    config=config,
                )
            else:
                return f"unknown tool: {name}"
        except Exception as exc:
            return f"obsidian error: {exc}"
        if isinstance(r, dict):
            if r.get("ok"):
                return r.get("preview") or f"{name} ok"
            return f"error: {r.get('error', 'unknown')}"
        return str(r)

    if name == "memory_summarize":
        names = args.get("names") or []
        if not isinstance(names, list):
            return "(memory_summarize: 'names' must be a list)"
        query = str(args.get("query", "")).strip()
        if not names:
            return "(memory_summarize: no names provided)"
        # Gather bodies. Cap each to keep the sub-call focused.
        blocks: list[str] = []
        for raw in names[:30]:
            body = vault_tools.read_node(vault, str(raw))
            blocks.append(f"=== {raw} ===\n{body[:3000]}")
        joined = "\n\n".join(blocks)
        sub_system = (
            "You are a focused summarizer. The user will give you a set of "
            "vault nodes and a query. Produce ONE tight paragraph (≤ 220 "
            "words) answering the query using ONLY those nodes.\n\n"
            "Non-negotiable rules:\n"
            "1. **Preserve patterns.** If several nodes describe the same "
            "kind of event (e.g. 3 separate appetite drops, 2 thunderstorm "
            "panics, 4 vet visits), say so explicitly — 'three times in the "
            "past month', 'recurring every spring', 'the second incident in "
            "a week'. NEVER collapse a pattern into a single 'one-time' "
            "event.\n"
            "2. **Keep specific dates and numbers** when the nodes give "
            "them. '32kg, down from 34' is better than 'lost weight'.\n"
            "3. **Flag what's ongoing vs resolved.** If the latest node on "
            "a topic leaves something open ('monitoring', 'messaged vet'), "
            "the summary must reflect that — don't round it up to 'fine'.\n"
            "4. **Use only the supplied nodes.** No invented facts, no "
            "general knowledge filler.\n"
            "5. **No node-name citations** unless the query asked for them."
        )
        sub_user = (
            f"Query: {query}\n\n"
            f"Nodes ({len(blocks)}):\n\n{joined[:24000]}"
        )
        try:
            summary = chat(
                role="model1",
                system=sub_system,
                messages=[{"role": "user", "content": sub_user}],
                config=config,
                max_tokens=768,
            )
        except Exception as exc:
            return f"(summarize failed: {exc})"
        return summary.strip() or "(sub-call returned nothing)"

    # ── utility tools (camera / voice / clock / timer / web) ────────────
    # These reach into the live process via core.runtime — handles are
    # set at TUI mount. Each tool fails gracefully if its backend isn't
    # available (e.g. no camera, no ddgs install).

    if name == "capture_camera":
        from core import runtime as _rt
        video = _rt.get("video")
        if video is None:
            return "(camera unavailable — no video backend in this process)"
        ok, reason = video.start()
        if not ok:
            return f"(camera failed to open: {reason})"
        time.sleep(1.0)  # let the user point it, grab a real burst of frames
        frames = video.stop(3)
        if not frames:
            return "(camera opened but captured no usable frames)"
        question = str(args.get("question") or "Describe what you see.")
        parts: list[dict[str, Any]] = [{"type": "text", "text": question}]
        for url in frames:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        role = "vision" if "vision" in (config.get("models") or {}) else "model1"
        try:
            description = chat(
                role=role,
                system=(
                    "You are a precise vision describer. Answer the user's "
                    "question about the attached image(s) in one tight paragraph. "
                    "Name what's visible concretely; skip filler."
                ),
                messages=[{"role": "user", "content": parts}],
                config=config,
                max_tokens=512,
            )
        except Exception as exc:
            return f"(vision sub-call failed: {exc})"
        return (description or "").strip() or "(vision returned no description)"

    if name == "mute_self":
        from core import runtime as _rt
        voice = _rt.get("voice")
        if voice is None:
            return "(no voice backend — already silent)"
        voice.muted = True
        dur = args.get("duration_seconds")
        if dur:
            try:
                secs = max(1, int(dur))
            except (TypeError, ValueError):
                secs = 0
            if secs:
                def _auto_unmute():
                    v = _rt.get("voice")
                    if v is not None:
                        v.muted = False
                    app = _rt.get("app")
                    if app is not None:
                        try:
                            app.call_from_thread(app._sys, "🔊 auto-unmuted.")
                        except Exception:
                            pass
                import threading as _th
                _th.Timer(float(secs), _auto_unmute).start()
                mins, rem = divmod(secs, 60)
                return f"muted for {mins}m {rem}s — I'll turn sound back on then."
        return "muted. say '/listen on' and ask me to unmute when you want audio back."

    if name == "unmute_self":
        from core import runtime as _rt
        voice = _rt.get("voice")
        if voice is None:
            return "(no voice backend)"
        voice.muted = False
        return "unmuted."

    if name == "current_time":
        from datetime import datetime as _dt
        now = _dt.now().astimezone()
        tz = now.tzname() or "local"
        return now.strftime("%A, %B %d %Y — %I:%M %p ") + tz

    if name == "set_timer":
        # Route relative timers through the same persistent cron store so
        # they show up in list_reminders and are cancellable by id. Before
        # this, set_timer used a bare threading.Timer with no handle —
        # once fired-and-forget, there was no way to cancel.
        from core import cron as _cron
        from datetime import datetime as _dt, timedelta as _td
        try:
            seconds = int(args.get("seconds", 0))
        except (TypeError, ValueError):
            seconds = 0
        message = str(args.get("message") or "timer done")
        if seconds <= 0:
            return "(timer needs a positive number of seconds)"
        fire_at = (_dt.now().astimezone() + _td(seconds=seconds)).isoformat(timespec="seconds")
        r = _cron.add(vault, message, once_at=fire_at)
        if not r.get("ok"):
            return f"error: {r.get('error')}"
        e = r["entry"]
        mins, rem = divmod(seconds, 60)
        return (
            f"timer set (id {e['id']}) — in {mins}m {rem}s I'll remind you: "
            f"{message!r}. Cancel with cancel_reminder(id='{e['id']}')."
        )

    if name == "schedule_reminder":
        from core import cron as _cron
        msg = str(args.get("message") or "").strip()
        cron_expr = args.get("cron") or None
        once_at = args.get("once_at") or None
        r = _cron.add(vault, msg, cron=cron_expr, once_at=once_at)
        if not r.get("ok"):
            return f"error: {r.get('error')}"
        e = r["entry"]
        when = f"cron `{e['cron']}`" if e.get("cron") else f"at {e.get('once_at')}"
        return f"reminder scheduled (id {e['id']}): {when} · {e['message']!r} · next: {e['next_fire']}"

    if name == "list_reminders":
        from core import cron as _cron
        entries = _cron.active(vault)
        if not entries:
            return "(no scheduled reminders)"
        lines = ["scheduled reminders:"]
        for e in entries:
            when = f"cron `{e['cron']}`" if e.get("cron") else f"once {e.get('once_at')}"
            lines.append(f"  {e['id']}  {when}  next: {e.get('next_fire')}  · {e['message']!r}")
        return "\n".join(lines)

    if name == "cancel_reminder":
        from core import cron as _cron
        rid = str(args.get("id") or "").strip()
        if not rid:
            return "(missing id)"
        ok = _cron.remove(vault, rid)
        return f"removed reminder {rid}." if ok else f"no reminder with id {rid}."

    if name == "web_search":
        try:
            from ddgs import DDGS
        except Exception:
            return "(web_search unavailable — pip install ddgs)"
        q = str(args.get("query") or "").strip()
        if not q:
            return "(empty query)"
        try:
            limit = min(max(int(args.get("max_results") or 5), 1), 10)
        except (TypeError, ValueError):
            limit = 5
        try:
            with DDGS() as d:
                results = list(d.text(q, max_results=limit))
        except Exception as exc:
            return f"(web_search error: {exc})"
        if not results:
            return "(no results)"
        lines: list[str] = []
        for r in results:
            title = (r.get("title") or "")[:120]
            href = r.get("href") or r.get("url") or ""
            body = (r.get("body") or "")[:280]
            lines.append(f"— {title}\n  {href}\n  {body}")
        return "\n\n".join(lines)

    return f"unknown tool: {name}"


def chat_with_tools(
    role: str,
    system: str,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    vault_path: str | Path,
    max_tokens: int = 2048,
    max_rounds: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Agentic chat: Model 1 can iteratively call memory tools during the turn.

    Returns (final_text, call_log). Falls back to plain chat() on the echo
    backend so offline tests still work.
    """
    provider_name, provider, model = _resolve(role, config)
    if provider_name == "echo":
        return chat(role, system, messages, config, max_tokens), []

    if max_rounds is None:
        max_rounds = int(
            (config.get("session") or {}).get("max_tool_rounds", 5)
        )

    vault = Path(vault_path)
    client = _get_client(provider_name, provider)
    retries, backoff = _retry_params(config)

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *messages,
    ]
    call_log: list[dict[str, Any]] = []

    schemas = model1_tool_schemas(config)

    for _ in range(max_rounds):
        resp = _create_with_retry(
            client,
            max_retries=retries, backoff_base_sec=backoff,
            model=model, max_tokens=max_tokens,
            messages=msgs,
            tools=schemas,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return strip_thinking(_normalize_content(msg.content)), call_log

        replay_content = strip_thinking(_normalize_content(msg.content))
        msgs.append({
            "role": "assistant",
            "content": replay_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = _model1_tool_dispatch(vault, tc.function.name, args, config)
            except Exception as exc:
                result = f"tool error: {exc}"
            # Record outcome to persistent tool-use log so the next turn's
            # system prompt can show the model what worked / didn't.
            try:
                from core import tool_memory as _tm
                _tm.log_call(vault, tc.function.name, args, str(result))
            except Exception:
                pass
            call_log.append({
                "tool": tc.function.name, "args": args,
                "result_chars": len(str(result)),
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:8000],
            })

    # Exhausted rounds — force a plain answer.
    resp = _create_with_retry(
        client,
        max_retries=retries, backoff_base_sec=backoff,
        model=model, max_tokens=max_tokens,
        messages=msgs + [{
            "role": "user",
            "content": "Max tool rounds reached. Answer now with what you have.",
        }],
    )
    return strip_thinking(_normalize_content(resp.choices[0].message.content)), call_log


def chat_with_tools_stream(
    role: str,
    system: str,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    vault_path: str | Path,
    max_tokens: int = 2048,
    max_rounds: int | None = None,
):
    """Streaming variant of chat_with_tools. Yields events:

      ("tool_call",   {"name": ..., "args": {...}, "result_chars": N})
      ("content",     "<partial text chunk>")
      ("final",       "<full final reply>")     — emitted once at the end

    Each API call is made with stream=True. We accumulate tool_call deltas
    until the round ends, then execute them; content deltas are forwarded
    to the caller immediately. On the non-tool-call round (the final
    answer), content tokens stream live — this is what gives the user
    "tokens appearing as she thinks" UX in agentic mode.
    """
    provider_name, provider, model = _resolve(role, config)
    if provider_name == "echo":
        reply, log = chat_with_tools(
            role=role, system=system, messages=messages,
            config=config, vault_path=vault_path, max_tokens=max_tokens,
            max_rounds=max_rounds,
        )
        for entry in log:
            yield ("tool_call", entry)
        yield ("content", reply)
        yield ("final", reply)
        return

    if max_rounds is None:
        max_rounds = int(
            (config.get("session") or {}).get("max_tool_rounds", 5)
        )
    vault = Path(vault_path)
    client = _get_client(provider_name, provider)
    retries, backoff = _retry_params(config)
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *messages,
    ]
    schemas = model1_tool_schemas(config)

    final_text_accum: list[str] = []

    for _round in range(max_rounds):
        # Accumulators for this round. Tool-call deltas arrive as
        # fragments indexed by tool-call index.
        content_buf: list[str] = []
        tool_call_bufs: dict[int, dict[str, Any]] = {}
        # Attempt loop for rate-limit retries.
        last_exc: Exception | None = None
        stream = None
        for attempt in range(retries):
            try:
                stream = client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    messages=msgs, tools=schemas, tool_choice="auto",
                    stream=True,
                )
                break
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limited(exc) or attempt == retries - 1:
                    raise
                time.sleep(backoff * (2 ** attempt))
        if stream is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("chat stream failed to initialize")

        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            # Text content delta.
            dc = getattr(delta, "content", None)
            if dc:
                text = _extract_text_only(dc)
                if text:
                    content_buf.append(text)
                    # Defer emitting content until we know this round
                    # WASN'T a tool-call round — see end-of-stream block.
            # Tool-call deltas — arrive piecemeal.
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = getattr(tc, "index", 0) or 0
                buf = tool_call_bufs.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""},
                )
                if getattr(tc, "id", None):
                    buf["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        buf["arguments"] += fn.arguments

        # Filter out hallucinated tool calls — Magistral occasionally
        # emits tool_call deltas whose "name" is a flag wrapped in
        # markdown (`[NOVEL: ...](#)` etc.). Those land in our dispatch
        # as garbage and clutter the UI. Keep only names we actually
        # expose in the schema.
        valid_names = {t["function"]["name"] for t in schemas}
        tool_call_bufs = {
            idx: buf for idx, buf in tool_call_bufs.items()
            if (buf.get("name") or "") in valid_names
        }

        if not tool_call_bufs:
            # This was the final round — stream the content now.
            full = "".join(content_buf)
            cleaned = strip_thinking(_normalize_content(full))
            final_text_accum.append(cleaned)
            if cleaned:
                yield ("content", cleaned)
            yield ("final", cleaned)
            return

        # Tool-call round — dispatch and continue looping.
        replay_content = strip_thinking(_normalize_content("".join(content_buf)))
        msgs.append({
            "role": "assistant",
            "content": replay_content,
            "tool_calls": [
                {
                    "id": buf["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": buf["name"] or "",
                        "arguments": buf["arguments"] or "{}",
                    },
                }
                for idx, buf in sorted(tool_call_bufs.items())
            ],
        })

        # Dispatch all tool calls in this round in parallel — when the
        # model calls e.g. memory_list + memory_search together, running
        # them sequentially doubles the latency. Each tool is pure I/O
        # or a model sub-call, so threads are fine.
        from concurrent.futures import ThreadPoolExecutor
        ordered = sorted(tool_call_bufs.items())

        def _run_one(idx_buf):
            idx, buf = idx_buf
            name = buf["name"] or ""
            try:
                args = json.loads(buf["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = _model1_tool_dispatch(vault, name, args, config)
            except Exception as exc:
                result = f"tool error: {exc}"
            try:
                from core import tool_memory as _tm
                _tm.log_call(vault, name, args, str(result))
            except Exception:
                pass
            return idx, buf, name, args, result

        max_workers = min(len(ordered), 6) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run_one, ordered))
        # Preserve original call order for the yielded events + appended msgs.
        for idx, buf, name, args, result in results:
            yield ("tool_call", {
                "name": name, "args": args,
                "result_chars": len(str(result)),
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": buf["id"] or f"call_{idx}",
                "content": str(result)[:8000],
            })

    # Exhausted rounds — one more shot, non-streaming, asking for an answer.
    resp = _create_with_retry(
        client, max_retries=retries, backoff_base_sec=backoff,
        model=model, max_tokens=max_tokens,
        messages=msgs + [{
            "role": "user",
            "content": "Max tool rounds reached. Answer now with what you have.",
        }],
    )
    fallback = strip_thinking(_normalize_content(resp.choices[0].message.content))
    yield ("content", fallback)
    yield ("final", fallback)


def routine(flag_summary: str, vault_files: list[tuple[str, str]],
            session_notes: str, config: dict[str, Any]) -> str:
    prompt = _load_prompt("reflection.md")
    user = (
        f"## Flagged moments\n{flag_summary}\n\n"
        f"## Retrieved vault nodes\n{_format_context(vault_files)}\n\n"
        f"## Session notes / transcript excerpt\n{session_notes}\n"
    )
    raw = _invoke("routine", prompt, [{"role": "user", "content": user}], config, 8192)
    return strip_thinking(raw)


def deep(vault_files: list[tuple[str, str]], metrics: dict[str, Any],
         triggers: list[str], config: dict[str, Any]) -> str:
    prompt = _load_prompt("meta_reflection.md")
    triggers_block = "\n".join(f"- {t}" for t in triggers) or "(no hard triggers — general audit)"
    metrics_block = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    user = (
        f"## Vault metrics\n{metrics_block}\n\n"
        f"## Monitor triggers\n{triggers_block}\n\n"
        f"## Vault sample (hubs, tensions, recent additions)\n{_format_context(vault_files)}\n"
    )
    raw = _invoke("deep", prompt, [{"role": "user", "content": user}], config, 16384)
    return strip_thinking(raw)


def deep_with_tools(
    vault_path: str | Path,
    vault_files: list[tuple[str, str]],
    metrics: dict[str, Any],
    triggers: list[str],
    config: dict[str, Any],
    max_rounds: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if max_rounds is None:
        max_rounds = int(
            (config.get("reflection") or {}).get("max_tool_rounds", 8)
        )
    """Deep reflection with a code runner attached.

    Instead of dumping every computation into the prompt, we expose the vault
    as a set of deterministic tools the model can call when it needs a fact.
    Returns (final_text, tool_call_log). The caller passes final_text to
    apply_writes() as usual.
    """
    provider_name, provider, model = _resolve("deep", config)
    if provider_name == "echo":
        text = deep(vault_files, metrics, triggers, config)
        return text, []

    vault = Path(vault_path)
    prompt = _load_prompt("meta_reflection.md")
    triggers_block = "\n".join(f"- {t}" for t in triggers) or "(no hard triggers — general audit)"
    metrics_block = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    user = (
        f"## Vault metrics\n{metrics_block}\n\n"
        f"## Monitor triggers\n{triggers_block}\n\n"
        f"## Vault sample (hubs, tensions, recent additions)\n{_format_context(vault_files)}\n\n"
        f"You have tools available — use them for any count, list, or lookup "
        f"you would otherwise have to guess. When you're done querying, emit "
        f"your <<WRITE>> blocks as your final message."
    )

    client = _get_client(provider_name, provider)
    retries, backoff = _retry_params(config)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user},
    ]

    call_log: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        resp = _create_with_retry(
            client,
            max_retries=retries,
            backoff_base_sec=backoff,
            model=model,
            max_tokens=16384,
            messages=messages,
            tools=vault_tools.TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return strip_thinking(_normalize_content(msg.content)), call_log

        # Strip reasoning chunks before replaying — Mistral's schema rejects
        # `thinking` parts in a replayed assistant message, and the internal
        # reasoning shouldn't be fed back to the model anyway.
        replay_content = strip_thinking(_normalize_content(msg.content))
        messages.append({
            "role": "assistant",
            "content": replay_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = vault_tools.call(vault, tc.function.name, args)
            call_log.append({"tool": tc.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str)[:8000],
            })

    # Exhausted rounds — force a final non-tool response.
    resp = _create_with_retry(
        client,
        max_retries=retries,
        backoff_base_sec=backoff,
        model=model,
        max_tokens=16384,
        messages=messages + [{
            "role": "user",
            "content": "Max tool rounds reached. Emit your final <<WRITE>> blocks now.",
        }],
    )
    return strip_thinking(_normalize_content(resp.choices[0].message.content)), call_log


def _user_name(vault: Path) -> str | None:
    """Return the user_name from the identity files (persona or self)."""
    for rel in ("_identity/persona.md", "_identity/self.md"):
        p = vault / rel
        if not p.exists():
            continue
        try:
            import yaml as _yaml
            text = p.read_text(encoding="utf-8")
            if not text.startswith("---"):
                continue
            parts = text.split("---", 2)
            if len(parts) < 3:
                continue
            fm = _yaml.safe_load(parts[1]) or {}
            name = fm.get("user_name")
            if name and str(name).lower() not in ("(unknown)", "(unset)", ""):
                return str(name)
        except Exception:
            continue
    return None


def _is_user_as_entity_attempt(rel_path: str, user_name: str | None) -> bool:
    """Detect the 'wrote the user as an entity' anti-pattern.

    Matches `entities/Ryan.md`, `entities/User Ryan.md`, `concepts/Ryan.md`,
    and case variants. Identity belongs in `_identity/self.md`, not here.
    """
    if not user_name:
        return False
    folder, _, filename = rel_path.partition("/")
    if folder not in ("entities", "concepts"):
        return False
    stem = filename.removesuffix(".md").strip()
    for prefix in ("user ", "the user "):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):].strip()
    return stem.lower() == user_name.lower()


_TENSION_SUFFIXES = (
    " discrepancy", " conflict", " contradiction", " dispute",
    " disagreement", " inconsistency", " tension",
)

_RESOLUTION_RE = re.compile(
    r"##\s+(?:Resolution|Resolved|Current state|Current|Latest)"
    r"[^\n]*\n(.+?)(?=\n##\s+|\n---|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_RECONCILE_MARKER = "<!-- reconciled-from-tension -->"


def _extract_entity_name(tension_stem: str) -> str:
    lower = tension_stem.lower()
    for suf in _TENSION_SUFFIXES:
        if lower.endswith(suf):
            return tension_stem[: -len(suf)].strip()
    return tension_stem


def reconcile_tensions(vault_path: str | Path) -> list[dict[str, str]]:
    """Mirror tension-node resolutions back into the affected entity bodies.

    Without this pass, `entities/Olga.md` can say 'Died in Kyiv' (the original
    fact) while `tensions/Olga death discrepancy.md` says 'Resolved: Rodez'.
    Retrieval pulls the entity first; the model sees the wrong fact. This pass
    appends the resolved fact to the entity so it's always visible.
    """
    vault = Path(vault_path)
    tensions_dir = vault / "tensions"
    if not tensions_dir.exists():
        return []

    from utils import frontmatter as _fm

    results: list[dict[str, str]] = []

    for tpath in tensions_dir.glob("*.md"):
        try:
            tfm, tbody = _fm.read(tpath)
        except Exception:
            continue

        m = _RESOLUTION_RE.search(tbody)
        if not m:
            continue
        resolution_text = m.group(1).strip()
        if not resolution_text or len(resolution_text) < 5:
            continue

        entity_name = _extract_entity_name(tpath.stem)
        if not entity_name:
            continue

        # Try to locate the entity. Exact match first, then case-insensitive
        # stem match, then substring.
        target: Path | None = None
        for folder in ("entities", "concepts", "decisions"):
            direct = vault / folder / f"{entity_name}.md"
            if direct.exists():
                target = direct
                break
        if target is None:
            for folder in ("entities", "concepts", "decisions"):
                for cand in (vault / folder).glob("*.md"):
                    if cand.stem.lower() == entity_name.lower():
                        target = cand
                        break
                if target:
                    break
        if target is None:
            for folder in ("entities", "concepts", "decisions"):
                for cand in (vault / folder).glob("*.md"):
                    if (cand.stem.lower() in entity_name.lower()
                            or entity_name.lower() in cand.stem.lower()):
                        target = cand
                        break
                if target:
                    break
        if target is None:
            continue

        try:
            efm, ebody = _fm.read(target)
        except Exception:
            continue

        # Idempotent — don't re-append the same resolution.
        marker_line = f"{_RECONCILE_MARKER} {tpath.stem}"
        if marker_line in ebody:
            continue

        block = (
            f"\n\n## Current state (reconciled)\n"
            f"{marker_line}\n"
            f"{resolution_text}\n"
            f"See [[tensions/{tpath.stem}]] for the contradiction history.\n"
        )
        _fm.write(target, efm, ebody.rstrip() + block)
        results.append({
            "tension": tpath.stem,
            "entity": target.stem,
            "folder": target.parent.name,
        })

    return results


_PLACEHOLDER_PATTERNS = re.compile(
    r"<\s*(?:preserve existing body|preserve existing|increment existing|"
    r"existing body preserved|existing frontmatter preserved|"
    r"remainder of existing body preserved|incremented)\s*>|"
    r"\.\.\. *existing body preserved *\.\.\.",
    re.IGNORECASE,
)


def _looks_like_placeholder_overwrite(content: str) -> bool:
    """True if the content is dominated by template placeholder tokens
    with little real prose. We allow a few placeholders for safety (they
    sometimes appear in frontmatter), but if the real-prose content is
    smaller than the placeholder bulk, this is a broken write."""
    hits = _PLACEHOLDER_PATTERNS.findall(content)
    if not hits:
        return False
    stripped = _PLACEHOLDER_PATTERNS.sub("", content)
    # Drop whitespace and markdown chrome so we measure real content.
    bare = re.sub(r"[\s#*\-\[\]]+", "", stripped)
    return len(bare) < 80


def apply_writes(
    model_output: str,
    vault_path: str | Path,
    *,
    similarity_threshold: float = 0.5,
    reconcile: bool = True,
) -> list[dict[str, str]]:
    vault = Path(vault_path)
    user_name = _user_name(vault)
    applied: list[dict[str, str]] = []
    cleaned = strip_thinking(model_output)

    for match in WRITE_BLOCK.finditer(cleaned):
        rel_path, action, content = match.group(1), match.group(2), match.group(3)
        folder = rel_path.split("/", 1)[0]
        if folder not in ALLOWED_FOLDERS:
            applied.append({"path": rel_path, "action": "rejected", "reason": f"bad folder: {folder}"})
            continue

        target = (vault / rel_path).resolve()
        try:
            target.relative_to(vault.resolve())
        except ValueError:
            applied.append({"path": rel_path, "action": "rejected", "reason": "path escapes vault"})
            continue

        # Personality is immutable. Reflection may NEVER touch persona.md —
        # only _identity/self.md, which accumulates relationship facts.
        if rel_path == "_identity/persona.md":
            applied.append({
                "path": rel_path, "action": "rejected",
                "reason": "persona is immutable — only _identity/self.md is writable",
            })
            continue

        # The user is never a regular memory node. Facts about them go to
        # _identity/self.md. Block entities/<user>.md, concepts/<user>.md,
        # and the "User <name>" variants the model keeps trying.
        if _is_user_as_entity_attempt(rel_path, user_name):
            applied.append({
                "path": rel_path, "action": "rejected",
                "reason": f"user '{user_name}' belongs in _identity/self.md, not {rel_path}",
            })
            continue

        if action == "delete":
            if target.exists():
                target.unlink()
            applied.append({"path": rel_path, "action": "deleted"})
            continue

        # Guard against the "placeholder overwrite" failure mode. The
        # reflection prompt's examples use tokens like
        # `<preserve existing body>`, `... existing body preserved ...`,
        # `<existing frontmatter preserved>`, `<increment existing>`.
        # If the model emits those verbatim instead of the real final
        # content, writing them to disk nukes the prior file. Reject
        # these so the target keeps its last good state. Also applies
        # to `<increment existing>` frontmatter placeholders.
        if _looks_like_placeholder_overwrite(content):
            applied.append({
                "path": rel_path, "action": "rejected",
                "reason": "content is mostly placeholders (e.g. <preserve existing body>)",
            })
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")

        entry: dict[str, str] = {"path": rel_path, "action": action}
        if action == "create":
            similar = _find_similar(target.parent, target.stem, similarity_threshold)
            if similar:
                entry["warning"] = "may duplicate: " + ", ".join(
                    f"{n} ({int(o*100)}%)" for n, o in similar[:3]
                )
        applied.append(entry)

    if reconcile:
        reconciled = reconcile_tensions(vault)
        for r in reconciled:
            applied.append({
                "path": f"{r['folder']}/{r['entity']}.md",
                "action": "reconciled",
                "source_tension": r["tension"],
            })

    # Refresh the semantic index if any write actually landed — new/changed
    # nodes are otherwise invisible to paraphrase retrieval until a manual
    # rebuild. Lazy: only re-encodes changed hashes. Silent on failure so
    # an embeddings-less install still works.
    if applied and any(a.get("action") in ("create", "update", "reconciled", "deleted")
                       for a in applied):
        # Invalidate the indexer cache — writes just happened, the next
        # retrieval must see them. Without this, cached metadata would
        # lag by one turn.
        try:
            from utils import indexer as _idx
            _idx.invalidate(vault)
        except Exception:
            pass
        try:
            from core import embeddings as _emb
            if _emb.is_available():
                _emb.build_index(vault)
        except Exception:
            pass
        try:
            from core import mood as _mood
            _mood.update_mood(vault)
        except Exception:
            pass

    return applied
