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
MODEL1_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search my memory for notes related to a topic. Returns the most "
                "relevant nodes as concatenated markdown. Call this when the user "
                "asks about something I might remember. Also call this iteratively "
                "when answering a question that requires chaining multiple facts "
                "(e.g. 'the spouse of the author of X' = one search per hop)."
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
]

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
    return THINK_BLOCK.sub("", text).strip()


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
    """
    out = []
    for m in messages:
        if (
            m.get("role") == "assistant"
            and not (m.get("content") or "").strip()
            and not m.get("tool_calls")
        ):
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

    for _ in range(max_rounds):
        resp = _create_with_retry(
            client,
            max_retries=retries, backoff_base_sec=backoff,
            model=model, max_tokens=max_tokens,
            messages=msgs,
            tools=MODEL1_TOOL_SCHEMAS,
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

    return applied
