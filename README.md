# meme

**A memory system with a point of view.**

Most "AI memory" is a searchable log of what was said. This is something
else: a system that turns each conversation into notes the assistant
itself has written — linked, decayed, contradicted, consolidated — so the
next conversation begins with a **mind**, not a transcript search.

The question this system is built to answer is not *"what did I say on
March 3rd?"* (though it can answer that too) — it's *"what do you
think, and why, and how did you come to think it?"*

---

## What this is for

You are talking to an AI across many sessions, over weeks and months.
You want it to:

- **Remember who you are** — name, preferences, role, how you like to be
  talked to. And hold its own identity consistently — Kai is Kai on day 1
  and day 500.
- **Synthesise across sessions.** Ask *"summarise my React project"* and
  get one coherent answer spanning a dozen conversations — not a pile
  of raw snippets.
- **Change its mind, deliberately.** When you contradict something it
  once believed, it doesn't silently overwrite or stack conflicts — it
  creates a tension node, resolves it explicitly, and carries the
  resolution forward.
- **Keep the raw, too.** Every conversation is also archived verbatim.
  So *"what did I say on March 3rd?"* is a lossless `grep` away.

That combination — synthesis *and* verbatim — is rare. Most memory
systems pick one.

---

## Why not just X?

| System | What it is | What it misses |
|---|---|---|
| **MemPalace** | Verbatim archive + embedding search | No persona, no contradictions, no synthesis across sessions |
| **mem0 / Letta** | Auto-extracted summaries in a vector store | No identity layer, embedding drift, opaque memory |
| **Claude Code memory** | Flat markdown prefs file | No graph, no decay, no epistemic types (decisions vs tensions) |
| **Obsidian + AI plugin** | Your notes, AI reads them | The AI doesn't write the notes — you do |

**meme** is: a persistent persona, a graph of synthesised nodes the AI
wrote itself, a verbatim transcript archive, a decay function that
forgets unimportant things, and explicit epistemic types — all in
plain markdown in a folder you can open in Obsidian.

---

## Quick start

```bash
git clone https://github.com/Himanshu8881212/meme.git
cd meme

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your MISTRAL_API_KEY (or any OpenAI-compatible provider)

python main.py init --persona june --user-name "<your name>"
python tui.py
```

Just start typing. Your first message auto-starts a session. Say `quit`
when you're done — it'll distill the conversation into memory on the
way out.

Open the `vault/` folder in Obsidian to watch the mind take shape.

---

## Three ways to use it

### 1. Interactive TUI
```bash
python tui.py
```
The main way to talk with the assistant. Agentic chat with multi-hop
memory tools. Auto-saves on exit. `/help` for commands.

### 2. MCP server — plug it into any chat client
```bash
python mcp_server.py
```
Exposes 7 tools: `memory_search`, `memory_read`, `memory_reflect`,
`memory_stats`, `memory_list_tags`, `memory_grep`,
`memory_transcripts_by_date`. Plug into Claude Desktop, Cursor, Zed,
or any MCP-capable agent — your memory is now available to whatever
model they use.

Claude Desktop config (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "meme": {
      "command": "/abs/path/meme/.venv/bin/python",
      "args": ["/abs/path/meme/mcp_server.py"]
    }
  }
}
```

### 3. Ingest any transcript
```bash
python main.py ingest path/to/chat.md --task "topic"
```
Feed it any conversation text — Slack export, ChatGPT history,
terminal log — and reflection distills it into memory.

---

## Architecture in five pieces

```
vault/
  _identity/         ← persona (immutable) + self (accumulating facts about you)
  _transcripts/      ← verbatim session archives (lossless, never decay)
  entities/          ← people, places, things I know about
  concepts/          ← ideas, patterns, theories
  decisions/         ← choices made and why
  episodes/          ← specific events that happened
  tensions/          ← contradictions I haven't fully resolved
  questions/         ← things I'm uncertain about
  procedures/        ← how to do things
  _meta/             ← running state, tag registry, session log
```

1. **Flag → Write cycle.** During a session the assistant drops inline
   tags: `[NOVEL: ...]`, `[SALIENT: ...]`, `[CONTRADICTION: ...]`,
   `[IDENTITY: ...]`. On session end, a reflection pass reads those
   flags + the transcript and writes distilled nodes.

2. **Two-tier reflection.**
   - *Routine* (per session): turns flags into vault nodes.
   - *Deep* (periodic): resolves tensions, merges duplicates, splits
     bloated hubs, consolidates tags, reconciles entity bodies with
     tension resolutions.

3. **Decay, not deletion.** Every node has a strength score:
   `importance × log(1+connections) × exp(-λ·days / log(1+access))`.
   Unused nodes fade but are never deleted. Accessed nodes reset their
   clock. Wikilink density protects central concepts.

4. **Hybrid retrieval.** Graph scoring (tag overlap + title match +
   decay + recency) plus BM25 over node bodies plus wikilink expansion.
   No embeddings — the graph is the retrieval mechanism, and the AI
   builds it as it reflects.

5. **Identity layer, orthogonal to memory.** `_identity/persona.md` is
   immutable (your chosen persona template). `_identity/self.md`
   accumulates what the assistant learns about *you* across sessions.
   Both are injected into every system prompt; neither competes for
   retrieval slots.

---

## Benchmarks

Honest results on public memory benchmarks (12-question samples,
Mistral backend):

| Benchmark | Score | What it tests |
|---|---|---|
| **Our synthesis eval** | 87.5% | Cross-session synthesis, contradiction, preference, identity, meta-knowledge |
| **[MemBench](https://aclanthology.org/2025.findings-acl.989/) reflective** | 73% | Pattern inference from dialogue |
| **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** | 75% | Verbatim recall + multi-session reasoning |
| **[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)** Conflict_Resolution | 10–20% | Multi-hop fact-override over 455 atemporal facts |

For context: MemPalace publishes 96.6% on LongMemEval — their
architecture (pure verbatim embedding search) is optimised for that
specific benchmark and scores nothing on the synthesis / identity
dimensions. We score competitively on LongMemEval *and* retain the
things that make this system a mind.

Benchmarks are runnable: `python bench_longmemeval.py`,
`python bench_membench.py`, `python bench_memoryagent.py`,
`python eval.py`.

---

## Personas

Out of the box:
- **june** — warm, curious, a little dry. A good default.
- **sage** — formal, scholarly, cites sources.
- **max** — direct, terse, peer-to-peer.
- **kai** — playful, opinionated, remembers people.
- **blank** — empty, grows through use.

Add your own by dropping a markdown file in `prompts/personas/`
with `name`, `pronouns`, `voice` frontmatter and a first-person body.

---

## Configuration

Everything's in `config/config.yaml`. Flip providers, swap models per
role, tune retrieval weights, set decay λ, change recovery thresholds.

The provider layer speaks the OpenAI-compatible `/v1/chat/completions`
protocol, so any of these work:

```yaml
providers:
  mistral:    { base_url: https://api.mistral.ai/v1,      api_key_env: MISTRAL_API_KEY }
  openai:     { base_url: https://api.openai.com/v1,      api_key_env: OPENAI_API_KEY }
  openrouter: { base_url: https://openrouter.ai/api/v1,   api_key_env: OPENROUTER_API_KEY }
  ollama:     { base_url: http://localhost:11434/v1 }
  echo:       { base_url: null }   # offline mode for tests
```

Run without a key by setting `MEMORY_BACKEND=echo` — useful for
verifying plumbing.

---

## What this system is NOT

- Not a document QA system. Feed it dumped Wikipedia and it will
  struggle — that's not what it's for.
- Not a fact-extraction pipeline. It synthesises opinions and
  relationships; it doesn't try to be objective.
- Not a replacement for embedding search on large corpora. Use
  MemPalace / mem0 / a vector DB if that's your need.

Use this when you want an AI that **is someone** — with persistent
identity, evolving beliefs, and a point of view that carries across
every conversation.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/        # 194 tests, ~4 seconds
pytest tests/ -m slow  # + 6 scale tests
```

Every test uses an offline `echo` backend — no API keys required.

---

## Project layout

```
main.py             CLI: init, decay, monitor, meta, ingest, index
tui.py              Interactive TUI
mcp_server.py       MCP stdio server (7 tools)
eval.py             Synthesis eval harness
bench_*.py          Public benchmark runners (LongMemEval, MemBench, MemoryAgentBench)
battery.py          8-scenario end-to-end smoke suite

config/config.yaml  Providers, models, thresholds
prompts/            System prompt, reflection prompts, persona templates
core/               retrieval, reflection, flagging, decay, monitor, tools
scheduler/          session lifecycle, crontab renderer
utils/              frontmatter, wikilinks, indexer, env loader
tests/              194 unit + integration tests
```

---

## License

MIT.
