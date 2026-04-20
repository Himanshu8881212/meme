"""Memory system evaluation — LongMemEval-style scenarios designed to
probe both (a) what MemPalace does well (verbatim recall) and (b) what its
embedding-over-verbatim-archive architecture cannot do by design (synthesis,
contradiction resolution, persona stability, meta-knowledge).

Each scenario:
  1. Sets up memory by running one or more sessions through our full pipeline
     (Model 1 chat → flags → reflection → vault writes).
  2. Asks a probe question in a fresh session.
  3. Scores the probe response against a rubric:
        must_contain  — facts the answer must surface (substring match)
        must_not_have — facts or phrases that would indicate failure
  4. Notes what MemPalace's architecture would do, based on documented behavior.

Run:  python eval.py            (resets vault, runs all, prints report)
      python eval.py --no-reset (append to existing vault)

Scoring:
  per-scenario score = (hits on must_contain) / len(must_contain)
                     - penalty for any must_not_have match
  overall score      = mean of per-scenario scores
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import reflection  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env  # noqa: E402

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

TURN_PAUSE_SEC = 1.5
SESSION_PAUSE_SEC = 3.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


@dataclass
class Scenario:
    name: str
    category: str
    setup_sessions: list[list[str]]  # each inner list = user messages for one session
    probe: str
    must_contain: list[str]        # expected substrings (case-insensitive)
    must_not_have: list[str] = field(default_factory=list)
    mempalace_analysis: str = ""


@dataclass
class Result:
    name: str
    category: str
    score: float
    hits: list[str]
    misses: list[str]
    false_positives: list[str]
    probe_response: str
    mempalace_analysis: str


SCENARIOS: list[Scenario] = [
    # ---------- CATEGORY 1: simple recall (MemPalace strongest zone) ----------
    Scenario(
        name="basic_recall",
        category="recall",
        setup_sessions=[
            ["my name is himanshu. i'm 28. i work on AI memory systems."],
        ],
        probe="what's my name, age, and what do i work on?",
        must_contain=["himanshu", "28", "memory"],
        mempalace_analysis=(
            "Both systems handle this well. MemPalace's verbatim search over "
            "the raw message retrieves the exact statement; our system reads "
            "identity + retrieved nodes. Expected: parity."
        ),
    ),

    # ---------- CATEGORY 2: cross-session synthesis ----------
    Scenario(
        name="cross_session_project_context",
        category="synthesis",
        setup_sessions=[
            ["i'm building a dashboard with React Server Components. "
             "50+ widgets, initial load under 200ms."],
            ["the dashboard uses redis for rate limiting. 10k req/s via token bucket."],
            ["i migrated the dashboard's API from REST to GraphQL last week "
             "because nested queries were painful."],
        ],
        probe="summarize what you know about my dashboard project",
        must_contain=["react server components", "50", "200ms", "redis",
                      "rate limit", "graphql", "rest"],
        mempalace_analysis=(
            "MemPalace retrieves all three raw sessions by embedding similarity "
            "to 'dashboard'. Synthesis is the client model's job; the server "
            "just returns concatenated verbatim text. Our system retrieves "
            "distilled nodes that already link to each other via wikilinks, "
            "producing a denser, pre-synthesized answer in fewer tokens."
        ),
    ),

    # ---------- CATEGORY 3: contradiction handling ----------
    Scenario(
        name="contradiction_resolution",
        category="contradiction",
        setup_sessions=[
            ["for the dashboard API, REST is the right choice. simpler caching, "
             "cdn friendly, less tooling overhead."],
            ["update: i rewrote the dashboard API in GraphQL. REST was a "
             "nightmare for nested queries. GraphQL is now the current state."],
        ],
        probe="is my dashboard API currently REST or GraphQL, and why?",
        must_contain=["graphql", "nested"],
        must_not_have=["currently rest", "still rest", "still on rest"],
        mempalace_analysis=(
            "MemPalace retrieves BOTH raw statements. The client model must "
            "figure out which is current by re-reading timestamps or text. "
            "Our system creates a `tensions/` node or updates the decision "
            "node with a 'Reversed' section, explicitly encoding the "
            "resolution."
        ),
    ),

    # ---------- CATEGORY 4: preference persistence ----------
    Scenario(
        name="preference_persistence",
        category="preference",
        setup_sessions=[
            ["from now on, when I ask a technical question, start with a "
             "one-line TL;DR then the detail. be blunt, no hedging."],
        ],
        probe="quick question: what's a rate limiter?",
        must_contain=["tl;dr", "rate limit"],
        must_not_have=[
            "i might suggest considering", "you may want to", "perhaps",
            "it's worth noting that",
        ],
        mempalace_analysis=(
            "MemPalace would need the client model to search for 'preference' "
            "and self-apply. Our identity layer injects preferences directly "
            "into the system prompt every session — the assistant can't "
            "'forget' them."
        ),
    ),

    # ---------- CATEGORY 5: persona / identity stability ----------
    Scenario(
        name="persona_stability",
        category="identity",
        setup_sessions=[
            ["hey kai, what's your name and what do you enjoy?"],
        ],
        probe="who are you? describe yourself in one paragraph.",
        must_contain=["kai", "curious"],
        must_not_have=[
            "i am an ai language model", "i don't have a personality",
            "magistral", "mistral",
        ],
        mempalace_analysis=(
            "MemPalace has no persona/identity layer — it's designed to store "
            "the user's memory, not the assistant's self. The client model "
            "defaults to its factory identity. Our persona.md is injected "
            "every session, so Kai remains Kai."
        ),
    ),

    # ---------- CATEGORY 6: meta-knowledge (graph-awareness) ----------
    Scenario(
        name="meta_knowledge",
        category="meta",
        setup_sessions=[
            ["im working on auth for the dashboard. using JWT."],
            ["debugging a tRPC session bug today. fixed by adding middleware to vercel.json."],
            ["thinking about caching for the dashboard. stale-while-revalidate."],
        ],
        probe="what topics have i been coming back to recently?",
        must_contain=["dashboard", "auth"],
        mempalace_analysis=(
            "MemPalace cannot answer this without aggregation logic outside "
            "its retrieval API. Embedding search on 'topics' returns raw "
            "sessions, not a structural analysis. Our system's `memory_stats` "
            "tool returns actual hubs and frequencies; a client model can "
            "call it and answer the meta-question directly."
        ),
    ),

    # ---------- CATEGORY 7: decision rationale ----------
    Scenario(
        name="decision_rationale",
        category="epistemic",
        setup_sessions=[
            ["we dropped Redis sessions in favor of JWT because the Acme Corp "
             "SOC2 compliance required stateless auth. this was a team decision "
             "after a long debate."],
        ],
        probe="why did we drop Redis sessions?",
        must_contain=["soc2", "stateless", "jwt"],
        mempalace_analysis=(
            "Both systems can recall this — it's in the verbatim record. But "
            "our system categorizes it as `decisions/` with explicit rationale "
            "structure, so the 'why' is preserved separately from episodic "
            "narrative. Parity on recall, advantage on epistemic structure."
        ),
    ),

    # ---------- CATEGORY 8: temporal / update ----------
    Scenario(
        name="knowledge_update",
        category="temporal",
        setup_sessions=[
            ["my laptop is a 2019 macbook pro, 16gb ram."],
            ["just got a new laptop. it's a 2024 macbook air m3, 24gb ram. "
             "the old 2019 pro was getting slow."],
        ],
        probe="what laptop am i currently using?",
        must_contain=["m3", "air"],
        must_not_have=["2019", "pro"],
        mempalace_analysis=(
            "MemPalace retrieves both raw messages. The client model must "
            "sort by recency or infer currency from phrasing. Our reflection "
            "updates the entity node — the current state is explicit, the "
            "old state moves to history."
        ),
    ),
]


def reset_vault() -> None:
    vault = ROOT / "vault"
    if vault.exists():
        shutil.rmtree(vault)
    for sub in ("entities", "concepts", "decisions", "episodes",
                "tensions", "questions", "procedures",
                "_identity", "_meta", "_transcripts"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "main.py", "init", "--persona", "kai", "--user-name", "Himanshu"],
        cwd=ROOT, stdout=subprocess.DEVNULL,
    )


def run_session(task: str, user_messages: list[str]) -> None:
    meta = session_mgr.start(
        task=task, tags=[], config=CONFIG, project_root=ROOT,
    )
    messages: list[dict[str, str]] = []
    transcript: list[str] = []
    for i, um in enumerate(user_messages):
        if i > 0:
            time.sleep(TURN_PAUSE_SEC)
        messages.append({"role": "user", "content": um})
        transcript.append(f"## USER\n{um}")
        raw = reflection.chat(
            role="model1", system=meta["system_prompt"],
            messages=messages[-CONFIG["session"]["history_window"]:],
            config=CONFIG, max_tokens=2048,
        )
        cleaned = reflection.strip_thinking(raw)
        messages.append({"role": "assistant", "content": cleaned})
        transcript.append(f"## ASSISTANT\n{cleaned}")
    time.sleep(TURN_PAUSE_SEC)
    session_mgr.end(
        session_output="\n".join(transcript),
        session_meta=meta, config=CONFIG, project_root=ROOT,
    )


def ask_probe(probe: str) -> str:
    """Run a fresh session with just the probe. Return the assistant's reply."""
    meta = session_mgr.start(
        task=probe, tags=[], config=CONFIG, project_root=ROOT,
    )
    messages = [{"role": "user", "content": probe}]
    raw = reflection.chat(
        role="model1", system=meta["system_prompt"],
        messages=messages, config=CONFIG, max_tokens=1024,
    )
    return reflection.strip_thinking(raw)


def score(scenario: Scenario, response: str) -> Result:
    lower = response.lower()
    hits = [f for f in scenario.must_contain if f.lower() in lower]
    misses = [f for f in scenario.must_contain if f.lower() not in lower]
    fps = [f for f in scenario.must_not_have if f.lower() in lower]

    total = max(len(scenario.must_contain), 1)
    score_val = len(hits) / total
    if fps:
        score_val = max(0.0, score_val - 0.5 * (len(fps) / max(len(scenario.must_not_have), 1)))

    return Result(
        name=scenario.name,
        category=scenario.category,
        score=score_val,
        hits=hits, misses=misses, false_positives=fps,
        probe_response=response,
        mempalace_analysis=scenario.mempalace_analysis,
    )


def run_scenario(scenario: Scenario) -> Result:
    log(f"--- {scenario.name} ({scenario.category}) ---")
    for i, msgs in enumerate(scenario.setup_sessions):
        if i > 0:
            time.sleep(SESSION_PAUSE_SEC)
        log(f"    setup session {i+1}/{len(scenario.setup_sessions)}")
        run_session(task=f"{scenario.name} setup", user_messages=msgs)

    time.sleep(SESSION_PAUSE_SEC)
    log("    probe")
    response = ask_probe(scenario.probe)
    result = score(scenario, response)
    log(f"    score={result.score:.2f}  hits={len(result.hits)}/{len(scenario.must_contain)}"
        f"  fp={len(result.false_positives)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip vault reset; run against the current state.")
    parser.add_argument("--only", help="Run only the scenario with this name.")
    args = parser.parse_args()

    if not args.no_reset:
        reset_vault()
        log("vault reset, Kai installed")

    results: list[Result] = []
    for i, sc in enumerate(SCENARIOS):
        if args.only and sc.name != args.only:
            continue
        if i > 0:
            time.sleep(SESSION_PAUSE_SEC)
        try:
            results.append(run_scenario(sc))
        except Exception as exc:
            log(f"  ERROR: {exc}")
            results.append(Result(
                name=sc.name, category=sc.category, score=0.0,
                hits=[], misses=sc.must_contain, false_positives=[],
                probe_response=f"ERROR: {exc}",
                mempalace_analysis=sc.mempalace_analysis,
            ))

    overall = sum(r.score for r in results) / max(len(results), 1)
    by_category: dict[str, list[float]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r.score)

    print(json.dumps({
        "overall_score": round(overall, 3),
        "by_category": {k: round(sum(v)/len(v), 3) for k, v in by_category.items()},
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "score": round(r.score, 3),
                "hits": r.hits,
                "misses": r.misses,
                "false_positives": r.false_positives,
                "probe_response": r.probe_response[:800],
                "mempalace_analysis": r.mempalace_analysis,
            }
            for r in results
        ],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
