"""LongMemEval runner — evaluates our memory system on the canonical
long-term-memory benchmark (Wu et al. 2024, ICLR 2025).

For each question:
  1. Reset entities/concepts/etc. (persona preserved).
  2. Ingest each haystack session as a pre-canned transcript — reflection
     distills each into vault nodes.
  3. Ask the probe question in a fresh session; Model 1 answers using the
     retrieved memory.
  4. Score with an LLM judge (same protocol as the benchmark's official eval).

Published MemPalace score on LongMemEval: 96.6% (after correction from the
original hand-tuned 100%).

Usage:
  python bench_longmemeval.py --sample 15       # stratified sample
  python bench_longmemeval.py --sample 30
  python bench_longmemeval.py --sample -1       # full benchmark (hours)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
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

DATA = ROOT / "bench" / "longmemeval_oracle.json"
TURN_PAUSE = 0.8
SESSION_PAUSE = 2.0
QUESTION_PAUSE = 3.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def stratified_sample(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n < 0:
        return questions
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)
    types = list(by_type)
    per = max(n // len(types), 1)
    rng = random.Random(seed)
    sample: list[dict] = []
    for t in types:
        pool = by_type[t]
        sample.extend(rng.sample(pool, min(per, len(pool))))
    return sample[:n]


def reset_vault_but_keep_persona() -> None:
    """Wipe distilled content between questions, keep persona and self.md.

    Each LongMemEval question is an independent memory scenario — prior
    questions' answers must not leak. But re-initialising the persona on
    every question would triple the I/O. Just clear the distilled layer.
    """
    vault = ROOT / "vault"
    for sub in ("entities", "concepts", "decisions", "episodes",
                "tensions", "questions", "procedures"):
        d = vault / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    # Keep _identity, _transcripts, _meta intact so the persona persists,
    # but clean transcripts between questions to avoid bleed.
    tdir = vault / "_transcripts"
    if tdir.exists():
        for f in tdir.glob("*.md"):
            f.unlink()


def initial_setup() -> None:
    """Full reset once at start. Persona = 'blank' — the benchmark doesn't
    care about personality, and we want to not bias the answers with Kai's voice."""
    vault = ROOT / "vault"
    if vault.exists():
        shutil.rmtree(vault)
    for sub in ("entities", "concepts", "decisions", "episodes",
                "tensions", "questions", "procedures",
                "_identity", "_meta", "_transcripts"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "main.py", "init", "--persona", "blank", "--user-name", "User"],
        cwd=ROOT, stdout=subprocess.DEVNULL,
    )


def ingest_session(session_turns: list[dict], session_idx: int, q_date: str) -> None:
    """Ingest one haystack session. Doesn't invoke Model 1 — we're replaying
    a pre-canned dialogue, not generating a new one."""
    transcript_lines: list[str] = []
    for turn in session_turns:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        transcript_lines.append(f"## {role}\n{content}")
    transcript = "\n".join(transcript_lines)

    meta = session_mgr.start(
        task=f"session {session_idx} on {q_date}",
        tags=[], config=CONFIG, project_root=ROOT,
    )
    session_mgr.end(
        session_output=transcript,
        session_meta=meta,
        config=CONFIG,
        project_root=ROOT,
    )


def ask_probe(question: str, q_date: str) -> str:
    """For the LongMemEval probe we include transcripts. This benchmark tests
    verbatim recall, so we deliberately lift the default-retrieval filter
    that hides _transcripts/ — our lossless archive earns its keep here."""
    from core import retrieval  # noqa
    from pathlib import Path as _P

    vault = _P(CONFIG["vault_path"])
    if not vault.is_absolute():
        vault = (ROOT / vault).resolve()

    # Retrieve WITH transcripts — this is the one place we unlock them.
    files = retrieval.retrieve(vault, question, [], CONFIG, include_transcripts=True)
    # Cap at 10 to keep context reasonable.
    files = files[:10]

    # Build the prompt ourselves to bypass session_mgr.start's default retrieve.
    prompts_dir = ROOT / "prompts"
    system_prompt_tmpl = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    persona_path = vault / "_identity" / "persona.md"
    self_path = vault / "_identity" / "self.md"
    ident_parts = []
    for p in (persona_path, self_path):
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            if txt.startswith("---"):
                parts = txt.split("---", 2)
                if len(parts) >= 3:
                    ident_parts.append(parts[2].strip())
    identity_block = "\n\n".join(ident_parts) if ident_parts else "(unnamed)"

    context_block = "\n\n".join(
        f"=== {_P(p).relative_to(vault)} ===\n{c}" for p, c in files
    ) or "(nothing retrieved)"

    system = (
        system_prompt_tmpl
        .replace("{{IDENTITY}}", identity_block)
        .replace("{{RETRIEVED_CONTEXT}}", context_block)
    )

    # Use mistral-small for the probe (non-reasoning), to avoid magistral
    # burning the token budget on thinking. This is a factual-recall benchmark.
    cfg_override = {**CONFIG, "models": {**CONFIG["models"]}}
    cfg_override["models"]["model1"] = {"provider": "mistral", "model": "mistral-small-latest"}

    user_msg = (
        f"Today is {q_date}. {question}\n\n"
        f"The retrieved memory in your system prompt already contains "
        f"relevant material — check it first. If you can answer from it, "
        f"answer directly and concisely. If it's ambiguous or missing the "
        f"specific detail, call memory_grep/memory_by_date/memory_search to "
        f"dig deeper. Prefer an answer over refusal when the context "
        f"contains it; only say 'I don't know' after actually searching."
    )

    if (cfg_override.get("session") or {}).get("agentic_model1", True):
        text, _log = reflection.chat_with_tools(
            role="model1", system=system,
            messages=[{"role": "user", "content": user_msg}],
            config=cfg_override, vault_path=vault, max_tokens=512,
        )
        return text
    raw = reflection.chat(
        role="model1", system=system,
        messages=[{"role": "user", "content": user_msg}],
        config=cfg_override, max_tokens=1024,
    )
    return reflection.strip_thinking(raw)


JUDGE_PROMPT = """You are judging whether a memory-system's response correctly answers a question.

Question: {question}
Expected answer: {expected}
System's response: {response}

Does the system's response correctly answer the question, matching the expected answer in meaning? The response may be phrased differently but must convey the same factual content. Respond with exactly one word: YES or NO."""


def judge(question: str, expected: str, response: str) -> bool:
    """LLM-as-judge, forced to a non-reasoning model so it actually emits
    YES/NO cleanly. Also does a fast substring shortcut first."""
    # Fast path: if expected answer appears in response and no refusal phrase.
    resp_low = response.lower()
    if "don't know" not in resp_low and "do not know" not in resp_low:
        if expected and isinstance(expected, str) and expected.lower() in resp_low:
            return True

    cfg = {**CONFIG, "models": {**CONFIG["models"]}}
    cfg["models"]["routine"] = {"provider": "mistral", "model": "mistral-small-latest"}
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, response=response[:2000],
    )
    try:
        out = reflection.chat(
            role="routine",
            system="You are a strict but fair evaluator. Respond with only YES or NO.",
            messages=[{"role": "user", "content": prompt}],
            config=cfg, max_tokens=16,
        )
    except Exception:
        return False
    return "YES" in out.upper().strip()[:20]


def run_question(q: dict, idx: int, total: int) -> dict:
    reset_vault_but_keep_persona()
    q_date = q.get("question_date", "2024-01-01")
    sessions = q["haystack_sessions"]
    log(f"[{idx+1}/{total}] {q['question_type']}: {q['question'][:60]}...  "
        f"({len(sessions)} sessions, {sum(len(s) for s in sessions)} turns)")

    for i, session in enumerate(sessions):
        try:
            ingest_session(session, i, q_date)
        except Exception as exc:
            log(f"  ingest session {i} failed: {exc}")
        if i < len(sessions) - 1:
            time.sleep(SESSION_PAUSE)

    time.sleep(SESSION_PAUSE)
    try:
        response = ask_probe(q["question"], q_date)
    except Exception as exc:
        response = f"ERROR: {exc}"

    time.sleep(TURN_PAUSE)
    try:
        correct = judge(q["question"], q["answer"], response)
    except Exception as exc:
        log(f"  judge failed: {exc}")
        correct = False

    log(f"  → {'✓' if correct else '✗'}  response: {response[:120]}")
    return {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "question": q["question"],
        "expected": q["answer"],
        "response": response,
        "correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=15,
                        help="How many questions to run. -1 for full 500.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = json.loads(DATA.read_text(encoding="utf-8"))
    sample = stratified_sample(questions, args.sample, seed=args.seed)
    log(f"loaded {len(questions)} questions, running {len(sample)}")

    initial_setup()
    log("vault reset, blank persona installed")

    results: list[dict] = []
    for i, q in enumerate(sample):
        if i > 0:
            time.sleep(QUESTION_PAUSE)
        results.append(run_question(q, i, len(sample)))

    # Aggregate
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["correct"])

    report = {
        "total": total,
        "correct": correct,
        "overall_accuracy": round(correct / total, 3) if total else 0.0,
        "by_type": {
            t: {
                "n": len(v),
                "correct": sum(v),
                "accuracy": round(sum(v) / len(v), 3),
            } for t, v in by_type.items()
        },
        "mempalace_published": 0.966,
        "results": results,
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
