"""MemBench (ACL 2025) runner — focused on reflective_memory, our core claim.

MemBench tests whether a memory system can *synthesize patterns* from a
dialogue, not just recall facts. Example:

    Dialogue: user discusses Raging Bull, 12 Angry Men, The Godfather ...
    Q:        "What kind of movies might I prefer to watch?"
    A:        "Drama"

This is inference over accumulated interaction — our system's designed use
case. Contrast with LongMemEval (verbatim recall) and MemoryAgentBench
(multi-hop fact recall).

Dataset:  github.com/import-myself/Membench (ACL Findings 2025)
Subsets:  First-person Participation, topics movie/food/book.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import reflection, retrieval  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env  # noqa: E402

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

DATA_DIR = ROOT / "bench" / "MemBench" / "MemData" / "FirstAgent"
TURN_PAUSE = 0.8
SESSION_PAUSE = 2.0
QUESTION_PAUSE = 2.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def initial_setup() -> None:
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


def reset_between_events() -> None:
    """Wipe distilled content between events; keep persona."""
    vault = ROOT / "vault"
    for sub in ("entities", "concepts", "decisions", "episodes",
                "tensions", "questions", "procedures", "_transcripts"):
        d = vault / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


def flatten_messages(message_list: list) -> list[dict]:
    """MemBench message_list is a list of lists. Flatten."""
    if not message_list:
        return []
    if isinstance(message_list[0], dict):
        return message_list  # already flat
    out = []
    for inner in message_list:
        if isinstance(inner, list):
            out.extend(inner)
        elif isinstance(inner, dict):
            out.append(inner)
    return out


def ingest_event(message_list: list, tid: int) -> None:
    msgs = flatten_messages(message_list)
    transcript_lines = []
    for m in msgs:
        user = m.get("user") or m.get("user_message") or ""
        assistant = m.get("assistant") or m.get("assistant_message") or ""
        place = m.get("place", "")
        tstamp = m.get("time", "")
        header = f" ({tstamp} at {place})" if tstamp else ""
        if user:
            transcript_lines.append(f"## USER{header}\n{user}")
        if assistant:
            transcript_lines.append(f"## ASSISTANT\n{assistant}")

    transcript = "\n".join(transcript_lines)
    meta = session_mgr.start(
        task=f"dialogue tid={tid}", tags=[], config=CONFIG, project_root=ROOT,
    )
    session_mgr.end(
        session_output=transcript, session_meta=meta,
        config=CONFIG, project_root=ROOT,
    )


def ask_probe(question: str, choices: dict) -> str:
    from pathlib import Path as _P
    vault = _P(CONFIG["vault_path"])
    if not vault.is_absolute():
        vault = (ROOT / vault).resolve()

    files = retrieval.retrieve(vault, question, [], CONFIG, include_transcripts=True)
    files = files[:8]

    prompts_dir = ROOT / "prompts"
    system_prompt_tmpl = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    persona_path = vault / "_identity" / "persona.md"
    self_path = vault / "_identity" / "self.md"
    ident_parts: list[str] = []
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

    choice_block = "\n".join(f"  {k}. {v}" for k, v in choices.items()) if choices else ""
    user_msg = (
        f"Based on our past conversations that you remember, answer:\n\n"
        f"Question: {question}\n\n"
        + (f"Options:\n{choice_block}\n\n"
           f"Answer with the letter (A/B/C/D) and a one-sentence reason.\n"
           if choices else "Answer concisely.\n")
    )

    cfg_override = {**CONFIG, "models": {**CONFIG["models"]}}
    cfg_override["models"]["model1"] = {"provider": "mistral", "model": "mistral-small-latest"}

    raw = reflection.chat(
        role="model1", system=system,
        messages=[{"role": "user", "content": user_msg}],
        config=cfg_override, max_tokens=256,
    )
    return reflection.strip_thinking(raw)


def score_mc(response: str, answer: str, ground_truth_letter: str | None) -> bool:
    """For multiple-choice: correct if response contains either the answer text
    (case-insensitive) or the ground-truth letter in a clear answer position."""
    resp = response.strip().lower()
    if answer and answer.lower() in resp:
        return True
    if ground_truth_letter:
        # Accept patterns like "A", "A.", "A)", "(A)", "answer: A", "option A"
        import re as _re
        letter = ground_truth_letter.upper()
        patterns = [
            rf"\b{letter}\b",
            rf"^{letter}[\.\):]",
            rf"answer[:\s]+{letter}",
            rf"option[:\s]+{letter}",
        ]
        for p in patterns:
            if _re.search(p, response, _re.IGNORECASE):
                return True
    return False


def run_subset(topic: str, n: int, seed: int) -> dict:
    path = DATA_DIR / "highlevel_rec.json"
    data = json.load(open(path))
    events = data.get(topic, [])
    rng = random.Random(seed)
    sampled = rng.sample(events, min(n, len(events)))
    log(f"[{topic}] {len(sampled)} events sampled from {len(events)}")

    results = []
    for i, ev in enumerate(sampled):
        tid = ev.get("tid")
        qa = ev.get("QA", {})
        q = qa.get("question", "")
        a = qa.get("answer", "")
        gt = qa.get("ground_truth")
        choices = qa.get("choices", {})

        log(f"  [{topic} {i+1}/{len(sampled)}] tid={tid}: {q[:60]}...")
        reset_between_events()
        try:
            ingest_event(ev["message_list"], tid)
        except Exception as exc:
            log(f"    ingest failed: {exc}")

        time.sleep(SESSION_PAUSE)
        try:
            response = ask_probe(q, choices)
        except Exception as exc:
            response = f"ERROR: {exc}"

        correct = score_mc(response, a, gt)
        log(f"    {'✓' if correct else '✗'} expected='{a}' ({gt}) → {response[:100]}")
        results.append({
            "topic": topic, "tid": tid, "question": q,
            "expected_answer": a, "expected_letter": gt,
            "choices": choices, "response": response, "correct": correct,
        })
        time.sleep(QUESTION_PAUSE)

    correct = sum(1 for r in results if r["correct"])
    return {
        "topic": topic, "n": len(results), "correct": correct,
        "accuracy": round(correct / max(len(results), 1), 3),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-topic", type=int, default=5,
                        help="Events per topic (movie/food/book).")
    parser.add_argument("--topics", nargs="+", default=["movie", "food", "book"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    initial_setup()
    log("vault initialized (blank persona)")

    reports = []
    for i, topic in enumerate(args.topics):
        if i > 0:
            time.sleep(QUESTION_PAUSE)
        reports.append(run_subset(topic, args.n_per_topic, args.seed))

    total_n = sum(r["n"] for r in reports)
    total_correct = sum(r["correct"] for r in reports)
    overall = round(total_correct / max(total_n, 1), 3)

    print(json.dumps({
        "benchmark": "MemBench / highlevel_rec (reflective memory)",
        "overall_accuracy": overall,
        "n_total": total_n,
        "correct_total": total_correct,
        "by_topic": {r["topic"]: {"n": r["n"], "correct": r["correct"], "accuracy": r["accuracy"]}
                     for r in reports},
        "results": [item for r in reports for item in r["results"]],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
