"""MemoryAgentBench (ICLR 2026) runner — focused on Conflict_Resolution.

Background:
- The benchmark's context is a numbered list of ~450 facts per row.
- Many facts are intentionally contradictory / later-overriding.
- Multi-hop questions require chaining facts, always using the latest version.
- This is EXACTLY the kind of conflict-handling our `tensions/` architecture
  was designed for — a genuine fit test.

Approach:
  1. Chunk the fact list into ~90-fact sessions.
  2. Ingest each chunk through our full pipeline (reflection writes vault).
  3. For each test question, probe with transcripts unlocked (lossless recall).
  4. Judge with LLM against the published answer.

We don't expect 96%+. We expect to measurably outperform a naive
synthesis-only config, and to show that Conflict_Resolution is a category
where our architecture has a real opinion.
"""
from __future__ import annotations

import argparse
import json
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

TURN_PAUSE = 0.8
SESSION_PAUSE = 2.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def chunk_facts(context: str, chunk_size: int = 90) -> list[str]:
    """Split a numbered-facts context into chunks for incremental ingestion.
    Keeps each chunk as a coherent 'session' narrative."""
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    header = lines[0] if "list of facts" in lines[0].lower() else "Here is a list of facts:"
    facts = [ln for ln in lines if ln[0:1].isdigit()]

    chunks = []
    for i in range(0, len(facts), chunk_size):
        chunk = [header] + facts[i:i + chunk_size]
        chunks.append("\n".join(chunk))
    return chunks


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


def ingest_chunk(chunk: str, idx: int) -> None:
    transcript = f"## USER\nHere are some facts to remember:\n\n{chunk}\n"
    meta = session_mgr.start(
        task=f"fact chunk {idx}", tags=["facts"], config=CONFIG, project_root=ROOT,
    )
    session_mgr.end(
        session_output=transcript,
        session_meta=meta,
        config=CONFIG, project_root=ROOT,
    )


def ask_probe(question: str) -> str:
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

    cfg_override = {**CONFIG, "models": {**CONFIG["models"]}}
    cfg_override["models"]["model1"] = {"provider": "mistral", "model": "mistral-small-latest"}

    user_msg = (
        f"Answer concisely using only facts from my memory. "
        f"For questions that require chaining multiple facts (e.g. 'the X of the Y of Z'), "
        f"call memory_search iteratively — one search per hop. "
        f"If the answer isn't in memory, say 'I don't know'.\n\n"
        f"Question: {question}"
    )

    agentic = (cfg_override.get("session") or {}).get("agentic_model1", True)
    if agentic:
        text, _log = reflection.chat_with_tools(
            role="model1", system=system,
            messages=[{"role": "user", "content": user_msg}],
            config=cfg_override, vault_path=vault, max_tokens=512,
        )
        return text
    raw = reflection.chat(
        role="model1", system=system,
        messages=[{"role": "user", "content": user_msg}],
        config=cfg_override, max_tokens=256,
    )
    return reflection.strip_thinking(raw)


JUDGE_PROMPT = """Question: {question}
Expected answer(s): {expected}
System's response: {response}

Does the system's response contain the expected answer? Accept synonyms and
paraphrases. Reply with just YES or NO."""


def judge(question: str, expected: list[str], response: str) -> bool:
    """LLM-as-judge — forced to use a non-reasoning model so it can emit
    YES/NO without burning tokens on thinking (which breaks on magistral)."""
    # Fast substring shortcut — if an expected answer appears in the response
    # and the response isn't a refusal, mark correct without an API call.
    resp_low = response.lower()
    if "don't know" not in resp_low and "do not know" not in resp_low:
        for e in expected:
            if e and e.lower() in resp_low:
                return True

    # Otherwise ask the non-reasoning judge.
    cfg = {**CONFIG, "models": {**CONFIG["models"]}}
    cfg["models"]["routine"] = {"provider": "mistral", "model": "mistral-small-latest"}
    out = reflection.chat(
        role="routine",
        system="You are a strict evaluator. Reply with only YES or NO.",
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(
                       question=question,
                       expected=" | ".join(expected),
                       response=response[:500],
                   )}],
        config=cfg, max_tokens=16,
    )
    return "YES" in reflection.strip_thinking(out).upper()[:20]


def run(split: str, row_idx: int, n_questions: int, chunk_size: int) -> dict:
    from datasets import load_dataset
    ds = load_dataset("ai-hyz/MemoryAgentBench")
    row = ds[split][row_idx]
    log(f"[{split} row {row_idx}] context={len(row['context'])} chars, "
        f"total_q={len(row['questions'])}")

    chunks = chunk_facts(row["context"], chunk_size=chunk_size)
    log(f"  split into {len(chunks)} chunks of ~{chunk_size} facts each")

    initial_setup()
    log("  vault initialized (blank persona)")

    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(SESSION_PAUSE)
        log(f"  ingesting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        try:
            ingest_chunk(chunk, i)
        except Exception as exc:
            log(f"    ingest failed: {exc}")

    time.sleep(SESSION_PAUSE)
    log(f"  running {n_questions} probe(s)")

    results = []
    for i in range(min(n_questions, len(row["questions"]))):
        q = row["questions"][i]
        a = row["answers"][i] if isinstance(row["answers"][i], list) else [row["answers"][i]]
        try:
            response = ask_probe(q)
        except Exception as exc:
            response = f"ERROR: {exc}"
        time.sleep(TURN_PAUSE)
        try:
            correct = judge(q, a, response)
        except Exception as exc:
            correct = False
        results.append({
            "question": q, "expected": a, "response": response, "correct": correct,
        })
        log(f"  Q{i+1} {'✓' if correct else '✗'}: {q[:60]}... → {response[:80]}")
        time.sleep(TURN_PAUSE)

    correct = sum(1 for r in results if r["correct"])
    return {
        "split": split,
        "row": row_idx,
        "total_chunks": len(chunks),
        "n": len(results),
        "correct": correct,
        "accuracy": round(correct / max(len(results), 1), 3),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="Conflict_Resolution",
                        choices=["Conflict_Resolution", "Accurate_Retrieval",
                                 "Test_Time_Learning", "Long_Range_Understanding"])
    parser.add_argument("--row", type=int, default=0, help="Which row of the split.")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to probe.")
    parser.add_argument("--chunk-size", type=int, default=90, help="Facts per ingestion chunk.")
    args = parser.parse_args()

    out = run(args.split, args.row, args.n, args.chunk_size)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
