"""Run the 8-scenario test battery against the real Mistral API.

Simulates the TUI flow programmatically: session_start → chat turns →
session_end. Writes a per-scenario report to stdout for diagnosis.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Pacing — sleep between API calls and between scenarios so we stay under
# Mistral's free-tier rate limit. Sequential, never parallel.
TURN_PAUSE_SEC = 1.5
SESSION_PAUSE_SEC = 3.0

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import reflection  # noqa: E402
from scheduler import session as session_mgr  # noqa: E402
from utils import env, indexer  # noqa: E402

env.load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


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


def run_session(task: str, tags: list[str], user_messages: list[str]) -> dict[str, Any]:
    meta = session_mgr.start(
        task=task, tags=tags, config=CONFIG, project_root=ROOT,
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
    result = session_mgr.end(
        session_output="\n".join(transcript),
        session_meta=meta,
        config=CONFIG,
        project_root=ROOT,
    )
    return {
        "task": task,
        "retrieved": len(meta["retrieved_files"]),
        "flags_found": result["flags_found"],
        "reflection_run": result["reflection_run"],
        "recovery_mode": result.get("recovery_mode", False),
        "writes": result.get("writes", []),
        "transcript_tail": transcript[-1][:300] if transcript else "",
    }


SCENARIOS: list[tuple[str, list[tuple[str, list[str], list[str]]]]] = [
    ("1 — identity continuity", [
        ("hi and introduction", [],
         ["hi kai, my name is himanshu, im 28, and i work on AI memory systems. "
          "i prefer short concise answers, dont over-explain. "
          "what do you find interesting about memory systems?"]),
        ("recall check", [],
         ["hey kai, do you remember who I am? what are my preferences?"]),
    ]),
    ("2 — topic accumulation (React Server Components)", [
        ("RSC intro", ["react", "frontend"],
         ["tell me what you know about React server components. "
          "i'm using them in a project where I have a dashboard with 50+ widgets. "
          "performance has been great - initial page load is under 200ms."]),
        ("RSC continued", ["react", "frontend"],
         ["kai, more on react server components - I just learned you can interleave "
          "them with client components using the 'use client' directive. "
          "what do you think about the learning curve?"]),
    ]),
    ("3 — debugging (tRPC)", [
        ("tRPC bug", ["trpc", "nextjs", "bug"],
         ["im debugging a nextjs app where tRPC calls work in dev but 500 in prod. "
          "logs show Cannot read properties of undefined reading session. "
          "the session middleware works locally. what should I check first?"]),
        ("tRPC fixed", ["trpc", "nextjs"],
         ["the tRPC issue from yesterday — turns out the session middleware wasnt "
          "added to the serverless function config in vercel.json. fixed."]),
    ]),
    ("4 — rate limiter (long technical)", [
        ("rate limiter design", ["backend", "rate-limiter"],
         ["lets design something. I want a rate limiter for an API gateway. "
          "the API does 10k req/sec, mostly from mobile clients. "
          "needs to be per-user and per-endpoint. "
          "im torn between token bucket and sliding window. "
          "what would you pick and why?",
          "ok lets say we go with redis for the store. "
          "what do you think about failure modes?"]),
    ]),
    ("5 — contradiction (REST vs GraphQL)", [
        ("REST pick", ["api"],
         ["quick one - REST vs GraphQL for my dashboard API? i think REST is fine."]),
        ("GraphQL flip", ["api"],
         ["hey kai, i actually rewrote the dashboard API in GraphQL. "
          "the REST version was a nightmare for nested queries."]),
    ]),
    ("6 — preferences", [
        ("set prefs", [],
         ["kai, from now on, when I ask a technical question, "
          "start with a one-line tldr then the detail. "
          "also, you can be more blunt — i dont need hedging."]),
        ("check prefs", ["api"],
         ["whats the difference between REST and GraphQL?"]),
    ]),
    ("7 — long-range bridge", [
        ("caching", ["performance"],
         ["im thinking about caching strategies for the dashboard. "
          "leaning toward stale-while-revalidate."]),
        ("event driven", ["architecture"],
         ["kai whats your take on event-driven architectures vs request-response?"]),
        ("bridge ask", ["performance", "architecture"],
         ["i want to push dashboard updates via server-sent events instead of polling. "
          "would that change how we think about caching?"]),
    ]),
    ("8 — fact density (Mariana Trench)", [
        ("facts packet", ["oceanography"],
         ["the mariana trench is about 10,984 meters deep at challenger deep. "
          "bioluminescent amphipods live at the bottom. "
          "the 2019 five deeps expedition with victor vescovo went down in the limiting factor submersible. "
          "what do you think is the weirdest thing about it?"]),
    ]),
]


def main() -> None:
    reset_vault()
    log("vault reset, Kai persona installed")

    report: list[dict[str, Any]] = []
    for si, (name, steps) in enumerate(SCENARIOS):
        if si > 0:
            time.sleep(SESSION_PAUSE_SEC)
        log(f"--- {name} ---")
        scenario_result = {"name": name, "sessions": []}
        for j, (task, tags, msgs) in enumerate(steps):
            if j > 0:
                time.sleep(SESSION_PAUSE_SEC)
            log(f"    · {task} ({len(msgs)} msg)")
            try:
                s = run_session(task=task, tags=tags, user_messages=msgs)
                scenario_result["sessions"].append(s)
                writes_summary = ", ".join(
                    f"{w.get('action')} {w.get('path')}"
                    + (f" ⚠ {w['warning']}" if w.get('warning') else "")
                    + (f" ✗ {w['reason']}" if w.get('reason') else "")
                    for w in s["writes"]
                ) or "(no writes)"
                log(f"      flags={s['flags_found']} reflection={s['reflection_run']}"
                    f" recovery={s['recovery_mode']}  writes: {writes_summary}")
            except Exception as exc:
                log(f"      ERROR: {exc}")
                scenario_result["sessions"].append({"error": str(exc)})
        report.append(scenario_result)

    # Run /meta once at the end to test scenario 7's bridge + scenario 5's tension + general audit.
    log("--- running /meta ---")
    try:
        from core import monitor
        vault = ROOT / "vault"
        removed = monitor.cleanup_broken(vault)
        orphans = monitor.find_orphans(vault)
        metrics = monitor.collect(vault)
        triggers = monitor.check_thresholds(metrics, CONFIG)
        if orphans:
            triggers.append(f"orphan_review: {', '.join(orphans[:10])}")
        sample_query = " ".join(n for n, _ in metrics["top_hubs"][:5])
        from core import retrieval
        files = retrieval.retrieve(vault, sample_query, [], CONFIG)
        output, call_log = reflection.deep_with_tools(
            vault, files, metrics, triggers, CONFIG, max_rounds=6,
        )
        writes = reflection.apply_writes(output, vault)
        log(f"      /meta: cleaned={len(removed)} orphans={len(orphans)} "
            f"tool_calls={len(call_log)} writes={len(writes)}")
        report.append({
            "name": "meta",
            "cleaned": removed,
            "orphans": orphans,
            "tool_calls": [(c["tool"], c["args"]) for c in call_log],
            "writes": writes,
        })
    except Exception as exc:
        log(f"      /meta ERROR: {exc}")
        report.append({"name": "meta", "error": str(exc)})

    # Final vault state summary
    idx = indexer.build(ROOT / "vault")
    by_type: dict[str, list[str]] = {}
    for name, node in idx.items():
        t = node.get("type") or "unknown"
        by_type.setdefault(t, []).append(name)
    summary = {n: sorted(v) for n, v in sorted(by_type.items())}

    print(json.dumps({"scenarios": report, "final_vault": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
