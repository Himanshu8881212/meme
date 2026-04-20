from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import decay, flagging, monitor, reflection, retrieval  # noqa: E402
from scheduler import cron, session  # noqa: E402
from utils import env, frontmatter, indexer  # noqa: E402

env.load_dotenv(ROOT / ".env")

CONFIG_PATH = ROOT / "config" / "config.yaml"


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _vault_path(config: dict[str, Any]) -> Path:
    p = Path(config["vault_path"])
    return p if p.is_absolute() else (ROOT / p).resolve()


def cmd_start(args: argparse.Namespace) -> None:
    config = _load_config()
    meta = session.start(
        task=args.task,
        tags=args.tags or [],
        config=config,
        project_root=ROOT,
    )
    state_path = Path(args.state or ROOT / "vault" / "_meta" / ".session_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sys.stdout.write(meta["system_prompt"])
    sys.stderr.write(
        f"\n[session started — {len(meta['retrieved_files'])} files retrieved, "
        f"state saved to {state_path}]\n"
    )


def cmd_end(args: argparse.Namespace) -> None:
    config = _load_config()
    state_path = Path(args.state or ROOT / "vault" / "_meta" / ".session_state.json")
    if not state_path.exists():
        sys.exit(f"no session state at {state_path} — run `start` first")

    meta = json.loads(state_path.read_text(encoding="utf-8"))

    if args.input == "-":
        output = sys.stdin.read()
    else:
        output = Path(args.input).read_text(encoding="utf-8")

    result = session.end(
        session_output=output,
        session_meta=meta,
        config=config,
        project_root=ROOT,
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k != "reflection_output"},
        indent=2,
    ))
    if args.verbose and result.get("reflection_output"):
        print("\n--- reflection output ---\n")
        print(result["reflection_output"])


def cmd_decay(args: argparse.Namespace) -> None:
    config = _load_config()
    vault = _vault_path(config)
    result = decay.run(
        vault_path=vault,
        lambda_=config["decay"]["lambda"],
        archive_threshold=config["decay"]["archive_threshold"],
    )
    print(json.dumps(result, indent=2))


def cmd_monitor(args: argparse.Namespace) -> None:
    config = _load_config()
    vault = _vault_path(config)
    metrics = monitor.collect(vault)
    triggers = monitor.check_thresholds(metrics, config)
    print(json.dumps({"metrics": metrics, "triggers": triggers}, indent=2, default=str))


def cmd_meta(args: argparse.Namespace) -> None:
    config = _load_config()
    vault = _vault_path(config)

    min_body = int((config.get("monitor") or {}).get("min_body_chars", 20))
    removed = monitor.cleanup_broken(vault, min_body_chars=min_body)
    reconciled = reflection.reconcile_tensions(vault)
    orphans = monitor.find_orphans(vault)

    metrics = monitor.collect(vault)
    triggers = monitor.check_thresholds(metrics, config)
    if orphans:
        triggers.append(f"orphan_review: {', '.join(orphans[:10])}")

    sample_query = " ".join(n for n, _ in metrics["top_hubs"][:5])
    files = retrieval.retrieve(vault, sample_query, [], config)

    output, call_log = reflection.deep_with_tools(vault, files, metrics, triggers, config)
    writes = reflection.apply_writes(
        output, vault,
        similarity_threshold=float(
            (config.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
        ),
    )

    print(json.dumps({
        "cleaned": removed,
        "reconciled": reconciled,
        "orphans": orphans,
        "triggers": triggers,
        "tool_calls": call_log,
        "writes": writes,
    }, indent=2, default=str))
    if args.verbose:
        print("\n--- reflection output ---\n")
        print(output)


def cmd_index(args: argparse.Namespace) -> None:
    config = _load_config()
    vault = _vault_path(config)
    idx = indexer.build(vault)
    print(json.dumps({"nodes": len(idx), "sample": list(idx)[:10]}, indent=2))


def cmd_crontab(args: argparse.Namespace) -> None:
    print(cron.render(ROOT))


def cmd_init(args: argparse.Namespace) -> None:
    """Seed a fresh vault: pick a persona, optional user name, minimal content.

    Default: preserve technical-seed content and just set identity.
    --fresh: archive all current vault content under _archive/ and start lean.
    """
    from datetime import date

    config = _load_config()
    vault = _vault_path(config)
    personas_dir = ROOT / "prompts" / "personas"
    persona_path = personas_dir / f"{args.persona}.md"
    if not persona_path.exists():
        available = ", ".join(p.stem for p in personas_dir.glob("*.md"))
        sys.exit(f"unknown persona '{args.persona}'. available: {available}")

    persona_fm, persona_body = frontmatter.read(persona_path)

    # Compose the active identity from the persona template.
    today = date.today().isoformat()
    identity_fm = {
        "type": "identity",
        "name": persona_fm.get("name", args.persona),
        "pronouns": persona_fm.get("pronouns", "(unset)"),
        "user_name": args.user_name or "(unknown)",
        "persona_template": args.persona,
        "created": today,
        "last_updated": today,
        "immutable_structure": True,
    }
    # Personalize the body by swapping any {user_name} marker if present.
    body = persona_body
    if args.user_name:
        body = body.replace("the user", args.user_name).replace(
            "The user", args.user_name
        )

    if args.fresh:
        archive = vault / "_archive" / today
        archive.mkdir(parents=True, exist_ok=True)
        moved = 0
        for folder in ("entities", "concepts", "decisions", "episodes",
                       "tensions", "questions", "procedures"):
            src = vault / folder
            if not src.exists():
                continue
            dst = archive / folder
            if any(src.iterdir()):
                src.rename(dst)
                (vault / folder).mkdir()
                moved += 1
        print(f"[init] --fresh: archived {moved} folder(s) to {archive.relative_to(vault)}")

    identity_dir = vault / "_identity"
    identity_dir.mkdir(parents=True, exist_ok=True)

    # persona.md — the immutable personality body. Reflection cannot touch this.
    persona_path = identity_dir / "persona.md"
    frontmatter.write(persona_path, {**identity_fm, "immutable": True}, body)

    # self.md — starts nearly empty. Reflection accumulates relationship +
    # user-given preferences here. Never touches persona.md.
    self_path = identity_dir / "self.md"
    user_line = f"Himanshu" if not args.user_name else args.user_name
    self_body = (
        f"## Who {user_line} is to me\n\n"
        f"(I'll accumulate what I learn about {user_line} here across our conversations.)\n\n"
        f"## Standing preferences\n\n"
        f"(Things {user_line} has asked me to keep in mind — none yet.)\n"
    ) if args.user_name else (
        "## Who the person I'm talking with is\n\n"
        "(I'll record what I learn across our conversations here once they tell me who they are.)\n\n"
        "## Standing preferences\n\n"
        "(Preferences they've asked me to hold — none yet.)\n"
    )
    frontmatter.write(self_path, {
        "type": "identity",
        "created": today,
        "last_updated": today,
    }, self_body)

    print(json.dumps({
        "persona": args.persona,
        "name": identity_fm["name"],
        "user_name": identity_fm["user_name"],
        "persona_file": str(persona_path.relative_to(ROOT)),
        "self_file": str(self_path.relative_to(ROOT)),
        "fresh": bool(args.fresh),
    }, indent=2))


def cmd_ingest(args: argparse.Namespace) -> None:
    config = _load_config()
    vault = _vault_path(config)

    if args.input == "-":
        transcript = sys.stdin.read()
    else:
        transcript = Path(args.input).read_text(encoding="utf-8")

    flags = flagging.extract(transcript)
    summary = flagging.summarize(flags)
    query = (args.task or "") + "\n" + summary
    files = retrieval.retrieve(vault, query, args.tags or [], config)

    if not flags and not args.force:
        print(json.dumps({"flags_found": 0, "reflection_run": False,
                          "note": "no flags found; use --force to reflect anyway"}, indent=2))
        return

    output = reflection.routine(
        flag_summary=summary or "(no inline flags — reflecting on raw transcript)",
        vault_files=files,
        session_notes=transcript[:4000],
        config=config,
    )
    writes = reflection.apply_writes(
        output, vault,
        similarity_threshold=float(
            (config.get("reflection") or {}).get("duplicate_similarity_threshold", 0.5)
        ),
    )
    print(json.dumps({"flags_found": len(flags), "writes": writes}, indent=2))
    if args.verbose:
        print("\n--- reflection output ---\n")
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="memory-system")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Begin a session — emit system prompt + memory context.")
    p_start.add_argument("task", help="Task description used for retrieval scoring.")
    p_start.add_argument("--tag", dest="tags", action="append", help="Tag hint for retrieval; repeatable.")
    p_start.add_argument("--state", help="Override session state file path.")
    p_start.set_defaults(func=cmd_start)

    p_end = sub.add_parser("end", help="End a session — parse flags, run routine reflection.")
    p_end.add_argument("input", help="Path to session transcript, or '-' for stdin.")
    p_end.add_argument("--state", help="Override session state file path.")
    p_end.add_argument("-v", "--verbose", action="store_true")
    p_end.set_defaults(func=cmd_end)

    p_decay = sub.add_parser("decay", help="Run decay pass over the vault.")
    p_decay.set_defaults(func=cmd_decay)

    p_mon = sub.add_parser("monitor", help="Collect vault health metrics + threshold triggers.")
    p_mon.set_defaults(func=cmd_monitor)

    p_meta = sub.add_parser("meta", help="Run deep (weekly) reflection.")
    p_meta.add_argument("-v", "--verbose", action="store_true")
    p_meta.set_defaults(func=cmd_meta)

    p_idx = sub.add_parser("index", help="Build and report vault index.")
    p_idx.set_defaults(func=cmd_index)

    p_cron = sub.add_parser("crontab", help="Print a crontab snippet for scheduled jobs.")
    p_cron.set_defaults(func=cmd_crontab)

    p_init = sub.add_parser("init", help="Set identity from a persona template. Fresh vault optional.")
    p_init.add_argument("--persona", default="june", help="Persona template: june, sage, max, kai, blank.")
    p_init.add_argument("--user-name", help="The human's name (goes into identity).")
    p_init.add_argument("--fresh", action="store_true",
                        help="Archive existing content under _archive/<date>/ and start lean.")
    p_init.set_defaults(func=cmd_init)

    p_ing = sub.add_parser("ingest", help="Reflect on an external transcript (any source).")
    p_ing.add_argument("input", help="Path to transcript file, or '-' for stdin.")
    p_ing.add_argument("--task", help="Task description used for retrieval scoring.")
    p_ing.add_argument("--tag", dest="tags", action="append", help="Tag hint; repeatable.")
    p_ing.add_argument("--force", action="store_true", help="Reflect even if no flags found.")
    p_ing.add_argument("-v", "--verbose", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
