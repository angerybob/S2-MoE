#!/usr/bin/env python3
"""Opt-in real Qwen3.5 GPU greedy comparisons; defaults to budgeted SSD I/O.

No downloads or global machine changes. Approximate expert reuse is disabled.
Every planned case must finish and match its same-prompt AR reference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time

PROMPTS = [
    {"id": "math", "text": "Solve step by step: A shop sells 12 notebooks at 3 dollars each. What is the total price?"},
    {"id": "science", "text": "Explain why the sky looks blue in two short sentences."},
    {"id": "code", "text": "Write a Python function that returns the nth Fibonacci number."},
    {"id": "chinese", "text": "用两句话解释为什么月球会出现不同的月相。"},
]


def stamp(path):
    value = path.stat()
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=("ssd", "resident"), default="ssd")
    ap.add_argument("--memory-budget-mib", type=int, default=24576)
    ap.add_argument("--runtime-reserve-mib", type=int, default=3072)
    ap.add_argument("--tokens", type=int, choices=(32, 64), default=64)
    ap.add_argument("--configs", nargs="+", default=["1:2", "4:2", "8:4"])
    ap.add_argument("--prompts", nargs="+", choices=[p["id"] for p in PROMPTS], default=[p["id"] for p in PROMPTS])
    ap.add_argument("--branches", type=int, default=1)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--repeat-math-ar", action="store_true")
    ap.add_argument("--expected-math", type=Path, help="frozen pre-fix AR output; also verify AR did not change")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    configs = [tuple(int(v) for v in text.split(":")) for text in args.configs]
    if not all(len(c) == 2 and 1 <= c[0] <= 31 and 1 <= c[1] <= 8 for c in configs):
        ap.error("configs must be depth:topk with depth 1..31 and topk 1..8")
    binary, model, output = args.binary.resolve(), args.model.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo))
    runtime = [binary, *sorted(binary.parent.glob("lib*.so"))]
    before = {str(p): {**stamp(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in runtime}
    model_before = stamp(model)
    prompts = [p for p in PROMPTS if p["id"] in args.prompts]
    schedule = []
    for prompt in prompts:
        schedule.append({"id": prompt["id"] + "-ar", "prompt": prompt, "mode": "AR", "depth": 0, "topk": 8})
        for depth, topk in configs:
            schedule.append({"id": f"{prompt['id']}-d{depth}-k{topk}", "prompt": prompt,
                             "mode": "SPEC", "depth": depth, "topk": topk})
    if args.repeat_math_ar and any(p["id"] == "math" for p in prompts):
        schedule.append({"id": "math-ar-repeat", "prompt": PROMPTS[0], "mode": "AR", "depth": 0, "topk": 8})
    rows = []
    for case in schedule:
        case["tokens_path"] = str(output / (case["id"] + "-tokens.json"))
        case["timing_path"] = str(output / (case["id"] + "-timing.json"))
        rows.append(f"{case['mode']} {case['depth']} {case['topk']} "
                    + " ".join(json.dumps(s, ensure_ascii=False) for s in (case["tokens_path"], case["timing_path"], case["prompt"]["text"])))
    plan = output / "plan.txt"
    plan.write_text("\n".join(rows) + "\n", encoding="utf-8")
    command = [str(binary), "-m", str(model), "-ngl", "99", "-c", "128", "-ub", "32", "-t", "8",
               "-n", str(args.tokens), "--ignore-eos", "--quiet", "--no-mmap", "--benchmark-plan", str(plan),
               "--draft-branches", str(args.branches), "--moe-reuse-expert-cap", "0"]
    if args.mode == "ssd":
        command += ["--ssd-moe", "--memory-budget-mib", str(args.memory_budget_mib),
                    "--runtime-reserve-mib", str(args.runtime_reserve_mib)]
    if args.prune:
        command += ["--prune", "1", "--prune-tpot", "5000", "--prune-bandwidth", "0.15", "--prune-score-thresh", "1"]
    expected = json.loads(args.expected_math.read_text()) if args.expected_math else None
    report = {"schema": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "source_revision": revision, "source_dirty": dirty, "model": {"path": str(model), **model_before},
              "runtime": before, "command": command, "planned_cases": len(schedule), "tokens_per_case": args.tokens,
              "expected_math_sha256": hashlib.sha256(args.expected_math.read_bytes()).hexdigest() if args.expected_math else None,
              "mode": args.mode, "memory_budget_mib": args.memory_budget_mib if args.mode == "ssd" else None,
              "runtime_reserve_mib": args.runtime_reserve_mib if args.mode == "ssd" else None,
              "approximate_reuse": False, "cases": []}
    references = {}
    index = 0

    def save():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")

    def harvest():
        nonlocal index
        while index < len(schedule):
            case = schedule[index]
            try:
                tokens = json.loads(Path(case["tokens_path"]).read_text())
                timing = json.loads(Path(case["timing_path"]).read_text())
            except (OSError, json.JSONDecodeError):
                return
            for path, initial in before.items():
                if stamp(Path(path)) != {k: initial[k] for k in ("size_bytes", "mtime_ns")}:
                    raise RuntimeError("runtime files changed during validation")
            if stamp(model) != model_before:
                raise RuntimeError("model changed during validation")
            prompt_id = case["prompt"]["id"]
            if case["mode"] == "AR":
                references.setdefault(prompt_id, tokens)
            reference = references[prompt_id]
            first_diff = next((i for i, (a, b) in enumerate(zip(reference, tokens)) if a != b), None)
            valid = (len(tokens) == len(reference) == args.tokens and first_diff is None
                     and not timing["audit"] and not timing["approximate_expert_reuse"])
            old_match = None
            if expected is not None and prompt_id == "math":
                old_match = tokens == expected[:args.tokens]
                valid = valid and old_match
            if args.mode == "ssd":
                valid = valid and (timing["ssd"] and timing["direct_io"] and timing["ssd_read_bytes"] > 0
                                   and timing["ssd_resident_hits"] > 0
                                   and timing["model_buffer_bytes"] + timing["runtime_accounted_bytes"] <= args.memory_budget_mib * 2**20)
            record = {**case, "tokens": tokens, "timing": timing, "valid": bool(valid), "first_diff": first_diff,
                      "matches_frozen_math_ar": old_match}
            report["cases"].append(record)
            index += 1
            save()
            logging.info("%s: valid=%s tokens=%s first_diff=%s frozen_AR=%s", case['id'], valid, len(tokens), first_diff, old_match)
            if not valid:
                raise RuntimeError(f"{case['id']}: token, residency, or frozen-AR check failed")

    save()
    try:
        with (output / "native.log").open("wb") as log:
            child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            start = time.monotonic()
            try:
                while child.poll() is None:
                    info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
                    if info["MemAvailable"] / 1024**2 < 2.5 or time.monotonic() - start > args.timeout:
                        raise RuntimeError("own validation memory/time guard reached")
                    harvest()
                    time.sleep(.25)
                harvest()
                if child.returncode or index != len(schedule):
                    raise RuntimeError(f"native exit {child.returncode}, completed {index}/{len(schedule)}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
