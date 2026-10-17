#!/usr/bin/env python3
"""Opt-in real-model token equivalence matrix; no downloads or timing claims."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


PROMPTS = [
    ("science", "Explain why the sky looks blue in two short sentences."),
    ("code", "Write a Python function that returns the nth Fibonacci number."),
    ("math", "Solve step by step: A shop sells 12 notebooks at 3 dollars each. What is the total price?"),
]
CONFIGS = [(1, 1), (4, 2), (8, 4)]


def stamp(path):
    stat = path.stat()
    return {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--ssd-moe", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200, help="timeout per process, seconds")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 64 or args.timeout <= 0:
        parser.error("tokens must be in [1,64] and timeout positive")
    binary, model, output = args.binary.resolve(), args.model.resolve(), args.output.resolve()
    binary_stamp, model_stamp = stamp(binary), stamp(model)
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              text=True, capture_output=True, check=False).stdout.strip()
    report = {"schema": 1, "started_at": datetime.now(timezone.utc).isoformat(), "source_revision": revision,
              "binary": binary_stamp, "model": model_stamp, "tokens_per_case": args.tokens,
              "gpu_layers": args.gpu_layers, "ssd_moe": args.ssd_moe,
              "shared_settings": {"context": 256, "ubatch": 32, "threads": 8, "temp": 0,
                                  "raw_prompts": True, "ignore_eos": True},
              "configs": CONFIGS, "prompts": dict(PROMPTS), "cases": [], "status": "running"}
    write_report(output, report)

    def run(label, prompt, depth, experts, ar=False):
        if stamp(binary) != binary_stamp or stamp(model) != model_stamp:
            raise RuntimeError("binary or model changed during the comparison")
        case_dir = output / label
        case_dir.mkdir()
        command = [str(binary), "-m", str(model), "-p", prompt, "-n", str(args.tokens),
                   "-ngl", str(args.gpu_layers), "-c", "256", "-ub", "32", "-t", "8", "--temp", "0",
                   "--ignore-eos", "--draft-max", str(depth), "--draft-expert-topk", str(experts),
                   "--tokens-file", str(case_dir / "tokens.json")]
        if args.ssd_moe:
            command.append("--ssd-moe")
        command.extend(["--ar"] if ar else ["--audit-state-copies"])
        sys.stdout.write(f"START {label}\n")
        sys.stdout.flush()
        with (case_dir / "run.log").open("wb") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, timeout=args.timeout, check=False)
                rc = completed.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        text = (case_dir / "run.log").read_text(encoding="utf-8", errors="replace")
        case = {"id": label, "ar": ar, "depth": depth, "experts": experts, "exit_code": rc,
                "log": f"{label}/run.log", "valid": False}
        token_path = case_dir / "tokens.json"
        if rc == 0 and token_path.is_file():
            tokens = json.loads(token_path.read_text(encoding="utf-8"))
            case["tokens"] = tokens
            case["token_count"] = len(tokens)
            matches = re.findall(r"NEXT_SPEC_RESULT generated=(\d+) accepted=(\d+) proposed=(\d+) rounds=(\d+) commit_replay=(\d+)", text)
            if len(matches) != 1:
                case["error"] = "missing/ambiguous native result"
            else:
                generated, accepted, proposed, rounds, replay = map(int, matches[0])
                history = [json.loads(item) for item in re.findall(r"S2MOE_HISTORY_RESTORE_JSON (\{[^\n]+\})", text)]
                case.update({"generated": generated, "accepted": accepted, "proposed": proposed,
                             "rounds": rounds, "commit_replay": replay, "commits": history,
                             "accepted_lengths": [item["prefix_tokens"] - 1 for item in history]})
                sane = generated == len(tokens) == args.tokens and replay == 0
                if not ar:
                    sane = sane and len(history) == rounds and all(
                        1 <= item["prefix_tokens"] <= item["verified_tokens"] <= depth + 1
                        and (item["verified_tokens"] == 1 or item.get("copy_audited") is True) for item in history)
                    sane = sane and sum(item["prefix_tokens"] - 1 for item in history) == accepted
                    sane = sane and sum(item["verified_tokens"] - 1 for item in history) == proposed
                case["valid"] = sane
        if not case["valid"] and "error" not in case:
            case["error"] = "process, output length, history audit, or accounting failed"
        report["cases"].append(case)
        write_report(output, report)
        sys.stdout.write(f"END {label} exit={rc} valid={case['valid']}\n")
        sys.stdout.flush()
        return case

    for prompt_id, prompt in PROMPTS:
        reference = run(f"{prompt_id}-ar", prompt, 0, 1, ar=True)
        for depth, experts in CONFIGS:
            case = run(f"{prompt_id}-d{depth}-k{experts}", prompt, depth, experts)
            actual, expected = case.get("tokens", []), reference.get("tokens", [])
            equal = bool(reference["valid"] and case["valid"] and actual == expected)
            case["matches_ar"] = equal
            if not equal:
                case["first_mismatch"] = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                                              min(len(actual), len(expected)))
            sys.stdout.write(f"COMPARE {case['id']} matches_ar={equal} accepted_lengths={case.get('accepted_lengths')}\n")
            sys.stdout.flush()
            write_report(output, report)
    speculative = [case for case in report["cases"] if not case["ar"]]
    commits = [item for case in speculative for item in case.get("commits", [])]
    report["coverage"] = {
        "comparisons": len(speculative), "matching": sum(case.get("matches_ar", False) for case in speculative),
        "zero_accept_rounds": sum(item["prefix_tokens"] == 1 and item["verified_tokens"] > 1 for item in commits),
        "partial_accept_rounds": sum(1 < item["prefix_tokens"] < item["verified_tokens"] for item in commits),
        "full_accept_rounds": sum(item["prefix_tokens"] == item["verified_tokens"] for item in commits),
    }
    report["status"] = "passed" if all(case["valid"] for case in report["cases"]) and all(
        case.get("matches_ar", False) for case in speculative) else "failed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_report(output, report)
    sys.stdout.write(json.dumps({"status": report["status"], "coverage": report["coverage"]}) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
