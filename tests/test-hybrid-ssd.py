#!/usr/bin/env python3
"""GPU integration: placement, direct I/O, pruning, tree state and reuse.

Use deterministic fixtures, or pass a real model for a separate recorded run.
No downloads, global cache drops, or power-policy changes.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default="test ab")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hot = args.output / "hot.json"
    hot.write_text(json.dumps({"gpu_experts": [[i, j] for i in range(args.layers)
                                               for j in range(args.experts // 2)]}))
    # SSD comparisons use the same non-routed weights and a partial resident
    # list in both AR and speculation. The resident oracle is only for fixtures.
    cases = [] if args.real else [("resident-ar", ["--ar"], True)]
    placement = ["--ssd-moe", "--hot-experts", str(hot)]
    cases += [("ssd-ar", [*placement, "--ar"], True)]
    if not args.real:
        cases += [
            ("ssd-all-cold-ar", ["--ssd-moe", "--ar"], True),
            ("ssd-budget-auto-ar", ["--ssd-moe", "--ar", "--memory-budget-mib", "512", "--runtime-reserve-mib", "511"], True),
        ]
    for depth, k in ((0, 1), (1, 1), (4, 1), (4, 2)):
        cases.append((f"ssd-d{depth}-k{k}", [*placement, "--draft-max", str(depth),
                     "--draft-expert-topk", str(k)], True))
    cases += [
        ("ssd-prune-stop", [*placement, "--draft-max", "4", "--prune", "1",
                            "--prune-conf-thresh", "1"], True),
        ("ssd-prune-tree", [*placement, "--draft-max", "4", "--draft-branches", "2",
                            "--draft-expert-topk", "1", "--prune", "1", "--prune-score-thresh", "0"], True),
        ("ssd-tree", [*placement, "--draft-max", "4", "--draft-branches", "2", "--draft-expert-topk", "1"], True),
        # Reuse modifies target routing. Check execution/telemetry separately;
        # its tokens must never be called an exact AR-equivalence result.
        ("ssd-reuse", [*placement, "--draft-max", "4", "--moe-reuse-runtime",
                       "--moe-reuse-expert-cap", "2"], False),
    ]
    report = {"status": "running", "model": str(args.model), "cases": []}
    reference = None
    failures = []
    for name, extra, exact in cases:
        tokens = args.output / f"{name}-tokens.json"
        timing = args.output / f"{name}-timing.json"
        command = [str(args.exe), "-m", str(args.model), "-p", args.prompt, "-n", str(args.tokens),
                   "-c", "128", "-ub", "32", "-ngl", "99", "--ignore-eos", "--quiet",
                   "--tokens-file", str(tokens), "--timings-file", str(timing), *extra]
        if not args.real and exact and "--ar" not in extra:
            command.append("--audit-state-copies")
        with (args.output / f"{name}.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        entry = {"name": name, "command": command, "returncode": result.returncode, "exact_mode": exact}
        if result.returncode:
            failures.append(f"{name}: process failed")
        else:
            actual = json.loads(tokens.read_text())
            metrics = json.loads(timing.read_text())
            if reference is None:
                reference = actual
            entry.update(tokens=actual, metrics=metrics)
            entry["matches_ar"] = actual == reference if exact else None
            if len(actual) != args.tokens or (exact and actual != reference):
                failures.append(f"{name}: token mismatch")
            if metrics["ssd"] and "all-cold" not in name and "budget-auto" not in name and not (
                    metrics["ssd_resident_hits"] > 0 and metrics["ssd_read_bytes"] > 0):
                failures.append(f"{name}: missing partial-residency evidence")
            if "all-cold" in name and not (metrics["ssd_resident_hits"] == 0 and metrics["ssd_read_bytes"] > 0):
                failures.append(f"{name}: cold experts unexpectedly resident")
            if "budget-auto" in name and not (metrics["ssd_resident_hits"] > 0 and metrics["ssd_read_bytes"] == 0):
                failures.append(f"{name}: automatic placement did not retain this entire tiny fixture")
            if "prune-stop" in name and metrics["pruned_extensions"] <= 0:
                failures.append(f"{name}: pruning did not execute")
        report["cases"].append(entry)
        report["failures"] = failures
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: rc={result.returncode} exact={entry.get('matches_ar')}", flush=True)
    report["status"] = "failed" if failures else "passed"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
