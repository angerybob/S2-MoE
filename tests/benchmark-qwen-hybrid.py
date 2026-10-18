#!/usr/bin/env python3
"""Opt-in Linux CPU benchmark of the experimental hybrid entry points (no audit)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time


def stamp(path):
    stat = path.stat()
    return {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def machine():
    def read(path):
        try:
            return Path(path).read_text().strip().replace("\0", "")
        except (OSError, TypeError):
            return None
    return {"model": read("/proc/device-tree/model"), "loadavg": read("/proc/loadavg"),
            "meminfo": read("/proc/meminfo"),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "cpu_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
            "thermal_millidegrees": {p.parent.name: read(p) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain why the sky looks blue in two short sentences.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--ubatch", type=int, default=32)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit-gib", type=int, default=46, help="child address-space cap; 0 disables")
    args = parser.parse_args()
    if (args.tokens < 2 or args.repeats < 1 or args.threads < 1 or args.timeout < 1
            or not all(1 <= d <= 31 and d + 1 <= args.ubatch for d in args.depths)
            or not all(k > 0 for k in args.experts)):
        parser.error("invalid generation, draft, or repetition limits")
    binary, model, output = args.binary.resolve(), args.model.resolve(), args.output.resolve()
    binary_stamp, model_stamp = stamp(binary), stamp(model)
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=source_root)
    report = {"schema": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "source_revision": revision, "source_dirty": bool(diff),
              "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
              "binary": {**binary_stamp, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
              "model": model_stamp, "prompt": args.prompt,
              "settings": {"tokens": args.tokens, "threads": args.threads, "context": args.context,
                           "ubatch": args.ubatch, "gpu_layers": 0, "audit": False, "quiet": True,
                           "greedy": True, "ignore_eos": True, "address_space_limit_gib": args.limit_gib},
              "throughput_definition": "(generated - 1) / post-prefill generation seconds; first token comes from prefill",
              "phase_definition": "draft includes its greedy selection; verify includes history writes; snapshot/restore/commit are separate API wall times",
              "start_machine": machine(), "cases": [], "status": "running"}

    def save():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")

    def limit():
        if args.limit_gib:
            resource.setrlimit(resource.RLIMIT_AS, (args.limit_gib * 1024**3,) * 2)

    reference = None

    def run(label, depth=0, experts=1, warmup=False):
        nonlocal reference
        if stamp(binary) != binary_stamp or stamp(model) != model_stamp:
            raise RuntimeError("binary or model changed during benchmark")
        case_dir = output / label
        case_dir.mkdir()
        command = [str(binary), "-m", str(model), "-p", args.prompt, "-n", str(args.tokens),
                   "-ngl", "0", "-c", str(args.context), "-ub", str(args.ubatch), "-t", str(args.threads),
                   "--ignore-eos", "--quiet", "--timings-file", str(case_dir / "timing.json"),
                   "--tokens-file", str(case_dir / "tokens.json"), "--draft-max", str(depth),
                   "--draft-expert-topk", str(experts)]
        if depth == 0:
            command.append("--ar")
        sys.stdout.write("START " + label + "\n")
        sys.stdout.flush()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        with (case_dir / "run.log").open("wb") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                       stdin=subprocess.DEVNULL, preexec_fn=limit, check=False)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        case = {"id": label, "depth": depth, "experts": experts, "warmup": warmup,
                "exit_code": completed.returncode, "process_seconds": time.monotonic() - start,
                "major_faults": after.ru_majflt - before.ru_majflt,
                "minor_faults": after.ru_minflt - before.ru_minflt,
                "input_blocks": after.ru_inblock - before.ru_inblock,
                "user_seconds": after.ru_utime - before.ru_utime,
                "system_seconds": after.ru_stime - before.ru_stime,
                "machine_after": machine(), "valid": False}
        report["cases"].append(case)
        save()
        if completed.returncode:
            raise RuntimeError(f"{label}: exit {completed.returncode}, see run.log")
        timing = json.loads((case_dir / "timing.json").read_text())
        tokens = json.loads((case_dir / "tokens.json").read_text())
        if reference is None:
            reference = tokens
        phases = sum(
            timing[k] for k in (
                "draft_ms", "verify_with_history_ms", "snapshot_ms",
                "restore_ms", "commit_ms", "ar_decode_ms", "sampling_ms",
            )
        )
        valid = (tokens == reference and len(tokens) == args.tokens and not timing["audit"]
                 and timing["quiet"] and timing["decode_tokens"] == args.tokens - 1
                 and 0 <= phases <= timing["generation_ms"] + 0.1)
        case.update({"timing": timing, "tokens": tokens, "valid": valid,
                     "decode_tokens_per_second": timing["decode_tokens"] * 1000 / timing["generation_ms"],
                     "other_ms": timing["generation_ms"] - phases})
        save()
        sys.stdout.write(f"END {label} valid={valid} tps={case['decode_tokens_per_second']:.4f}\n")
        sys.stdout.flush()
        if not valid:
            raise RuntimeError(f"{label}: token equivalence or timing accounting failed")

    save()
    try:
        # Same full output as measured runs, excluded from aggregates. Warms the
        # model pages used by the AR trajectory without clearing global caches.
        run("warmup-ar", warmup=True)
        order = [(d, k) for d in args.depths for k in args.experts]
        rng = random.Random(35018)
        for repetition in range(args.repeats):
            run(f"r{repetition + 1}-ar")
            rng.shuffle(order)
            for depth, experts in order:
                run(f"r{repetition + 1}-d{depth}-k{experts}", depth, experts)
        run("final-ar")
        report.update({"status": "passed", "finished_at": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report["end_machine"] = machine()
        save()


if __name__ == "__main__":
    main()
