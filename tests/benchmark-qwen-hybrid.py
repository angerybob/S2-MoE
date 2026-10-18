#!/usr/bin/env python3
"""Opt-in Linux CPU/CUDA benchmark of experimental hybrid entry points (no audit)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
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
    parser.add_argument("--gpu-layers", type=int, default=99, help="99: full GPU offload; 0: CPU only")
    parser.add_argument("--ssd-moe", action="store_true")
    parser.add_argument("--memory-budget-mib", type=int, default=0)
    parser.add_argument("--runtime-reserve-mib", type=int, default=2048)
    parser.add_argument("--hot-experts", type=Path)
    parser.add_argument("--draft-branches", type=int, default=1)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--prune-score-thresh", type=float, default=1)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resident-model", action="store_true", help="load once, fresh contexts for each case")
    parser.add_argument("--confirm-prompts", action="store_true", help="resident code/math checks at depths 1/8, top-k 4")
    parser.add_argument("--min-available-gib", type=float, default=2.5, help="resident-run physical memory guard")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit-gib", type=int, default=0, help="CPU-only child address-space cap; 0 disables")
    args = parser.parse_args()
    placement = []
    if args.ssd_moe:
        if args.gpu_layers < 99:
            parser.error("SSD hybrid execution requires -ngl 99")
        placement.append("--ssd-moe")
    if args.memory_budget_mib:
        placement += ["--memory-budget-mib", str(args.memory_budget_mib),
                      "--runtime-reserve-mib", str(args.runtime_reserve_mib)]
    if args.hot_experts:
        placement += ["--hot-experts", str(args.hot_experts.resolve())]
    placement += ["--draft-branches", str(args.draft_branches)]
    if args.prune:
        placement += ["--prune", "1", "--prune-score-thresh", str(args.prune_score_thresh)]
    if (args.tokens < 2 or args.repeats < 1 or args.threads < 1 or args.timeout < 1
            or args.gpu_layers < 0 or args.limit_gib < 0
            or not all(1 <= d <= 31 and d + 1 <= args.ubatch for d in args.depths)
            or not all(k > 0 for k in args.experts)):
        parser.error("invalid generation, draft, or repetition limits")
    if args.gpu_layers > 0 and args.limit_gib:
        parser.error("CUDA reserves large virtual address ranges; use --limit-gib 0")
    if args.confirm_prompts and not args.resident_model:
        parser.error("--confirm-prompts requires --resident-model")
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
                           "ubatch": args.ubatch, "gpu_layers": args.gpu_layers, "audit": False, "quiet": True,
                           "resident_model": args.resident_model,
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

    prompts = {"main": args.prompt}
    if args.confirm_prompts:
        prompts.update({"code": "Write a Python function that returns the nth Fibonacci number.",
                        "math": "Solve step by step: A shop sells 12 notebooks at 3 dollars each. What is the total price?"})
    report["prompts"] = prompts
    references = {}

    def prompt_id(label):
        return label.split("-")[1] if label.startswith("confirm-") else "main"

    def run(label, depth=0, experts=1, warmup=False):
        if stamp(binary) != binary_stamp or stamp(model) != model_stamp:
            raise RuntimeError("binary or model changed during benchmark")
        case_dir = output / label
        case_dir.mkdir()
        command = [str(binary), "-m", str(model), "-p", args.prompt, "-n", str(args.tokens),
                   "-ngl", str(args.gpu_layers), "-c", str(args.context), "-ub", str(args.ubatch), "-t", str(args.threads),
                   "--ignore-eos", "--quiet", "--timings-file", str(case_dir / "timing.json"),
                   "--tokens-file", str(case_dir / "tokens.json"), "--draft-max", str(depth),
                   "--draft-expert-topk", str(experts), *placement]
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
        collect(case)

    def collect(case):
        label = case["id"]
        case["prompt_id"] = prompt_id(label)
        case_dir = output / label
        report["cases"].append(case)
        save()
        if case["exit_code"]:
            raise RuntimeError(f"{label}: exit {case['exit_code']}, see run.log")
        timing = json.loads((case_dir / "timing.json").read_text())
        tokens = json.loads((case_dir / "tokens.json").read_text())
        reference = references.setdefault(case["prompt_id"], tokens)
        phases = sum(
            timing[k] for k in (
                "draft_ms", "verify_with_history_ms", "snapshot_ms",
                "restore_ms", "commit_ms", "ar_decode_ms", "sampling_ms",
            )
        )
        valid = (tokens == reference and len(tokens) == args.tokens and not timing["audit"]
                 and timing["quiet"] and timing["decode_tokens"] == args.tokens - 1
                 and not timing.get("logits_traced", False) and timing.get("batch_probe_at", -1) < 0
                 and 0 <= phases <= timing["generation_ms"] + 0.1)
        case.update({"timing": timing, "tokens": tokens, "valid": valid,
                     "decode_tokens_per_second": timing["decode_tokens"] * 1000 / timing["generation_ms"],
                     "other_ms": timing["generation_ms"] - phases})
        save()
        sys.stdout.write(f"END {label} valid={valid} tps={case['decode_tokens_per_second']:.4f}\n")
        sys.stdout.flush()
        if not valid:
            raise RuntimeError(f"{label}: token equivalence or timing accounting failed")

    def schedule():
        items = [("warmup-ar", 0, 1, True)]
        order = [(d, k) for d in args.depths for k in args.experts]
        rng = random.Random(35018)
        for repetition in range(args.repeats):
            items.append((f"r{repetition + 1}-ar", 0, 1, False))
            rng.shuffle(order)
            items.extend((f"r{repetition + 1}-d{d}-k{k}", d, k, False) for d, k in order)
        items.append(("final-ar", 0, 1, False))
        if args.confirm_prompts:
            for name in ("code", "math"):
                items.extend([(f"confirm-{name}-warmup-ar", 0, 1, True),
                              (f"confirm-{name}-ar", 0, 1, False),
                              (f"confirm-{name}-d1-k4", 1, 4, False),
                              (f"confirm-{name}-d8-k4", 8, 4, False),
                              (f"confirm-{name}-final-ar", 0, 1, False)])
        return items

    def resident(items):
        rows = []
        for label, depth, experts, _ in items:
            case_dir = output / label
            case_dir.mkdir()
            paths = [json.dumps(str(case_dir / name), ensure_ascii=False) for name in ("tokens.json", "timing.json")]
            mode = "AR" if depth == 0 else "SPEC"
            rows.append(f"{mode} {depth} {experts} " + " ".join(paths)
                        + " " + json.dumps(prompts[prompt_id(label)], ensure_ascii=False))
        plan = output / "plan.txt"
        plan.write_text("\n".join(rows) + "\n", encoding="utf-8")
        command = [str(binary), "-m", str(model), "-p", args.prompt, "-n", str(args.tokens),
                   "-ngl", str(args.gpu_layers), "-c", str(args.context), "-ub", str(args.ubatch),
                   "-t", str(args.threads), "--ignore-eos", "--quiet", "--benchmark-plan", str(plan), *placement]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        next_case = 0
        low = None

        def harvest():
            nonlocal next_case
            while next_case < len(items):
                label, depth, experts, warmup = items[next_case]
                case_dir = output / label
                try:
                    json.loads((case_dir / "tokens.json").read_text())
                    json.loads((case_dir / "timing.json").read_text())
                except (OSError, json.JSONDecodeError):
                    break
                if stamp(binary) != binary_stamp or stamp(model) != model_stamp:
                    raise RuntimeError("binary or model changed during benchmark")
                collect({"id": label, "depth": depth, "experts": experts, "warmup": warmup,
                         "exit_code": 0, "log": "resident.log", "process_seconds": None,
                         "major_faults": None, "minor_faults": None, "input_blocks": None,
                         "user_seconds": None, "system_seconds": None,
                         "machine_after": machine(), "valid": False})
                next_case += 1

        with (output / "resident.log").open("wb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, preexec_fn=limit, start_new_session=True)
            try:
                while child.poll() is None:
                    mem = {line.split(":")[0]: int(line.split()[1])
                           for line in Path("/proc/meminfo").read_text().splitlines()}
                    available = mem["MemAvailable"] / 1024**2
                    low = available if low is None else min(low, available)
                    if available < args.min_available_gib or time.monotonic() - start > args.timeout:
                        raise RuntimeError("resident child reached memory guard or timeout")
                    harvest()
                    time.sleep(0.25)
                harvest()
                if child.returncode or next_case != len(items):
                    raise RuntimeError(f"resident child failed: exit {child.returncode}, {next_case}/{len(items)} cases")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report["resident_process"] = {"seconds": time.monotonic() - start, "min_available_gib": low,
                                      "major_faults": after.ru_majflt - before.ru_majflt,
                                      "minor_faults": after.ru_minflt - before.ru_minflt,
                                      "model_load_ms": report["cases"][0]["timing"]["model_load_ms"]}

    save()
    try:
        # Excluded full-length AR warmup; measured AR brackets each sweep.
        items = schedule()
        if args.resident_model:
            resident(items)
        else:
            for label, depth, experts, warmup in items:
                run(label, depth, experts, warmup)
        report.update({"status": "passed", "finished_at": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report["end_machine"] = machine()
        save()


if __name__ == "__main__":
    main()
