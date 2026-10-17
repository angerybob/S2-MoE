#!/usr/bin/env python3
"""Compare the same ordinary AR driver linked to candidate and pinned upstream."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROMPTS = [
    "Explain why the sky looks blue in two short sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Solve step by step: A shop sells 12 notebooks at 3 dollars each. What is the total price?",
]
TOKENIZER_PROMPTS = [
    "Hello, world!", "I'm testing 1234567890.", "cafe\u0301 naive\u0308", "na\u0308ive coo\u0308perate",
    "你好，世界！", "اَلْعَرَبِيَّة", "हिन्दी १२३", "ภาษาไทย", "emoji 👩‍💻 🇨🇳",
    "line1\n\n  line2\tend", "<|im_start|>user\nHello<|im_end|>", " A\u030a Å e\u0301 é",
]


def stamp(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 64 or args.timeout <= 0:
        parser.error("tokens must be in [1,64] and timeout positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate, reference, model = args.candidate.resolve(), args.reference.resolve(), args.model.resolve()
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    report = {"schema": 1, "source_revision": revision, "source_dirty": dirty,
              "reference_revision": args.reference_revision, "model": stamp(model),
              "candidate_binary": stamp(candidate), "reference_binary": stamp(reference),
              "mode": "tokenizer" if args.tokenizer_only else "ordinary_ar",
              "tokens": 0 if args.tokenizer_only else args.tokens, "cases": [], "status": "running"}
    env = dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1", LLAMA_GRAPH_REUSE_DISABLE="1")
    for key in ("QWEN35_TRACE", "QWEN35_DUMP_LOGITS", "QWEN35_FORCE_TOKENS"):
        env.pop(key, None)

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    save()
    for index, prompt in enumerate(TOKENIZER_PROMPTS if args.tokenizer_only else PROMPTS):
        results = {}
        variants = [("upstream", reference, 32), ("candidate", candidate, 32)]
        if not args.tokenizer_only:
            variants.append(("candidate_serial_prefill", candidate, 1))
        for label, binary, ubatch in variants:
            dst = output / f"{index}-{label}.json"
            command = [str(binary), str(model), prompt, str(dst), str(report["tokens"]), str(ubatch)]
            with (output / f"{index}-{label}.log").open("wb") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=False)
            if completed.returncode:
                report["status"] = "failed"
                report["error"] = f"{index}/{label}: exit {completed.returncode}"
                save()
                return 1
            results[label] = json.loads(dst.read_text(encoding="utf-8"))
        expected = results["upstream"]
        comparisons = {label: result == expected and len(result["generated_tokens"]) == report["tokens"]
                       for label, result in results.items() if label != "upstream"}
        report["cases"].append({"prompt": prompt, "results": results, "matches_reference": comparisons})
        save()
        sys.stdout.write(f"case {index}: {comparisons}\n")
        sys.stdout.flush()
    comparisons = [equal for case in report["cases"] for equal in case["matches_reference"].values()]
    report["coverage"] = {"comparisons": len(comparisons), "matching": sum(comparisons)}
    report["status"] = "passed" if comparisons and all(comparisons) else "failed"
    save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
