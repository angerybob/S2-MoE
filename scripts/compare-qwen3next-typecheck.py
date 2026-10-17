#!/usr/bin/env python3
"""Compare original-CI Pyright diagnostics in one shared Python environment."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=False, **kwargs)


def dependencies(root):
    paths = set(root.rglob("requirements*.txt"))
    paths.update((root / "requirements").glob("*.txt"))
    paths.update(root / name for name in ("pyrightconfig.json", "pyproject.toml",
                                         ".github/workflows/python-type-check.yml"))
    return {str(path.relative_to(root)): path.read_bytes() for path in paths if path.is_file()}


def check(root, executable, output, label):
    result = command([str(executable), "--level", "warning", "--warnings", "--outputjson"], cwd=root)
    (output / f"{label}-stdout.json").write_text(result.stdout, encoding="utf-8")
    (output / f"{label}-stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"{label}: Pyright exited {result.returncode}; comparison is inconclusive")
    report = json.loads(result.stdout)
    normalized = []
    for diag in report["generalDiagnostics"]:
        if diag["severity"] not in ("error", "warning"):
            continue
        filename = Path(diag["file"]).resolve()
        normalized.append({
            "file": str(filename.relative_to(root)) if filename.is_relative_to(root) else str(filename),
            "severity": diag["severity"], "range": diag.get("range"), "rule": diag.get("rule"),
            "message": diag["message"].replace(str(root), "<repo>"),
        })
    return {"revision": command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip(),
            "exit_code": result.returncode, "summary": report["summary"], "diagnostics": normalized}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pyright", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, candidate, output = args.base.resolve(), args.candidate.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if dependencies(base) != dependencies(candidate):
        raise RuntimeError("dependency inputs or the original type-check configuration differ")
    version = command([str(args.pyright.resolve()), "--version"]).stdout.strip()
    if version != "pyright 1.1.382":
        raise RuntimeError(f"expected original-CI pyright 1.1.382, got {version!r}")
    freeze = command([sys.executable, "-m", "pip", "freeze", "--all"])
    if freeze.returncode:
        raise RuntimeError("cannot record installed Python dependencies")
    (output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    baseline = check(base, args.pyright.resolve(), output, "base")
    branch = check(candidate, args.pyright.resolve(), output, "candidate")
    before = Counter(json.dumps(d, sort_keys=True) for d in baseline["diagnostics"])
    after = Counter(json.dumps(d, sort_keys=True) for d in branch["diagnostics"])
    added = [json.loads(d) for d in (after - before).elements()]
    removed = [json.loads(d) for d in (before - after).elements()]
    result = {"schema": 1, "python": sys.version, "platform": platform.platform(), "pyright": version,
              "same_dependency_inputs": True, "shared_dependency_environment": True,
              "cli_flags": ["--level", "warning", "--warnings", "--outputjson"],
              "base": baseline, "candidate": branch, "added": added, "removed": removed,
              "diagnostics_identical": before == after, "new_diagnostics": len(added)}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = (f"## Qwen3-Next / orin type-check comparison\n\n"
               f"Same Python process environment and dependency inputs; {version}.\n\n"
               f"| Revision | Errors | Warnings | Exit |\n|---|---:|---:|---:|\n"
               f"| orin `{baseline['revision'][:9]}` | {baseline['summary']['errorCount']} | "
               f"{baseline['summary']['warningCount']} | {baseline['exit_code']} |\n"
               f"| candidate `{branch['revision'][:9]}` | {branch['summary']['errorCount']} | "
               f"{branch['summary']['warningCount']} | {branch['exit_code']} |\n\n"
               f"New diagnostics: **{len(added)}**; removed: **{len(removed)}**; "
               f"identical normalized diagnostics: **{before == after}**.\n\n"
               f"Raw reports and `pip-freeze.txt` are in the uploaded artifact. "
               f"A successful comparison does not mean the repository type-check passes.\n")
    (output / "summary.md").write_text(summary, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary)
    sys.stdout.write(summary)
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
