#!/usr/bin/env python3
"""Check the actual greedy loop against AR, including zero draft depth."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe, model = sys.argv[1:3]
gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)

    def run(extra, filename):
        path = directory / filename
        command = [exe, "-m", model, "-p", "test ab", "-ngl", gpu, "-n", "40", "-c", "128",
                   "--ignore-eos", "--tokens-file", str(path), *extra]
        result = subprocess.run(command, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return json.loads(path.read_text())

    reference = run(["--ar"], "ar.json")
    assert len(reference) == 40
    for depth in (0, 1, 3, 8, 31):
        for experts in (1, 2):
            actual = run(["--draft-max", str(depth), "--draft-expert-topk", str(experts), "--audit-state-copies"],
                         f"d{depth}-k{experts}.json")
            if actual != reference:
                raise AssertionError(f"greedy token mismatch: depth={depth}, experts={experts}")
            print(f"PASS greedy AR equality depth={depth}, experts={experts}, tokens={len(actual)}")
    for args in (["--draft-max", "32"], ["--draft-max", "8", "-ub", "4"],
                 ["--draft-skip-attn", "0"], ["-np", "2"], ["--temp", "0.8"], ["--prune", "1"]):
        result = subprocess.run([exe, "-m", model, *args], capture_output=True)
        if result.returncode == 0:
            raise AssertionError(f"unsupported configuration accepted: {args}")
    print("PASS unsupported CLI configurations rejected")
