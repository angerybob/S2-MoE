#!/usr/bin/env python3
"""Exercise architecture dispatch and the public JSON runner end to end."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    entry, dedicated, model, runner = [Path(arg).resolve() for arg in sys.argv[1:]]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        reference = directory / "reference.json"
        result = subprocess.run([str(dedicated), "-m", str(model), "-ngl", "0", "-n", "16", "-c", "128",
                                 "--ar", "--ignore-eos", "--quiet", "-p", "test ab", "--tokens-file", str(reference)],
                                capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        output = directory / "tree.json"
        config = {"engine": "hybrid", "binary_path": str(entry), "model": str(model), "prompt": "test ab",
                  "n_gpu_layers": 0, "n_ctx": 128, "n_predict": 16, "ignore_eos": True, "quiet": True,
                  "tokens_file": str(output), "speculative": {"draft_max": 4, "draft_expert_topk": 1, "branches": 2},
                  "pruning": {"enable": True, "score_thresh": 0}}
        source = directory / "config.json"
        source.write_text(json.dumps(config))
        result = subprocess.run([sys.executable, str(runner), "--config", str(source)], capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        if json.loads(reference.read_text()) != json.loads(output.read_text()):
            raise AssertionError("main/JSON hybrid dispatcher differs from dedicated AR")
    print("PASS public main entry + hybrid JSON configuration + adaptive tree against dedicated AR")


if __name__ == "__main__":
    main()
