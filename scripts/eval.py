import json
import os
import re
import subprocess
import time
from pathlib import Path


LLAMA_BIN = "./build/bin/llama-speculative"
MODEL_PATH = "/opt/pretrained_models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat-Q8_0.gguf"
DRAFT_MODEL_PATH = "/opt/pretrained_models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat-draft-q4.gguf"

DATASET_PATH = "datastes/gsm8K.jsonl"
PROMPT_FIELD = "question"
HUMAN_EVAL_TURNS_FIELD = "turns"
HUMAN_EVAL_TASK_ID_FIELD = "question_id"
MAX_QUESTIONS = 10 

N_TOKENS = 1024
NGL = 99
TEMP = 0
DRAFT_MAX = 5
MOE_REUSE_STRENGTH = 0.8
MOE_REUSE_EXPERT_CAP = 8
DRAFT_SHARE_KV = True
PRUNE = 0
PRUNE_MAX_DEPTH = 10
PRUNE_MAX_NODES = 10

CUDA_VISIBLE_DEVICES = "0,1,2,3"  
TIMEOUT_SEC = 300

OUTPUT_DIR = "answers"
OUTPUT_FILE = "gsm8K_answers.jsonl"
OUTPUT_FORMAT = "gsm8k"  # "gsm8k" or "human_eval"

BASELINE_TOKENS_PER_SECOND = 60


def build_command(prompt):
    cmd = [
        LLAMA_BIN,
        "-m",
        MODEL_PATH,
        "-md",
        DRAFT_MODEL_PATH,
        "-p",
        prompt,
        "-n",
        str(N_TOKENS),
        "-ngl",
        str(NGL),
        "--temp",
        str(TEMP),
        "--draft-max",
        str(DRAFT_MAX),
        "--moe-reuse-strength",
        str(MOE_REUSE_STRENGTH),
        "--moe-reuse-expert-cap",
        str(MOE_REUSE_EXPERT_CAP),
    ]
    if DRAFT_SHARE_KV:
        cmd.append("--draft-share-kv")
    if PRUNE is not None:
        cmd.extend(["--prune", str(PRUNE)])
        cmd.extend(["--prune-max-depth", str(PRUNE_MAX_DEPTH)])
        cmd.extend(["--prune-max-nodes", str(PRUNE_MAX_NODES)])
    return cmd


def parse_perf(stderr_text):
    def _match(pattern):
        m = re.search(pattern, stderr_text)
        if not m:
            return None
        return float(m.group(1)), int(m.group(2))

    total = _match(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
    prompt = _match(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
    eval_t = _match(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
    accept_percent = None
    m = re.search(r"accept\s*=\s*([0-9.]+)%", stderr_text)
    if m:
        accept_percent = float(m.group(1))
    else:
        m = re.search(r"acceptance rate:\s*([0-9.]+)", stderr_text)
        if m:
            val = float(m.group(1))
            accept_percent = val * 100.0 if val <= 1.0 else val

    metrics = {}
    if total:
        total_ms, total_tokens = total
        metrics["total_time_ms"] = total_ms
        metrics["total_tokens"] = total_tokens
        if total_ms > 0:
            metrics["total_tokens_per_second"] = total_tokens / (total_ms / 1000.0)
    if prompt:
        prompt_ms, prompt_tokens = prompt
        metrics["prompt_time_ms"] = prompt_ms
        metrics["prompt_tokens"] = prompt_tokens
        if prompt_ms > 0:
            metrics["prompt_tokens_per_second"] = prompt_tokens / (prompt_ms / 1000.0)
    if eval_t:
        eval_ms, eval_tokens = eval_t
        metrics["decode_time_ms"] = eval_ms
        metrics["decode_tokens"] = eval_tokens
        if eval_ms > 0:
            metrics["decode_tokens_per_second"] = eval_tokens / (eval_ms / 1000.0)
    if accept_percent is not None:
        metrics["acceptance_rate_percent"] = accept_percent

    if BASELINE_TOKENS_PER_SECOND and metrics.get("total_tokens_per_second"):
        metrics["speedup_vs_baseline"] = (
            metrics["total_tokens_per_second"] / BASELINE_TOKENS_PER_SECOND
        )

    return metrics


def load_jsonl(path, max_items):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_items is not None and idx >= max_items:
                return
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def get_prompt(item):
    if OUTPUT_FORMAT == "human_eval":
        turns = item.get(HUMAN_EVAL_TURNS_FIELD)
        if not isinstance(turns, list) or not turns:
            raise KeyError("Missing or empty 'turns' for human_eval")
        return turns[0]
    prompt = item.get(PROMPT_FIELD)
    if prompt is None:
        raise KeyError(f"Missing field '{PROMPT_FIELD}'")
    return prompt


def get_task_id(item, default_idx):
    if OUTPUT_FORMAT == "human_eval":
        return item.get(HUMAN_EVAL_TASK_ID_FIELD, str(default_idx))
    return default_idx


def main():
    dataset_path = Path(DATASET_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE
    if output_path.exists():
        output_path.unlink()

    env = os.environ.copy()
    if CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

    results = []
    start_all = time.time()
    throughput_values = []
    acceptance_values = []
    speedup_values = []

    for idx, item in load_jsonl(dataset_path, MAX_QUESTIONS):
        prompt = get_prompt(item)
        task_id = get_task_id(item, idx)

        cmd = build_command(prompt)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=TIMEOUT_SEC,
        )

        metrics = parse_perf(proc.stderr)
        if OUTPUT_FORMAT == "human_eval":
            record = {
                "task_id": task_id,
                "completion": proc.stdout,
            }
        elif OUTPUT_FORMAT == "gsm8k":
            record = {
                "id": idx,
                "prediction": proc.stdout,
                "metrics": metrics,
            }
        else:
            raise ValueError(f"Unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")
        results.append(record)

        if metrics.get("total_tokens_per_second"):
            throughput_values.append(metrics["total_tokens_per_second"])
        if metrics.get("acceptance_rate_percent") is not None:
            acceptance_values.append(metrics["acceptance_rate_percent"])
        if metrics.get("speedup_vs_baseline"):
            speedup_values.append(metrics["speedup_vs_baseline"])

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_sec = time.time() - start_all
    summary = {
        "summary": True,
        "count": len(results),
        "wall_time_sec": total_sec,
    }
    if throughput_values:
        summary["avg_tokens_per_second"] = sum(throughput_values) / len(throughput_values)
    if acceptance_values:
        summary["avg_acceptance_rate_percent"] = sum(acceptance_values) / len(acceptance_values)
    if speedup_values:
        summary["avg_speedup_vs_baseline"] = sum(speedup_values) / len(speedup_values)

    if OUTPUT_FORMAT == "gsm8k":
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_path}")
    if summary.get("avg_tokens_per_second") is not None:
        print(f"Average throughput: {summary['avg_tokens_per_second']:.2f} tokens/s")
    if summary.get("avg_acceptance_rate_percent") is not None:
        print(f"Average acceptance rate: {summary['avg_acceptance_rate_percent']:.2f}%")
    if summary.get("avg_speedup_vs_baseline") is not None:
        print(f"Average speedup: {summary['avg_speedup_vs_baseline']:.2f}x")


if __name__ == "__main__":
    main()
