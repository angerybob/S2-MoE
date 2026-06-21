#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from bayes_opt import BayesianOptimization


DECODED_SPEED_RE = re.compile(r"decoded\s+\d+\s+tokens.*?speed:\s*([0-9.]+)\s*t/s", re.IGNORECASE)
ACCEPT_RATE_RE = re.compile(r"accept\s*=\s*([0-9.]+)%", re.IGNORECASE)
SEARCH_RESULT_RE = re.compile(r"^SEARCH_RESULT\s+(\{.*\})\s*$")


def parse_tokens_per_s(output: str) -> float:
    last = None
    for line in output.splitlines():
        match = DECODED_SPEED_RE.search(line)
        if match:
            last = float(match.group(1))
    if last is None:
        raise RuntimeError("failed to parse throughput from llama-speculative output")
    return last


def parse_accept_rate(output: str) -> Optional[float]:
    last = None
    for line in output.splitlines():
        match = ACCEPT_RATE_RE.search(line)
        if match:
            last = float(match.group(1))
    return last


def format_int_list(values: List[int]) -> str:
    return ",".join(str(v) for v in values)


class SpeculativeRunner:
    def __init__(
        self,
        base_cmd: List[str],
        prompt: str,
        verbose: bool,
        env: Optional[Dict[str, str]] = None,
    ):
        cmd = list(base_cmd)
        cmd.append("--search-stdin")
        self.prompt = prompt
        self.verbose = verbose
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to start llama-speculative process")

    def evaluate(
        self,
        attn_skip: List[int],
        mlp_skip: List[int],
        expert_topk: Optional[int],
        layer_topk: Optional[List[int]],
    ) -> Tuple[float, Optional[float]]:
        payload = {
            "prompt": self.prompt,
            "draft_skip_attn": attn_skip,
            "draft_skip_mlp": mlp_skip,
        }
        if expert_topk is not None:
            payload["draft_expert_topk"] = expert_topk
        if layer_topk:
            if any(k <= 0 for k in layer_topk):
                raise ValueError("layer_topk must be >= 1 for all layers")
            payload["draft_layer_topk"] = layer_topk
        line = json.dumps(payload, ensure_ascii=False)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

        output_lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "" and self.proc.poll() is not None:
                raise RuntimeError("llama-speculative exited before returning a result")
            if line:
                output_lines.append(line)
                if self.verbose:
                    print(line, end="", flush=True)
                match = SEARCH_RESULT_RE.match(line.strip())
                if match:
                    result = json.loads(match.group(1))
                    return float(result["tokens_per_s"]), result.get("accept_rate")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


class LayerSkippingSearching:
    def __init__(
        self,
        base_cmd: List[str],
        n_layers: int,
        expert_topk_range: Optional[Tuple[int, int]] = None,
        per_layer_expert_topk_range: Optional[Tuple[int, int]] = None,
        probe_topk: int = 3,
        verbose: bool = False,
        env: Optional[Dict[str, str]] = None,
        runner: Optional[SpeculativeRunner] = None,
        skip_threshold: float = 0.6,
    ):
        self.base_cmd = base_cmd
        self.n_layers = n_layers
        self.candidate_layers = list(range(0, n_layers))
        self.expert_topk_range = expert_topk_range
        self.per_layer_expert_topk_range = per_layer_expert_topk_range
        self.probe_topk = probe_topk
        self.verbose = verbose
        self.env = env
        self.runner = runner
        self.skip_threshold = skip_threshold
        self._eval_count = 0

        self.pbounds: Dict[str, Tuple[float, float]] = {
            f"x{i}": (0.0, 1.0) for i in range(len(self.candidate_layers) * 2)
        }
        if self.expert_topk_range is not None:
            self.pbounds["expert_topk"] = (float(self.expert_topk_range[0]), float(self.expert_topk_range[1]))
        if self.per_layer_expert_topk_range is not None:
            for idx in range(len(self.candidate_layers)):
                self.pbounds[f"expert_topk_{idx}"] = (
                    float(self.per_layer_expert_topk_range[0]),
                    float(self.per_layer_expert_topk_range[1]),
                )

        self.optimizer = BayesianOptimization(
            f=self._black_box_evaluate_function,
            pbounds=self.pbounds,
            random_state=1,
            verbose=1,
        )

    def _black_box_evaluate_function(self, **kargs):
        self._eval_count += 1
        n = len(self.candidate_layers)
        attn_scores = []
        mlp_scores = []
        for idx, layer_id in enumerate(self.candidate_layers):
            attn_scores.append((kargs[f"x{idx}"], layer_id))
            mlp_scores.append((kargs[f"x{idx + n}"], layer_id))

        attn_skip_layers = [layer_id for score, layer_id in attn_scores if score > self.skip_threshold]
        mlp_skip_layers = [layer_id for score, layer_id in mlp_scores if score > self.skip_threshold]

        # no skip count limiting

        expert_topk = None
        if self.expert_topk_range is not None:
            expert_topk = int(round(kargs["expert_topk"]))
            expert_topk = max(self.expert_topk_range[0], min(self.expert_topk_range[1], expert_topk))

        layer_topk = None
        if self.per_layer_expert_topk_range is not None:
            layer_topk = [self.probe_topk] * self.n_layers
            for idx, layer_id in enumerate(self.candidate_layers):
                k = int(round(kargs[f"expert_topk_{idx}"]))
                k = max(self.per_layer_expert_topk_range[0], min(self.per_layer_expert_topk_range[1], k))
                layer_topk[layer_id] = k

        if not self.runner:
            raise RuntimeError("SpeculativeRunner is not initialized")
        tokens_per_s, accept_rate = self.runner.evaluate(
            attn_skip_layers,
            mlp_skip_layers,
            expert_topk,
            layer_topk,
        )

        accept_str = f"{accept_rate:.3f}%" if accept_rate is not None else "n/a"
        print(
            f"[Eval {self._eval_count}] {tokens_per_s:.3f} t/s accept={accept_str} "
            f"attn_skip={len(attn_skip_layers)} mlp_skip={len(mlp_skip_layers)} "
            f"expert_topk={expert_topk} layer_topk={'set' if layer_topk else 'none'} "
            f"attn_ids={attn_skip_layers} mlp_ids={mlp_skip_layers}"
        )
        if layer_topk is not None:
            print(f"[Eval {self._eval_count}] layer_topk={layer_topk}")

        return tokens_per_s

    def probe(self, attn_skip_layers: List[int], mlp_skip_layers: List[int]):
        params = {f"x{i}": 0.0 for i in range(len(self.candidate_layers) * 2)}
        for lid in attn_skip_layers:
            if lid in self.candidate_layers:
                idx = self.candidate_layers.index(lid)
                params[f"x{idx}"] = 1.0
        for lid in mlp_skip_layers:
            if lid in self.candidate_layers:
                idx = self.candidate_layers.index(lid)
                params[f"x{idx + len(self.candidate_layers)}"] = 1.0
        if self.expert_topk_range is not None:
            params["expert_topk"] = float(self.probe_topk)
        if self.per_layer_expert_topk_range is not None:
            for idx in range(len(self.candidate_layers)):
                params[f"expert_topk_{idx}"] = float(self.probe_topk)
        self.optimizer.probe(params=params, lazy=True)

    def search(self, n_iter: int):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        return self.get_solution()

    def get_solution(self):
        skip_attn_layers = []
        for idx, layer_id in enumerate(self.candidate_layers):
            if self.optimizer.max["params"][f"x{idx}"] > 0.5:
                skip_attn_layers.append(layer_id)

        skip_mlp_layers = []
        offset = len(self.candidate_layers)
        for idx, layer_id in enumerate(self.candidate_layers):
            if self.optimizer.max["params"][f"x{idx + offset}"] > 0.5:
                skip_mlp_layers.append(layer_id)

        best_expert_topk = None
        if self.expert_topk_range is not None:
            best_expert_topk = int(round(self.optimizer.max["params"]["expert_topk"]))
            best_expert_topk = max(self.expert_topk_range[0], min(self.expert_topk_range[1], best_expert_topk))

        best_layer_topk = None
        if self.per_layer_expert_topk_range is not None:
            best_layer_topk = [self.probe_topk] * self.n_layers
            for idx, layer_id in enumerate(self.candidate_layers):
                k = int(round(self.optimizer.max["params"][f"expert_topk_{idx}"]))
                k = max(self.per_layer_expert_topk_range[0], min(self.per_layer_expert_topk_range[1], k))
                best_layer_topk[layer_id] = k

        return skip_attn_layers, skip_mlp_layers, best_expert_topk, best_layer_topk


def build_base_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        args.binary,
        "-m",
        args.model,
        "-md",
        args.draft_model,
        "-p",
        args.prompt,
        "--parallel",
        str(args.parallel),
        "--moe-reuse-strength",
        str(args.moe_reuse_strength),
        "--moe-reuse-expert-cap",
        str(args.moe_reuse_expert_cap),
        "--prune",
        str(args.prune),
        "--prune-max-depth",
        str(args.prune_max_depth),
        "--prune-max-nodes",
        str(args.prune_max_nodes),
        "--prune-budget",
        str(args.prune_budget),
        "--prune-m-route",
        str(args.prune_m_route),
        "--prune-k-tgt",
        str(args.prune_k_tgt),
        "--prune-beta",
        str(args.prune_beta),
        "--prune-gamma",
        str(args.prune_gamma),
        "--prune-lambda",
        str(args.prune_lambda),
        "--prune-tpot",
        str(args.prune_tpot),
        "--prune-eps",
        str(args.prune_eps),
        "--prune-expert-bytes",
        str(args.prune_expert_bytes),
        "--prune-bandwidth",
        str(args.prune_bandwidth),
        "--prune-expert-max",
        str(args.prune_expert_max),
        "--draft",
        str(args.draft_tokens),
        "--temp",
        str(args.temp),
        "--draft-p-split",
        str(args.draft_p_split),
        "-ngl",
        str(args.ngl),
        "-ngld",
        str(args.ngld),
        "-n",
        str(args.n_predict),
    ]
    if args.draft_share_kv:
        cmd.append("--draft-share-kv")
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Bayesian search for draft skip layers and MoE top-k.")
    parser.add_argument("--binary", default="./build/bin/llama-speculative")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n-layers", type=int, default=48)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--moe-reuse-strength", type=float, default=1.4)
    parser.add_argument("--moe-reuse-expert-cap", type=int, default=14)
    parser.add_argument("--prune", type=int, default=1)
    parser.add_argument("--prune-max-depth", type=int, default=10)
    parser.add_argument("--prune-max-nodes", type=int, default=60)
    parser.add_argument("--prune-budget", type=float, default=145.0)
    parser.add_argument("--prune-m-route", type=int, default=8)
    parser.add_argument("--prune-k-tgt", type=int, default=8)
    parser.add_argument("--prune-beta", type=float, default=0.8)
    parser.add_argument("--prune-gamma", type=float, default=2.0)
    parser.add_argument("--prune-lambda", type=float, default=1.0)
    parser.add_argument("--prune-tpot", type=float, default=54.0)
    parser.add_argument("--prune-eps", type=float, default=24.0)
    parser.add_argument("--prune-expert-bytes", type=float, default=16.384)
    parser.add_argument("--prune-bandwidth", type=float, default=204.8)
    parser.add_argument("--prune-expert-max", type=int, default=384)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--draft-p-split", type=float, default=0.1)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ngld", type=int, default=99)
    parser.add_argument("--n-predict", type=int, default=512)
    parser.add_argument("--draft-share-kv", action="store_true")
    parser.add_argument("--expert-topk-range", type=str, default="3,3")
    parser.add_argument("--per-layer-topk-range", type=str, default="")
    parser.add_argument("--probe-topk", type=int, default=3)
    parser.add_argument("--skip-threshold", type=float, default=0.5)
    parser.add_argument("--verbose-run", action="store_true")
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    expert_topk_range = None
    if args.expert_topk_range:
        parts = [int(p) for p in args.expert_topk_range.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--expert-topk-range must be: min,max")
        expert_topk_range = (parts[0], parts[1])

    per_layer_expert_topk_range = None
    if args.per_layer_topk_range:
        parts = [int(p) for p in args.per_layer_topk_range.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--per-layer-topk-range must be: min,max")
        per_layer_expert_topk_range = (parts[0], parts[1])

    base_cmd = build_base_cmd(args)
    env = None
    if args.cuda_visible_devices:
        env = dict(**os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    runner = SpeculativeRunner(
        base_cmd=base_cmd,
        prompt=args.prompt,
        verbose=args.verbose_run,
        env=env,
    )
    try:
        searcher = LayerSkippingSearching(
            base_cmd=base_cmd,
            n_layers=args.n_layers,
            expert_topk_range=expert_topk_range,
            per_layer_expert_topk_range=per_layer_expert_topk_range,
            probe_topk=args.probe_topk,
            verbose=args.verbose_run,
            env=env,
            runner=runner,
            skip_threshold=args.skip_threshold,
        )

        searcher.probe(attn_skip_layers=[], mlp_skip_layers=[])
        skip_attn, skip_mlp, topk, layer_topk = searcher.search(n_iter=args.n_iter)
    finally:
        runner.close()

    best = {
        "attn_skip_layers": skip_attn,
        "mlp_skip_layers": skip_mlp,
        "expert_topk": topk,
        "layer_topk": layer_topk,
    }
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
