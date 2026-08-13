import json
import subprocess
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run llama-speculative from a JSON config.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_config_olmoe2.json"),
        help="Path to config JSON file.",
    )
    args = parser.parse_args()

    config_file = args.config
    
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    executable = config.get("binary_path", "./llama-speculative")
    cmd = [executable]

    # 3. 解析基础模型参数
    if "model" in config:
        cmd.extend(["-m", config["model"]])
    if "draft_model" in config:
        cmd.extend(["-md", config["draft_model"]])
    if "prompt" in config:
        cmd.extend(["-p", config["prompt"]])
    if "datasets" in config:
        cmd.extend(["--dataset", config["datasets"]])
    if "questions_limit" in config:
        cmd.extend(["--n-questions", str(config["questions_limit"])])
    if "chat_template_file" in config:
        cmd.extend(["--jinja", config["chat_template_file"]])
    if "dataset" in config:
        cmd.extend(["--dataset", config["dataset"]])
    if "n_questions" in config:
        cmd.extend(["--n-questions", str(config["n_questions"])])
    if "hot_experts" in config:
        cmd.extend(["--hot-experts", config["hot_experts"]])
    cmd.append("--no-mmap")
        
    # 4. 解析常规生成参数 (params)
    params = config.get("params", {})
    
    # 值映射 (Key -> Flag)
    arg_map = {
        "n_predict": "-n",
        "n_ctx":"-c",
        "temp": "--temp",
        "min_p":"--min-p",
        "n_gpu_layers": "-ngl",
        "n_gpu_layers_draft": "-ngld",
        "threads": "-t",
        "parallel": "--parallel"  # [新增] 并行参数
    }
    
    for key, flag in arg_map.items():
        if key in params:
            cmd.extend([flag, str(params[key])])
            
    # [新增] 布尔 Flag 处理 (-kvu)
    if params.get("kv_unified", False):
        cmd.append("-kvu")
    if params.get("no_warmup", False):
        cmd.append("--no-warmup")
    if params.get("ssd_moe", False):
        cmd.append("--ssd-moe")
    if params.get("accept_log", False):
        cmd.append("--acc-log")
    if params.get("no_mmap", False):
        cmd.append("--no-mmap")
    # 5. 解析投机采样参数 (speculative)
    spec = config.get("speculative", {})
    if "n_draft" in spec:
        cmd.extend(["--draft", str(spec["n_draft"])])
    if "p_split" in spec:
        cmd.extend(["--draft-p-split", str(spec["p_split"])])
    if spec.get("share_kv", False):
        cmd.append("--draft-share-kv")
    if "draft_expert_topk" in spec:
        cmd.extend(["--draft-expert-topk",str(spec["draft_expert_topk"])])
    if spec.get("moe_utility_spec", False):
        cmd.append("--moe-utility-spec")
    if "utility_test_iters" in spec:
        cmd.extend(["--spec-utility-test-iters", str(spec["utility_test_iters"])])
    if "utility_set_iters" in spec:
        cmd.extend(["--spec-utility-set-iters", str(spec["utility_set_iters"])])
    # 6. 解析 MoE 参数 (moe)
    moe = config.get("moe", {})
    if "reuse_strength" in moe:
        cmd.extend(["--moe-reuse-strength", str(moe["reuse_strength"])])
    if "reuse_expert_cap" in moe:
        cmd.extend(["--moe-reuse-expert-cap", str(moe["reuse_expert_cap"])])
    if moe.get("reuse_runtime", False):
        cmd.append("--moe-reuse-runtime")

    # 7. 解析剪枝参数 (pruning)
    prune = config.get("pruning", {})
    
    # 处理开关 --prune N
    if "enable" in prune:
        cmd.extend(["--prune", str(prune["enable"])])
    # [新增] 解析 trace 开关
    if prune.get("trace", False):
        cmd.append("--prune-trace")
    # 映射表
    prune_map = {
        "max_nodes":    "--prune-max-nodes", # [新增]
        "max_depth":    "--prune-max-depth", # [顺便加上]
        "budget_b":     "--prune-budget",
        "m_route":      "--prune-m-route",
        "k_tgt":        "--prune-k-tgt",
        "beta":         "--prune-beta",
        "gamma":        "--prune-gamma",
        "lambda":       "--prune-lambda",
        "tpot":         "--prune-tpot",
        "eps":          "--prune-eps",
        "expert_bytes": "--prune-expert-bytes",
        "bandwidth":    "--prune-bandwidth",
        "expert_max":   "--prune-expert-max"
    }
    
    for key, flag in prune_map.items():
        if key in prune:
            cmd.extend([flag, str(prune[key])])

    # 8. 打印并执行
    print("=" * 60)
    print("Executing command:")
    print(" ".join(cmd))
    # print(cmd)
    print("=" * 60)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',    # [新增] 明确指定编码
            errors='replace'     # [新增] 关键！遇到无法解码的字节，用  替换，而不是崩溃
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error execution failed: {e}")

if __name__ == "__main__":
    main()
