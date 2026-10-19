"""JSON configuration adapter for the hybrid S2-MoE driver."""


def build_command(config):
    """Build an argv list; reject unsupported settings instead of ignoring them."""
    flags = {
        "model": "-m", "draft_model": "--model-draft", "prompt": "-p", "n_predict": "-n",
        "n_gpu_layers": "-ngl", "n_ctx": "-c", "n_ubatch": "-ub", "threads": "-t",
        "memory_budget_mib": "--memory-budget-mib", "runtime_reserve_mib": "--runtime-reserve-mib",
        "hot_experts": "--hot-experts", "tokens_file": "--tokens-file", "timings_file": "--timings-file",
    }
    switches = {"ssd_moe": "--ssd-moe", "ar": "--ar", "ignore_eos": "--ignore-eos", "quiet": "--quiet"}
    sections = {
        "speculative": {"draft_max": "--draft-max", "draft_expert_topk": "--draft-expert-topk", "branches": "--draft-branches"},
        "pruning": {"enable": "--prune", "max_nodes": "--prune-max-nodes", "max_depth": "--prune-max-depth",
                    "budget_b": "--prune-budget", "tpot": "--prune-tpot", "bandwidth": "--prune-bandwidth",
                    "score_thresh": "--prune-score-thresh", "conf_thresh": "--prune-conf-thresh", "lambda": "--prune-lambda"},
        "moe": {"reuse_strength": "--moe-reuse-strength", "reuse_expert_cap": "--moe-reuse-expert-cap"},
    }
    allowed = set(flags) | set(switches) | set(sections) | {"engine", "binary_path"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unsupported hybrid configuration keys: {sorted(unknown)}")
    command = [config.get("binary_path", "./build/bin/llama-speculative")]
    for key, flag in flags.items():
        if key in config:
            command += [flag, str(config[key])]
    for key, flag in switches.items():
        if config.get(key, False):
            command.append(flag)
    for section, mapping in sections.items():
        values = config.get(section, {})
        extra = set(values) - set(mapping) - ({"reuse_runtime"} if section == "moe" else set())
        if extra:
            raise ValueError(f"unsupported hybrid {section} keys: {sorted(extra)}")
        for key, flag in mapping.items():
            if key in values:
                value = int(values[key]) if isinstance(values[key], bool) else values[key]
                command += [flag, str(value)]
    if config.get("moe", {}).get("reuse_runtime", False):
        command.append("--moe-reuse-runtime")
    return command
