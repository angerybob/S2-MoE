# MoE 效用驱动投机解码（Cascade 风格）

本仓库在 `examples/speculative/speculative.cpp` 中实现了论文 *Utility-Driven Speculative Decoding for Mixture-of-Experts*（arXiv:2506.20675，文中系统名 **Cascade**）的**简化版**思路：按「效用」在多个草稿长度 **K** 之间做 **测试（test）/ 固定（set）** 调度，并在 **K=0**（关闭投机）时拉长 set 阶段以减少探测开销（回退）。

> 说明：完整论文在 vLLM 等服务框架上的实现与此处 llama.cpp 示例二进制不同；此处用**可观测统计量**近似「收益 / 验证代价」，用于在 MoE 上避免固定大 **K** 带来的验证访存放大。

---

## 主要改动（文件）

| 文件 | 内容 |
|------|------|
| `common/common.h` | `common_params_speculative` 增加 `moe_utility_spec`、`utility_test_iters`、`utility_set_iters` |
| `common/arg.cpp` | 注册命令行选项 `--moe-utility-spec`、`--spec-utility-test-iters`、`--spec-utility-set-iters` |
| `examples/speculative/speculative.cpp` | `MoeUtilityCascadeState`：test/set 状态机；解码循环用动态 `n_draft_active`；`on_verify_done` 记录验证耗时；剪枝模式下 **K=0** 时修正 `MAX_NODES` 下限 |
| `examples/speculative/run_spec.py` | 从 JSON `speculative` 段传递上述参数 |

---

## 行为概要

1. **草稿长度上限**：仍由 `--draft` / `n_draft`（JSON 里 `speculative.n_draft`）给出，记为 **cap**。
2. **候选 K**：在 `{0, 1, max(2, cap/2), cap}` 上去重排序（最多 4 个不同值）。
3. **Test 阶段**：对每个候选 K，连续采样 **utility_test_iters** 次（默认 4）  
   - 单次样本：**本轮 verify 循环内产出的 token 数 ÷ 上一轮 target 验证耗时（微秒）**（越大表示单位验证时间产出越多，作为效用代理）。
   - 对每个 K 算平均效用，取 **best_k**。
4. **Set 阶段**：固定使用 **best_k** 跑 **utility_set_iters** 步（默认 16）。  
   - 若 **best_k == 0**，下一轮 set 长度乘以 2（上限 ×8），以减少「无投机」时仍频繁做 test 的开销。
5. Set 结束后重新进入下一轮 **Test**，重复上述过程。

未开启 `--moe-utility-spec` 时，行为与原先一致：始终使用固定的 `n_max`（cap）作为草稿长度。

---

## 命令行用法

在原有 `llama-speculative` 参数基础上增加：

```text
--moe-utility-spec                 开启效用驱动自适应 K
--spec-utility-test-iters N       每个候选 K 在 test 阶段的样本数（默认 4）
--spec-utility-set-iters N       set 阶段迭代步数（默认 16）
```

示例（需自备模型路径）：

```bash
./build/bin/llama-speculative -m /path/to/target.gguf -md /path/to/draft.gguf \
  -p "Hello" -n 128 --draft 8 --moe-utility-spec \
  --spec-utility-test-iters 4 --spec-utility-set-iters 16
```

运行日志中可出现类似：

- 启动：`MoE utility-driven speculation (Cascade-style): --draft cap=... test_iters=... set_iters=...`
- 每轮 test 结束：`moe-utility-spec: test round done, best_k=... best_avg_tokens_per_us=...`
- 重新进入 test：`moe-utility-spec: restarting test phase (cap=...)`

---

## `run_spec.py` / JSON 配置

在配置文件的 **`speculative`** 对象中增加字段（与现有 `n_draft`、`p_split` 等并列）：

```json
"speculative": {
  "n_draft": 8,
  "p_split": 0.1,
  "share_kv": true,
  "moe_utility_spec": true,
  "utility_test_iters": 4,
  "utility_set_iters": 16
}
```

- `moe_utility_spec`: `true` 时追加 `--moe-utility-spec`
- `utility_test_iters` / `utility_set_iters`: 可选，会映射到对应 CLI

---

## 编译

在已配置好的 build 目录下增量编译即可（勿随意 `cmake -B` 全量重配，除非你明确需要）：

```bash
conda activate llamacpp   # 或你的环境名
cmake --build build --parallel
```

产物：`build/bin/llama-speculative`。

---

## 参考

- Saxena et al., *Utility-Driven Speculative Decoding for Mixture-of-Experts*, arXiv:2506.20675.
