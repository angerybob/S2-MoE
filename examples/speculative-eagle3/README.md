### 1. Hidden States Extraction
./build/bin/llama-speculative-eagle3-tgt-hs-extraction ~/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf all-layers
./build/bin/llama-speculative-eagle3-tgt-hs-extraction ~/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf
当前硬编码提取层 [2, n_layer/2, n_layer-3] (对36层模型为[2,18,33])，maybe 后续需要修改，但至少目前能跑

### 2. D2T map test
map存的是diff，token_id_tgt = token_id_draft + d2t_map[token_id_draft]
这个没什么好测试的，知道就不会错了，可以用静态map存，这样不用每次都查然后加

### 3. 简化的单元测试 
./build/bin/llama-speculative-eagle3-simple-predict ~/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf ~/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf

### 4. embd 共享
./build/bin/llama-speculative-eagle3-embd-sharing-test ~/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf ~/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf

### 5. 单步预测
./build/bin/llama-speculative-eagle3-simple-logic ~/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf ~/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf

### 6. 连续 预测
./build/bin/llama-speculative-eagle3-conti-predict /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf
目前硬编码prompt以及测试长度20

### 7. 多 prompt 单步测试
./build/bin/llama-speculative-eagle3-batch-test /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf

### 8. 综合测试-top1-chain，严格接收
./build/bin/llama-speculative-eagle3-comprehensive-fix /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf > tmp.txt 2>&1
3-top1-chain: Mean Speculation Length: 0.25

./build/bin/llama-speculative-eagle3-comprehensive /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf >> tmp.txt 2>&1 && bash clear.sh

### 8.1 综合测试-top1-chain，Spec接收（接受条件：target_prob / draft_prob >= uniform_prob）
./build/bin/llama-speculative-eagle3-comprehensive-speclike /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf > tmp.txt 2>&1
Mean Speculation Length: 0.35 
注：这里使用的prompt做了修改，如果严格接收使用相同的prompt，结果Mean Speculation Length: 0.11

**TODO: 稀疏树实现**
- 🔄 实现EAGLE3稀疏树 speculative decoding (类似choices.py中的tree结构)
- 🎯 扩展从单token验证到多token树形验证，提升acceptance rate
- 📊 构建候选树: `[0], [1], [2]` → `[0,0], [0,1], [0,2]` → `[0,0,0], [0,0,1]...`
对着sglang猛抄就完事了

注意可能还会有些问题，比如norm之类的，某些commit遇到过可能的问题，但是目前来看难以debug
./build/bin/llama-speculative-eagle3-simple-input-analysis /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf
这个可能不是很有必要

另外值得一提的是，查看终端输出时有很多不必要的输出，这里附赠一个清理脚本，当然如果build出问题还是得慢慢看完整日志的
```bash
# !/bin/bash

sed -i '/^load/d' tmp.txt
sed -i '/^common/d' tmp.txt
sed -i '/^llama/d' tmp.txt
sed -i '/^print_info/d' tmp.txt
sed -i '/^create_tensor/d' tmp.txt
sed -i '/^EAGLE3/d' tmp.txt
sed -i '/^graph_reserve/d' tmp.txt
sed -i '/^Draft predicted/d' tmp.txt
sed -i '/^=== DRAFT PHASE START/d' tmp.txt
sed -i '/^=== DRAFT PHASE END/d' tmp.txt
```

最新的修复dft hs更新
单步Mean Speculation Length: 0.22
3步Mean Speculation Length: 0.25
步数设置少一点有
单步Mean Speculation Length: 0.33
Mean Speculation Length: 0.33

值得参考的是sglang强制设置top1单步预测，Accept length = 1.82，也就是top1准确率 82%，现在显然低了
没辙😇

[Eagle3 DEBUG] Target Hidden States Input: max=4288.000000, min=-1112.000000
[Eagle3 DEBUG] Hidden States after FC: max=202.000000, min=-102.000000
[Eagle3 DEBUG] Embedding after Norm: max=3.937500, min=-4.437500
[Eagle3 DEBUG] Hidden States after Norm: max=7.562500, min=-11.250000
[Eagle3 DEBUG] Before Cat -> Embeds: max=3.937500, min=-4.437500
[Eagle3 DEBUG] Before Cat -> Hidden States: max=7.562500, min=-11.250000

更新了hs提取api，现在是底层获取全token的三层hs，交给顶层决定使用哪些idx的hs，避免了dft hs的污染问题。
最新版的comprehensive可以看到日志输出，应该hs已经没问题了



### 统一 10 token 测试
comprehensive-fix.cpp 1-token-top4 预测

=== Statistics ===
Total Drafted: 28
Total Accepted: 3
Acceptance Rate: 10.71%
Mean Speculation Length: 0.43

real    0m9.951s
user    0m22.795s
sys     0m1.947s

comprehensive-fix.cpp 2-token-top4*top4 预测

=== Statistics ===
Total Drafted: 224
Total Accepted: 3
Acceptance Rate: 1.34%
Mean Speculation Length: 0.43

real    0m21.682s
user    1m6.631s
sys     0m2.893s


comprehensive.cpp 2-token预测

=== Statistics ===
Total Drafted: 16
Total Accepted: 2
Acceptance Rate: 12.50%
Verification Steps: 8
Mean Speculation Length: 0.25

real    0m6.086s
user    0m13.949s
sys     0m1.312s

comprehensive.cpp 1-token预测

=== Statistics ===
Total Drafted: 8
Total Accepted: 2
Acceptance Rate: 25.00%
Verification Steps: 8
Mean Speculation Length: 0.25

real    0m5.772s
user    0m12.807s
sys     0m1.316s

补充 50-token 树形测试 

=== Statistics ===
Total Drafted: 1152
Total Accepted: 14
Acceptance Rate: 1.22%
Mean Speculation Length: 0.39

可以看到基本是有效的

加了采样：20 步 2-token-top4*top4

=== Statistics ===
Total Drafted: 352
Total Accepted: 9
Probabilistic Accepts: 7
Acceptance Rate: 2.56%
Mean Speculation Length: 0.82


./build/bin/llama-speculative-eagle3-comprehensive /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle-F32.gguf

./build/bin/llama-speculative-eagle3-comprehensive /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle.gguf


以下都是 GPU 单卡 4090 测试

./build/bin/llama-speculative-eagle3-comprehensive /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle-F32.gguf

50-token-1t1-gpu
Mean Speculation Length: 0.22
real    0m7.246s
user    0m6.195s
sys     0m2.842s

50-token-2t1-gpu
Mean Speculation Length: 0.25
real    0m8.255s
user    0m7.443s
sys     0m2.829s

make -C build llama-speculative-eagle3-comprehensive-fix -j$(nproc)

./build/bin/llama-speculative-eagle3-comprehensive-fix /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle-F32.gguf

./build/bin/llama-bench -m /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf -p 0 -n 128 -ngl 999 -b 1

| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 4B all F32               |  14.98 GiB |     4.02 B | CUDA       | 999 |       1 |           tg128 |         54.38 ± 0.02 |


make -C build llama-speculative-eagle3-bench -j$(nproc)

./build/bin/llama-speculative-eagle3-bench /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle-F32.gguf 5

1t1略有加速
=== EAGLE3 Benchmark Results (256 Target Forwards) ===
Prompt:     29 tokens, 1487.61 TPS
Generation: 313 tokens, 51.17 TPS
Overall:    342 tokens, 55.58 TPS
Target Forwards: 256, Generated/Forward: 1.22
Acceptance Rate: 22.27%, Mean Spec Length: 0.22
Prob Accepts: 6/57 (10.5% of accepts)

make -C build llama-speculative-eagle3-tree-bench -j$(nproc)

./build/bin/llama-speculative-eagle3-tree-bench /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-F32.gguf /data/home/tianjianyang/models/Qwen3-4B-Instruct-2507-Eagle3/eagle-F32.gguf 5

=== EAGLE3 Tree Benchmark Results (128 Target Forwards) ===
Tree Config: TOP_K_L1=4, TOP_K_L2=4, TOTAL_LANES=16
Prompt:     29 tokens, 1090.85 TPS
Generation: 204 tokens, 32.96 TPS
Overall:    233 tokens, 37.40 TPS
Target Forwards: 128, Generated/Forward: 1.59
Acceptance Rate: 1.86%, Mean Spec Length: 0.59
Prob Accepts: 25/76 (32.9% of accepts)

寄

=== EAGLE3 Tree Benchmark Results (128 Target Forwards) ===
Tree Config: TOP_K_L1=8, TOTAL_LANES=8 (1-step parallel)
Prompt:     29 tokens, 1262.01 TPS
Generation: 212 tokens, 46.67 TPS
Overall:    241 tokens, 52.62 TPS
Target Forwards: 128, Generated/Forward: 1.66
Acceptance Rate: 8.20%, Mean Spec Length: 0.66
Prob Accepts: 27/84 (32.1% of accepts)

--- Rank Acceptance Distribution ---
Rank 0: 29 accepts (35.4%)
Rank 1: 17 accepts (20.7%)
Rank 2: 9 accepts (11.0%)
Rank 3: 7 accepts (8.5%)
Rank 4: 5 accepts (6.1%)
Rank 5: 8 accepts (9.8%)
Rank 6: 3 accepts (3.7%)
Rank 7: 4 accepts (4.9%)

=== EAGLE3 Tree Benchmark Results (256 Target Forwards) ===
Tree Config: TOP_K_L1=3, TOTAL_LANES=3 (1-step parallel)
Prompt:     29 tokens, 1350.46 TPS
Generation: 394 tokens, 52.19 TPS
Overall:    423 tokens, 55.76 TPS
Target Forwards: 256, Generated/Forward: 1.54
Acceptance Rate: 17.97%, Mean Spec Length: 0.54
Prob Accepts: 25/138 (18.1% of accepts)

--- Rank Acceptance Distribution ---
Rank 0: 63 accepts (54.8%)
Rank 1: 33 accepts (28.7%)
Rank 2: 19 accepts (16.5%)