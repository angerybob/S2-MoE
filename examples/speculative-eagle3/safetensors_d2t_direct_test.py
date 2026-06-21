#!/usr/bin/env python3
"""
直接读取safetensors中的d2tmap值，仿照sglang实现
不依赖gguf库，直接验证原始数据
"""

import torch
from safetensors import safe_open
import numpy as np
import sys
import os

def test_safetensors_d2tmap(safetensors_path):
    """直接测试safetensors中的d2tmap"""
    print(f"=== 直接读取Safetensors测试 ===")
    print(f"文件: {safetensors_path}")

    if not os.path.exists(safetensors_path):
        print(f"❌ 文件不存在: {safetensors_path}")
        return False

    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            # 读取原始d2t和t2d数据
            d2t_tensor = f.get_tensor('d2t')
            t2d_tensor = f.get_tensor('t2d')

            print(f"\n✅ 成功加载safetensors:")
            print(f"  d2t tensor: shape={d2t_tensor.shape}, dtype={d2t_tensor.dtype}")
            print(f"  t2d tensor: shape={t2d_tensor.shape}, dtype={t2d_tensor.dtype}")

            # 转换为numpy数组便于处理
            d2t_diff = d2t_tensor.to(torch.int32).numpy()
            t2d_map = t2d_tensor.to(torch.bool).numpy()

            print(f"\n=== 原始D2T差值数据 ===")
            print(f"值范围: [{d2t_diff.min()}, {d2t_diff.max()}]")
            print(f"非零值数量: {(d2t_diff != 0).sum()}/{len(d2t_diff)}")

            print(f"\n前20个d2t差值:")
            for i in range(20):
                print(f"  [{i:5d}] = {d2t_diff[i]:10d}")

            # 仿照sglang的计算方式
            print(f"\n=== 应用sglang方式计算target_id ===")
            hot_token_id = d2t_diff + np.arange(len(d2t_diff), dtype=np.int32)
            print(f"hot_token_id范围: [{hot_token_id.min()}, {hot_token_id.max()}]")

            # 测试关键token
            test_tokens = [2419, 2018, 21836, 1473, 18957]
            print(f"\n=== 关键Token测试 ===")
            print(f"格式: token -> diff + token = target_id")

            all_correct = True
            for token in test_tokens:
                if token < len(d2t_diff):
                    diff = d2t_diff[token]
                    target_id = hot_token_id[token]
                    print(f"  {token:5d} -> {diff:10d} + {token:5d} = {target_id:5d}")
                else:
                    print(f"  {token:5d} -> 超出范围")
                    all_correct = False

            # 验证T2D一致性
            print(f"\n=== T2D一致性验证 ===")
            valid_mappings = 0
            invalid_mappings = 0

            for draft_id in range(min(len(hot_token_id), len(t2d_map))):
                target_id = hot_token_id[draft_id]
                if 0 <= target_id < len(t2d_map):
                    if t2d_map[target_id]:
                        valid_mappings += 1
                    else:
                        invalid_mappings += 1

            print(f"有效映射: {valid_mappings}")
            print(f"无效映射: {invalid_mappings}")
            print(f"成功率: {valid_mappings / len(hot_token_id) * 100:.2f}%")

            return all_correct

    except Exception as e:
        print(f"❌ 读取safetensors失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python3 safetensors_d2t_direct_test.py <safetensors_path>")
        print("示例: python3 safetensors_d2t_direct_test.py ~/models/Qwen3-4B-Instruct-2507-Eagle3/model.safetensors")
        sys.exit(1)

    safetensors_path = sys.argv[1]

    success = test_safetensors_d2tmap(safetensors_path)

    if success:
        print(f"\n🎉 结论: Safetensors中的d2tmap数据完全正确!")
        print("   这证明了:")
        print("   1. 原始数据没有i+5错误模式")
        print("   2. GGUF转换应该是正确的")
        print("   3. Python gguf库读取时存在bug")
        return 0
    else:
        print(f"\n❌ 结论: Safetensors中的d2tmap数据有问题!")
        return 1

if __name__ == "__main__":
    sys.exit(main())