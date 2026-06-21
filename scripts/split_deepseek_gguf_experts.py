#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np

import gguf


EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|down|up)_exps\.weight$")


def copy_field(writer: gguf.GGUFWriter, key: str, field) -> None:
    if key.startswith("GGUF.") or key == "general.architecture":
        return

    types = field.types
    vtype = types[0]
    sub_type = types[1] if vtype == gguf.GGUFValueType.ARRAY and len(types) > 1 else None
    writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)


def split_merged_expert_tensor(writer: gguf.GGUFWriter, tensor) -> int:
    match = EXPERT_RE.match(tensor.name)
    if match is None:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        return 1

    layer, proj = int(match.group(1)), match.group(2)
    if len(tensor.shape) != 3:
        raise ValueError(f"{tensor.name}: expected 3D expert tensor, got shape {tensor.shape}")

    n_expert = int(tensor.shape[-1])
    if tensor.data.shape[0] != n_expert:
        raise ValueError(
            f"{tensor.name}: expected numpy expert axis 0 to have {n_expert} entries, "
            f"got data shape {tensor.data.shape}"
        )

    for expert_id in range(n_expert):
        data = np.ascontiguousarray(tensor.data[expert_id])
        name = f"blk.{layer}.ffn_{proj}.{expert_id:03d}.weight"
        writer.add_tensor(name, data, raw_dtype=tensor.tensor_type)

    return n_expert


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DeepSeek-V2 merged MoE expert tensors in a GGUF file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.input)
    arch = reader.fields["general.architecture"].contents()
    if arch != "deepseek2":
        raise ValueError(f"expected deepseek2 architecture, got {arch!r}")

    writer = gguf.GGUFWriter(args.output, arch=arch)
    for key, field in reader.fields.items():
        copy_field(writer, key, field)

    total_in = 0
    total_out = 0
    split_groups = 0
    for tensor in reader.tensors:
        total_in += 1
        produced = split_merged_expert_tensor(writer, tensor)
        total_out += produced
        if produced != 1:
            split_groups += 1
            if args.progress:
                print(f"split {tensor.name}: {produced} experts")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=args.progress)
    writer.close()

    print(
        f"wrote {args.output}: input_tensors={total_in} output_tensors={total_out} "
        f"split_groups={split_groups}"
    )


if __name__ == "__main__":
    main()
