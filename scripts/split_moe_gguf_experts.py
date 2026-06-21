#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf  # noqa: E402

LOG = logging.getLogger("split-moe-gguf")

EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

        if field.name == gguf.Keys.General.ALIGNMENT:
            writer.data_alignment = int(field.contents())


def split_name(name: str, expert: int) -> str:
    match = EXPERT_TENSOR_RE.match(name)
    if match is None:
        raise ValueError(f"not a split MoE expert tensor: {name}")

    layer, kind = match.groups()
    return f"blk.{layer}.ffn_{kind}.{expert:03d}.weight"


def add_tensor_infos(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> list[tuple[str, Any]]:
    write_plan: list[tuple[str, Any]] = []

    for tensor in reader.tensors:
        match = EXPERT_TENSOR_RE.match(tensor.name)
        if match is None:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.n_bytes, tensor.tensor_type)
            write_plan.append((tensor.name, tensor))
            continue

        n_expert = tensor.data.shape[0]
        for expert in range(n_expert):
            data = tensor.data[expert]
            name = split_name(tensor.name, expert)
            writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)
            write_plan.append((name, (tensor, expert)))

    return write_plan


def tensor_data(entry: Any) -> np.ndarray:
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
        tensor, expert = entry
        data = tensor.data[expert]
    else:
        data = entry.data

    return data if data.flags.c_contiguous else np.ascontiguousarray(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite merged MoE expert tensors as one GGUF tensor per expert")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    reader = gguf.GGUFReader(args.input, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer)

    plan = add_tensor_infos(reader, writer)
    n_split_src = sum(1 for tensor in reader.tensors if EXPERT_TENSOR_RE.match(tensor.name))
    LOG.info("input tensors: %d, split source tensors: %d, output tensors: %d", len(reader.tensors), n_split_src, len(plan))

    total_bytes = sum(tensor_data(entry).nbytes for _, entry in plan)
    bar = tqdm(desc="Writing", total=total_bytes, unit="B", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for _, entry in plan:
        data = tensor_data(entry)
        writer.write_tensor_data(data)
        bar.update(data.nbytes)

    writer.close()
    bar.close()


if __name__ == "__main__":
    main()
