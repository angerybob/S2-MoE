#!/usr/bin/env bash
# 自回归 llama-cli（与 ./run_spec_stream.sh 对应：投机 decoding 用后者）。须在仓库根目录执行：
#   ./run_cli_stream.sh
#   CONFIG=examples/speculative/cli_config_ds2.json ./run_cli_stream.sh
#   STREAM_PYTHON_MODE=fence ./run_cli_stream.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-examples/speculative/cli_config_olmoe_demo.json}"
# 默认写入实现文件，避免覆盖本仓库里带 # pyright 的稳定入口 streamed_generated_cli.py
OUT_PY="${OUT_PY:-streamed_generated_cli_impl.py}"
STREAM_PYTHON_MODE="${STREAM_PYTHON_MODE:-fence}"

exec python3 examples/speculative/run_cli2.py \
  --config "$CONFIG" \
  --stream-python "$OUT_PY" \
  # --stream-python-mode "$STREAM_PYTHON_MODE" \
  "$@"
