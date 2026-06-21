"""从 llama-speculative 的混合 stdout 里截取每条样本的 assistant 正文（无加载/perf 等日志）。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_RE_Q_START = re.compile(r"=== Processing Question (\d+) \(Dataset\) ===")
# 与 llama.cpp common_chat / 各模型 chat 模板中「轮到 assistant 生成」前的后缀对齐；长串放前面便于同位置 tie-break
_ASSISTANT_MARKERS: tuple[str, ...] = (
    "<|im_start|>assistant<|im_sep|>",
    # Llama 3（与 llama-chat.cpp LLM_CHAT_TEMPLATE_LLAMA_3 一致）
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|im_start|>assistant",
    "<|start|>assistant",
    "<start_of_turn>model\n",
    "<start_of_turn>model",
    "<start_of_turn>assistant\n",
    "<start_of_turn>assistant",
    "<s>assistant\n",
    "<|assistant|>\n",
    "<|assistant|>",
    "[/INST]\n",
    "[/INST]",
    "<|im_assistant|>assistant<|im_middle|>",
    "<seed:bos>assistant\n",
    "<seed:bos>assistant",
    "### Assistant\n",
    "### Assistant",
    # 简单「User: / Assistant:」拼接（无 ChatML 头，常见于部分模板或回退格式）
    "Assistant:\n",
    "Assistant:",
    "assistant:\n",
    "assistant:",
)
# [SOFTMAX DEBUG]、[GETROWS DEBUG] 等：整段须在方括号内闭合
_RE_DEBUG_LINE = re.compile(r"^\s*\[[^\]\n]*DEBUG[^\]\n]*\]")
# llama 彩色输出：CSI SGR / 光标等（否则答案里每个 token 都带 \x1b[36m 等，JSONL 巨大且不可读）
_RE_ANSI = re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
# 流式输出若把 ESC 与 “[36m” 拆开，会残留裸的 SGR 片段
_RE_ORPHAN_SGR = re.compile(r"(?<!\x1b)\[(\d{1,4}(?:;\d{1,4})*)m")

# llama 在 stdout 里打印的 decode / KV 错误，不应进入答案 JSONL
_LLAMA_STDERR_MARKERS = (
    "llama_decode: failed",
    "decode: failed to find a memory slot",
    "decode: failed to initialize",
    "init: the tokens of sequence",
    "inconsistent sequence positions",
    "failed to initialize batch",
    "it is required that the sequence positions remain consecutive",
    "the last position stored in the memory module",
    "starting position of Y =",
    "llama_decode: failed to decode",
)

_RE_GPTOSS_CHANNEL_PREFIX = re.compile(r"<\|channel\|>\s*(analysis|final|commentary)\s*<\|message\|>", re.IGNORECASE)
_RE_GPTOSS_CONTROL = re.compile(r"<\|(?:channel|message|end|return|call)\|>")


def _strip_ansi(s: str) -> str:
    return _RE_ANSI.sub("", s)


def _strip_orphan_sgr(s: str) -> str:
    return _RE_ORPHAN_SGR.sub("", s)


def _line_has_llama_error_noise(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    return any(m in t for m in _LLAMA_STDERR_MARKERS)


def _strip_special_tail(s: str) -> str:
    s = s.rstrip()
    for _ in range(8):
        changed = False
        for tok in (
            "<|endoftext|>",
            "<|return|>",
            "<|end|>",
            "<|call|>",
            "<｜end▁of▁sentence｜>",
            "<｜begin▁of▁sentence｜>",
            "</think>",
            "#",
        ):
            if s.endswith(tok):
                s = s[: -len(tok)].rstrip()
                changed = True
        if not changed:
            break
    return s.strip()


def _strip_gptoss_channels(s: str) -> str:
    final_match = re.search(r"<\|channel\|>\s*final\s*<\|message\|>", s, flags=re.IGNORECASE)
    if final_match:
        final_text = _RE_GPTOSS_CONTROL.sub("", s[final_match.end() :]).strip()
        if final_text:
            s = s[final_match.end() :]
    s = _RE_GPTOSS_CHANNEL_PREFIX.sub("", s)
    s = _RE_GPTOSS_CONTROL.sub("", s)
    return s


def _find_assistant_body_start(segment: str, answer_pos: int) -> int | None:
    """
    在 segment[:answer_pos] 中，用各模板后缀的最后一次出现作为「模型开始续写」的位置
    （避免题面里偶然出现较短子串时误切）。
    """
    head = segment[:answer_pos]
    best_i = -1
    best_m = ""
    for m in _ASSISTANT_MARKERS:
        i = head.rfind(m)
        if i < 0:
            continue
        if i > best_i or (i == best_i and len(m) > len(best_m)):
            best_i, best_m = i, m
    if best_i < 0:
        return None
    pos = best_i + len(best_m)
    while pos < len(segment) and segment[pos] in "\r\n":
        pos += 1
    return pos


def _line_is_debug_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(_RE_DEBUG_LINE.match(s))


def _clean_answer(raw: str) -> str:
    s = _strip_orphan_sgr(_strip_ansi(raw.replace("\r\n", "\n")))
    s = _strip_gptoss_channels(s)
    lines = s.split("\n")
    out_lines: list[str] = []
    for line in lines:
        if _line_has_llama_error_noise(line):
            break
        if _line_is_debug_noise(line):
            continue
        out_lines.append(line)
    s = "\n".join(out_lines).strip()
    s = _strip_orphan_sgr(s)
    return _strip_special_tail(s)


def extract_dataset_answers(log: str) -> list[dict[str, Any]]:
    """按 speculative.cpp 打印格式解析各题 assistant 生成内容。"""
    results: list[dict[str, Any]] = []
    pos = 0
    while True:
        m = _RE_Q_START.search(log, pos)
        if not m:
            break
        qnum = int(m.group(1))
        start = m.end()
        m_next = _RE_Q_START.search(log, start)
        seg_end = m_next.start() if m_next else len(log)
        segment = log[start:seg_end]
        ans_needle = f"=== Answer {qnum} ==="
        j = segment.find(ans_needle)
        if j < 0:
            pos = seg_end
            continue
        body_start = _find_assistant_body_start(segment, j)
        if body_start is None:
            pos = seg_end
            continue
        raw = segment[body_start:j]
        results.append({"question_index": qnum, "answer": _clean_answer(raw)})
        pos = seg_end
    return results


def write_answers_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    if p.parent and str(p.parent) not in (".", ""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
