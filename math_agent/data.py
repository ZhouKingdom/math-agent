"""Strict, deterministic conversion of ReTool SFT and DAPO RL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SYSTEM_PROMPT = """You are a mathematical problem-solving agent with one Python tool.
Use code_interpreter only when computation helps. The tool is stateless: every call must be a complete Python program and cannot rely on files or variables from earlier calls.
Reason carefully, check the result, and end with exactly one final line in the form:
Answer: \\boxed{answer}"""

TOOL_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute a complete, stateless Python program and return its stdout or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "A complete Python program. Print any result that should be observed.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]

_RETOOL_QUESTION_MARKER = "*user question:*"
_RETOOL_REMINDER = "Remember to place the final answer"
_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)
_DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
_PAIR_RE = re.compile(r"<code>(.*?)</code>\s*<interpreter>(.*?)</interpreter>", re.DOTALL)
_SPECIAL_TAG_RE = re.compile(r"</?(?:code|interpreter)>")
_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class DataFormatError(ValueError):
    """Raised when a source row cannot be converted without guessing."""


def _sample_id(namespace: str, text: str) -> str:
    # Mathematical whitespace can be semantically significant inside TeX.
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"{namespace}-{digest[:24]}"


def extract_retool_question(content: str) -> str:
    if _RETOOL_QUESTION_MARKER not in content:
        raise DataFormatError("ReTool user message has no '*user question:*' marker")
    question = content.split(_RETOOL_QUESTION_MARKER, 1)[1]
    reminder_pos = question.rfind(_RETOOL_REMINDER)
    if reminder_pos < 0:
        raise DataFormatError("ReTool user message has no final-answer reminder")
    question = question[:reminder_pos].strip()
    if not question:
        raise DataFormatError("ReTool question is empty")
    return question


def extract_dapo_question(content: str) -> str:
    prefix = _DAPO_PREFIX + "\n\n"
    suffix = "\n\n" + _DAPO_SUFFIX
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise DataFormatError("DAPO prompt wrapper differs from the audited format")
    question = content[len(prefix) : -len(suffix)].strip()
    if not question:
        raise DataFormatError("DAPO question is empty")
    return question


def _strip_python_fence(raw_code: str) -> str:
    code = raw_code.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", code, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
    if not code:
        raise DataFormatError("ReTool trajectory contains an empty code block")
    return code


def _boxed_expression(text: str) -> str:
    start = text.rfind("\\boxed{")
    if start < 0:
        raise DataFormatError("final answer block has no \\boxed expression")

    cursor = start + len("\\boxed{")
    depth = 1
    while cursor < len(text):
        char = text[cursor]
        if char == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
            depth += 1
        elif char == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start : cursor + 1]
        cursor += 1
    raise DataFormatError("final \\boxed expression has unbalanced braces")


def _split_final_answer(tail: str) -> tuple[str, str]:
    matches = list(_ANSWER_BLOCK_RE.finditer(tail))
    if len(matches) != 1 or tail[matches[0].end() :].strip():
        raise DataFormatError("ReTool trajectory must end in exactly one <answer> block")
    answer_match = matches[0]
    reasoning = tail[: answer_match.start()].strip()
    boxed = _boxed_expression(answer_match.group(1))
    return reasoning, f"Answer: {boxed}"


def parse_retool_assistant(content: str) -> tuple[list[dict[str, Any]], int]:
    """Convert legacy code/interpreter tags to Qwen's structured tool messages."""

    tag_counts = (
        content.count("<code>"),
        content.count("</code>"),
        content.count("<interpreter>"),
        content.count("</interpreter>"),
    )
    if len(set(tag_counts)) != 1 or tag_counts[0] == 0:
        raise DataFormatError(f"unbalanced ReTool tags: {tag_counts}")

    messages: list[dict[str, Any]] = []
    cursor = 0
    pair_count = 0
    for pair_count, match in enumerate(_PAIR_RE.finditer(content), start=1):
        reasoning = content[cursor : match.start()]
        if _SPECIAL_TAG_RE.search(reasoning):
            raise DataFormatError("ReTool tool tags are not in code/result pairs")

        code = _strip_python_fence(match.group(1))
        observation = match.group(2).strip()
        call_id = f"call_{pair_count - 1}"
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "code_interpreter",
                            "arguments": {"code": code},
                        },
                        "id": call_id,
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "code_interpreter",
                "tool_call_id": call_id,
                "content": observation,
            }
        )
        cursor = match.end()

    if pair_count != tag_counts[0] or _SPECIAL_TAG_RE.search(content[cursor:]):
        raise DataFormatError("ReTool tool tags are malformed or out of order")

    reasoning, answer = _split_final_answer(content[cursor:])
    messages.append(
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": answer,
        }
    )
    return messages, pair_count


def convert_retool_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    source_messages = row.get("messages")
    if not isinstance(source_messages, list) or len(source_messages) != 2:
        raise DataFormatError("ReTool row must contain exactly one user and one assistant message")
    if [message.get("role") for message in source_messages] != ["user", "assistant"]:
        raise DataFormatError("ReTool roles must be user, assistant")

    question = extract_retool_question(source_messages[0].get("content", ""))
    trajectory, tool_calls = parse_retool_assistant(source_messages[1].get("content", ""))
    return {
        "sample_id": _sample_id("retool", question),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            *trajectory,
        ],
        "tools": TOOL_SPEC,
        "metadata": {
            "source": "ReTool-SFT",
            "source_row": row_index,
            "tool_calls": tool_calls,
            "protocol": "qwen3_hermes_code_interpreter_v1",
        },
    }


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
            # NamedTemporaryFile defaults to 0600. These generated datasets are
            # shared artifacts, so keep them readable after a root-run container
            # atomically replaces a file in a host checkout.
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return count


def prepare_retool(input_path: Path, output_path: Path, max_rows: int | None = None) -> dict[str, Any]:
    table = pq.read_table(input_path, columns=["messages"])
    converted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_index, row in enumerate(table.to_pylist()):
        try:
            record = convert_retool_row(row, row_index)
            if record["sample_id"] in seen_ids:
                rejected.append({"source_row": row_index, "reason": "duplicate_question"})
                continue
            seen_ids.add(record["sample_id"])
            converted.append(record)
        except DataFormatError as error:
            rejected.append({"source_row": row_index, "reason": str(error)})
        if max_rows is not None and len(converted) >= max_rows:
            break

    written = _atomic_write_jsonl(output_path, converted)
    return {
        "source": str(input_path),
        "output": str(output_path),
        "source_rows_seen": row_index + 1 if table.num_rows else 0,
        "written": written,
        "rejected": len(rejected),
        "rejections": rejected,
    }


def _read_dapo_message(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
        raise DataFormatError("DAPO prompt must contain exactly one user message")
    question = extract_dapo_question(prompt[0].get("content", ""))
    reward_model = row.get("reward_model") or {}
    answer = reward_model.get("ground_truth")
    index = (row.get("extra_info") or {}).get("index")
    if answer is None or index is None:
        raise DataFormatError("DAPO row is missing ground truth or index")
    return question, str(answer), str(index)


def prepare_dapo(input_path: Path, output_path: Path, max_rows: int | None = None) -> dict[str, Any]:
    """Deduplicate DAPO's 100x physical rows and remove conflicting labels."""

    parquet = pq.ParquetFile(input_path)
    by_question: dict[str, dict[str, Any]] = {}
    physical_rows = 0
    invalid_rows = 0

    columns = ["prompt", "reward_model", "extra_info", "data_source"]
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            physical_rows += 1
            try:
                question, answer, source_index = _read_dapo_message(row)
            except DataFormatError:
                invalid_rows += 1
                continue
            # Deduplicate exact extracted questions only. More aggressive
            # whitespace/TeX normalization can merge distinct expressions.
            key = question
            entry = by_question.setdefault(
                key,
                {
                    "question": question,
                    "answers": set(),
                    "indices": set(),
                    "first_seen": physical_rows,
                },
            )
            entry["answers"].add(answer)
            entry["indices"].add(source_index)

    conflicts = [entry for entry in by_question.values() if len(entry["answers"]) != 1]
    clean = [entry for entry in by_question.values() if len(entry["answers"]) == 1]
    clean.sort(key=lambda entry: entry["first_seen"])
    if max_rows is not None:
        clean = clean[:max_rows]

    def records() -> Iterable[dict[str, Any]]:
        for entry in clean:
            question = entry["question"]
            yield {
                "sample_id": _sample_id("dapo", question),
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "label": next(iter(entry["answers"])),
                "tools": TOOL_SPEC,
                "metadata": {
                    "source": "DAPO-Math-17k",
                    "source_indices": sorted(entry["indices"]),
                    "rm_type": "dapo",
                    "protocol": "qwen3_hermes_code_interpreter_v1",
                },
            }

    written = _atomic_write_jsonl(output_path, records())
    conflict_report = [
        {
            "sample_id": _sample_id("dapo-conflict", entry["question"]),
            "answers": sorted(entry["answers"]),
            "source_indices": sorted(entry["indices"]),
        }
        for entry in sorted(conflicts, key=lambda item: item["first_seen"])
    ]
    return {
        "source": str(input_path),
        "output": str(output_path),
        "physical_rows": physical_rows,
        "unique_questions": len(by_question),
        "conflicting_questions_dropped": len(conflicts),
        "invalid_rows": invalid_rows,
        "written": written,
        "conflicts": conflict_report,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("retool", "dapo"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--max-rows", type=_positive_int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "retool":
        report = prepare_retool(args.input, args.output, args.max_rows)
    else:
        report = prepare_dapo(args.input, args.output, args.max_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
