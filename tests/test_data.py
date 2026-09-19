import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from math_agent.data import (
    DataFormatError,
    SYSTEM_PROMPT,
    convert_retool_row,
    extract_dapo_question,
    extract_retool_question,
    parse_retool_assistant,
    prepare_dapo,
    prepare_retool,
)


def _retool_user(question: str) -> str:
    return (
        "Instructions that are deliberately discarded.\n\n"
        f"*user question:*\n{question}\n\n"
        "Remember to place the final answer in the requested format."
    )


def _retool_answer(answer: str = "5") -> str:
    return (
        "I will calculate it.\n"
        "<code>\n```python\nprint(2 + 3)\n```\n</code>\n"
        "<interpreter>\n5\n</interpreter>\n"
        "The result checks out.\n"
        f"<answer>\n\\boxed{{{answer}}}\n</answer>"
    )


def _dapo_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        f"{question}\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )


def _dapo_row(question: str, answer: str, index: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": _dapo_prompt(question)}],
        "reward_model": {"ground_truth": answer, "style": "test"},
        "extra_info": {"index": index},
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extract_source_wrappers() -> None:
    assert extract_retool_question(_retool_user("What is 2+3?")) == "What is 2+3?"
    assert extract_dapo_question(_dapo_prompt("What is 2+3?")) == "What is 2+3?"


def test_retool_is_converted_to_qwen_tool_messages() -> None:
    messages, calls = parse_retool_assistant(_retool_answer())
    assert calls == 1
    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant"]
    assert messages[0]["reasoning_content"] == "I will calculate it."
    assert messages[0]["tool_calls"][0]["function"]["arguments"]["code"] == "print(2 + 3)"
    assert messages[1]["content"] == "5"
    assert messages[-1]["content"] == r"Answer: \boxed{5}"

    row = {"messages": [{"role": "user", "content": _retool_user("Q")}, {"role": "assistant", "content": _retool_answer()}]}
    converted = convert_retool_row(row, 7)
    assert converted["messages"][0]["content"] == SYSTEM_PROMPT
    assert converted["metadata"]["source_row"] == 7
    assert converted["tools"][0]["function"]["name"] == "code_interpreter"


def test_retool_malformed_tags_fail_closed() -> None:
    malformed = _retool_answer().replace("</interpreter>", "", 1)
    with pytest.raises(DataFormatError, match="unbalanced"):
        parse_retool_assistant(malformed)


def test_prepare_retool_rejects_duplicate_questions(tmp_path: Path) -> None:
    source = tmp_path / "retool.parquet"
    output = tmp_path / "retool.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": _retool_user("Q")}, {"role": "assistant", "content": _retool_answer()}]},
        {"messages": [{"role": "user", "content": _retool_user(" Q ")}, {"role": "assistant", "content": _retool_answer()}]},
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    report = prepare_retool(source, output)
    assert report["written"] == 1
    assert report["rejected"] == 1
    assert report["rejections"][0]["reason"] == "duplicate_question"
    assert output.stat().st_mode & 0o777 == 0o644


def test_prepare_dapo_drops_all_conflicting_labels(tmp_path: Path) -> None:
    source = tmp_path / "dapo.parquet"
    output = tmp_path / "dapo.jsonl"
    rows = [
        _dapo_row("Same question", "1", "a"),
        _dapo_row("Same question", "2", "b"),
        _dapo_row("Clean question", "3", "c"),
        _dapo_row("Clean question", "3", "c"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    report = prepare_dapo(source, output)
    records = _read_jsonl(output)
    assert report["physical_rows"] == 4
    assert report["unique_questions"] == 2
    assert report["conflicting_questions_dropped"] == 1
    assert report["written"] == 1
    assert records[0]["label"] == "3"
    assert records[0]["prompt"][1]["content"] == "Clean question"
