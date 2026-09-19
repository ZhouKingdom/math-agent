"""Strict final-answer reward for the plain GRPO baseline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _consume_boxed(text: str) -> tuple[str, str] | None:
    if not text.startswith("\\boxed{"):
        return None
    depth = 1
    cursor = len("\\boxed{")
    while cursor < len(text):
        char = text[cursor]
        if char == "{" and text[cursor - 1] != "\\":
            depth += 1
        elif char == "}" and text[cursor - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[len("\\boxed{") : cursor], text[cursor + 1 :]
        cursor += 1
    return None


def extract_final_answer(response: str) -> str | None:
    """Accept an exact final answer followed only by a known EOS token."""

    response = response.rstrip()
    for terminal_token in ("<|im_end|>", "<|endoftext|>"):
        if response.endswith(terminal_token):
            response = response[: -len(terminal_token)].rstrip()
            break

    lines = response.splitlines()
    if not lines:
        return None
    final_line = lines[-1].strip()
    prefix = "Answer:"
    if not final_line.startswith(prefix):
        return None
    parsed = _consume_boxed(final_line[len(prefix) :].strip())
    if parsed is None:
        return None
    answer, remainder = parsed
    return answer.strip() if not remainder.strip() else None


def _normalize_answer(answer: str) -> str:
    normalized = "".join(answer.split())
    if normalized.startswith("$") and normalized.endswith("$") and len(normalized) >= 2:
        normalized = normalized[1:-1]
    return normalized.replace(",", "")


def compute_reward(
    response: str,
    label: str,
    canonical_scorer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return correctness only; tool usage and trajectory length never change score."""

    prediction = extract_final_answer(response)
    if prediction is None:
        return {"score": -1.0, "acc": False, "pred": "", "format_valid": False}

    if canonical_scorer is None:
        correct = _normalize_answer(prediction) == _normalize_answer(str(label))
        return {
            "score": 1.0 if correct else -1.0,
            "acc": correct,
            "pred": prediction,
            "format_valid": True,
        }

    result = canonical_scorer(response, str(label), strict_box_verify=True)
    if isinstance(result, dict):
        result = dict(result)
        correct = float(result.get("score", -1.0)) > 0
        result["score"] = 1.0 if correct else -1.0
        result["acc"] = bool(result.get("acc", correct))
        result["pred"] = result.get("pred") or prediction
        result["format_valid"] = True
        return result
    correct = bool(result)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": prediction,
        "format_valid": True,
    }


async def reward_func(args: Any, sample: Any, **_: Any) -> dict[str, Any]:
    """slime ``--custom-rm-path`` entry point."""

    try:
        from slime.rollout.rm_hub.math_dapo_utils import compute_score as dapo_score
    except ImportError:
        dapo_score = None
    return compute_reward(sample.response or "", sample.label or "", canonical_scorer=dapo_score)
