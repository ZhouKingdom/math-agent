"""Qwen/Hermes multi-turn tool rollout compatible with slime."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from math_agent.data import TOOL_SPEC
from math_agent.python_tool import DockerPythonTool
from math_agent.reward import extract_final_answer


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    code: str


@dataclass(frozen=True)
class FinalAnswer:
    answer: str


@dataclass(frozen=True)
class InvalidAction:
    reason: str


Action = ToolCall | FinalAnswer | InvalidAction


def parse_model_action(text: str) -> Action:
    """Parse one strict Hermes call or one canonical final answer."""

    blocks = _TOOL_CALL_RE.findall(text)
    open_count = text.count("<tool_call>")
    close_count = text.count("</tool_call>")
    answer = extract_final_answer(text)

    if open_count or close_count:
        if open_count != 1 or close_count != 1 or len(blocks) != 1:
            return InvalidAction("expected exactly one complete <tool_call> block")
        if answer is not None:
            return InvalidAction("a turn cannot contain both a tool call and a final answer")
        block_match = _TOOL_CALL_RE.search(text)
        assert block_match is not None
        if text[block_match.end() :].strip():
            return InvalidAction("unexpected text after </tool_call>")
        try:
            payload = json.loads(blocks[0])
        except json.JSONDecodeError as error:
            return InvalidAction(f"tool call is not valid JSON: {error.msg}")
        if not isinstance(payload, dict) or payload.get("name") != "code_interpreter":
            return InvalidAction("only code_interpreter is available")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"code"}:
            return InvalidAction("code_interpreter arguments must contain only 'code'")
        code = arguments["code"]
        if not isinstance(code, str) or not code.strip():
            return InvalidAction("code_interpreter code must be a non-empty string")
        return ToolCall(code=code)

    if answer is not None:
        return FinalAnswer(answer=answer)
    return InvalidAction("turn ended without a valid tool call or canonical final answer")


def sanitize_tool_observation(observation: str) -> str:
    """Prevent stdout from injecting Qwen chat or tool delimiter tokens."""

    replacements = {
        "<|": "< |",
        "|>": "| >",
        "<tool_call>": "<tool_call >",
        "</tool_call>": "</tool_call >",
        "<tool_response>": "<tool_response >",
        "</tool_response>": "</tool_response >",
    }
    sanitized = observation
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def format_tool_observation(observation: str) -> str:
    """Render exactly the tool-response suffix used by Qwen3's chat template."""

    content = sanitize_tool_observation(observation).strip()
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{content}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _with_tool_stop(sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    params = dict(sampling_params)
    stop = params.get("stop")
    if stop is None:
        stops: list[str] = []
    elif isinstance(stop, str):
        stops = [stop]
    else:
        stops = list(stop)
    if "</tool_call>" not in stops:
        stops.append("</tool_call>")
    params["stop"] = stops
    params["no_stop_trim"] = True
    params["max_new_tokens"] = max_new_tokens
    return params


_TOOL: DockerPythonTool | None = None
_TOOL_SEMAPHORE: asyncio.Semaphore | None = None


async def _execute_python(code: str):
    global _TOOL, _TOOL_SEMAPHORE
    if _TOOL is None:
        _TOOL = DockerPythonTool()
    if _TOOL_SEMAPHORE is None:
        concurrency = int(os.environ.get("MATH_AGENT_TOOL_CONCURRENCY", "8"))
        if concurrency <= 0:
            raise ValueError("MATH_AGENT_TOOL_CONCURRENCY must be positive")
        _TOOL_SEMAPHORE = asyncio.Semaphore(concurrency)
    async with _TOOL_SEMAPHORE:
        return await _TOOL.execute(code)


def _reset_sample(sample: Any) -> None:
    sample.tokens = []
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.reward = None


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """slime custom generation entry point for a stateless Python math agent."""

    if getattr(args, "partial_rollout", False):
        raise ValueError("Math Agent baseline does not support partial rollout")

    from slime.rollout.sglang_rollout import GenerateState, get_model_url
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    _reset_sample(sample)
    sample.status = Sample.Status.PENDING
    state = GenerateState(args)

    if isinstance(sample.prompt, str):
        messages = [{"role": "user", "content": sample.prompt}]
    else:
        messages = sample.prompt
    if not isinstance(messages, list) or not messages:
        raise ValueError("sample.prompt must be a non-empty conversation")

    tools = sample.metadata.get("tools") if isinstance(sample.metadata, dict) else None
    tools = tools or TOOL_SPEC
    template_kwargs = dict(getattr(sample, "apply_chat_template_kwargs", {}) or {})
    template_kwargs.setdefault("enable_thinking", True)
    prompt_text = state.tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_ids)

    max_context = getattr(args, "rollout_max_context_len", None)
    if max_context is None:
        max_context = int(args.context_parallel_size) * int(args.max_tokens_per_gpu)
    response_budget = int(sampling_params.get("max_new_tokens") or args.rollout_max_response_len)
    max_tool_calls = int(os.environ.get("MATH_AGENT_MAX_TOOL_CALLS", "4"))
    if max_tool_calls <= 0:
        raise ValueError("MATH_AGENT_MAX_TOOL_CALLS must be positive")

    rollout_meta = sample.metadata.setdefault("math_agent_rollout", {})
    rollout_meta.update({"tool_calls": 0, "protocol": "qwen3_hermes_code_interpreter_v1"})
    url = get_model_url(args, "actor", "/generate")

    for _turn in range(max_tool_calls + 1):
        remaining = min(response_budget - sample.response_length, max_context - len(sample.tokens))
        if remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            rollout_meta["terminal_reason"] = "token_budget"
            break

        payload = {
            "input_ids": sample.tokens,
            "sampling_params": _with_tool_stop(sampling_params, remaining),
            "return_logprob": True,
        }
        output = await post(url, payload)
        meta_info = output.get("meta_info") or {}
        finish_type = (meta_info.get("finish_reason") or {}).get("type")
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            rollout_meta["terminal_reason"] = "server_abort"
            return sample

        token_log_probs = meta_info.get("output_token_logprobs")
        if token_log_probs is None:
            sample.status = Sample.Status.ABORTED
            rollout_meta["terminal_reason"] = "missing_logprobs"
            return sample
        generated_ids = [item[1] for item in token_log_probs]
        generated_log_probs = [item[0] for item in token_log_probs]
        generated_text = output.get("text")
        if generated_text is None:
            generated_text = state.tokenizer.decode(generated_ids)
        sample.append_response_tokens(
            args,
            tokens=generated_ids,
            log_probs=generated_log_probs,
            trainable=True,
            meta_info=meta_info,
            text=generated_text,
        )

        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
            rollout_meta["terminal_reason"] = "model_length"
            break

        action = parse_model_action(generated_text)
        if isinstance(action, FinalAnswer):
            sample.status = Sample.Status.COMPLETED
            rollout_meta["terminal_reason"] = "final_answer"
            break
        if isinstance(action, InvalidAction):
            sample.status = Sample.Status.COMPLETED
            rollout_meta["terminal_reason"] = "invalid_action"
            rollout_meta["error"] = action.reason
            break

        if rollout_meta["tool_calls"] >= max_tool_calls:
            sample.status = Sample.Status.COMPLETED
            rollout_meta["terminal_reason"] = "tool_call_limit"
            break

        result = await _execute_python(action.code)
        rollout_meta["tool_calls"] += 1
        rollout_meta.setdefault("tool_results", []).append(
            {
                "ok": result.ok,
                "error_type": result.error_type,
                "exit_code": result.exit_code,
                "truncated": result.truncated,
                "duration_seconds": result.duration_seconds,
            }
        )
        observation_text = format_tool_observation(result.observation())
        observation_ids = state.tokenizer(observation_text, add_special_tokens=False)["input_ids"]
        available = min(response_budget - sample.response_length, max_context - len(sample.tokens))
        if len(observation_ids) > available:
            observation_ids = observation_ids[: max(0, available)]
            observation_text = state.tokenizer.decode(observation_ids)
            sample.append_response_tokens(
                args,
                tokens=observation_ids,
                trainable=False,
                text=observation_text,
            )
            sample.status = Sample.Status.TRUNCATED
            rollout_meta["terminal_reason"] = "observation_length"
            break

        sample.append_response_tokens(
            args,
            tokens=observation_ids,
            trainable=False,
            text=observation_text,
        )
        sample.status = Sample.Status.PENDING
    else:
        sample.status = Sample.Status.COMPLETED
        rollout_meta["terminal_reason"] = "turn_limit"

    return sample
