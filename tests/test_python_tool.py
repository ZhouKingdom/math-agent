import asyncio
import os

import pytest

from math_agent.python_tool import DockerPythonTool, PythonToolConfig, ToolResult


def test_docker_command_enforces_isolation_limits() -> None:
    tool = DockerPythonTool(PythonToolConfig(image="python:test"))
    command = tool.build_command("math-agent-test")
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges:true" in command
    assert "--pids-limit 64" in joined
    assert "--memory 512m --memory-swap 512m" in joined
    assert "--user 65534:65534" in joined
    assert command[-4:] == ["python:test", "python", "-I", "-S", "-"][-4:]


def test_invalid_code_is_rejected_before_docker() -> None:
    tool = DockerPythonTool()
    result = asyncio.run(tool.execute("  "))
    assert not result.ok
    assert result.error_type == "invalid_code"


def test_tool_result_observation_has_bounded_error_shape() -> None:
    result = ToolResult(False, stderr="trace", error_type="runtime_error", truncated=True)
    assert result.observation() == "runtime_error: trace [output truncated]"


@pytest.mark.skipif(os.environ.get("MATH_AGENT_RUN_DOCKER_TESTS") != "1", reason="live Docker test is opt-in")
def test_live_container_is_stateless_networkless_and_read_only() -> None:
    code = """from pathlib import Path
import socket

print(6 * 7)
try:
    Path('/forbidden').write_text('x')
except OSError as error:
    print('root_write_blocked', type(error).__name__)
try:
    socket.create_connection(('1.1.1.1', 53), timeout=0.5)
except OSError as error:
    print('network_blocked', type(error).__name__)
"""
    result = asyncio.run(DockerPythonTool().execute(code))
    assert result.ok, result.observation()
    assert "42" in result.stdout
    assert "root_write_blocked" in result.stdout
    assert "network_blocked" in result.stdout
