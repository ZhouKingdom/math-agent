"""A stateless Python tool executed in a resource-limited Docker container."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class PythonToolConfig:
    image: str = "python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76"
    docker_binary: str = "docker"
    timeout_seconds: float = 10.0
    memory: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 64
    tmpfs_size: str = "64m"
    max_code_bytes: int = 64 * 1024
    max_output_bytes: int = 32 * 1024

    @classmethod
    def from_env(cls) -> "PythonToolConfig":
        return cls(
            image=os.environ.get("MATH_AGENT_TOOL_IMAGE", cls.image),
            docker_binary=os.environ.get("MATH_AGENT_DOCKER_BINARY", cls.docker_binary),
            timeout_seconds=float(os.environ.get("MATH_AGENT_TOOL_TIMEOUT", cls.timeout_seconds)),
            memory=os.environ.get("MATH_AGENT_TOOL_MEMORY", cls.memory),
            cpus=float(os.environ.get("MATH_AGENT_TOOL_CPUS", cls.cpus)),
            pids_limit=int(os.environ.get("MATH_AGENT_TOOL_PIDS", cls.pids_limit)),
            tmpfs_size=os.environ.get("MATH_AGENT_TOOL_TMPFS", cls.tmpfs_size),
            max_code_bytes=int(os.environ.get("MATH_AGENT_TOOL_MAX_CODE_BYTES", cls.max_code_bytes)),
            max_output_bytes=int(os.environ.get("MATH_AGENT_TOOL_MAX_OUTPUT_BYTES", cls.max_output_bytes)),
        )

    def validate(self) -> None:
        if not self.image.strip():
            raise ValueError("tool image must not be empty")
        if self.timeout_seconds <= 0 or self.cpus <= 0:
            raise ValueError("tool timeout and CPU limit must be positive")
        if self.pids_limit <= 0 or self.max_code_bytes <= 0 or self.max_output_bytes <= 0:
            raise ValueError("tool PID, code, and output limits must be positive")


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error_type: str | None = None
    truncated: bool = False
    duration_seconds: float = 0.0

    def observation(self) -> str:
        if self.ok:
            return self.stdout.rstrip() or "(program completed with no output)"
        detail = self.stderr.rstrip() or self.stdout.rstrip() or "no diagnostic output"
        kind = self.error_type or "execution_error"
        suffix = " [output truncated]" if self.truncated else ""
        return f"{kind}: {detail}{suffix}"


class DockerPythonTool:
    """Execute each call in a new, networkless, read-only container."""

    def __init__(self, config: PythonToolConfig | None = None) -> None:
        self.config = config or PythonToolConfig.from_env()
        self.config.validate()

    def build_command(self, container_name: str) -> list[str]:
        config = self.config
        return [
            config.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(config.pids_limit),
            "--memory",
            config.memory,
            "--memory-swap",
            config.memory,
            "--cpus",
            str(config.cpus),
            "--user",
            "65534:65534",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={config.tmpfs_size}",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONIOENCODING=utf-8",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            config.image,
            "python",
            "-I",
            "-S",
            "-",
        ]

    async def _remove_container(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.docker_binary,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=5)
        except (FileNotFoundError, TimeoutError):
            return

    async def execute(self, code: str) -> ToolResult:
        started = time.monotonic()
        if not isinstance(code, str) or not code.strip():
            return ToolResult(False, error_type="invalid_code", stderr="code must be a non-empty string")
        encoded_code = code.encode("utf-8")
        if len(encoded_code) > self.config.max_code_bytes:
            return ToolResult(
                False,
                error_type="code_too_large",
                stderr=f"code exceeds {self.config.max_code_bytes} bytes",
            )

        container_name = f"math-agent-{uuid.uuid4().hex}"
        try:
            process = await asyncio.create_subprocess_exec(
                *self.build_command(container_name),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as error:
            return ToolResult(
                False,
                error_type="sandbox_unavailable",
                stderr=str(error),
                duration_seconds=time.monotonic() - started,
            )

        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        process.stdin.write(encoded_code)
        await process.stdin.drain()
        process.stdin.close()

        stdout = bytearray()
        stderr = bytearray()
        total_bytes = 0
        output_lock = asyncio.Lock()
        truncated = False

        async def read_limited(stream: asyncio.StreamReader, destination: bytearray) -> None:
            nonlocal total_bytes, truncated
            while chunk := await stream.read(8192):
                async with output_lock:
                    remaining = self.config.max_output_bytes - total_bytes
                    if remaining > 0:
                        destination.extend(chunk[:remaining])
                        total_bytes += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        truncated = True
                        if process.returncode is None:
                            process.kill()
                        return

        readers = [
            asyncio.create_task(read_limited(process.stdout, stdout)),
            asyncio.create_task(read_limited(process.stderr, stderr)),
        ]
        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.gather(process.wait(), *readers),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            timed_out = True
            if process.returncode is None:
                process.kill()
            await self._remove_container(container_name)
            await process.wait()
            await asyncio.gather(*readers, return_exceptions=True)
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()

        if truncated:
            await self._remove_container(container_name)

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        duration = time.monotonic() - started
        if timed_out:
            return ToolResult(
                False,
                stdout=stdout_text,
                stderr=stderr_text or f"execution exceeded {self.config.timeout_seconds:g} seconds",
                exit_code=process.returncode,
                error_type="timeout",
                truncated=truncated,
                duration_seconds=duration,
            )
        if truncated:
            return ToolResult(
                False,
                stdout=stdout_text,
                stderr=stderr_text or f"output exceeded {self.config.max_output_bytes} bytes",
                exit_code=process.returncode,
                error_type="output_limit",
                truncated=True,
                duration_seconds=duration,
            )
        if process.returncode != 0:
            return ToolResult(
                False,
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=process.returncode,
                error_type="runtime_error",
                duration_seconds=duration,
            )
        return ToolResult(
            True,
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=process.returncode,
            duration_seconds=duration,
        )
