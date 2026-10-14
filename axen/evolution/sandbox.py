"""Fail-closed OS containment for general Python. No host subprocess fallback."""
from __future__ import annotations
from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from .sandbox_policy import SandboxPolicy


@dataclass
class SandboxExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_seconds: float = 0.0
    timed_out: bool = False
    memory_exceeded: bool = False
    error: str | None = None

    @property
    def is_success(self):
        return self.exit_code == 0 and not self.error and not self.timed_out and not self.memory_exceeded


class SandboxExecutor:
    """Linux container backend; operator must pre-pull the trusted image.

    No network, host mounts or inherited secrets. Non-root, read-only filesystem,
    capped CPU/RAM/PIDs and no capabilities. Shares the host kernel, not a VM.
    """
    def __init__(self, policy=None, *, image="python:3.13-slim"):
        self.policy = policy or SandboxPolicy()
        self.image = image
        self._active = {}; self._lock = threading.Lock()

    def run_code_snippet(self, code, timeout_seconds=None, env_overrides=None, *, cancel=lambda: False):
        if env_overrides: return SandboxExecutionResult(error="Environment overrides forbidden")
        if not isinstance(code, str) or len(code) > 65536:
            return SandboxExecutionResult(error="Code budget exceeded")
        timeout = min(timeout_seconds or self.policy.timeout_seconds, self.policy.timeout_seconds)
        docker = shutil.which("docker")
        if not docker:
            return SandboxExecutionResult(error="Docker containment unavailable; host execution refused")
        name = "axen-practice-" + uuid.uuid4().hex
        command = [docker, "run", "--rm", "--pull=never", "--name", name, "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
            "--memory=" + str(self.policy.max_memory_mb) + "m", "--memory-swap=" + str(self.policy.max_memory_mb) + "m",
            "--cpus=0.5", "--pids-limit=32", "--ulimit", "nofile=64:64", "--log-driver=none",
            "-i", self.image, "python", "-I", "-B", "-"]
        started = time.perf_counter()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                with self._lock:
                    if len(self._active) >= self.policy.max_concurrent_sandboxes:
                        return SandboxExecutionResult(error="Sandbox concurrency budget exceeded")
                    if cancel(): return SandboxExecutionResult(error="Cancelled before execution")
                    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=errors,
                        env={k: v for k, v in os.environ.items() if k in ("PATH", "SystemRoot", "USERPROFILE")},
                        creationflags=flags)
                    self._active[name] = proc
                proc.stdin.write(code.encode()); proc.stdin.close()
                timed_out = False; cancelled = False
                while proc.poll() is None:
                    cancelled = cancel()
                    timed_out = time.perf_counter() - started >= timeout
                    if cancelled or timed_out or output.tell() + errors.tell() > 1_000_000:
                        subprocess.run([docker, "rm", "-f", name], timeout=5, capture_output=True, creationflags=flags)
                        proc.kill(); break
                    time.sleep(0.05)
                proc.wait(timeout=5)
                output.seek(0); errors.seek(0)
                return SandboxExecutionResult(output.read(65536).decode(errors="replace"),
                    errors.read(65536).decode(errors="replace"), proc.returncode,
                    time.perf_counter() - started, timed_out, proc.returncode == 137,
                    "Cancelled" if cancelled else ("Execution timed out" if timed_out else None))
            except (OSError, subprocess.SubprocessError) as exc:
                return SandboxExecutionResult(error="Containment execution failed: " + type(exc).__name__)
            finally:
                with self._lock: active = self._active.pop(name, None)
                if active and active.poll() is None: active.kill(); active.wait(timeout=5)
                try:
                    subprocess.run([docker, "rm", "-f", name], timeout=5, capture_output=True, creationflags=flags)
                except (OSError, subprocess.SubprocessError): pass

