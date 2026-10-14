"""
axen/coding/command_runner.py
==============================
Secure command runner for the AXEN Coding Agent:
- Commands execute strictly inside the approved workspace root directory
- Hard timeouts to prevent hung processes
- Strict security blacklist blocking dangerous system commands
- Captures exit code, stdout, stderr, and elapsed duration
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shlex
import subprocess
import time
import os
import shutil
import tempfile
import sys
import uuid
from typing import List, Optional

from .permissions import OperationType, PermissionPolicy
from .workspace import Workspace

logger = logging.getLogger("axen.coding.command")


class CommandSecurityError(PermissionError):
    """Raised when an execution violates command safety rules."""
    pass


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# Dangerous commands that must never be executed autonomously
_DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\b(?:rm|del)\s+[\/\-]*(?:rf|s|q|\*)\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\b(?:shutdown|reboot)\b", re.IGNORECASE),
    re.compile(r"\b(?:netsh|reg|sc|taskkill)\b", re.IGNORECASE),
    re.compile(r"\b(?:chmod|chown)\s+[\/\-]*[0-7]{3,4}\b", re.IGNORECASE),
    re.compile(r"\bpowershell\b.*?(?:-enc|-encodedcommand)\b", re.IGNORECASE),
    re.compile(r"\|\s*(?:sh|bash|powershell|cmd)\b", re.IGNORECASE),
    re.compile(r"\b(?:curl|wget)\b.*?\b(?:sh|bash)\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
]


class CommandRunner:
    """Executes validated commands strictly inside the approved workspace."""

    def __init__(self, workspace: Workspace, default_timeout: float = 30.0) -> None:
        self.workspace = workspace
        self.default_timeout = default_timeout

    def validate_command(self, command_str: str) -> None:
        """Inspect command string for dangerous operations."""
        cmd_clean = command_str.strip()
        if not cmd_clean:
            raise CommandSecurityError("Empty command string.")

        for pat in _DANGEROUS_COMMAND_PATTERNS:
            if pat.search(cmd_clean):
                raise CommandSecurityError(
                    f"Blocked dangerous command pattern in: '{command_str}'."
                )

    def run_sandboxed(self, command, timeout=45.0):
        """Untrusted tests run in an ephemeral, offline container, never on the host."""
        image = os.getenv("AEXN_SANDBOX_IMAGE")
        docker = shutil.which("docker")
        if not image or not docker:
            raise CommandSecurityError("No external sandbox configured. Set AEXN_SANDBOX_IMAGE to an existing local Docker image with the project's test dependencies.")
        args = list(command) if isinstance(command, list) else shlex.split(command)
        self.validate_command(" ".join(args))
        if args[0] == sys.executable:
            args[0] = "python"
        with tempfile.TemporaryDirectory(prefix="aexn-sandbox-") as folder:
            copied = 0
            for source in self.workspace.list_files(max_files=2000):
                if source.name.startswith('.') or any(word in source.name.lower() for word in ('credential', 'secret', 'token', 'password')) or source.suffix.lower() not in ('.py', '.js', '.ts', '.json', '.txt', '.toml', '.yaml', '.yml', '.html', '.css', '.md', '.cfg', '.ini'):
                    continue
                copied += source.stat().st_size
                if copied > 50 * 1024 * 1024:
                    raise CommandSecurityError("Sandbox source copy exceeds 50 MB")
                target = Path(folder) / source.relative_to(self.workspace.root_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            container_name = 'aexn-' + uuid.uuid4().hex
            argv = [docker, "run", "--name", container_name, "--rm", "--pull=never", "--network=none", "--read-only",
                    "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
                    "--memory=1g", "--cpus=2", "--tmpfs=/tmp", "-v", f"{folder}:/workspace:rw",
                    "-w", "/workspace", image, *args]
            started = time.perf_counter()
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False)
                return CommandResult(" ".join(args), result.returncode, result.stdout, result.stderr,
                                     time.perf_counter() - started)
            except subprocess.TimeoutExpired:
                return CommandResult(" ".join(args), -1, "", "Sandbox timed out", time.perf_counter() - started, True)
            finally:
                try:
                    subprocess.run([docker, 'rm', '-f', container_name], capture_output=True, timeout=10, shell=False)
                except (OSError, subprocess.TimeoutExpired):
                    logger.warning("Could not confirm sandbox cleanup: %s", container_name)

    def run(
        self,
        command: str | List[str],
        timeout: Optional[float] = None,
        env_vars: Optional[dict[str, str]] = None,
    ) -> CommandResult:
        """
        Execute command inside the workspace directory.
        Raises PermissionError if EXECUTE permission is missing.
        """
        self.workspace.policy.require(OperationType.EXECUTE_COMMAND, str(command))

        if isinstance(command, list):
            cmd_str = " ".join(command)
            cmd_args = command
        else:
            cmd_str = command
            cmd_args = command  # On Windows, pass string to shell=True or split

        self.validate_command(cmd_str)

        t_start = time.perf_counter()
        use_timeout = timeout or self.default_timeout

        try:
            # Execute within the workspace root
            proc = subprocess.run(
                cmd_args,
                cwd=str(self.workspace.root_path),
                capture_output=True,
                text=True,
                shell=isinstance(cmd_args, str),
                timeout=use_timeout,
                env=env_vars,
            )
            duration = time.perf_counter() - t_start
            return CommandResult(
                command=cmd_str,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - t_start
            logger.warning("Command timed out after %.2f seconds: %s", use_timeout, cmd_str)
            return CommandResult(
                command=cmd_str,
                exit_code=-1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\n[Process timed out after {use_timeout}s]",
                duration_seconds=duration,
                timed_out=True,
            )
        except Exception as exc:
            duration = time.perf_counter() - t_start
            return CommandResult(
                command=cmd_str,
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {exc}",
                duration_seconds=duration,
                timed_out=False,
            )
