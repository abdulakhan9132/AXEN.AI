"""Workspace-scoped Code Agent tools using the existing tool runtime."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
import ntpath
from pathlib import Path, PureWindowsPath
import shlex
import sqlite3
import threading
import uuid

from app.agent.memory import MemoryStore
from app.agent.tools import ExecutionState, ToolDefinition, ToolExecutionContext, ToolPermission, ToolResult
from axen.coding.command_runner import CommandRunner
from axen.coding.permissions import PermissionLevel, PermissionPolicy
from axen.coding.workspace import Workspace, WorkspaceSecurityError
from axen.tools.command_tool import _validate_command_tokens


@dataclass(frozen=True)
class PreparedAction:
    tool: str
    arguments_json: str
    original_hash: str | None = None


class CodeAgentTools:
    """Approval tickets are one-use, server-owned and bound to the exact action."""

    def __init__(self, workspace, *, memory_store=None, command_runner=None):
        self.workspace = Workspace(workspace, policy=PermissionPolicy(PermissionLevel.EXECUTE))
        self.memory = memory_store if memory_store is not None else MemoryStore()
        self.runner = command_runner if command_runner is not None else CommandRunner(self.workspace)
        self._pending: dict[str, PreparedAction] = {}
        self._grants: dict[str, PreparedAction] = {}
        self._read_paths: set[str] = set()
        self._lock = threading.RLock()
        with sqlite3.connect(str(self.memory.db_path)) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS code_agent_actions (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, tool TEXT NOT NULL,
                arguments TEXT NOT NULL, decision TEXT NOT NULL, detail TEXT NOT NULL)""")
        from app.tools.registry import build_registry
        self.registry = build_registry(code_tools=self)

    def _audit(self, tool, arguments, decision, detail=""):
        with sqlite3.connect(str(self.memory.db_path)) as connection:
            connection.execute("INSERT INTO code_agent_actions VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, datetime.now(timezone.utc).isoformat(), str(tool),
                 json.dumps(arguments, ensure_ascii=False, default=str), decision, str(detail)))

    def resolve_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip() or '\x00' in path:
            raise WorkspaceSecurityError("A nonempty relative path is required")
        windows = PureWindowsPath(path)
        parts = path.replace('\\', '/').split('/')
        if windows.drive or windows.root or path.startswith('/') or '..' in parts or ':' in path:
            raise WorkspaceSecurityError("Absolute paths and '..' are not allowed")
        for part in parts:
            if part not in ('', '.') and (part.endswith((' ', '.')) or ntpath.isreserved(part)):
                raise WorkspaceSecurityError("Windows device names and ambiguous paths are not allowed")
        target = self.workspace.resolve_path(path.replace('\\', '/'))
        if target.exists() and target.is_file() and target.stat().st_nlink > 1:
            raise WorkspaceSecurityError("Hard-linked files are not allowed")
        return target

    def _relative(self, path):
        return self.resolve_path(path).relative_to(self.workspace.root_path).as_posix()

    def command_arguments(self, command):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("A nonempty command is required")
        parts = shlex.split(command, posix=False)
        parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in ('\"', "'") else part for part in parts]
        if PureWindowsPath(parts[0]).drive or '/' in parts[0] or '\\' in parts[0]:
            raise PermissionError("Use an allowlisted program name, not an executable path")
        executable, arguments = _validate_command_tokens(parts[0], parts[1:])
        for argument in arguments:
            value = argument.split('=', 1)[-1]
            if PureWindowsPath(value).drive or PureWindowsPath(value).root or value.startswith('/') and not argument.startswith('-'):
                raise WorkspaceSecurityError("Command paths must be relative to WORKSPACE")
            if '/' in value or '\\' in value or Path(value).suffix in ('.py', '.js', '.json'):
                self.resolve_path(value)
        return [executable, *[argument.replace('\\', '/') for argument in arguments]]

    def _validate(self, action):
        if not isinstance(action, dict):
            raise ValueError("Action must be one JSON object")
        tool = action.get('tool')
        keys = {'list_files': {'path'}, 'read_file': {'path'},
                'write_file': {'path', 'content'}, 'run_command': {'command'}}
        if not isinstance(tool, str) or tool not in keys or set(action) != {'tool', *keys[tool]}:
            raise ValueError("Unknown tool or invalid action fields")
        arguments = {key: action[key] for key in keys[tool]}
        if tool == 'run_command':
            self.command_arguments(arguments['command'])
        else:
            path = self._relative(arguments['path'])
            arguments['path'] = path
            if tool != 'list_files' and path == '.':
                raise ValueError("A file path is required")
            if tool == 'write_file':
                content = arguments['content']
                if not isinstance(content, str) or len(content.encode('utf-8')) > self.workspace.max_file_size_bytes:
                    raise ValueError("File content is invalid or exceeds the size limit")
        return tool, arguments

    @staticmethod
    def _hash(content):
        return None if content is None else hashlib.sha256(content.encode('utf-8')).hexdigest()

    def prepare(self, action):
        """Validate before asking approval, and prepare a review without writing."""
        try:
            tool, arguments = self._validate(action)
            original = None
            diff = None
            if tool == 'write_file':
                target = self.resolve_path(arguments['path'])
                if target.exists():
                    if arguments['path'] not in self._read_paths:
                        raise PermissionError("Read the existing file before editing it")
                    original = self.workspace.read_file(arguments['path'])
                diff = ''.join(difflib.unified_diff((original or '').splitlines(keepends=True),
                    arguments['content'].splitlines(keepends=True), fromfile=arguments['path'],
                    tofile=arguments['path']))
            request_id = uuid.uuid4().hex
            prepared = PreparedAction(tool, json.dumps(arguments, sort_keys=True), self._hash(original))
            self._audit(tool, arguments, 'pending' if tool in ('write_file', 'run_command') else 'requested')
            with self._lock:
                self._pending[request_id] = prepared
            result = {'request_id': request_id, 'tool': tool, 'arguments': arguments,
                      'requires_approval': tool in ('write_file', 'run_command')}
            if diff is not None:
                result['diff'] = diff
            return result
        except (ValueError, OSError, PermissionError) as error:
            self._audit(action.get('tool', 'invalid') if isinstance(action, dict) else 'invalid', action, 'blocked', error)
            raise

    def execute(self, request_id, *, approved=None):
        """Called by the server after a user decision, never by model arguments."""
        with self._lock:
            prepared = self._pending.pop(request_id, None)
        if prepared is None:
            raise ValueError("Unknown or already consumed permission request")
        arguments = json.loads(prepared.arguments_json)
        risky = prepared.tool in ('write_file', 'run_command')
        if risky and approved is not True:
            self._audit(prepared.tool, arguments, 'denied')
            return ToolResult(tool_name=prepared.tool, success=False,
                error='Permission denied. Adapt your plan or finish.', state=ExecutionState.FAILED)
        if risky:
            with self._lock:
                self._grants[request_id] = prepared
        self._audit(prepared.tool, arguments, 'approved' if risky else 'authorized')
        context = ToolExecutionContext(workspace_path=str(self.workspace.root_path),
            timeout_seconds=30, authorized_permissions={ToolPermission.READ_ONLY,
                ToolPermission.LOCAL_FILE, ToolPermission.COMMAND_EXECUTION},
            metadata={'code_request_id': request_id}, max_output_bytes=100_000)
        try:
            return self.registry.execute(prepared.tool, arguments, context=context)
        finally:
            with self._lock:
                self._grants.pop(request_id, None)

    def _handler(self, tool, arguments, context=None):
        decision = 'automatic'
        try:
            if 'tool' in arguments:
                raise ValueError("Tool name cannot be overridden by arguments")
            _, safe = self._validate({'tool': tool, **arguments})
            prepared = None
            if tool in ('write_file', 'run_command'):
                request_id = context.metadata.get('code_request_id') if context else None
                with self._lock:
                    prepared = self._grants.pop(request_id, None)
                if not prepared or prepared.tool != tool or prepared.arguments_json != json.dumps(safe, sort_keys=True):
                    raise PermissionError("This exact action requires a fresh user approval")
                decision = 'approved'
            if tool == 'list_files':
                target = self.resolve_path(safe['path'])
                if not target.is_dir():
                    raise NotADirectoryError(safe['path'])
                files = self.workspace.list_files(safe['path'], max_files=500)
                result = {'success': True, 'files': [self._relative(str(path.relative_to(self.workspace.root_path))) for path in files
                    if path.stat().st_nlink == 1]}
            elif tool == 'read_file':
                result = {'success': True, 'path': safe['path'], 'content': self.workspace.read_file(safe['path'])}
                self._read_paths.add(safe['path'])
            elif tool == 'write_file':
                target = self.resolve_path(safe['path'])
                original = self.workspace.read_file(safe['path']) if target.exists() else None
                if self._hash(original) != prepared.original_hash:
                    raise PermissionError("File changed since the approval preview; request approval again")
                count = self.workspace.write_file(safe['path'], safe['content'])
                result = {'success': True, 'path': safe['path'], 'characters_written': count}
            else:
                # Don't copy hard-linked project files into the execution snapshot.
                for file in self.workspace.list_files(max_files=2000):
                    self.resolve_path(str(file.relative_to(self.workspace.root_path)))
                run = self.runner.run_sandboxed(self.command_arguments(safe['command']), timeout=30)
                result = {'success': run.success, 'stdout': run.stdout, 'stderr': run.stderr,
                          'exit_code': run.exit_code, 'timeout': run.timed_out}
                if not run.success:
                    result['error'] = run.stderr or 'Command failed'
            self._audit(tool, safe, decision, json.dumps(result, ensure_ascii=False)[:3000])
            if tool == 'run_command':
                return ToolResult(tool_name=tool, success=run.success, output=result,
                    error=result.get('error'), state=ExecutionState.TIMEOUT if run.timed_out else
                        ExecutionState.COMPLETED if run.success else ExecutionState.FAILED,
                    metadata={'exit_code': run.exit_code, 'stdout': run.stdout, 'stderr': run.stderr})
            return result
        except Exception as error:
            self._audit(tool, arguments, 'blocked' if decision == 'automatic' else decision, error)
            return {'success': False, 'error': str(error)}

    def definitions(self):
        tools = []
        for name, fields in {'list_files': {'path': 'string'}, 'read_file': {'path': 'string'},
                             'write_file': {'path': 'string', 'content': 'string'},
                             'run_command': {'command': 'string'}}.items():
            def handler(arguments, context=None, _name=name):
                return self._handler(_name, arguments, context)
            risky = name in ('write_file', 'run_command')
            tools.append(ToolDefinition(name=name, description=f'{name} inside the selected workspace',
                input_schema={'type': 'object', 'properties': {key: {'type': value} for key, value in fields.items()},
                              'required': list(fields), 'additionalProperties': False},
                permission=(ToolPermission.COMMAND_EXECUTION if name == 'run_command' else
                    ToolPermission.LOCAL_FILE if name == 'write_file' else ToolPermission.READ_ONLY).value,
                requires_approval=risky, risk='high' if risky else 'low', handler=handler))
        return tools

    @staticmethod
    def model_result(result):
        if isinstance(result, ToolResult):
            payload = {'tool': result.tool_name, 'success': result.success,
                       'result': result.output, 'error': result.error}
        else:
            payload = result
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= 3000:
            return text
        length = 2900
        while True:
            wrapped = json.dumps({'truncated': True, 'result_text': text[:length]}, ensure_ascii=False)
            if len(wrapped) <= 3000:
                return wrapped
            length = max(0, length - (len(wrapped) - 3000))
