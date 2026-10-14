"""
axen/security/target_gate.py
===========================
Authorized Target Gate for AXEN Ethical Hacking Agent.

Enforces Section 3 of Ethical Hacking Charter:
- Explicit distinction between AUTHORIZED TARGET and UNAUTHORIZED TARGET
- Hard prohibition against unauthorized scanning or exploitation
- Local sandbox, test lab, and internal workspace target verification
"""

from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .models import AuthorizedTarget

logger = logging.getLogger("axen.security.gate")


class TargetAuthorizationError(PermissionError):
    """Raised when an operation attempts to target an unauthorized host or path."""
    pass


class SecurityTargetGate:
    """
    Validates and enforces authorization boundaries for all security testing.
    """

    def __init__(self, workspace_root: Optional[Path | str] = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self._targets: Dict[str, AuthorizedTarget] = {}
        self._init_default_targets()

    def _init_default_targets(self) -> None:
        """Register the safe local workspace and loopback interfaces by default."""
        # 1. Local Workspace Root Target
        ws_target = AuthorizedTarget(
            target_id="tgt_local_workspace",
            name="AXEN Local Repository & Software Workspace",
            target_type="local_workspace",
            endpoint=str(self.workspace_root),
            authorized_by="system_initialization",
            allowed_actions=["static_analysis", "ast_audit", "source_inspection", "safe_simulation"],
        )
        self.register_target(ws_target)

        # 2. Local Loopback API Target
        loopback_target = AuthorizedTarget(
            target_id="tgt_local_loopback",
            name="AXEN Localhost Test API (Port 8000)",
            target_type="local_api",
            endpoint="http://127.0.0.1:8000",
            authorized_by="system_initialization",
            allowed_actions=["api_audit", "header_inspection", "input_fuzzing"],
        )
        self.register_target(loopback_target)

    def register_target(self, target: AuthorizedTarget) -> None:
        """Register an authorized laboratory target."""
        self._targets[target.target_id] = target
        logger.info("Registered authorized security target: %s [%s]", target.name, target.endpoint)

    def revoke_target(self, target_id: str) -> bool:
        """Revoke authorization for a target."""
        if target_id in self._targets:
            self._targets[target_id].is_active = False
            return True
        return False

    def list_authorized_targets(self) -> List[AuthorizedTarget]:
        """Return list of all currently active authorized targets."""
        return [t for t in self._targets.values() if t.is_active]

    def validate_target(self, target_str: str, required_action: Optional[str] = None) -> AuthorizedTarget:
        """
        Verify that a target file, directory, or endpoint is explicitly authorized.

        Raises:
            TargetAuthorizationError if target is unauthorized.
        """
        if not target_str or not str(target_str).strip():
            raise TargetAuthorizationError("Security target cannot be empty.")

        clean_target = str(target_str).strip()

        # Check URL targets
        if clean_target.startswith(("http://", "https://")):
            return self._validate_url_target(clean_target, required_action)

        # Check Filesystem path targets
        return self._validate_path_target(clean_target, required_action)

    def _validate_url_target(self, url_str: str, required_action: Optional[str]) -> AuthorizedTarget:
        parsed = urlparse(url_str)
        hostname = (parsed.hostname or "").lower()

        # Strictly restrict to local loopback hosts (127.0.0.1, localhost)
        is_loopback = False
        if hostname in ("localhost", "127.0.0.1", "::1"):
            is_loopback = True
        else:
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_loopback:
                    is_loopback = True
            except ValueError:
                pass

        if not is_loopback:
            raise TargetAuthorizationError(
                f"UNAUTHORIZED TARGET: '{url_str}' is an external or remote network host. "
                "Ethical security testing is strictly restricted to authorized local laboratory targets."
            )

        for target in self._targets.values():
            if not target.is_active or target.target_type != "local_api":
                continue
            parsed_tgt = urlparse(target.endpoint)
            if parsed.hostname == parsed_tgt.hostname and (parsed.port or 80) == (parsed_tgt.port or 80):
                if required_action and required_action not in target.allowed_actions:
                    raise TargetAuthorizationError(
                        f"Action '{required_action}' is not permitted on target '{target.name}'."
                    )
                return target

        # If it is loopback but unmapped, allow as dynamic local laboratory target
        dyn_target = AuthorizedTarget(
            name=f"Dynamic Localhost ({hostname}:{parsed.port})",
            target_type="local_api",
            endpoint=url_str,
            authorized_by="loopback_policy",
        )
        return dyn_target

    def _validate_path_target(self, path_str: str, required_action: Optional[str]) -> AuthorizedTarget:
        try:
            resolved_path = Path(path_str).resolve()
        except Exception as exc:
            raise TargetAuthorizationError(f"Invalid target path '{path_str}': {exc}")

        # Check if contained inside any registered local workspace target
        for target in self._targets.values():
            if not target.is_active:
                continue
            try:
                tgt_path = Path(target.endpoint).resolve()
                if tgt_path.is_file() and resolved_path == tgt_path:
                    return target
                if tgt_path.is_dir() and (resolved_path == tgt_path or tgt_path in resolved_path.parents):
                    if required_action and required_action not in target.allowed_actions:
                        raise TargetAuthorizationError(
                            f"Action '{required_action}' is not permitted on target '{target.name}'."
                        )
                    return target
            except Exception:
                continue

        # Out-of-bounds path rejection (e.g. system directories, other users' homes)
        raise TargetAuthorizationError(
            f"UNAUTHORIZED TARGET: Path '{path_str}' falls outside authorized laboratory workspaces."
        )
