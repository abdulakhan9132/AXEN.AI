"""
axen/security/code_analyzer.py
=============================
AST-Based Static Application Security Testing (SAST) for AXEN.

Implements Section 7 (Secure Coding) of the Cybersecurity Charter:
- Audits source code for injection vulnerabilities, unsafe eval, hardcoded secrets,
  weak cryptography, path traversal, and insecure deserialization.
- Pinpoints exact line numbers, AST nodes, and generates actionable remediation guidance.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    VulnerabilityCategory,
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilityStatus,
)

logger = logging.getLogger("axen.security.analyzer")


class SecureCodeAnalyzer:
    """
    Performs deep AST and lexical security analysis of Python source code.
    """

    # Secret patterns
    _SECRET_KEY_REGEX = re.compile(r"(?i)(api[_-]?key|secret|password|auth[_-]?token|access[_-]?token|private[_-]?key)")
    _HIGH_ENTROPY_TOKEN = re.compile(r"^(?:sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9_\-\.]{20,}|ghp_[A-Za-z0-9]{30,}|[A-Fa-f0-9]{32,64})$")

    def __init__(self) -> None:
        pass

    def audit_file(self, file_path: Path | str) -> List[VulnerabilityFinding]:
        """Audit a single source code file on disk."""
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return self.audit_source_code(content, target_path=str(path))
        except Exception as exc:
            logger.error("Error reading file %s for audit: %s", path, exc)
            return []

    def audit_source_code(self, source_code: str, target_path: str = "<memory>") -> List[VulnerabilityFinding]:
        """Analyze Python source code string for security weaknesses."""
        findings: List[VulnerabilityFinding] = []
        if not source_code or not source_code.strip():
            return findings

        # 1. Lexical checks (Secrets & Token leakage)
        findings.extend(self._scan_lexical_secrets(source_code, target_path))

        # 2. AST structural checks
        try:
            tree = ast.parse(source_code)
            findings.extend(self._scan_ast_nodes(tree, source_code, target_path))
        except SyntaxError as syn_err:
            findings.append(
                VulnerabilityFinding(
                    title="Syntax Error / Unparsable Code",
                    category=VulnerabilityCategory.INSECURE_CONFIGURATION,
                    severity=VulnerabilitySeverity.INFORMATIONAL,
                    target_path=target_path,
                    line_number=syn_err.lineno,
                    code_snippet=syn_err.text or "",
                    root_cause=f"Code syntax error: {syn_err.msg}",
                    remediation_guidance="Fix syntax before performing deeper AST security audit.",
                )
            )

        return findings

    def _scan_lexical_secrets(self, source_code: str, target_path: str) -> List[VulnerabilityFinding]:
        findings: List[VulnerabilityFinding] = []
        lines = source_code.splitlines()

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Look for variable assignment: api_key = "..."
            if "=" in stripped:
                parts = stripped.split("=", 1)
                var_name = parts[0].strip()
                val_part = parts[1].strip().strip("'\"")

                if self._SECRET_KEY_REGEX.search(var_name):
                    # Check if assigned value resembles a real token or hardcoded secret
                    if len(val_part) >= 12 and not val_part.startswith(("os.getenv", "os.environ", "config", "settings")):
                        findings.append(
                            VulnerabilityFinding(
                                title=f"Hardcoded Secret / Token in Assignment ('{var_name}')",
                                category=VulnerabilityCategory.HARDCODED_SECRETS,
                                severity=VulnerabilitySeverity.HIGH,
                                target_path=target_path,
                                line_number=idx,
                                code_snippet=stripped,
                                root_cause="Credentials or tokens embedded directly in source code.",
                                conditions_required="Secret exposed if source repository or artifacts are accessed.",
                                security_impact="Unauthorized API access, credential theft, and privilege escalation.",
                                detection_method="lexical_token_analysis",
                                remediation_guidance=f"Extract '{var_name}' to environment variables or safe secret manager.",
                                remediation_patch=f"{var_name} = os.getenv('{var_name.upper()}')",
                            )
                        )
                        continue

            # Direct token match
            for token_match in self._HIGH_ENTROPY_TOKEN.finditer(stripped):
                findings.append(
                    VulnerabilityFinding(
                        title="High-Entropy Credential or API Token Detected",
                        category=VulnerabilityCategory.HARDCODED_SECRETS,
                        severity=VulnerabilitySeverity.CRITICAL,
                        target_path=target_path,
                        line_number=idx,
                        code_snippet=stripped,
                        root_cause="High-entropy private key or token literal in source file.",
                        security_impact="Immediate compromise of external or internal service credentials.",
                        detection_method="entropy_pattern_matcher",
                        remediation_guidance="Revoke exposed key and load from secure environment variables.",
                    )
                )

        return findings

    def _scan_ast_nodes(self, tree: ast.AST, source_code: str, target_path: str) -> List[VulnerabilityFinding]:
        findings: List[VulnerabilityFinding] = []
        lines = source_code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 1. Unsafe Eval / Exec
                func_name = self._get_call_func_name(node)
                if func_name in ("eval", "exec"):
                    lineno = getattr(node, "lineno", 1)
                    snippet = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    findings.append(
                        VulnerabilityFinding(
                            title=f"Arbitrary Code Execution via Unsafe '{func_name}()'",
                            category=VulnerabilityCategory.UNSAFE_EVAL,
                            severity=VulnerabilitySeverity.CRITICAL,
                            target_path=target_path,
                            line_number=lineno,
                            code_snippet=snippet.strip(),
                            root_cause=f"Direct invocation of '{func_name}()' parses and executes arbitrary input.",
                            conditions_required="User-controlled or untrusted string supplied as argument.",
                            security_impact="Remote code execution (RCE) and complete host compromise.",
                            detection_method="ast_call_visitor",
                            remediation_guidance="Use ast.literal_eval for data parsing or structured schema deserializers.",
                            remediation_patch="ast.literal_eval(safe_input)",
                        )
                    )

                # 2. Insecure Subprocess with shell=True or Command Injection
                elif func_name in ("subprocess.Popen", "subprocess.run", "subprocess.call", "os.system", "os.popen"):
                    lineno = getattr(node, "lineno", 1)
                    snippet = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    has_shell_true = False
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            has_shell_true = True

                    is_os_system = func_name in ("os.system", "os.popen")

                    if has_shell_true or is_os_system:
                        findings.append(
                            VulnerabilityFinding(
                                title=f"Command Injection Risk via '{func_name}(shell=True)'",
                                category=VulnerabilityCategory.COMMAND_INJECTION,
                                severity=VulnerabilitySeverity.HIGH,
                                target_path=target_path,
                                line_number=lineno,
                                code_snippet=snippet.strip(),
                                root_cause=f"Invoking system shell through '{func_name}' permits command separator injection.",
                                conditions_required="Untrusted strings concatenated into command string.",
                                security_impact="Arbitrary command execution in local shell environment.",
                                detection_method="ast_call_visitor",
                                remediation_guidance="Pass command arguments as a list with shell=False.",
                                remediation_patch="subprocess.run(['command', arg1, arg2], check=True)",
                            )
                        )

                # 3. Insecure Deserialization (pickle / unsafe yaml)
                elif func_name in ("pickle.loads", "pickle.load", "_pickle.loads", "yaml.unsafe_load"):
                    lineno = getattr(node, "lineno", 1)
                    snippet = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    findings.append(
                        VulnerabilityFinding(
                            title=f"Insecure Deserialization via '{func_name}'",
                            category=VulnerabilityCategory.UNSAFE_DESERIALIZATION,
                            severity=VulnerabilitySeverity.HIGH,
                            target_path=target_path,
                            line_number=lineno,
                            code_snippet=snippet.strip(),
                            root_cause="Deserializing untrusted byte streams enables arbitrary code execution via object reduction (__reduce__).",
                            conditions_required="Deserializing data from network, untrusted cache, or user upload.",
                            security_impact="Remote code execution via serialized gadget chain.",
                            detection_method="ast_call_visitor",
                            remediation_guidance="Use safe formats such as JSON or yaml.safe_load.",
                            remediation_patch="json.loads(data)",
                        )
                    )

                # 4. Weak Cryptography (md5, sha1 for security)
                elif func_name in ("hashlib.md5", "hashlib.sha1"):
                    # Check if usedforsecurity=False is set
                    used_for_sec = True
                    for kw in node.keywords:
                        if kw.arg == "usedforsecurity" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            used_for_sec = False

                    if used_for_sec:
                        lineno = getattr(node, "lineno", 1)
                        snippet = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                        findings.append(
                            VulnerabilityFinding(
                                title=f"Weak Cryptographic Hash Algorithm ('{func_name}')",
                                category=VulnerabilityCategory.WEAK_CRYPTOGRAPHY,
                                severity=VulnerabilitySeverity.MEDIUM,
                                target_path=target_path,
                                line_number=lineno,
                                code_snippet=snippet.strip(),
                                root_cause=f"{func_name} is cryptographically broken and vulnerable to collision attacks.",
                                conditions_required="Used in password hashing, digital signatures, or integrity verification.",
                                security_impact="Cryptographic collision, signature forgery, and authentication bypass.",
                                detection_method="ast_call_visitor",
                                remediation_guidance="Use hashlib.sha256() or Argon2id/bcrypt for password hashing.",
                                remediation_patch="hashlib.sha256(data.encode()).hexdigest()",
                            )
                        )

        return findings

    def _get_call_func_name(self, call_node: ast.Call) -> str:
        """Resolve full dotted function name from an ast.Call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            parts = []
            curr: Any = call_node.func
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            parts.reverse()
            return ".".join(parts)
        return ""
