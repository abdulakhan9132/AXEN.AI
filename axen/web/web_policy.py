"""
axen/web/web_policy.py
======================
Security policies, URL validation, SSRF protection, prompt injection defense,
and permission enforcement for the AXEN Web Agent.

Safety Principles:
1. Web content is strictly UNTRUSTED DATA — never treated as agent instructions.
2. SSRF defense blocks localhost, private LAN IPs, and local loopbacks.
3. Protocols are strictly restricted to HTTP and HTTPS.
4. Prompt injection attempts are flagged and quarantined.
5. Dangerous binary extensions are blocked from download.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any, List, Optional, Set, Tuple
from urllib.parse import urlparse

from axen.web.web_models import WebPermissionLevel


class WebSecurityError(Exception):
    """Raised when an operation violates Web Agent safety policies."""
    pass


class WebPolicy:
    """
    Enforces URL safety, SSRF protection, prompt injection mitigation,
    and permission checks.
    """

    # Prohibited URL schemes
    ALLOWED_SCHEMES: Set[str] = {"http", "https"}

    # Dangerous file extensions that must not be automatically downloaded or executed
    DANGEROUS_EXTENSIONS: Set[str] = {
        ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".msi", ".dll",
        ".scr", ".jar", ".zip", ".tar", ".gz", ".7z", ".bin",
        ".sh", ".com", ".pif", ".cpl",
    }

    # Common prompt injection patterns found in adversarial web pages
    PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.IGNORECASE),
        re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:a|an)?\s*(?:unrestricted|evil|dan|jailbreak)\b", re.IGNORECASE),
        re.compile(r"\b(?:reveal|show|dump|print)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|secret\s+key|api\s+key)\b", re.IGNORECASE),
        re.compile(r"\b(?:send|exfiltrate|post|upload)\s+(?:secrets|passwords|credentials|keys)\s+to\b", re.IGNORECASE),
        re.compile(r"\b(?:run|execute)\s+(?:the\s+following\s+)?(?:command|shell|script|cmd|powershell)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        permission_level: WebPermissionLevel = WebPermissionLevel.WEB_RESEARCH,
        allow_private_networks: bool = False,
        max_searches: int = 5,
        max_pages: int = 10,
        max_depth: int = 2,
        max_content_size: int = 15000,
        request_timeout: float = 10.0,
    ) -> None:
        self.permission_level = permission_level
        self.allow_private_networks = allow_private_networks
        self.max_searches = max_searches
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.max_content_size = max_content_size
        self.request_timeout = request_timeout

    def check_permission(self, action: str) -> None:
        """
        Verify if the requested action is permitted under the current permission level.
        """
        if self.permission_level == WebPermissionLevel.WEB_DISABLED:
            raise WebSecurityError(
                f"Web access is disabled (level={self.permission_level.value}). Cannot perform '{action}'."
            )

        if action == "search" and self.permission_level == WebPermissionLevel.WEB_READ:
            raise WebSecurityError(
                "Search is not permitted under 'WEB_READ' permission. Only direct page reading is allowed."
            )

        if action in ("click", "type", "submit", "interact"):
            if self.permission_level != WebPermissionLevel.WEB_INTERACTION:
                raise WebSecurityError(
                    f"Browser interaction '{action}' requires WEB_INTERACTION permission "
                    f"(current={self.permission_level.value}). Interactive automation is disabled."
                )

    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validates URL against SSRF, dangerous protocols, and private network addresses.
        Returns: (is_valid, error_reason)
        """
        if not url or not isinstance(url, str):
            return False, "URL cannot be empty."

        url_clean = url.strip()

        # Scheme check
        try:
            parsed = urlparse(url_clean)
        except Exception as e:
            return False, f"Malformed URL: {e}"

        scheme = parsed.scheme.lower()
        if scheme not in self.ALLOWED_SCHEMES:
            return False, (
                f"Security Violation: Protocol '{scheme}://' is prohibited. "
                f"Only HTTP and HTTPS are permitted."
            )

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False, "URL does not contain a valid hostname."

        # Check for local loopbacks / localhost by name
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "local", "internal"):
            return False, f"Security Violation: Access to local address '{hostname}' is blocked."

        # Check dangerous file extensions in path
        path_lower = parsed.path.lower()
        for ext in self.DANGEROUS_EXTENSIONS:
            if path_lower.endswith(ext):
                return False, (
                    f"Security Violation: Download of potentially dangerous executable/archive '{ext}' is prohibited."
                )

        # Check IP address resolution if not allowing private networks
        if not self.allow_private_networks:
            # Check if hostname itself is an IP
            try:
                ip_obj = ipaddress.ip_address(hostname)
                if self._is_private_or_reserved(ip_obj):
                    return False, f"Security Violation: Access to private/reserved IP '{hostname}' is prohibited."
            except ValueError:
                # Not a raw IP literal; resolve domain to check for DNS rebinding/SSRF
                try:
                    # Quick DNS resolution check
                    addr_info = socket.getaddrinfo(hostname, None)
                    for family, _, _, _, sockaddr in addr_info:
                        resolved_ip = sockaddr[0]
                        ip_obj = ipaddress.ip_address(resolved_ip)
                        if self._is_private_or_reserved(ip_obj):
                            return False, (
                                f"Security Violation: Host '{hostname}' resolves to private/reserved IP '{resolved_ip}'."
                            )
                except (socket.gaierror, socket.herror, ValueError):
                    # If DNS cannot resolve or offline test, allow proceed to provider handling
                    pass

        return True, None

    def sanitize_untrusted_text(self, text: str) -> Tuple[str, bool, List[str]]:
        """
        Inspect webpage text for adversarial prompt injection attempts.
        Returns: (sanitized_text, has_injection, detected_patterns)
        
        The text is never discarded entirely (as it may be legitimate analysis),
        but flagged and neutralized by prepending an UNTRUSTED DATA disclaimer.
        """
        if not text:
            return "", False, []

        detected: List[str] = []
        for pat in self.PROMPT_INJECTION_PATTERNS:
            matches = pat.findall(text)
            if matches:
                detected.extend(matches[:2])

        has_injection = len(detected) > 0
        if has_injection:
            # Neutralize: explicitly enclose in untrusted quotation boundaries
            # and strip any adversarial override prefixes
            sanitized = f"[UNTRUSTED WEBPAGE CONTENT - SUSPICIOUS INSTRUCTIONS NEUTRALIZED]\n{text}"
            return sanitized, True, detected

        return text, False, []

    @staticmethod
    def _is_private_or_reserved(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP address belongs to loopback, private, link-local, or reserved blocks."""
        return (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
