"""Optional read-only official APIs. Secrets stay in Windows Credential Manager.

Consent and revocation are trusted application actions, never model text.
No browser, cookie, password-discovery, posting or write API exists here.
"""
from __future__ import annotations

import re
import json
import sys
import threading
import time

import requests


class WindowsCredentialVault:
    def __init__(self):
        if sys.platform != "win32": raise RuntimeError("Windows OS credential vault required")
        try:
            from keyring.backends.Windows import WinVaultKeyring
            self.backend = WinVaultKeyring()
        except ImportError as exc:
            raise RuntimeError("Install keyring; plaintext credential fallback is forbidden") from exc

    def put(self, service, token): self.backend.set_password("AXEN-Learning", service, token)
    def get(self, service): return self.backend.get_password("AXEN-Learning", service)
    def delete(self, service):
        if self.get(service): self.backend.delete_password("AXEN-Learning", service)


class AccountConnectorManager:
    supported = {"github": frozenset({"public_read"}), "youtube": frozenset({"metadata_read"})}

    def __init__(self, store, vault=None, transport=None):
        self.store, self.vault = store, vault
        self.transport = transport or requests.Session()
        self.transport.trust_env = False
        self.lock = threading.Lock(); self.last = {}; self.calls = {}

    def _vault(self):
        if self.vault is None: self.vault = WindowsCredentialVault()
        return self.vault

    def authorize(self, service, scopes, *, consent=False):
        if not consent: raise PermissionError("Explicit account authorization required")
        if service not in self.supported or not scopes or not set(scopes) <= self.supported[service]:
            raise PermissionError("Only supported read-only capabilities may be authorized")
        # Tokens are entered separately into Windows vault by the operator CLI,
        # never into dashboard, LLM context, events or SQLite.
        if not self._vault().get(service): raise ValueError("No operator-provisioned OS credential found")
        self.store.set_state("connector_" + service, {"authorized": True, "scopes": scopes})
        self.store.event("connector_authorized", {"service": service, "scopes": scopes})

    def revoke(self, service):
        if service not in self.supported: raise ValueError("Unsupported service")
        with self.lock:
            self.store.set_state("connector_" + service, {"authorized": False})
            self._vault().delete(service)
            self.store.event("connector_revoked", {"service": service})

    def status(self):
        return {service: self.store.state("connector_" + service, {"authorized": False})
                for service in self.supported}

    def get(self, service, resource, *, params=None):
        consent = self.store.state("connector_" + service, {})
        if not consent.get("authorized"): raise PermissionError("Connector not authorized or revoked")
        if service == "github":
            # Only the public repository metadata API is implemented; never
            # resolve arbitrary private files or operator-provided API URLs.
            if not re.fullmatch(r"repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", resource):
                raise PermissionError("Only repository metadata may be read")
            url = "https://api.github.com/" + resource
            if params: raise PermissionError("No arbitrary API parameters")
            headers = {"Accept": "application/vnd.github+json"}
        elif service == "youtube":
            if resource not in ("search", "videos"): raise PermissionError("Only official video metadata API may be read")
            allowed = {"part", "q", "type", "maxResults", "id"}
            if not params or not set(params) <= allowed: raise PermissionError("Unsupported YouTube parameters")
            url = "https://www.googleapis.com/youtube/v3/" + resource
            params = dict(params); headers = {}
        else: raise PermissionError("Unsupported connector")
        if not self._vault().get(service): raise PermissionError("Credential was removed")
        with self.lock:
            if not self.store.state("connector_" + service, {}).get("authorized"):
                raise PermissionError("Connector revoked before request")
            secret = self._vault().get(service)
            if not secret: raise PermissionError("Credential was removed")
            if service == "github": headers["Authorization"] = "Bearer " + secret
            else: params["key"] = secret
            count = self.calls.get(service, 0)
            if count >= 20: raise PermissionError("Service session request budget exhausted")
            if time.monotonic() - self.last.get(service, 0) < 2:
                raise PermissionError("Service rate limit; retry later")
            self.last[service] = time.monotonic(); self.calls[service] = count + 1
            # Holding lock serializes revocation-sensitive calls within manager.
            try:
                with self.transport.get(url, headers=headers, params=params, timeout=8,
                                        allow_redirects=False, stream=True) as response:
                    if response.status_code != 200: raise ValueError("Official API denied read access")
                    chunks = []; size = 0
                    for chunk in response.iter_content(8192):
                        size += len(chunk)
                        if size > 512_000: raise ValueError("API response budget exceeded")
                        chunks.append(chunk)
                    result = json.loads(b"".join(chunks))
            except requests.RequestException:
                raise ValueError("Official API request unavailable") from None
        if service == "github" and result.get("private") is not False:
            raise PermissionError("Private repositories are not authorized")
        self.store.event("connector_read", {"service": service, "resource": resource})
        return result


class YouTubeMetadataProvider:
    def __init__(self, connectors): self.connectors = connectors

    def search(self, question, limit=2):
        result = self.connectors.get("youtube", "search", params={"part": "snippet", "q": question[:200],
            "type": "video", "maxResults": min(2, limit)})
        return [{"id": item["id"]["videoId"], "title": item["snippet"]["title"],
                 "author": item["snippet"]["channelTitle"], "duration": None,
                 "url": "https://www.youtube.com/watch?v=" + item["id"]["videoId"],
                 "duration_limitation": "Search API does not return duration; videos API requires another budgeted request"}
                for item in result.get("items", [])[:limit]]

    def authorized_captions(self, video_id):
        raise PermissionError("Metadata key does not authorize caption downloads. Supply an explicitly licensed transcript.")
