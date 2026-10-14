"""Budgeted public-document research with explicit trust and access boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from html.parser import HTMLParser
import ipaddress
import re
import socket
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests

from .learning_store import now


class ResearchUnavailable(ValueError):
    pass


class _Text(HTMLParser):
    def __init__(self, section=None):
        super().__init__(); self.parts = []; self.hidden = 0
        self.section = section; self.started = not section

    def handle_starttag(self, tag, attrs):
        if self.section and dict(attrs).get("id") == self.section: self.started = True
        if tag in ("script", "style", "nav"): self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav") and self.hidden: self.hidden -= 1

    def handle_data(self, data):
        if self.started and not self.hidden: self.parts.append(data)


@dataclass
class ResearchDocument:
    url: str
    retrieved_at: str
    text: str
    content_hash: str
    access: str = "approved_public_document"
    verification_status: str = "untrusted_observation"

    def to_dict(self): return asdict(self)


class WebResearchAgent:
    user_agent = "AXEN-Learning/2.0"

    def __init__(self, domains, *, max_sources=5, interval_seconds=2,
                 max_requests=15, timeout=8, session=None, resolve=socket.getaddrinfo,
                 cancel=lambda: False):
        self.domains = frozenset(domains)
        self.max_sources, self.max_requests = max_sources, max_requests
        self.interval, self.timeout = max(1, interval_seconds), min(timeout, 10)
        self.session = session or requests.Session()
        # Never inherit browser cookies, netrc credentials or local proxy credentials.
        self.session.trust_env = False
        self.resolve, self.cancel = resolve, cancel
        self.requests = 0; self.sources = 0; self.last_request = 0
        self.robots = {}

    def validate_url(self, url):
        p = urlsplit(url)
        if (p.scheme != "https" or not p.hostname or p.hostname not in self.domains
                or p.username or p.password or p.port not in (None, 443)):
            raise ResearchUnavailable("Only HTTPS on exact approved public domains is permitted")
        if p.query:
            raise ResearchUnavailable("Research URLs must not contain query credentials or tracking parameters")
        try:
            addresses = self.resolve(p.hostname, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ResearchUnavailable("Source DNS unavailable") from exc
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ResearchUnavailable("Private, reserved or local network address refused")
        return p

    def _get(self, url):
        self.validate_url(url)
        if self.cancel(): raise ResearchUnavailable("Research cancelled")
        if self.requests >= self.max_requests:
            raise ResearchUnavailable("Network request budget exhausted")
        wait = self.interval - (time.monotonic() - self.last_request)
        while wait > 0:
            if self.cancel(): raise ResearchUnavailable("Research cancelled")
            time.sleep(min(wait, 0.1))
            wait = self.interval - (time.monotonic() - self.last_request)
        self.requests += 1; self.last_request = time.monotonic()
        try:
            with self.session.get(url, headers={"User-Agent": self.user_agent},
                                  timeout=self.timeout, allow_redirects=False, stream=True) as response:
                # Redirects require operator selection of the final URL. Never follow
                # an approved host redirect into a private or unauthorised target.
                if 300 <= response.status_code < 400:
                    raise ResearchUnavailable("Redirect refused; configure the canonical public URL")
                if response.status_code == 404 and url.endswith("/robots.txt"):
                    return "", "text/plain"
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";")[0]
                if content_type not in ("text/plain", "text/html", "application/xhtml+xml"):
                    raise ResearchUnavailable("Only text documentation is supported")
                chunks = []; size = 0
                for chunk in response.iter_content(8192):
                    if self.cancel(): raise ResearchUnavailable("Research cancelled")
                    size += len(chunk)
                    if size > 512_000: raise ResearchUnavailable("Document byte budget exceeded")
                    chunks.append(chunk)
                return b"".join(chunks).decode("utf-8", errors="replace"), content_type
        except requests.RequestException as exc:
            # Do not log request headers or credentials in exceptions.
            raise ResearchUnavailable("Source request failed: " + type(exc).__name__) from exc

    def retrieve(self, url):
        if self.sources >= self.max_sources: raise ResearchUnavailable("Research source budget exhausted")
        p = self.validate_url(url)
        origin = urlunsplit((p.scheme, p.netloc, "", "", ""))
        if origin not in self.robots:
            body, _ = self._get(origin + "/robots.txt")
            robot = RobotFileParser(); robot.parse(body.splitlines())
            self.robots[origin] = robot
        robot = self.robots[origin]
        if not robot.can_fetch(self.user_agent, url):
            raise ResearchUnavailable("robots.txt prohibits retrieval")
        delay = robot.crawl_delay(self.user_agent)
        rate = robot.request_rate(self.user_agent)
        if delay: self.interval = max(self.interval, delay)
        if rate: self.interval = max(self.interval, rate.seconds / rate.requests)
        self.sources += 1
        body, content_type = self._get(url)
        if content_type != "text/plain":
            parser = _Text(p.fragment); parser.feed(body); body = " ".join(parser.parts)
        # Store only a bounded excerpt relevant to the exercise, not full articles.
        match = re.search(r"(?:range\(|stop.*(?:excluded|exclusive)|(?:excluded|exclusive).*stop)", body, re.I)
        start = max(0, match.start() - 300) if match else 0
        excerpt = re.sub(r"\s+", " ", body[start:start + 3000])
        return ResearchDocument(url, now(), excerpt,
                                hashlib.sha256(body.encode()).hexdigest())

    def search(self, question, approved_catalog):
        """Search operator-curated source metadata, without scraping search engines."""
        words = set(re.findall(r"[a-z]{3,}", question.lower()))
        ranked = sorted(approved_catalog, key=lambda s: len(words & set(
            re.findall(r"[a-z]{3,}", (s.get("topic", "") + " " + s["url"]).lower()))), reverse=True)
        return ranked[:self.max_sources]


class VideoLearningAgent:
    """Official-API provider adapter and explicitly licensed transcript intake.

    No unofficial transcript scraping. A provider must enforce OAuth and official
    API quotas; an unavailable caption yields a limitation, never a learned skill.
    """
    def __init__(self, provider=None, max_videos=2):
        self.provider, self.max_videos = provider, max_videos

    def search(self, question):
        if self.provider is None:
            return {"videos": [], "limitation": "No authorized YouTube API provider connected"}
        videos = self.provider.search(question, limit=self.max_videos)
        return {"videos": videos[:self.max_videos]}

    def study(self, video, *, licensed_transcript=None, authorization=False):
        p = urlsplit(video.get("url", ""))
        if p.scheme != "https" or p.hostname not in ("www.youtube.com", "youtube.com", "youtu.be"):
            raise ResearchUnavailable("Invalid educational video URL")
        transcript = None
        if licensed_transcript is not None:
            if not authorization: raise PermissionError("Transcript use requires explicit authorization")
            transcript = licensed_transcript
        elif self.provider is not None:
            try: transcript = self.provider.authorized_captions(video["id"])
            except PermissionError: pass
        if not transcript:
            return {"video": video, "retrieved_at": now(), "verified": False,
                    "limitation": "No authorized transcript accessible; use approved documentation"}
        if len(transcript) > 100_000: raise ResearchUnavailable("Transcript budget exceeded")
        return {"video": video, "retrieved_at": now(), "text": transcript[:3000],
                "verified": False, "status": "untrusted_observation_requires_documentation_and_practice"}
