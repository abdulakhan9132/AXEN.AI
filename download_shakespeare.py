#!/usr/bin/env python3
r"""Download a public-domain complete-works Shakespeare corpus for AEXN GPT.

Source: Project Gutenberg eBook #100 (The Complete Works of William Shakespeare)
https://www.gutenberg.org/ebooks/100

Run from your AEXN AI project root:
    python download_shakespeare.py
    # or on Windows:
    .venv\Scripts\python.exe download_shakespeare.py

Outputs (default: data/shakespeare/):
  shakespeare_complete_raw.txt     Original Project Gutenberg text
  shakespeare_training.txt         Text between Gutenberg start/end markers
  shakespeare_corpus_info.json     File sizes, line counts, source and SHA-256

This script uses only Python's standard library and does not start model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCE_PAGE = "https://www.gutenberg.org/ebooks/100"
TEXT_URLS = (
    "https://www.gutenberg.org/ebooks/100.txt.utf-8",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
)
START = re.compile(r"(?im)^\*{3}\s*START OF .*PROJECT GUTENBERG EBOOK.*\*{3}\s*$")
END = re.compile(r"(?im)^\*{3}\s*END OF .*PROJECT GUTENBERG EBOOK.*\*{3}\s*$")


def download_complete_works() -> tuple[str, bytes]:
    failures = []
    for url in TEXT_URLS:
        for attempt in range(2):
            try:
                req = Request(url, headers={"User-Agent": "AEXN-education-corpus/1.0 (Python urllib)", "Accept": "text/plain"})
                with urlopen(req, timeout=90) as response:
                    raw = response.read()
                if len(raw) < 1_000_000 or raw.lstrip().startswith((b"<!DOCTYPE", b"<html")):
                    raise ValueError("Response is unexpectedly short or is HTML, not the complete text")
                return url, raw
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                failures.append(f"{url}: {exc}")
                if attempt == 0:
                    time.sleep(2)
    raise RuntimeError("Could not download Gutenberg #100. Check internet access.\n" + "\n".join(failures))


def remove_gutenberg_boilerplate(text: str) -> str:
    """Return source text between START/END markers, preserving all internal lines."""
    begin = START.search(text)
    if not begin:
        raise ValueError("Couldn't find Gutenberg START marker; refusing to label the file as cleaned")
    ending = END.search(text, begin.end())
    if not ending:
        raise ValueError("Couldn't find Gutenberg END marker; refusing to label the file as cleaned")
    body = text[begin.end():ending.start()].strip("\r\n")
    if len(body) < 1_000_000:
        raise ValueError("Extracted text is too short for the complete works")
    return body + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Shakespeare complete works for AEXN model training")
    parser.add_argument("--output-dir", default="data/shakespeare", help="Output folder (default: data/shakespeare)")
    args = parser.parse_args()
    out = Path(args.output_dir)
    print("Downloading the complete works from Project Gutenberg #100...")
    try:
        url, raw = download_complete_works()
        raw_text = raw.decode("utf-8-sig")
        cleaned = remove_gutenberg_boilerplate(raw_text)
    except (RuntimeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    original = out / "shakespeare_complete_raw.txt"
    training = out / "shakespeare_training.txt"
    info_path = out / "shakespeare_corpus_info.json"
    original.write_bytes(raw)
    training.write_text(cleaned, encoding="utf-8", newline="\n")
    details = {
        "title": "The Complete Works of William Shakespeare",
        "source_catalog": SOURCE_PAGE,
        "source_text": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_notice": "Project Gutenberg lists eBook #100 as public domain in the USA; verify laws where you live.",
        "raw_bytes": len(raw),
        "raw_lines": len(raw_text.splitlines()),
        "cleaned_bytes": len(cleaned.encode("utf-8")),
        "cleaned_lines": len(cleaned.splitlines()),
        "cleaned_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        "count_note": "Line count varies with source edition, blank lines, and formatting; not guaranteed to be 167,205.",
    }
    info_path.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Downloaded from: {url}")
    print(f"Original: {original.resolve()} ({details['raw_lines']:,} lines)")
    print(f"Training text: {training.resolve()} ({details['cleaned_lines']:,} lines)")
    print(f"Metadata: {info_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
