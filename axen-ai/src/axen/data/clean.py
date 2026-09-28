from __future__ import annotations

import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and cleaned_lines:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    cleaned = "\n".join(cleaned_lines)
    return cleaned.strip()


def clean_text_file(source: str | Path, output: str | Path | None = None) -> str:
    source_path = Path(source)
    raw = source_path.read_text(encoding="utf-8")
    cleaned = normalize_text(raw)
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
    return cleaned
