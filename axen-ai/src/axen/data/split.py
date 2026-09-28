from __future__ import annotations

from pathlib import Path


def split_text_file(source: str | Path, train_path: str | Path, val_path: str | Path, split: float = 0.9) -> dict[str, int]:
    source_path = Path(source)
    text = source_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Source file is empty: {source_path}")

    chunks = [segment for segment in text.split("\n\n") if segment.strip()]
    if len(chunks) < 2:
        raise ValueError("Need at least two non-empty text blocks to split safely.")

    split_index = max(1, int(len(chunks) * split))
    train_chunks = chunks[:split_index]
    val_chunks = chunks[split_index:]

    train_out = Path(train_path)
    val_out = Path(val_path)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_out.write_text("\n\n".join(train_chunks).strip() + "\n", encoding="utf-8")
    val_out.write_text("\n\n".join(val_chunks).strip() + "\n", encoding="utf-8")

    return {"train_tokens": sum(len(chunk.split()) for chunk in train_chunks), "val_tokens": sum(len(chunk.split()) for chunk in val_chunks)}
