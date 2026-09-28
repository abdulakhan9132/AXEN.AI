from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from axen.tokenizer.tokenizer import Tokenizer

from .clean import normalize_text

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv"}


@dataclass
class DataRecord:
    source: str
    text: str


@dataclass
class IngestionReport:
    records: list[DataRecord]
    empty_files: list[str]
    malformed_files: dict[str, str]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _read_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]
    if path.suffix.lower() == ".json":
        return list(_strings(json.loads(path.read_text(encoding="utf-8"))))
    if path.suffix.lower() == ".jsonl":
        return [
            text
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for text in _strings(json.loads(line))
        ]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [" ".join(cell for cell in row if cell) for row in csv.reader(handle)]
    raise ValueError(f"Unsupported data type: {path.suffix}")


def load_records(raw_dir: str | Path = "data/raw") -> list[DataRecord]:
    return inspect_raw_data(raw_dir).records


def inspect_raw_data(raw_dir: str | Path = "data/raw") -> IngestionReport:
    root = Path(raw_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    records: list[DataRecord] = []
    empty_files: list[str] = []
    malformed_files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = str(path.relative_to(root))
            try:
                file_records = _read_file(path)
            except (OSError, UnicodeError, ValueError, csv.Error, json.JSONDecodeError) as exc:
                malformed_files[source] = f"{type(exc).__name__}: {exc}"
                continue
            if not file_records or not any(text.strip() for text in file_records):
                empty_files.append(source)
                continue
            records.extend(DataRecord(source, text) for text in file_records)
    return IngestionReport(records, empty_files, malformed_files)


def clean_records(records: Iterable[DataRecord]) -> tuple[list[DataRecord], int]:
    cleaned: list[DataRecord] = []
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        text = normalize_text(record.text)
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        cleaned.append(DataRecord(record.source, text))
    return cleaned, duplicates


def split_records(
    records: list[DataRecord],
    seed: int = 42,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
) -> dict[str, list[DataRecord]]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and leave room for the test split.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        train_end = 1 if total else 0
        validation_end = train_end
    else:
        train_end = max(1, int(total * train_ratio))
        validation_count = max(1, int(total * validation_ratio))
        test_count = total - train_end - validation_count
        if test_count < 1:
            train_end -= 1
            test_count = 1
        validation_end = train_end + validation_count
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def _write_tokens(path: Path, tokens: list[int]) -> None:
    path.write_text(json.dumps(tokens), encoding="utf-8")


def prepare_data(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ingestion = inspect_raw_data(raw_dir)
    records, duplicates = clean_records(ingestion.records)
    splits = split_records(records, seed, train_ratio, validation_ratio)
    tokenizer = Tokenizer().fit(record.text for record in splits["train"])
    split_tokens: dict[str, list[int]] = {}
    for name, split in splits.items():
        text = "\n\n".join(record.text for record in split)
        split_tokens[name] = tokenizer.encode(text, add_special_tokens=False)
        _write_tokens(output / f"{name}.json", split_tokens[name])
        (output / f"{name}.txt").write_text(text, encoding="utf-8")
    tokenizer.save(output / "tokenizer.json")

    split_token_counts = {name: len(values) for name, values in split_tokens.items()}
    lengths = [len(record.text) for record in records]
    split_warnings: list[str] = []
    if len(records) < 3:
        split_warnings.append("Fewer than 3 unique documents: validation/test coverage is not statistically meaningful.")
    if not splits["validation"]:
        split_warnings.append("Validation split is empty.")
    if not splits["test"]:
        split_warnings.append("Test split is empty.")
    source_paths = sorted({record.source for record in records})
    manifest = {
        "dataset_version": "2",
        "source_files": [
            {
                "name": source,
                "sha256": hashlib.sha256((Path(raw_dir) / source).read_bytes()).hexdigest(),
            }
            for source in source_paths
        ],
        "document_count": len(records),
        "character_count": sum(lengths),
        "estimated_token_count": sum(split_token_counts.values()),
        "token_count": sum(split_token_counts.values()),
        "vocabulary_size": tokenizer.vocab_size,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_token_counts": split_token_counts,
        "duplicate_count": duplicates,
        "empty_files": ingestion.empty_files,
        "malformed_files": ingestion.malformed_files,
        "average_document_length": (sum(lengths) / len(lengths)) if lengths else 0,
        "longest_document": max(lengths, default=0),
        "shortest_document": min(lengths, default=0),
        "split_warnings": split_warnings,
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def preview_data(raw_dir: str | Path = "data/raw", seed: int = 42) -> dict[str, Any]:
    ingestion = inspect_raw_data(raw_dir)
    raw_records = ingestion.records
    records, duplicates = clean_records(raw_records)
    splits = split_records(records, seed)
    files = sorted({record.source for record in raw_records})
    empty_files = [
        source for source in files
        if not any(record.source == source and record.text.strip() for record in raw_records)
    ]
    return {
        "files": files,
        "file_count": len(files),
        "file_types": sorted({Path(name).suffix.lower() for name in files}),
        "empty_files": sorted(set(empty_files) | set(ingestion.empty_files)),
        "malformed_files": ingestion.malformed_files,
        "duplicate_count": duplicates,
        "raw_records": len(raw_records),
        "clean_records": len(records),
        "characters": sum(len(record.text) for record in records),
        "estimated_tokens": sum(max(1, len(record.text) // 4) for record in records),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "preview": [asdict(record) for record in records[:3]],
    }
