import json
from pathlib import Path

from axen.data.clean import normalize_text
from axen.data.pipeline import prepare_data, preview_data


def test_normalization_preserves_paragraph_boundaries():
    assert normalize_text("  Café\r\n\r\n  next   paragraph  ") == "Café\n\nnext paragraph"


def test_preview_supports_local_file_formats(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.txt").write_text("alpha beta", encoding="utf-8")
    (raw / "notes.md").write_text("# heading\nalpha", encoding="utf-8")
    (raw / "facts.json").write_text(json.dumps({"text": "json value"}), encoding="utf-8")
    (raw / "facts.jsonl").write_text('{"text":"jsonl value"}\n', encoding="utf-8")
    (raw / "facts.csv").write_text("name,value\nrow,cell\n", encoding="utf-8")

    result = preview_data(raw)

    assert result["file_count"] == 5
    assert result["characters"] > 0
    assert result["split_counts"]["train"] > 0


def test_prepare_fits_tokenizer_on_training_and_writes_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    (raw / "a.txt").write_text("alpha\n\nbeta", encoding="utf-8")
    (raw / "b.txt").write_text("gamma\n\ndelta", encoding="utf-8")
    (raw / "duplicate.txt").write_text("alpha\n\nbeta", encoding="utf-8")

    manifest = prepare_data(raw, processed, seed=7)

    assert manifest["duplicate_count"] == 1
    assert (processed / "tokenizer.json").exists()
    assert (processed / "manifest.json").exists()
    assert (processed / "train.json").exists()
    assert "alpha" not in (processed / "manifest.json").read_text(encoding="utf-8")


def test_prepare_reports_empty_and_malformed_files(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    (raw / "valid.txt").write_text("first\n\nsecond", encoding="utf-8")
    (raw / "empty.txt").write_text("  \n\n", encoding="utf-8")
    (raw / "broken.json").write_text("{broken", encoding="utf-8")

    manifest = prepare_data(raw, processed, seed=11)

    assert manifest["empty_files"] == ["empty.txt"]
    assert "broken.json" in manifest["malformed_files"]
    assert manifest["split_token_counts"]["train"] > 0
    assert manifest["split_warnings"]

    preview = preview_data(raw, seed=11)
    assert preview["empty_files"] == ["empty.txt"]
    assert "broken.json" in preview["malformed_files"]
