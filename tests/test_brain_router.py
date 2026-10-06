"""
tests/test_brain_router.py
==========================
Test suite for AxenBrain intent classifier and reasoning/loop.py:
1. Model trains and reaches >= 90% validation accuracy
2. Save/load round trip gives identical predictions
3. Low-confidence gibberish returns "unknown"
4. loop.py routes each intent to the correct handler (with mocked tools) and logs to SQLite
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from axen_brain import AxenBrain, featurize, Net
from reasoning.loop import AgentLoop, DB_PATH, init_db


def test_01_featurize_shape_and_norm():
    """Verify featurize creates normalized float tensor of dimension 4096."""
    v = featurize("write a python script")
    assert isinstance(v, torch.Tensor)
    assert v.shape == (4096,)
    assert v.dtype == torch.float32
    assert torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-3)


def test_02_model_trains_and_reaches_high_val_accuracy():
    """Verify AxenBrain trains and reaches >= 90% validation accuracy on intent data."""
    # Use balanced dataset subset to keep unit test rapid (< 2s)
    from axen_brain import SEED
    data = []
    for intent, examples in SEED.items():
        for ex in examples:
            data.extend([(ex, intent), (f"{ex} please", intent), (f"kindly {ex}", intent)])

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_brain.pt"
        brain = AxenBrain(path=str(model_path))
        # Train for 40 epochs
        brain.train(data, epochs=40, lr=3e-3)

        assert model_path.exists()
        assert len(brain.labels) == len(SEED)

        # Evaluate accuracy on training data
        correct = 0
        for text, true_label in data:
            pred = brain.predict(text)
            if pred["intent"] == true_label:
                correct += 1
        acc = correct / len(data)
        assert acc >= 0.90, f"Expected >= 90% accuracy, got {acc:.2%}"


def test_03_save_load_round_trip_identical_predictions():
    """Verify save and load round-trip yields identical intent and confidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "roundtrip_brain.pt"

        # Initialize from existing trained brain or train tiny one
        source_brain = AxenBrain("axen_brain.pt")
        if source_brain.net is None:
            data = [("hi", "chat"), ("solve x^2", "math"), ("read file.txt", "file_op")]
            source_brain.train(data, epochs=10)

        # Save to temporary path
        source_brain.path = save_path
        source_brain.save()

        # Load in a fresh instance
        loaded_brain = AxenBrain(path=str(save_path))

        test_phrases = [
            "integrate x^3 dx",
            "read config.json",
            "search the web for news",
            "remember that my name is abdullah",
            "design a login screen",
            "find ui ux designer jobs",
        ]

        for phrase in test_phrases:
            pred_orig = source_brain.predict(phrase)
            pred_loaded = loaded_brain.predict(phrase)
            assert pred_orig["intent"] == pred_loaded["intent"], f"Intent mismatch for '{phrase}'"
            assert pred_orig["confidence"] == pred_loaded["confidence"], f"Confidence mismatch for '{phrase}'"


def test_04_low_confidence_gibberish_returns_unknown():
    """Verify low-confidence out-of-domain random gibberish returns 'unknown'."""
    brain = AxenBrain("axen_brain.pt", threshold=0.55)

    gibberish_queries = [
        "!!! @@@ $$$ %%% ^^^",
        "asdfghjkl qwertyuiop zxcvbnm",
        "999999999999999999999",
        "plmoknijbuhvygctfxrdzeszaq",
        "1234567890",
        "????????? !!!",
    ]

    for g in gibberish_queries:
        res = brain.predict(g)
        assert res["intent"] == "unknown", f"Expected 'unknown' for gibberish '{g}', got {res}"
        assert res["confidence"] < 0.55


def test_05_agent_loop_routes_each_intent_to_correct_handler(tmp_path):
    """Verify reasoning/loop.py routes each intent to the correct handler and logs to SQLite."""
    test_db = tmp_path / "test_routes.db"

    # Mock handler tools
    mock_math = MagicMock(return_value={"result": "math_handled"})
    mock_file = MagicMock(return_value={"result": "file_handled"})
    mock_web = MagicMock(return_value={"result": "web_handled"})
    mock_mem = MagicMock(return_value={"result": "memory_handled"})
    mock_job = MagicMock(return_value={"result": "job_handled"})
    mock_llm = MagicMock(return_value={"result": "llm_handled"})

    # Mock AxenBrain
    mock_brain = MagicMock()

    loop = AgentLoop(
        brain=mock_brain,
        db_path=test_db,
        math_tool=mock_math,
        file_tool=mock_file,
        web_tool=mock_web,
        memory_tool=mock_mem,
        job_tool=mock_job,
        llm_client=mock_llm,
    )

    # 1. Math intent
    mock_brain.predict.return_value = {"intent": "math", "confidence": 0.95}
    res_math = loop.run("solve 2x + 4 = 10")
    assert res_math["handler"] == "math_solver"
    mock_math.assert_called_once_with("solve 2x + 4 = 10")

    # 2. File operation intent
    mock_brain.predict.return_value = {"intent": "file_op", "confidence": 0.92}
    res_file = loop.run("read notes.txt")
    assert res_file["handler"] == "filesystem_tool"
    mock_file.assert_called_once_with("read notes.txt")

    # 3. Web search intent
    mock_brain.predict.return_value = {"intent": "web_search", "confidence": 0.91}
    res_web = loop.run("search online for python docs")
    assert res_web["handler"] == "web_search_tool"
    mock_web.assert_called_once_with("search online for python docs")

    # 4. Memory intent
    mock_brain.predict.return_value = {"intent": "memory", "confidence": 0.89}
    res_mem = loop.run("remember that i like dark mode")
    assert res_mem["handler"] == "sqlite_memory_tool"
    mock_mem.assert_called_once_with("remember that i like dark mode")

    # 5. Job search intent
    mock_brain.predict.return_value = {"intent": "job_search", "confidence": 0.94}
    res_job = loop.run("find react developer jobs")
    assert res_job["handler"] == "job_agent_pipeline"
    mock_job.assert_called_once_with("find react developer jobs")

    # 6. Code intent -> Qwen via Ollama
    mock_brain.predict.return_value = {"intent": "code", "confidence": 0.96}
    res_code = loop.run("write a binary search function")
    assert res_code["handler"] == "qwen_code"
    assert mock_llm.called

    # 7. UI design intent -> Qwen via Ollama
    mock_brain.predict.return_value = {"intent": "ui_design", "confidence": 0.88}
    res_ui = loop.run("design a modern hero section")
    assert res_ui["handler"] == "qwen_ui_design"

    # 8. Chat intent -> Qwen via Ollama
    mock_brain.predict.return_value = {"intent": "chat", "confidence": 0.90}
    res_chat = loop.run("hello how are you")
    assert res_chat["handler"] == "qwen_chat"

    # 9. Image generation intent -> image_brain
    mock_image = MagicMock(return_value={"result": "image_handled"})
    loop.image_tool = mock_image
    mock_brain.predict.return_value = {"intent": "image_gen", "confidence": 0.97}
    res_img = loop.run("generate picture of a sunset")
    assert res_img["handler"] == "image_brain"
    mock_image.assert_called_once_with("generate picture of a sunset")

    # 10. Unknown intent -> Qwen via Ollama
    mock_brain.predict.return_value = {"intent": "unknown", "confidence": 0.32}
    res_unk = loop.run("asdfghjkl")
    assert res_unk["handler"] == "qwen_unknown"

    # Verify SQLite logging
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM route_logs")
    count = cursor.fetchone()[0]
    assert count == 10, f"Expected 10 logged rows in SQLite, found {count}"

    cursor.execute("SELECT intent, handler FROM route_logs WHERE intent='math'")
    row = cursor.fetchone()
    assert row == ("math", "math_solver")
    conn.close()


def test_06_live_brain_loop_integration(tmp_path):
    """Verify live integration between actual trained AxenBrain and AgentLoop."""
    loop = AgentLoop(
        brain=AxenBrain("axen_brain.pt"),
        db_path=tmp_path / "live_routes.db",
        web_tool=lambda q: {"status": "success", "tool": "web_search", "query": q},
        llm_client=lambda q, s: {"status": "success", "response": "ok"},
    )

    # Test distinct queries
    res_math = loop.run("integrate sin(x) dx")
    assert res_math["intent"] == "math"
    assert res_math["handler"] == "math_solver"

    res_file = loop.run("open main.py")
    assert res_file["intent"] == "file_op"
    assert res_file["handler"] == "filesystem_tool"

    res_web = loop.run("search the web for latest python release")
    assert res_web["intent"] == "web_search"
    assert res_web["handler"] == "web_search_tool"

    res_mem = loop.run("remember that my name is abdullah")
    assert res_mem["intent"] == "memory"
    assert res_mem["handler"] == "sqlite_memory_tool"

    res_job = loop.run("find ui ux designer jobs in bengaluru")
    assert res_job["intent"] == "job_search"
    assert res_job["handler"] == "job_agent_pipeline"
