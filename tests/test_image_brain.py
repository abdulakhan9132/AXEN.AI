"""
tests/test_image_brain.py
=========================
Unit tests for AXEN Phase 11 Image Generation (ImageBrain):
- Pipeline is completely mocked to ensure fast, deterministic tests without GPU or network.
- Tests verify:
  1. generate() saves a valid PNG file and returns the expected dictionary shape.
  2. enhance_prompt() falls back gracefully to the original idea when Ollama is offline.
  3. CUDA Out of Memory (OOM) handling retries once at 384x384 resolution and unloads.
  4. reasoning/loop.py routes 'image_gen' requests to ImageBrain and logs to SQLite.
  5. axen_brain router classifies at least 20 sample image generation queries correctly.
  6. POST /api/image endpoint returns a valid URL and path.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from axen_brain import AxenBrain
from image_brain import ImageBrain, PRESETS
from reasoning.loop import AgentLoop


# ---------------------------------------------------------------------------
# Test 1: generate() saves PNG and returns {"prompt": str, "paths": [str]}
# ---------------------------------------------------------------------------
def test_01_generate_saves_png_and_returns_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = ImageBrain("turbo", out_dir=tmpdir)

        # Mock the diffusers pipeline
        mock_pipe = MagicMock()
        mock_img = Image.new("RGB", (512, 512), color=(255, 120, 60))
        mock_result = MagicMock()
        mock_result.images = [mock_img]
        mock_pipe.return_value = mock_result

        brain.pipe = mock_pipe

        # Run generate without enhance to test raw pipeline flow
        res = brain.generate("a serene sunset over mountain peak", enhance=False, seed=42, n=1)

        assert isinstance(res, dict)
        assert "prompt" in res
        assert "paths" in res
        assert res["prompt"] == "a serene sunset over mountain peak"
        assert len(res["paths"]) == 1

        saved_path = Path(res["paths"][0])
        assert saved_path.exists()
        assert saved_path.suffix == ".png"
        assert saved_path.stat().st_size > 0

        # Verify image dimensions
        with Image.open(saved_path) as loaded_img:
            assert loaded_img.size == (512, 512)


# ---------------------------------------------------------------------------
# Test 2: enhance_prompt falls back to original idea when Ollama is unreachable
# ---------------------------------------------------------------------------
def test_02_enhance_prompt_offline_fallback():
    brain = ImageBrain("turbo")

    # Mock requests.post to raise ConnectionError
    with patch("requests.post", side_effect=Exception("Connection refused")):
        expanded = brain.enhance_prompt("cyberpunk neon city")
        assert expanded == "cyberpunk neon city"


# ---------------------------------------------------------------------------
# Test 3: OOM handling retries once at 384x384 and unloads pipeline
# ---------------------------------------------------------------------------
def test_03_oom_handling_retries_at_384_and_unloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = ImageBrain("turbo", out_dir=tmpdir)

        call_count = 0
        mock_img = Image.new("RGB", (384, 384), color=(0, 255, 128))

        def side_effect_pipe(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("width") == 512:
                # Trigger CUDA Out of Memory simulation on first attempt
                raise RuntimeError("CUDA error: out of memory. Tried to allocate 1.20 GiB")
            # Retry at 384 succeeds
            res = MagicMock()
            res.images = [mock_img]
            return res

        mock_pipe = MagicMock(side_effect=side_effect_pipe)
        brain.pipe = mock_pipe

        # Mock _load so reload re-assigns the mock
        def mock_load():
            brain.pipe = mock_pipe
        brain._load = mock_load

        unload_spy = MagicMock(wraps=brain.unload)
        brain.unload = unload_spy

        res = brain.generate("space nebula in deep cosmos", enhance=False)

        assert call_count == 2
        assert unload_spy.call_count >= 1
        assert len(res["paths"]) == 1
        assert Path(res["paths"][0]).exists()


# ---------------------------------------------------------------------------
# Test 4: loop.py routes 'image_gen' to ImageBrain and logs to SQLite
# ---------------------------------------------------------------------------
def test_04_loop_routes_image_gen_and_logs(tmp_path):
    test_db = tmp_path / "test_image_routes.db"
    mock_brain = MagicMock()
    mock_brain.predict.return_value = {"intent": "image_gen", "confidence": 0.96}

    mock_image_tool = MagicMock(return_value={
        "status": "success",
        "tool": "image_brain",
        "prompt": "sunset mountains",
        "paths": [str(tmp_path / "sunset.png")],
        "url": "/images/sunset.png",
        "elapsed_sec": 1.25,
        "reply": "Done. Prompt used: sunset mountains\nSaved: sunset.png\nURL: /images/sunset.png"
    })

    loop = AgentLoop(
        brain=mock_brain,
        db_path=test_db,
        image_tool=mock_image_tool,
    )

    res = loop.run("generate a sunset over the mountains")

    assert res["intent"] == "image_gen"
    assert res["handler"] == "image_brain"
    mock_image_tool.assert_called_once_with("generate a sunset over the mountains")

    # Verify SQLite route_logs
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT intent, handler FROM route_logs WHERE intent='image_gen'")
    row = cursor.fetchone()
    assert row == ("image_gen", "image_brain")
    conn.close()


# ---------------------------------------------------------------------------
# Test 5: Router classifies at least 20 sample image requests as image_gen
# ---------------------------------------------------------------------------
def test_05_router_classifies_sample_image_requests():
    brain = AxenBrain("axen_brain.pt")
    assert brain.net is not None, "Trained router model axen_brain.pt not found"

    sample_queries = [
        "generate an image of a sunset over mountains",
        "draw a logo for my brand",
        "create a poster for a music festival",
        "make a picture of a cyberpunk city",
        "generate app icon art",
        "ek logo bana de mere channel ke liye",
        "generate pic of a red rose with water droplets",
        "make an illustration of a robot holding a flower",
        "create wallpaper for mobile phone with dark aesthetic",
        "render a 3d model of a sports car",
        "paint a watercolor landscape of a lake at dawn",
        "make an anime style character with silver hair",
        "generate a portrait of an ancient warrior",
        "create concept art of an alien planet with two moons",
        "draw a vintage travel poster for tokyo",
        "bhai ek sundar tasveer bana do pahadon ki",
        "isometric 3d illustration of a modern bedroom",
        "photorealistic portrait of an elderly watchmaker",
        "draw me an avatar of a cybernetic wolf",
        "generate picture of a lion wearing a golden crown",
        "generate an image of a cozy cabin in the woods",
        "draw a colorful graffiti logo of the word AXEN",
    ]

    assert len(sample_queries) >= 20

    correct = 0
    for q in sample_queries:
        pred = brain.predict(q)
        if pred["intent"] == "image_gen":
            correct += 1

    accuracy = correct / len(sample_queries)
    assert accuracy >= 0.90, f"Router classified {correct}/{len(sample_queries)} ({accuracy:.1%}) as image_gen"


# ---------------------------------------------------------------------------
# Test 6: POST /api/image endpoint returns valid URL
# ---------------------------------------------------------------------------
def test_06_fastapi_image_endpoint():
    from fastapi.testclient import TestClient
    from app.ui.app import app

    client = TestClient(app)

    mock_generate_result = {
        "prompt": "enhanced sunset prompt",
        "paths": ["outputs/images/sunset_12345_0.png"]
    }

    with patch("image_brain.ImageBrain.generate", return_value=mock_generate_result):
        resp = client.post("/api/image", json={"prompt": "sunset", "enhance": True, "preset": "turbo"})

        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data
        assert "prompt" in data
        assert data["url"] == "/images/sunset_12345_0.png"
        assert data["prompt"] == "enhanced sunset prompt"
