"""Automated tests for AXEN Milestone 1:
Environment, Hardware Detection, Config Presets, Polyglot Structure, and Makefile.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from scripts.hardware_check import (
    HardwareProfile,
    detect_hardware,
    evaluate_capabilities,
    format_text_report,
)


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_hardware_detection():
    """Verify hardware inspection retrieves realistic values."""
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.physical_cores >= 1
    assert hw.logical_cores >= hw.physical_cores
    assert hw.total_ram_gb > 0
    assert hw.available_ram_gb >= 0
    assert isinstance(hw.gpu_available, bool)
    if hw.gpu_available:
        assert hw.gpu_name is not None
        assert hw.vram_gb is not None and hw.vram_gb > 0


def test_hardware_evaluation_recommendations():
    """Test recommendation logic under both GPU and CPU simulated profiles."""
    # Test CPU-only scenario
    cpu_profile = HardwareProfile(
        cpu_model="Simulated CPU",
        physical_cores=4,
        logical_cores=8,
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        gpu_available=False,
    )
    cpu_analysis = evaluate_capabilities(cpu_profile)
    assert "tiny" in cpu_analysis["training_recommendation"]["preset"].lower()
    assert "Qwen" in cpu_analysis["inference_recommendation"]["model"]

    # Test 4GB VRAM GPU scenario (like RTX 3050 Laptop)
    gpu_profile_4gb = HardwareProfile(
        cpu_model="Simulated CPU",
        physical_cores=6,
        logical_cores=12,
        total_ram_gb=8.0,
        available_ram_gb=1.0,
        gpu_available=True,
        gpu_name="NVIDIA GeForce RTX 3050 Laptop GPU",
        vram_gb=4.0,
        cuda_version="12.4",
        bf16_supported=True,
        fp16_supported=True,
    )
    gpu_analysis = evaluate_capabilities(gpu_profile_4gb)
    assert "tiny" in gpu_analysis["training_recommendation"]["preset"].lower()
    assert gpu_analysis["ram_warning"] is not None  # Should warn about low RAM (<1.5 GB)

    report_text = format_text_report(gpu_analysis)
    assert "AXEN SYSTEM & HARDWARE REPORT" in report_text
    assert "RTX 3050" in report_text


def test_configs_exist_and_valid_yaml():
    """Verify configuration YAML files exist and have required schema."""
    model_cfg_path = ROOT_DIR / "configs" / "model.yaml"
    presets_cfg_path = ROOT_DIR / "configs" / "training_presets.yaml"

    assert model_cfg_path.exists(), "configs/model.yaml does not exist"
    assert presets_cfg_path.exists(), "configs/training_presets.yaml does not exist"

    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    assert "llm" in model_cfg
    assert "default_model" in model_cfg["llm"]
    assert "agent" in model_cfg
    assert model_cfg["agent"]["max_steps"] == 8

    with open(presets_cfg_path, "r", encoding="utf-8") as f:
        presets_cfg = yaml.safe_load(f)
    assert "presets" in presets_cfg
    for name in ("tiny", "small", "medium"):
        assert name in presets_cfg["presets"], f"Missing preset: {name}"
        preset = presets_cfg["presets"][name]
        assert "architecture" in preset
        assert "training" in preset
        assert preset["architecture"]["vocab_size"] > 0
        assert preset["architecture"]["hidden_dim"] > 0
        assert preset["architecture"]["num_layers"] > 0


def test_training_presets_parameter_formula():
    """Verify exact theoretical parameter calculation matches the declared parameter counts."""
    presets_cfg_path = ROOT_DIR / "configs" / "training_presets.yaml"
    with open(presets_cfg_path, "r", encoding="utf-8") as f:
        presets_cfg = yaml.safe_load(f)

    for name, p in presets_cfg["presets"].items():
        arch = p["architecture"]
        v = arch["vocab_size"]
        d = arch["hidden_dim"]
        layers = arch["num_layers"]
        d_ff = arch["intermediate_dim"]

        # Decoder-only transformer with SwiGLU and tied embeddings:
        # Embedding: v * d
        # Per layer attention: 4 * d^2 (Q, K, V, O projections)
        # Per layer MLP: 3 * d * d_ff (Gate, Up, Down projections in SwiGLU)
        # Per layer RMSNorm: 2 * d
        # Final RMSNorm: d
        calculated = (v * d) + layers * (4 * (d**2) + 3 * d * d_ff + 2 * d) + d
        declared = p["params_millions"] * 1e6
        # Check within 2% margin
        ratio = calculated / declared
        assert 0.98 <= ratio <= 1.02, f"Preset {name} param mismatch: calc={calculated}, decl={declared}"


def test_polyglot_and_makefile():
    """Verify polyglot folder and Makefile are present and well-structured."""
    polyglot_readme = ROOT_DIR / "polyglot" / "README.md"
    makefile = ROOT_DIR / "Makefile"

    assert polyglot_readme.exists(), "polyglot/README.md missing"
    assert makefile.exists(), "Makefile missing"

    content = makefile.read_text(encoding="utf-8")
    for target in ("setup:", "check-hardware:", "test:", "run:"):
        assert target in content, f"Makefile missing target: {target}"
