"""
tests/test_neural_trainer_evolution.py
======================================
Tests for DatasetCurator and EvolutionNeuralTrainer safety gating.
"""

import tempfile
from pathlib import Path
import pytest

from axen.evolution.dataset_curator import DatasetCurator
from axen.evolution.neural_trainer import EvolutionNeuralTrainer


def test_dataset_curator_rejection_rules():
    curator = DatasetCurator(min_instruction_len=10)

    # 1. Unverified AI outputs rejected
    valid, reason = curator.validate_sample(
        instruction="What is photosynthesis in detail?",
        output="Photosynthesis is the process...",
        is_verified=False,
    )
    assert not valid
    assert "unverified" in reason.lower()

    # 2. Too short instruction rejected
    valid, reason = curator.validate_sample(
        instruction="Hi",
        output="Hello there!",
        is_verified=True,
    )
    assert not valid
    assert "short" in reason.lower()

    # 3. Verified valid sample accepted
    valid, reason = curator.validate_sample(
        instruction="Explain how binary search operates on arrays.",
        output="Binary search repeatedly divides the search interval in half.",
        is_verified=True,
    )
    assert valid
    assert reason is None


def test_dataset_curator_deduplication():
    curator = DatasetCurator()
    samples = [
        {"instruction": "Calculate 5 plus 5", "output": "10", "verified": True},
        {"instruction": "Calculate 5 plus 5", "output": "10", "verified": True},
    ]
    curated = curator.curate_samples(samples)
    assert len(curated) == 1  # Duplicate removed


def test_convert_tool_trace():
    curator = DatasetCurator()
    trace_sample = curator.convert_tool_trace(
        user_prompt="Add 123 and 456",
        tool_name="calculator",
        arguments={"expression": "123 + 456"},
        observation="579",
        final_answer="The answer is 579.",
        verifier_passed=True,
    )
    assert trace_sample is not None
    assert "calculator" in trace_sample.output
    assert trace_sample.source == "tool_trace_verified"


def test_evolution_neural_trainer_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = EvolutionNeuralTrainer(staging_dir=tmp_dir)
        train_samples = [
            {"instruction": "Explain Python GIL.", "output": "Global Interpreter Lock.", "verified": True}
        ]
        val_samples = [
            {"instruction": "Explain Python GIL.", "output": "Global Interpreter Lock.", "verified": True}
        ]

        result = trainer.train_candidate_weights(
            experiment_id="exp_neural_01",
            base_model=None,  # Headless test run
            train_samples=train_samples,
            val_samples=val_samples,
        )

        assert result.experiment_id == "exp_neural_01"
        assert result.is_accepted is False
        assert result.staged_checkpoint_path is None
        assert "No real base model" in result.error


def test_real_candidate_changes_weights_and_preserves_base(tmp_path):
    import torch
    from axen.model.foundation import AXENFoundationModel, FoundationModelConfig
    from axen.model.trainer import TrainingConfig
    from axen.evolution.neural_trainer import weight_hash
    torch.manual_seed(42)
    model = AXENFoundationModel(FoundationModelConfig(context_length=48, embedding_dim=8,
        num_layers=1, num_heads=2, ff_dim=16, dropout=0, device="cpu"))
    before = weight_hash(model)
    def sample(prompt, output):
        return {"instruction": prompt, "output": output, "verified": True,
                "source": "test_verified", "metadata": {"provenance": "arithmetic assertion in fixed test fixture"}}
    trainer = EvolutionNeuralTrainer(staging_dir=str(tmp_path))
    result = trainer.train_candidate_weights("real_candidate", model,
        [sample("Sum zero through one", "1" * 2)], [sample("Sum zero through two", "3" * 2)],
        held_out_samples=[sample("Sum zero through three", "6" * 2)], max_steps=2,
        training_config=TrainingConfig(learning_rate=0.001, batch_size=1))
    assert result.error is None, result.error
    assert result.metadata["weights_changed"]
    assert weight_hash(model) == before
    assert Path(result.staged_checkpoint_path).is_file()
    checkpoint = torch.load(result.staged_checkpoint_path, weights_only=True)
    assert checkpoint["step"] == 2
    assert result.metadata["held_out_candidate_loss"] > 0
    assert result.metadata["activation_status"] == "not_activated"


def test_candidate_rejects_overlapping_or_unprovenanced_data(tmp_path):
    from axen.model.foundation import AXENFoundationModel, FoundationModelConfig
    model = AXENFoundationModel(FoundationModelConfig(context_length=16, embedding_dim=8,
        num_layers=1, num_heads=2, ff_dim=16, device="cpu"))
    sample = {"instruction": "What is a range?", "output": "An interval", "verified": True,
              "source": "test", "metadata": {"provenance": "fixture"}}
    trainer = EvolutionNeuralTrainer(staging_dir=str(tmp_path))
    result = trainer.train_candidate_weights("overlap", model, [sample], [sample], held_out_samples=[sample])
    assert not result.is_accepted and "overlap" in result.error
    assert not list(tmp_path.rglob("*.pt"))
