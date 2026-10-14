"""Real bounded candidate training. Never mutates or activates the base model."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import copy
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any

import torch

from axen.model.trainer import FoundationTrainer, TrainingConfig
from .dataset_curator import DatasetCurator
from .models import ApprovalRequest, Experiment, ExperimentStatus


@dataclass
class TrainingExperimentResult:
    experiment_id: str
    baseline_loss: float | None = None
    candidate_loss: float | None = None
    improvement_delta: float = 0.0
    is_accepted: bool = False  # eligible for human review, never activated
    staged_checkpoint_path: str | None = None
    epochs_completed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EvolutionNeuralTrainer:
    def __init__(self, curator=None, store=None, staging_dir="checkpoints/evolution_staging"):
        self.curator = curator or DatasetCurator()
        self.store = store
        self.staging_dir = Path(staging_dir).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def train_candidate_weights(self, experiment_id, base_model, train_samples, val_samples,
                               training_config=None, min_improvement_threshold=0.001, *,
                               held_out_samples=None, replay_samples=None, max_steps=15,
                               cancel=lambda: False, event=lambda kind, data: None):
        result = TrainingExperimentResult(experiment_id)
        start = time.perf_counter()
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", experiment_id): raise ValueError("Invalid experiment ID")
            if base_model is None:
                raise ValueError("No real base model supplied; simulated training forbidden")
            if not held_out_samples:
                raise ValueError("Independent held-out records required")

            if not 1 <= max_steps <= 30: raise ValueError("Training step budget must be 1..30")
            if not math.isfinite(min_improvement_threshold) or min_improvement_threshold <= 0:
                raise ValueError("Positive finite improvement threshold required")
            splits = []
            hashes = []
            for samples in (list(train_samples) + list(replay_samples or []), val_samples, held_out_samples):
                # Per-split curation does not silently erase validation duplicates.
                curated = DatasetCurator().curate_samples(samples)
                if not curated: raise ValueError("Each split requires verified examples")
                if any(not s.source or s.source == "batch_curation" or not s.metadata.get("provenance") for s in curated):
                    raise ValueError("Every training record requires explicit source and provenance")
                splits.append([dict(s.to_record(), input=s.instruction, target=s.output) for s in curated])
                hashes.append({s.content_hash for s in curated})
            if any(hashes[a] & hashes[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Training, validation and held-out examples overlap")
            if sum(p.numel() for p in base_model.parameters()) * 32 > 512 * 1024**2:
                raise ValueError("Candidate model exceeds 512 MB training estimate")
            if base_model.config.context_length > 256: raise ValueError("Context length exceeds bounded experiment limit")
            config = replace(training_config or TrainingConfig(), batch_size=1, epochs=1,
                             checkpoint_dir=str(self.staging_dir / experiment_id), device="cpu")
            if not 0 < config.learning_rate <= 0.01: raise ValueError("Learning rate outside bounded range")
            original_hash = weight_hash(base_model)
            candidate = copy.deepcopy(base_model).to("cpu")
            candidate._device = torch.device("cpu")
            candidate.config = replace(candidate.config, device="cpu")
            before_hash = weight_hash(candidate)
            trainer = FoundationTrainer(candidate, splits[0], splits[1], config)
            holdout = FoundationTrainer(candidate, splits[0], splits[2], config)
            result.baseline_loss = trainer.evaluate()
            held_before = holdout.evaluate()
            if not all(math.isfinite(v) and v > 0 for v in (result.baseline_loss, held_before)):
                raise ValueError("Invalid baseline evaluation")
            steps = 0
            while steps < max_steps:
                for inputs, targets in trainer.train_loader:
                    if cancel(): raise ValueError("Training cancelled")
                    loss = trainer.train_step(inputs, targets); steps += 1
                    event("training_step", {"step": steps, "max_steps": max_steps,
                          "training_loss": loss, "gpu_usage_mb": 0, "device": "cpu"})
                    if steps >= max_steps: break
            if cancel(): raise ValueError("Training cancelled")
            result.candidate_loss = trainer.evaluate()
            held_after = holdout.evaluate()
            changed_hash = weight_hash(candidate)
            if changed_hash == before_hash: raise ValueError("Model parameters did not change")
            if weight_hash(base_model) != original_hash: raise ValueError("Base model unexpectedly mutated")
            if not all(math.isfinite(v) and v > 0 for v in (result.candidate_loss, held_after)):
                raise ValueError("Invalid candidate evaluation")
            path = self.staging_dir / experiment_id / "candidate_weights.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            provenance = {"split_hashes": [sorted(h) for h in hashes], "base_weight_hash": original_hash,
                          "candidate_weight_hash": changed_hash, "verified_sources":
                          [[s["source"] for s in split] for split in splits]}
            candidate.save_checkpoint(path, optimizer=trainer.optimizer, step=steps, metadata=provenance)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not all(torch.equal(t.cpu(), checkpoint["model_state"][n]) for n, t in candidate.state_dict().items()):
                raise ValueError("Checkpoint weights do not match trained candidate")
            result.staged_checkpoint_path = str(path)
            result.improvement_delta = result.baseline_loss - result.candidate_loss
            result.is_accepted = (result.improvement_delta >= min_improvement_threshold and held_after <= held_before)
            result.metadata = dict(provenance, steps_completed=steps, weights_changed=True,
                held_out_baseline_loss=held_before, held_out_candidate_loss=held_after,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), activation_status="not_activated",
                acceptance_meaning="eligible_for_human_review", replay_count=len(replay_samples or []))
            if self.store:
                exp = Experiment(experiment_id=experiment_id, title="Candidate neural training",
                                 metadata=result.metadata, status=ExperimentStatus.AWAITING_APPROVAL if result.is_accepted else ExperimentStatus.FAILED)
                self.store.save_experiment(exp)
                if result.is_accepted:
                    self.store.save_approval(ApprovalRequest(experiment_id=experiment_id,
                        request_type="weight_activation", title="Review staged neural checkpoint",
                        benchmark_summary={"baseline_loss": result.baseline_loss, "candidate_loss": result.candidate_loss,
                                           "held_out_baseline": held_before, "held_out_candidate": held_after},
                        metadata={"checkpoint_path": str(path), "checkpoint_sha256": result.metadata["checkpoint_sha256"]}))
            event("training_evaluation", {"baseline_loss": result.baseline_loss, "validation_loss": result.candidate_loss,
                  "held_out_before": held_before, "held_out_after": held_after, "eligible_for_review": result.is_accepted})
        except Exception as exc:
            result.error = str(exc); result.is_accepted = False
        result.duration_seconds = time.perf_counter() - start
        return result
