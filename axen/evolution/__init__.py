"""
axen/evolution/__init__.py
==========================
AXEN Self-Evolution Engine v1.

Provides controlled self-improvement inspired by Darwin–Gödel Machine,
HyperAgents, and AlphaEvolve.  All changes require human approval.

Public surface:
    from axen.evolution import (
        EvolutionController, EvolutionStore,
        Experiment, Candidate, BenchmarkResult, EvolutionRecord, ApprovalRequest,
        TaskAgent, MetaAgent,
        EvolutionaryCodeSearch, CandidateEvaluator, BenchmarkSuite,
        EvolutionNeuralTrainer, DatasetCurator,
        SandboxExecutor, SandboxPolicy, ApprovalGate,
    )
"""

from .approval import ApprovalGate
from .benchmark_suite import BenchmarkSuite, TestCase
from .candidate_generator import CandidateGenerator
from .candidate_selector import CandidateSelector
from .code_search import EvolutionaryCodeSearch
from .controller import EvolutionController
from .dataset_curator import CuratedSample, DatasetCurator
from .evaluator import CandidateEvaluator
from .history import EvolutionHistory
from .meta_agent import MetaAgent, StrategyProposal
from .models import (
    ApprovalRequest,
    ApprovalStatus,
    BenchmarkResult,
    Candidate,
    CandidateStatus,
    EvolutionRecord,
    Experiment,
    ExperimentStatus,
)
# Candidate training is optional and expensive on an 8 GB laptop. Importing the
# learning API must not load PyTorch, CUDA or the neural trainer into memory.
from .permissions import AgentRole, EvolutionPermission, EvolutionSecurityPolicy
from .sandbox import SandboxExecutionResult, SandboxExecutor
from .sandbox_policy import SandboxPolicy
from .store import EvolutionStore
from .task_agent import TaskAgent, TaskExecutionResult
from .weakness_detector import Weakness, WeaknessDetector

__version__ = "1.0.0"


def __getattr__(name):
    if name in ("EvolutionNeuralTrainer", "TrainingExperimentResult"):
        from . import neural_trainer
        return getattr(neural_trainer, name)
    raise AttributeError(name)

__all__ = [
    # Models
    "ExperimentStatus",
    "CandidateStatus",
    "ApprovalStatus",
    "Experiment",
    "Candidate",
    "BenchmarkResult",
    "EvolutionRecord",
    "ApprovalRequest",
    # Store
    "EvolutionStore",
    # History
    "EvolutionHistory",
    # Permissions
    "AgentRole",
    "EvolutionPermission",
    "EvolutionSecurityPolicy",
    # Agents
    "TaskAgent",
    "TaskExecutionResult",
    "MetaAgent",
    "StrategyProposal",
    # Search & Benchmarking
    "TestCase",
    "BenchmarkSuite",
    "CandidateEvaluator",
    "CandidateGenerator",
    "CandidateSelector",
    "EvolutionaryCodeSearch",
    "Weakness",
    "WeaknessDetector",
    # Neural Training
    "CuratedSample",
    "DatasetCurator",
    "EvolutionNeuralTrainer",
    "TrainingExperimentResult",
    # Sandbox & Security
    "SandboxPolicy",
    "SandboxExecutor",
    "SandboxExecutionResult",
    "ApprovalGate",
    # Controller
    "EvolutionController",
]
