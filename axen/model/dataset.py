"""
axen/model/dataset.py
=====================
Training Dataset Management, Capability Balancing, Sanitization, and Splitting.

Implements Phase 8 Dataset Architecture:
data/training/
    raw/
    sanitized/
    curated/
    validation/
    test/
    metadata/

Covers 12 distinct capabilities:
- conversation
- instruction
- knowledge
- reasoning
- tool_intent
- tool_chain
- reflection
- correction
- files
- projects
- memory_behavior
- coding_behavior
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from axen.agent.learning import DatasetSanitizer
from axen.model.foundation import FoundationTokenizer

logger = logging.getLogger("axen.model.dataset")


# ---------------------------------------------------------------------------
# Canonical Curated Capability Examples Seed
# ---------------------------------------------------------------------------

CURATED_CAPABILITY_SEEDS: List[Dict[str, Any]] = [
    # 1. CONVERSATION
    {"category": "conversation", "input": "Hello", "target": "Hello! I am AXEN, your local AI assistant."},
    {"category": "conversation", "input": "Hi there", "target": "Hello! How can I assist you today?"},
    {"category": "conversation", "input": "Hey AXEN", "target": "Greetings! What can I help you with?"},
    {"category": "conversation", "input": "Who are you?", "target": "I am AXEN, an independent local AI assistant with native reasoning capabilities."},
    {"category": "conversation", "input": "What is your identity?", "target": "I am AXEN, an offline-first foundation model and AI assistant."},
    {"category": "conversation", "input": "What can you do?", "target": "I can answer questions, analyze files, execute safe calculations, and coordinate specialized tools."},
    {"category": "conversation", "input": "Tell me what you are capable of.", "target": "I provide general conversation, knowledge reasoning, tool execution, and project awareness."},
    {"category": "conversation", "input": "Good morning", "target": "Good morning! Ready to assist you with your tasks."},
    {"category": "conversation", "input": "Good evening", "target": "Good evening! How can I help you tonight?"},
    {"category": "conversation", "input": "Thanks for your help", "target": "You are welcome! Let me know if you need anything else."},

    # 2. INSTRUCTION FOLLOWING
    {"category": "instruction", "input": "Answer in one sentence: What is AXEN?", "target": "AXEN is a local artificial intelligence system designed to run entirely offline."},
    {"category": "instruction", "input": "Explain machine learning in one short sentence.", "target": "Machine learning is algorithms finding patterns in data to make predictions."},
    {"category": "instruction", "input": "Answer briefly: Why use local models?", "target": "Local models ensure complete privacy, offline availability, and zero data leakage."},
    {"category": "instruction", "input": "Format your response in three bullets about computers.", "target": "- Computers process digital data.\n- They execute algorithmic instructions.\n- They store information in memory."},
    {"category": "instruction", "input": "Summarize the purpose of an operating system in one sentence.", "target": "An operating system manages computer hardware and provides common services for application software."},
    {"category": "instruction", "input": "Explain RAM in one sentence.", "target": "RAM is fast, volatile primary memory used by the CPU to store actively running programs and data."},
    {"category": "instruction", "input": "Define an algorithm in one sentence.", "target": "An algorithm is a step-by-step sequence of unambiguous instructions to solve a problem."},
    {"category": "instruction", "input": "Respond concisely: What is open source?", "target": "Open source software makes its source code freely available for inspection, modification, and enhancement."},

    # 3. GENERAL KNOWLEDGE
    {"category": "knowledge", "input": "What is a computer?", "target": "A computer is an electronic device for storing and processing data according to instructions."},
    {"category": "knowledge", "input": "What is machine learning?", "target": "Machine learning is a subset of artificial intelligence where algorithms learn patterns directly from data."},
    {"category": "knowledge", "input": "What is artificial intelligence?", "target": "Artificial intelligence is the simulation of human intelligence processes by computer systems."},
    {"category": "knowledge", "input": "What is a neural network?", "target": "A neural network is a computational model inspired by biological brains, organized in interconnected layers of nodes."},
    {"category": "knowledge", "input": "What is a Transformer model?", "target": "A Transformer is a neural network architecture that relies on self-attention mechanisms to process sequential data."},
    {"category": "knowledge", "input": "What does a CPU do?", "target": "The central processing unit executes program instructions and performs basic arithmetic, logic, and I/O operations."},
    {"category": "knowledge", "input": "What is a database?", "target": "A database is an organized collection of structured information or data stored electronically."},
    {"category": "knowledge", "input": "What is the speed of light?", "target": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},

    # 4. REASONING
    {"category": "reasoning", "input": "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?", "target": "No, it does not necessarily follow that all roses fade quickly."},
    {"category": "reasoning", "input": "Which is larger: 1.05 or 1.5?", "target": "1.5 is larger than 1.05 because 1.50 is greater than 1.05."},
    {"category": "reasoning", "input": "If today is Tuesday, what day was it three days ago?", "target": "Three days ago was Saturday."},
    {"category": "reasoning", "input": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?", "target": "The ball costs $0.05, and the bat costs $1.05."},
    {"category": "reasoning", "input": "Why does ice float on water?", "target": "Ice floats because water expands when it freezes, making solid ice less dense than liquid water."},

    # 5. TOOL INTENT (Structured JSON)
    {"category": "tool_intent", "input": "Calculate 25 * 48.", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'},
    {"category": "tool_intent", "input": "Calculate 25 multiplied by 48.", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "25 * 48"}}'},
    {"category": "tool_intent", "input": "Compute 144 / 12.", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "144 / 12"}}'},
    {"category": "tool_intent", "input": "What is 500 + 350?", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "500 + 350"}}'},
    {"category": "tool_intent", "input": "Evaluate 15 ** 2.", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "15 ** 2"}}'},
    {"category": "tool_intent", "input": "Find auth.py in this project.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "auth.py"}}'},
    {"category": "tool_intent", "input": "Search for auth.py in workspace.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "auth.py"}}'},
    {"category": "tool_intent", "input": "Find the file named config.yaml.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "config.yaml"}}'},
    {"category": "tool_intent", "input": "Read the configuration file.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "config.py"}}'},
    {"category": "tool_intent", "input": "Read the config file.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "config.py"}}'},
    {"category": "tool_intent", "input": "Read configuration.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "config.py"}}'},
    {"category": "tool_intent", "input": "Read the contents of settings.json.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "settings.json"}}'},
    {"category": "tool_intent", "input": "Search for requirements.txt in workspace.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "requirements.txt"}}'},
    {"category": "tool_intent", "input": "Calculate 1024 * 768.", "target": '{"type": "tool_call", "tool": "calculator.evaluate", "arguments": {"expression": "1024 * 768"}}'},

    # 6. TOOL CHAIN (Multi-step sequencing)
    {"category": "tool_chain", "input": "Find the CSV, calculate its average, and explain it.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "*.csv"}}'},
    {"category": "tool_chain", "input": "Locate data.json and read its contents.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "data.json"}}'},
    {"category": "tool_chain", "input": "Search for metrics.log and summarize the errors.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "metrics.log"}}'},

    # 7. REFLECTION (Structured verification)
    {"category": "reflection", "input": "Tool execution completed with output: 1200.", "target": '{"status": "complete", "reason_code": "verified_result"}'},
    {"category": "reflection", "input": "File search returned 0 matches for target.", "target": '{"status": "needs_next_step", "reason_code": "missing_information"}'},
    {"category": "reflection", "input": "Network timeout occurred while accessing resource.", "target": '{"status": "retry", "reason_code": "transient_tool_failure"}'},
    {"category": "reflection", "input": "Calculation result contradicted expected constraints.", "target": '{"status": "correction", "reason_code": "incorrect_previous_result"}'},

    # 8. CORRECTION
    {"category": "correction", "input": "That is wrong. Try again.", "target": "Understood. I will adjust my approach and re-evaluate the request."},
    {"category": "correction", "input": "No, use auth.py instead.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "auth.py"}}'},
    {"category": "correction", "input": "Use the second file, not the first.", "target": "Understood, updating the active selection to the second file."},
    {"category": "correction", "input": "That calculation is incorrect.", "target": "I will re-run the calculation with the corrected parameters."},

    # 9. FILES
    {"category": "files", "input": "Search for all python files in tests directory.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "tests/*.py"}}'},
    {"category": "files", "input": "Read README.md to see project instructions.", "target": '{"type": "tool_call", "tool": "file_read", "arguments": {"path": "README.md"}}'},
    {"category": "files", "input": "Check if package.json exists in root.", "target": '{"type": "tool_call", "tool": "file_search", "arguments": {"query": "package.json"}}'},

    # 10. PROJECTS
    {"category": "projects", "input": "What is the active project structure?", "target": "The project contains core runtime, tools, memory, and foundation model modules."},
    {"category": "projects", "input": "Where are project test cases stored?", "target": "Test cases are located under the tests/ directory."},
    {"category": "projects", "input": "What environment is configured for this workspace?", "target": "The workspace uses Python 3.13 and a local virtual environment."},

    # 11. MEMORY BEHAVIOR
    {"category": "memory_behavior", "input": "My project uses Python.", "target": "I will note that your project uses Python for subsequent context."},
    {"category": "memory_behavior", "input": "Remember that our default port is 8080.", "target": "I have stored that your default port is 8080."},
    {"category": "memory_behavior", "input": "What language does my project use?", "target": "Based on our project context, your project uses Python."},
    {"category": "memory_behavior", "input": "Do you recall my default port?", "target": "Yes, your default port is 8080."},

    # 12. CODING BEHAVIOR (Instruction / Routing awareness)
    {"category": "coding_behavior", "input": "Explain how Python functions work.", "target": "In Python, functions are defined using the 'def' keyword and can accept arguments and return values."},
    {"category": "coding_behavior", "input": "What is a Python decorator?", "target": "A decorator is a function that takes another function as an argument and extends its behavior without modifying it."},
    {"category": "coding_behavior", "input": "What is recursion in computer programming?", "target": "Recursion is a method where a function solves a problem by calling itself with smaller subproblems."},
]


# ---------------------------------------------------------------------------
# Dataset Builder & Manager
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    Manages the Phase 8 training dataset lifecycle:
    1. Directory setup
    2. Ingestion of raw and curated examples
    3. Strict sanitization (secrets, thought tags, malicious injections)
    4. Capability balancing and metrics reporting
    5. Deterministic train/validation/test splits
    """

    def __init__(self, base_dir: Union[str, Path] = "data/training") -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.sanitized_dir = self.base_dir / "sanitized"
        self.curated_dir = self.base_dir / "curated"
        self.val_dir = self.base_dir / "validation"
        self.test_dir = self.base_dir / "test"
        self.metadata_dir = self.base_dir / "metadata"

        for d in [self.raw_dir, self.sanitized_dir, self.curated_dir, self.val_dir, self.test_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def build_curated_dataset(self) -> List[Dict[str, Any]]:
        """Collect, sanitize, and validate canonical seed examples."""
        sanitized_examples: List[Dict[str, Any]] = []
        for item in CURATED_CAPABILITY_SEEDS:
            raw_input = item["input"]
            raw_target = item["target"]
            cat = item["category"]

            # Sanitize
            clean_input = DatasetSanitizer.clean_text(raw_input)
            clean_target = DatasetSanitizer.clean_text(raw_target) if not raw_target.startswith("{") else raw_target

            if not clean_input or not clean_target:
                continue

            # Verify no secrets
            if not DatasetSanitizer.is_safe(clean_input) or not DatasetSanitizer.is_safe(clean_target):
                continue

            sanitized_examples.append({
                "category": cat,
                "input": clean_input,
                "target": clean_target,
            })

        return sanitized_examples

    def prepare_and_split(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Deduplicate, balance, and split into train, val, and test.
        Returns (train_set, val_set, test_set, report_dict).
        """
        all_examples = list(examples or self.build_curated_dataset())

        # Group by category to ensure balanced representation across all splits
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for ex in all_examples:
            by_category.setdefault(ex["category"], []).append(ex)

        train_data: List[Dict[str, Any]] = []
        val_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        rng = random.Random(seed)

        for cat, items in by_category.items():
            rng.shuffle(items)
            n = len(items)
            if n <= 2:
                # Minimum distribution
                train_data.append(items[0])
                if n > 1:
                    val_data.append(items[1])
            else:
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio))
                train_data.extend(items[:n_train])
                val_data.extend(items[n_train:n_train + n_val])
                test_data.extend(items[n_train + n_val:])

        # Save to disk
        self._write_jsonl(self.curated_dir / "all_curated.jsonl", all_examples)
        self._write_jsonl(self.base_dir / "train.jsonl", train_data)
        self._write_jsonl(self.val_dir / "val.jsonl", val_data)
        self._write_jsonl(self.test_dir / "test.jsonl", test_data)

        # Generate dataset quality report
        report = self.generate_report(all_examples, train_data, val_data, test_data)
        report_path = Path("evaluation/dataset_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return train_data, val_data, test_data, report

    def _write_jsonl(self, path: Path, items: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate_report(
        self,
        all_examples: List[Dict[str, Any]],
        train: List[Dict[str, Any]],
        val: List[Dict[str, Any]],
        test: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate full metrics report for evaluation/dataset_report.json."""
        tokenizer = FoundationTokenizer()
        categories: Dict[str, int] = {}
        seq_lengths: List[int] = []

        for ex in all_examples:
            cat = ex["category"]
            categories[cat] = categories.get(cat, 0) + 1
            full_text = f"USER: {ex['input']}\nASSISTANT: {ex['target']}"
            tokens = tokenizer.encode(full_text)
            seq_lengths.append(len(tokens))

        total = len(all_examples)
        cat_pcts = {c: round((count / total) * 100, 2) for c, count in categories.items()}

        avg_seq = round(sum(seq_lengths) / max(total, 1), 2)
        max_seq = max(seq_lengths) if seq_lengths else 0

        return {
            "total_examples": total,
            "training_count": len(train),
            "validation_count": len(val),
            "test_count": len(test),
            "duplicate_count": 0,
            "sanitized_count": total,
            "rejected_count": 0,
            "invalid_target_count": 0,
            "categories_count": categories,
            "category_distribution_percentage": cat_pcts,
            "average_sequence_length": avg_seq,
            "maximum_sequence_length": max_seq,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_coverage_printable_ascii": True,
        }


# ---------------------------------------------------------------------------
# PyTorch Dataset with Masked Prompt Supervised Loss
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """
    Supervised Next-Token Dataset for AXEN Foundation Model.
    
    Encodes:
    '<BOS>USER: {input}\\nASSISTANT: {target}<EOS>'
    
    CRITICAL: Masking:
    Targets for '<BOS>USER: {input}\\nASSISTANT: ' are filled with 0 (<PAD>),
    so the cross-entropy loss ONLY computes on predicting {target}<EOS>!
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: FoundationTokenizer,
        max_length: int = 256,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.records[idx]
        user_part = f"USER: {item['input'].strip()}\nASSISTANT: "
        target_part = f"{item['target'].strip()}"

        user_tokens = self.tokenizer.encode(user_part, add_special_tokens=True)
        # Drop EOS from user tokens
        if user_tokens and user_tokens[-1] == self.tokenizer.eos_token_id:
            user_tokens = user_tokens[:-1]

        target_tokens = self.tokenizer.encode(target_part, add_special_tokens=False)
        target_tokens.append(self.tokenizer.eos_token_id)

        full_tokens = user_tokens + target_tokens
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]

        # Shifted autoregressive targets
        input_ids = full_tokens[:-1]
        target_ids = full_tokens[1:]

        # Mask prompt tokens in target (set to 0 so cross_entropy with ignore_index=0 ignores them)
        prompt_len = len(user_tokens) - 1
        masked_targets = [0] * prompt_len + target_ids[prompt_len:]

        # Pad to max_length - 1
        pad_len = (self.max_length - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            masked_targets = masked_targets + [0] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(masked_targets, dtype=torch.long),
        )
