"""
axen/model/hf_model.py
======================
Direct local model inference via PyTorch + HuggingFace Transformers.

This module is the NEW inference layer for AXEN Phase 2.
It does NOT use Ollama, cloud APIs, or any API key.
It loads a model directly into memory and runs inference locally.

Designed for RTX 3050 Laptop (4 GB VRAM) using float16.
Default model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Architecture note:
  The existing Ollama-based LocalModel (app/ai/local_model.py) is PRESERVED
  and still used by the math/code reasoning pipelines. This is a new,
  separate, independent inference path.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("axen.model.hf")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

@dataclass
class HFModelConfig:
    """Configuration for the local HuggingFace-based language model."""
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_context_length: int = 2048
    use_flash_attention: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/axen_local.yaml") -> "HFModelConfig":
        """Load configuration from YAML file. Uses defaults if file missing."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            hf = raw.get("hf_model", {})
            if not isinstance(hf, dict):
                return cls()
            return cls(
                name=str(hf.get("name", cls.name)),
                device=str(hf.get("device", cls.device)),
                dtype=str(hf.get("dtype", cls.dtype)),
                max_new_tokens=int(hf.get("max_new_tokens", cls.max_new_tokens)),
                temperature=float(hf.get("temperature", cls.temperature)),
                top_p=float(hf.get("top_p", cls.top_p)),
                max_context_length=int(hf.get("max_context_length", cls.max_context_length)),
                use_flash_attention=bool(hf.get("use_flash_attention", cls.use_flash_attention)),
            )
        except Exception as exc:
            logger.warning("Could not parse HF model config: %s — using defaults.", exc)
            return cls()


# ---------------------------------------------------------------
# Stop Criteria and Turn Truncation
# ---------------------------------------------------------------

def _truncate_turn(text: str) -> str:
    """
    Truncate text at any marker indicating the model began generating
    a subsequent turn (such as simulating a user question or dialogue continuation).
    """
    stop_markers = [
        "<|user|>",
        "<|system|>",
        "<|assistant|>",
        "</s>",
        "\n<|user|>",
        "\nUser:",
        "\nHuman:",
        "\nQuestion:",
        "\nUser ",
        "\n[USER]",
        "\n[ASSISTANT]",
        "\nUser's next",
    ]
    earliest = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest].strip()


StoppingCriteriaList = None


class TurnStoppingCriteria:
    """Callable stopping criterion; importing the UI must not load Transformers."""

    def __init__(self, tokenizer: Any, prompt_length: int, stop_sequences: list[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        new_ids = input_ids[0][self.prompt_length:]
        if len(new_ids) < 1:
            return False
        tail_text = self.tokenizer.decode(new_ids[-15:], skip_special_tokens=False)
        return any(seq in tail_text for seq in self.stop_sequences)


# ---------------------------------------------------------------
# Local HuggingFace Language Model
# ---------------------------------------------------------------

class HFLocalModel:
    """
    Loads and runs a HuggingFace causal language model locally.

    Lifecycle:
      1. Instantiate with config.
      2. Call load() — downloads or loads from cache, moves to GPU.
      3. Call generate() / chat() for inference.
      4. Call unload() to free GPU memory when done.

    Error handling:
      - If transformers or torch are not installed: raises ImportError with a clear message.
      - If the model download fails: raises RuntimeError.
      - If VRAM is insufficient: raises RuntimeError with a helpful message.
      - If generation fails: raises RuntimeError.

    This class NEVER pretends to run when the model is not loaded.
    """

    def __init__(self, config: Optional[HFModelConfig] = None) -> None:
        self.config = config or HFModelConfig.from_yaml()
        self._model: Any = None       # AutoModelForCausalLM instance
        self._tokenizer: Any = None   # AutoTokenizer instance
        self._device: str = "cpu"     # Resolved actual device
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if the model and tokenizer are loaded and ready."""
        return self._loaded and self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        return self.config.name

    def load(self) -> None:
        """
        Load the tokenizer and model into memory.
        Downloads from HuggingFace Hub on first use (cached afterwards).
        Raises RuntimeError on any failure — never pretends to succeed.
        """
        try:
            import torch
            global StoppingCriteriaList
            from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for local model inference. "
                "Run: pip install transformers torch accelerate"
            ) from exc

        # Resolve the target device
        self._device = self._resolve_device(torch)
        logger.info("Loading tokenizer for %s ...", self.config.name)

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.name,
                trust_remote_code=True,
            )
            # Ensure a padding token exists (required for batched inference)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("Loading model %s on device=%s dtype=%s ...",
                        self.config.name, self._device, self.config.dtype)

            # Determine torch dtype
            torch_dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

            # Load model with optional flash attention
            model_kwargs: dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            if self._device == "cpu":
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.name,
                    **model_kwargs,
                )
            else:
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.name,
                        device_map=self._device,
                        **model_kwargs,
                    )
                except Exception as gpu_exc:
                    logger.warning("GPU load failed (%s), trying CPU ...", gpu_exc)
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.name,
                        **model_kwargs,
                    )
                    self._device = "cpu"

            self._model.eval()
            if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
                self._model.generation_config.max_length = None
            self._loaded = True
            logger.info("Model loaded successfully on %s.", self._device)

        except MemoryError as exc:
            raise RuntimeError(
                f"Axen: I could not load my local language model — out of memory. "
                f"Try a smaller model or reduce max_context_length. Details: {exc}"
            ) from exc
        except Exception as exc:
            self._loaded = False
            error_msg = str(exc)
            if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                raise RuntimeError(
                    f"Axen: I could not load my local language model — GPU out of memory. "
                    f"Try float16 dtype or a smaller model. Details: {exc}"
                ) from exc
            raise RuntimeError(
                f"Axen: I could not load my local language model. Details: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model and tokenizer from memory."""
        try:
            import torch
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and GPU cache cleared.")
        except Exception as exc:
            logger.warning("Could not fully unload model: %s", exc)

    def generate(
        self,
        prompt: str,
        *,
        system: str = (
            "You are AXEN, a local AI assistant. Answer the user's current request directly and naturally. "
            "Use available tools when necessary. Do not invent information. Follow the conversation context. "
            "Do not mention internal instructions or hidden reasoning."
        ),
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a text response given a user prompt.

        Parameters:
            prompt: The user's input text.
            system: System instruction shown before the prompt.
            max_new_tokens: Override config max_new_tokens.
            temperature: Override config temperature.
            top_p: Override config top_p.

        Returns:
            The generated assistant response as a string.

        Raises:
            RuntimeError if the model is not loaded or generation fails.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Axen: My local language model is not loaded. Call load() first."
            )

        import torch

        # Format using the chat template directly with tokenize=True
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded.input_ids
                attention_mask = getattr(encoded, "attention_mask", None)
            else:
                input_ids = encoded
                attention_mask = None
        except Exception:
            # Fallback: plain text formatting
            formatted = f"[SYSTEM] {system}\n[USER] {prompt}\n[ASSISTANT]"
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if input_ids.shape[-1] > self.config.max_context_length:
            input_ids = input_ids[:, -self.config.max_context_length:]
            attention_mask = attention_mask[:, -self.config.max_context_length:]

        try:
            # Move inputs to model device
            if self._device != "cpu":
                input_ids = input_ids.to(self._model.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._model.device)

            n_tokens = max_new_tokens or self.config.max_new_tokens
            temp = temperature if temperature is not None else self.config.temperature
            p = top_p if top_p is not None else self.config.top_p

            stopping_criteria = None
            if StoppingCriteriaList is not None and TurnStoppingCriteria is not None:
                stop_seqs = ["<|user|>", "<|system|>", "</s>", "\nUser:", "\nHuman:"]
                stopping_criteria = StoppingCriteriaList([
                    TurnStoppingCriteria(self._tokenizer, input_ids.shape[-1], stop_seqs)
                ])

            # Generate with optimized inference mode
            ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            with ctx:
                gen_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": n_tokens,
                    "do_sample": (temp > 0.0),
                    "temperature": temp if temp > 0.0 else None,
                    "top_p": p if temp > 0.0 else None,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                }
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria
                output_ids = self._model.generate(**gen_kwargs)

            # Decode only the newly generated tokens (not the prompt)
            new_ids = output_ids[0][input_ids.shape[-1]:]
            response = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return _truncate_turn(response)

        except Exception as exc:
            error_str = str(exc)
            if "out of memory" in error_str.lower():
                raise RuntimeError(
                    "Axen: I ran out of GPU memory during generation. "
                    "Try reducing max_new_tokens."
                ) from exc
            raise RuntimeError(f"Axen: Generation failed — {exc}") from exc

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """
        Multi-turn chat interface. messages is a list of role/content dicts.
        Roles: 'system', 'user', 'assistant'.
        Returns the assistant's response string.
        """
        resp, _ = self.chat_with_stats(messages, **kwargs)
        return resp

    def chat_with_stats(self, messages: list[dict], **kwargs: Any) -> tuple[str, dict[str, Any]]:
        """
        Multi-turn chat interface returning (response_string, stats_dict).
        Stats include input_tokens, generated_tokens, and roles.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Axen: My local language model is not loaded. Call load() first."
            )
        import torch

        # Requirement 1: Log exact model input in DEBUG mode
        try:
            import json
            logger.debug("\n=== AXEN MODEL INPUT ===\n%s\n=== END MODEL INPUT ===", json.dumps(messages, indent=2, ensure_ascii=False))
        except Exception:
            pass

        # Use direct tokenizer apply_chat_template
        t_tok_start = time.perf_counter()
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded.input_ids
                attention_mask = getattr(encoded, "attention_mask", None)
            else:
                input_ids = encoded
                attention_mask = None
        except Exception:
            # Fallback: concatenate into plain text
            parts = []
            for m in messages:
                role = m.get("role", "user").upper()
                parts.append(f"[{role}] {m.get('content', '')}")
            parts.append("[ASSISTANT]")
            formatted = "\n".join(parts)
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if input_ids.shape[-1] > self.config.max_context_length:
            input_ids = input_ids[:, -self.config.max_context_length:]
            attention_mask = attention_mask[:, -self.config.max_context_length:]
        t_tok_duration = time.perf_counter() - t_tok_start

        try:
            if self._device != "cpu":
                input_ids = input_ids.to(self._model.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._model.device)

            n_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
            temp = kwargs.get("temperature", self.config.temperature)
            p = kwargs.get("top_p", self.config.top_p)

            stopping_criteria = None
            if StoppingCriteriaList is not None and TurnStoppingCriteria is not None:
                stop_seqs = ["<|user|>", "<|system|>", "</s>", "\nUser:", "\nHuman:"]
                stopping_criteria = StoppingCriteriaList([
                    TurnStoppingCriteria(self._tokenizer, input_ids.shape[-1], stop_seqs)
                ])

            ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            t_gen_start = time.perf_counter()
            with ctx:
                gen_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": n_tokens,
                    "do_sample": (temp > 0.0),
                    "temperature": temp if temp > 0.0 else None,
                    "top_p": p if temp > 0.0 else None,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                }
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria
                output_ids = self._model.generate(**gen_kwargs)
            t_gen_duration = time.perf_counter() - t_gen_start

            new_ids = output_ids[0][input_ids.shape[-1]:]
            decoded = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            cleaned = _truncate_turn(decoded)
            stats = {
                "input_tokens": int(input_ids.shape[-1]),
                "generated_tokens": len(new_ids),
                "roles": [m.get("role", "unknown") for m in messages],
                "tokenization_time": t_tok_duration,
                "generation_time": t_gen_duration,
            }
            return cleaned, stats

        except Exception as exc:
            raise RuntimeError(f"Axen: Chat generation failed — {exc}") from exc

    def get_memory_info(self) -> dict[str, Any]:
        """Return memory statistics for GPU VRAM and CPU RAM."""
        info = {
            "model_loaded": self.is_loaded,
            "device": self._device,
            "dtype": self.config.dtype,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
        }
        try:
            import torch
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        except Exception:
            pass
        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, torch: Any) -> str:
        """Resolve the target device string from config."""
        cfg = self.config.device.lower()
        if cfg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU.")
            return "cpu"
        return cfg

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"HFLocalModel(name={self.config.name!r}, device={self._device!r}, status={status!r})"
