"""
axen_chat.py
=============
Interactive CLI chat interface for AXEN Phase 2 Local Assistant.

Usage:
    python axen_chat.py                   # Load model + start chat
    python axen_chat.py --no-model        # Tool-only mode (no LLM)
    python axen_chat.py --model <name>    # Use a different HF model
    python axen_chat.py --status          # Print status and exit

Commands inside chat:
    /quit or /exit    — Exit the chat
    /reset            — Clear conversation memory
    /status           — Print current status
    /remember <fact>  — Save a fact to long-term memory
    /recall           — List all saved facts
    /tools            — List available tools
    /help             — Show this help

This file is SELF-CONTAINED — it does not require the older AXEN systems
and does not modify them. It imports only from the new `axen/` package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import `axen`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axen import AssistantController, HFModelConfig

# -----------------------------------------------------------------------
# Logging: suppress noisy library logs by default
# -----------------------------------------------------------------------
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
# Keep axen-internal INFO logs
logging.getLogger("axen").setLevel(logging.ERROR)


# Ensure stdout and stderr handle UTF-8 characters on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------
# ANSI colors (Windows 10+/11 via colorama fallback)
# -----------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init()
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = Style.RESET_ALL

    def _c(text: str, color: str = CYAN) -> str:
        return f"{color}{text}{RESET}"
except ImportError:
    CYAN = GREEN = YELLOW = RED = RESET = ""

    def _c(text: str, color: str = "") -> str:
        return text


# -----------------------------------------------------------------------
# Banners & helpers
# -----------------------------------------------------------------------

def _banner() -> None:
    print(_c("""
╔══════════════════════════════════════════════════════╗
║          AXEN  —  Local AI Assistant  v2.0           ║
║    Runs fully offline · No cloud · No API keys       ║
╚══════════════════════════════════════════════════════╝
""", CYAN))
    print("  Type /help for commands, /quit to exit.\n")


def _show_help() -> None:
    print(_c("""
Commands:
  /quit   or /exit    — Exit the chat
  /reset              — Clear conversation history
  /status             — Show model and memory status
  /remember <fact>    — Save a fact permanently
  /recall             — List all saved facts
  /tools              — List available tools
  /help               — Show this message
""", YELLOW))


def _axen_say(msg: str, color: str = "") -> None:
    """Print an AXEN status message during startup."""
    print(_c(f"  AXEN: {msg}", color or GREEN))


def _axen_err(msg: str) -> None:
    print(_c(f"  ⚠  AXEN: {msg}", RED))


# -----------------------------------------------------------------------
# Model loading — step-by-step visible progress
# -----------------------------------------------------------------------

def load_model_with_progress(controller: AssistantController, config: HFModelConfig) -> bool:
    """
    Load the local HuggingFace model with step-by-step console feedback.

    Returns True on success, False on failure.
    Never pretends the model is loaded when it isn't.
    """
    # Step 1 — announce
    _axen_say(f"Loading my local language model...")
    _axen_say(f"Model  : {config.name}")

    # Step 2 — check CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            _axen_say(f"Device : cuda  ({device_name}, {vram_gb:.1f} GB VRAM)")
        else:
            _axen_say(f"Device : cpu   (CUDA not available — this will be slow)")
    except ImportError:
        _axen_err(
            "PyTorch is not installed in this environment.\n"
            "  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "       pip install transformers accelerate"
        )
        return False

    _axen_say(f"Dtype  : {config.dtype}")
    _axen_say(f"Max tokens : {config.max_new_tokens}")

    # Step 3 — load tokenizer + model (single synchronous call)
    _axen_say("Loading tokenizer...", YELLOW)
    try:
        # Monkey-patch: hook into HFLocalModel to get per-step feedback
        _do_load_with_hooks(controller, config)
        return True
    except ImportError as exc:
        _axen_err(str(exc))
        _axen_err(
            "Install missing packages:\n"
            "    pip install transformers accelerate sentencepiece tokenizers"
        )
        return False
    except RuntimeError as exc:
        _axen_err(f"I could not load my local language model.")
        _axen_err(f"Details: {exc}")
        return False
    except Exception as exc:
        _axen_err(f"Unexpected error during model load: {exc}")
        return False


def _do_load_with_hooks(controller: AssistantController, config: HFModelConfig) -> None:
    """
    Call controller.load_model() with step-level console messages.
    We print feedback before and after tokenizer/model loading by
    temporarily wrapping the internal methods.
    """
    import torch
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required. "
            "Run: pip install transformers accelerate"
        ) from exc

    model_obj = controller.model
    cfg = model_obj.config

    # --- Resolve device ---
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device not in ("auto",) and not cfg.device.startswith("cuda"):
        device_str = cfg.device
    torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.float32

    # --- Tokenizer ---
    _axen_say("Loading tokenizer...", YELLOW)
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _axen_say("Tokenizer loaded.", GREEN)

    # --- Model ---
    _axen_say("Loading model weights (this takes ~20–60 s on first CUDA load)...", YELLOW)

    model_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_str != "cpu":
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.name, device_map=device_str, **model_kwargs
            )
        except Exception as gpu_exc:
            _axen_say(f"GPU load failed ({gpu_exc}), retrying on CPU ...", YELLOW)
            hf_model = AutoModelForCausalLM.from_pretrained(cfg.name, **model_kwargs)
            device_str = "cpu"
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.name, **model_kwargs)

    hf_model.eval()
    if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
        hf_model.generation_config.max_length = None

    # --- Inject into model object ---
    model_obj._tokenizer = tokenizer
    model_obj._model = hf_model
    model_obj._device = device_str
    model_obj._loaded = True
    controller._model_loaded = True

    # --- Confirm ---
    if device_str.startswith("cuda"):
        mem_used = torch.cuda.memory_allocated(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _axen_say(
            f"Model loaded on {device_str} "
            f"(VRAM used: {mem_used:.1f}/{mem_total:.1f} GB).",
            GREEN,
        )
    else:
        _axen_say("Model loaded on CPU.", GREEN)
    _axen_say("I am ready.", GREEN)


# -----------------------------------------------------------------------
# Chat command handler
# -----------------------------------------------------------------------

def _handle_command(cmd: str, controller: AssistantController) -> bool:
    """Handle a /command. Returns True if the loop should exit."""
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit"):
        print("\nGoodbye! 👋")
        return True

    elif name == "/reset":
        controller.reset_conversation()
        print(_c("  Conversation memory cleared.", GREEN))

    elif name == "/status":
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        except ImportError:
            cuda_available = False
            cuda_device = "N/A"

        st = controller.status()
        print()
        print(_c("  ─── AXEN Status ───", CYAN))
        print(f"  Model         : {controller.model.model_name}")
        print(f"  Model loaded  : {st['model_loaded']}")
        print(f"  CUDA available: {cuda_available}  ({cuda_device})")
        print(f"  Device in use : {controller.model._device}")
        print(f"  Conv. turns   : {st['conversation_turns']}")
        print(f"  Ctx tokens ≈  : {st['memory_tokens_estimate']}")
        print(f"  Long-term facts: {st['long_term_facts']}")
        print(f"  Tools         : {', '.join(st['tools'])}")
        print()

    elif name == "/remember":
        if not arg:
            print("  Usage: /remember <fact>")
        else:
            controller.long_memory.remember("user_note", arg)
            print(_c(f'  Saved: "{arg}"', GREEN))

    elif name == "/recall":
        facts = controller.long_memory.all_facts()
        if not facts:
            print("  No saved facts yet.")
        else:
            for k, v in facts.items():
                print(f"  {k}: {v}")

    elif name == "/tools":
        print()
        for t in controller.tools.list_tools():
            risk_color = GREEN if t["risk"] == "low" else YELLOW
            print(_c(f"  [{t['risk'].upper()}]", risk_color) + f" {t['name']} — {t['description']}")
        print()

    elif name == "/help":
        _show_help()

    else:
        print(f"  Unknown command: {name}. Type /help for commands.")

    return False


# -----------------------------------------------------------------------
# Main chat loop
# -----------------------------------------------------------------------

def chat_loop(controller: AssistantController, debug: bool = False) -> None:
    """Main REPL loop."""
    _banner()

    while True:
        try:
            user_input = input(_c("You: ", CYAN)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_exit = _handle_command(user_input, controller)
            if should_exit:
                break
            continue

        response = controller.respond(user_input, debug=debug)

        if debug:
            state = getattr(controller, "last_agent_state", None)
            d = controller.last_debug_info
            messages = controller.memory.to_messages()
            sys_content = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            print()
            print(_c("=== AXEN INPUT ===", YELLOW))
            print(f"SYSTEM: {sys_content}")
            print(f"USER: {user_input}")
            print()
            print(_c("=== AXEN OUTPUT ===", YELLOW))
            print(response)
            print()
            print(_c("AXEN DEBUG:", YELLOW))
            print(f"Task: {d.get('task_type', 'unknown')}")
            tool_name = d.get("selected_tool") or "none"
            print(f"Tool: {tool_name}")
            if state and state.tool_calls:
                tc_args = state.tool_calls[0].get("arguments", {})
                if "expression" in tc_args:
                    print(f"Expression: {tc_args['expression']}")
                elif "problem" in tc_args:
                    print(f"Problem: {tc_args['problem']}")
                elif "code" in tc_args:
                    print(f"Code: {tc_args['code']}")
                elif "path" in tc_args:
                    print(f"Path: {tc_args['path']}")
            result_val = d.get("tool_result")
            if result_val and result_val != "none":
                print(f"Result: {result_val}")
            print(f"Verification: {d.get('verification', 'passed')}")
            print(f"Status: {d.get('final_status', 'completed')}")
            print()

        print(_c("Axen: ", GREEN) + response)
        print()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AXEN Local AI Assistant — Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Start in tool-only mode without loading the language model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HuggingFace model name (overrides config). "
            "E.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit (does not load model).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show safe diagnostic information (tokens, roles, model) during chat.",
    )
    args = parser.parse_args()

    # Build config — apply CLI --model override if given
    config = HFModelConfig.from_yaml()
    if args.model:
        config = HFModelConfig(
            name=args.model,
            device=config.device,
            dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            max_context_length=config.max_context_length,
        )

    controller = AssistantController(config=config)

    # ── Status mode ──────────────────────────────────────────────────
    if args.status:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        except ImportError:
            cuda_available = False
            cuda_device = "N/A"

        print()
        print(_c("  ─── AXEN Status ───", CYAN))
        print(f"  Configured model : {config.name}")
        print(f"  Model loaded     : False  (--status skips model load)")
        print(f"  CUDA available   : {cuda_available}  ({cuda_device})")
        print(f"  Configured device: {config.device}")
        print(f"  Dtype            : {config.dtype}")
        print(f"  Max new tokens   : {config.max_new_tokens}")
        print(f"  Tools            : {', '.join(controller.tools.names())}")
        print()
        return

    # ── No-model mode ─────────────────────────────────────────────────
    if args.no_model:
        print()
        _axen_say("Starting in tool-only mode (language model NOT loaded).", YELLOW)
        _axen_say("Calculator and Python executor are available.", YELLOW)
        print()
        chat_loop(controller, debug=args.debug)
        return

    # ── Normal mode: load the model ───────────────────────────────────
    print()
    success = load_model_with_progress(controller, config)
    print()

    if not success:
        _axen_err(
            "Model could not be loaded. "
            "Falling back to tool-only mode.\n"
            "  Run 'python axen_chat.py --no-model' to start in tool-only mode intentionally."
        )
        print()

    chat_loop(controller, debug=args.debug)


if __name__ == "__main__":
    main()
