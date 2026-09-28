"""
axen/agent/loop.py
==================
Core Agentic Reasoning Loop for AXEN Phase 3.

Implements the complete cycle:
UNDERSTAND
→ CLASSIFY
→ PLAN
→ CHOOSE TOOLS
→ EXECUTE
→ OBSERVE RESULTS
→ VERIFY
→ RESPOND

Safety & limits:
- MAX_STEPS = 8
- MAX_TOOL_ATTEMPTS = 3
- Bounded retries on failure
- No private chain-of-thought stored or exposed
- Never fabricates tool output or calculations
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .classifier import TaskClassifier
from .planner import Planner
from .state import AgentState
from .verifier import Verifier

logger = logging.getLogger("axen.agent.loop")

MAX_STEPS = 8
MAX_TOOL_ATTEMPTS = 3


class AgentLoop:
    """Coordinates the full agentic loop across controller subsystems."""

    def __init__(
        self,
        tools: Any,
        memory: Any,
        long_memory: Any,
        model: Any,
        classifier: Optional[TaskClassifier] = None,
        planner: Optional[Planner] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.tools = tools
        self.memory = memory
        self.long_memory = long_memory
        self.model = model
        self.classifier = classifier or TaskClassifier()
        self.planner = planner or Planner()
        self.verifier = verifier or Verifier()

    def run(self, user_request: str, *, verbose: bool = False, debug: bool = False) -> tuple[str, AgentState]:
        """
        Execute the agentic reasoning loop for a user request.

        Returns:
            (final_answer_string, final_agent_state)
        """
        state = AgentState(user_request=user_request, status="understanding")

        # 1. UNDERSTAND & CLASSIFY
        task_type, tool_required, selected_tool = self.classifier.classify(user_request)
        state.task_type = task_type
        state.metadata["tool_required"] = tool_required
        state.metadata["selected_tool"] = selected_tool

        # 2. PLAN
        plan = self.planner.create_plan(task_type, user_request, selected_tool)
        state.plan = plan
        state.status = "planned"

        # 3. CONVERSATIONAL SHORT-CIRCUIT (No tool needed)
        if not tool_required or not selected_tool:
            answer = self._run_conversation(user_request, state)
            state.final_answer = answer
            state.status = "completed"
            return answer, state

        # 4. TOOL EXECUTION & VERIFICATION LOOP
        step_count = 0
        latest_observation: Optional[str] = None
        has_verification_failure = False

        for step in plan:
            if step_count >= MAX_STEPS:
                logger.warning("Agent loop reached MAX_STEPS limit (%d)", MAX_STEPS)
                break
            step_count += 1
            state.current_step = step_count

            step_tool = step.get("tool")
            step_args = step.get("arguments", {})

            if not step_tool:
                continue

            # Execute tool with bounded retry
            obs, success, err, verif = self._execute_with_retry(step_tool, step_args, user_request, state)
            state.verification = verif
            latest_observation = obs

            if not success or (verif and not verif.get("passed", False)):
                has_verification_failure = True
                state.status = "verification_failed"
                err_msg = err or (verif.get("detail") if verif else "Execution failed.")
                answer = f"Error: {err_msg}"
                state.final_answer = answer
                return answer, state

        # 5. SYNTHESIZE / FINALIZE RESPONSE
        if state.task_type == "mixed":
            final_answer = self._synthesize_mixed_response(user_request, latest_observation, state)
        elif state.task_type == "calculation":
            final_answer = f"The answer is {latest_observation}."
        elif latest_observation is not None:
            final_answer = str(latest_observation)
        else:
            final_answer = "Task completed."

        state.final_answer = final_answer
        state.status = "completed"
        return final_answer, state

    # ------------------------------------------------------------------
    # Step Execution with Bounded Retry
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_request: str,
        state: AgentState,
    ) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """
        Execute a tool with up to MAX_TOOL_ATTEMPTS attempts.
        Returns (observation, success, error, verification_dict).
        """
        attempts = 0
        last_error = None
        last_verif = None

        while attempts < MAX_TOOL_ATTEMPTS:
            attempts += 1

            # Dispatch tool
            if tool_name == "calculator":
                obs, ok, err, verif = self._run_calculator(user_request, state)
            elif tool_name == "math_solver":
                obs, ok, err, verif = self._run_math_solver(user_request, state)
            elif tool_name == "python":
                obs, ok, err, verif = self._run_python(user_request, state)
            elif tool_name == "file":
                obs, ok, err, verif = self._run_file(args, user_request, state)
            elif tool_name == "memory":
                obs, ok, err, verif = self._run_memory(user_request, state)
            elif tool_name == "llm":
                obs = self._run_conversation(user_request, state)
                ok, err, verif = True, None, {"passed": True, "status": "passed"}
            else:
                ok, err = False, f"Unknown tool: {tool_name}"
                obs, verif = None, None

            if ok and (verif is None or verif.get("passed", False)):
                return obs, True, None, verif

            last_error = err
            last_verif = verif

            # Check if recoverable
            if not self._is_recoverable_error(err):
                break

        return None, False, last_error, last_verif

    def _is_recoverable_error(self, error: Optional[str]) -> bool:
        """Determine if an error is transient / recoverable."""
        if not error:
            return False
        err_lower = error.lower()
        if "syntax error" in err_lower or "division by zero" in err_lower:
            return False  # Non-recoverable invalid math/syntax
        if "blocked" in err_lower or "security" in err_lower:
            return False  # Security violations never retry
        return True

    # ------------------------------------------------------------------
    # Specific Tool Handlers
    # ------------------------------------------------------------------

    def _run_calculator(self, req: str, state: AgentState) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """Normalize, evaluate, and verify arithmetic expressions."""
        expr = self._extract_math_expression(req)
        if not expr:
            return None, False, f"Could not parse arithmetic expression from '{req}'.", None

        state.add_tool_call("calculator", {"expression": expr})
        raw_res = self.tools.call("calculator.evaluate", {"expression": expr})

        if not raw_res.success:
            state.add_tool_result("calculator", None, success=False, error=raw_res.error)
            verif = self.verifier.verify_tool_result("calculator", {"expression": expr}, {"success": False, "error": raw_res.error})
            return None, False, raw_res.error, verif

        output = raw_res.output
        display = str(output.get("display", output.get("result", "")))
        state.add_tool_result("calculator", display, success=True)

        verif = self.verifier.verify_tool_result("calculator", {"expression": expr}, output)
        return display, True, None, verif

    def _run_math_solver(self, req: str, state: AgentState) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """Run SymPy MathSolver and verify symbolic proof."""
        # Normalize request for MathSolver:
        # e.g., "Differentiate x^3 + 2x" -> "derivative of x^3 + 2*x"
        norm_req = req
        if re.search(r"\bdifferentiate\b", norm_req, re.I) and "derivative of" not in norm_req.lower():
            norm_req = re.sub(r"\bdifferentiate\b", "derivative of", norm_req, flags=re.I)
        # Convert implicit multiplication: 2x -> 2*x, 3y -> 3*y
        norm_req = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", norm_req)

        state.add_tool_call("math_solver", {"problem": norm_req})
        try:
            from reasoning.math.solver import MathSolver
            solver = MathSolver()
            res = solver.solve(norm_req)
            if res.verified:
                answer_str = str(res.answer)
                state.add_tool_result("math_solver", answer_str, success=True)
                verif = {"passed": True, "status": "passed", "detail": str(res.verification)}
                return answer_str, True, None, verif
            else:
                state.add_tool_result("math_solver", None, success=False, error=res.verification)
                verif = {"passed": False, "status": "verification_failed", "detail": res.verification}
                return None, False, res.verification, verif
        except Exception as exc:
            state.add_tool_result("math_solver", None, success=False, error=str(exc))
            return None, False, str(exc), None

    def _run_python(self, req: str, state: AgentState) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """Extract and safely execute Python code."""
        code = self._extract_python_code(req)
        if not code:
            return None, False, "No valid Python code found in request.", None

        state.add_tool_call("python", {"code": code})
        res = self.tools.call("python.execute", {"code": code})

        if not res.success:
            state.add_tool_result("python", None, success=False, error=res.error)
            verif = {"passed": False, "status": "verification_failed", "detail": res.error}
            return None, False, res.error, verif

        stdout = res.output.get("stdout", "")
        stderr = res.output.get("stderr", "")
        formatted = f"```\n{stdout}\n```" if stdout else "_(no output)_"
        if stderr:
            formatted += f"\n**stderr:**\n```\n{stderr}\n```"

        state.add_tool_result("python", stdout, success=True)
        verif = self.verifier.verify_tool_result("python", {"code": code}, res.output)
        return formatted, True, None, verif

    def _run_file(self, args: dict[str, Any], req: str, state: AgentState) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """Safely read and verify local file."""
        path = args.get("path")
        if not path:
            m = re.search(r"\b(?:read|show|view|open)\s+(?:this\s+)?file[:\s]+([^\s,;]+)", req, re.I)
            path = m.group(1).strip().strip("'\"") if m else ""

        if not path:
            return None, False, "Missing file path.", None

        state.add_tool_call("file", {"path": path})
        res = self.tools.call("file.read", {"path": path})
        if not res.success:
            state.add_tool_result("file", None, success=False, error=res.error)
            verif = {"passed": False, "status": "verification_failed", "detail": res.error}
            return None, False, res.error, verif

        content = res.output.get("content", "")
        state.add_tool_result("file", f"{len(content)} chars", success=True)
        verif = self.verifier.verify_tool_result("file", {"path": path}, res.output)
        return f"Contents of `{path}`:\n\n```\n{content}\n```", True, None, verif

    def _run_memory(self, req: str, state: AgentState) -> tuple[Optional[str], bool, Optional[str], Optional[dict[str, Any]]]:
        """Store or recall persistent user facts."""
        state.add_tool_call("memory", {"raw_input": req})
        # Store
        if re.search(r"\b(remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind)\b", req, re.I):
            m = re.search(r"(?:remember\s+that|please\s+note|save\s+that|note\s+that|keep\s+in\s+mind)[,:\s]+(.+)", req, re.I)
            if m:
                fact = m.group(1).strip().rstrip(".")
                key = re.sub(r"[^a-z0-9_]", "_", fact[:40].lower()).strip("_")
                self.long_memory.remember(key, fact)
                msg = f'Got it — I\'ve saved that: "{fact}"'
                state.add_tool_result("memory", msg, success=True)
                return msg, True, None, {"passed": True, "status": "passed"}

        # Recall
        facts = self.long_memory.all_facts()
        if not facts:
            msg = "I don't have any saved notes about you yet."
        else:
            lines = [f"- **{k}**: {v}" for k, v in list(facts.items())[-10:]]
            msg = "Here's what I remember:\n" + "\n".join(lines)
        state.add_tool_result("memory", f"{len(facts)} facts", success=True)
        return msg, True, None, {"passed": True, "status": "passed"}

    def _run_conversation(self, req: str, state: AgentState) -> str:
        """Call local LLM with conversation history and anti-fabrication guardrails."""
        if not getattr(self.model, "is_loaded", False):
            return (
                "My local language model isn't loaded yet. "
                "For deterministic tasks (math, code), I can still help. "
                "For open conversation, start me with `controller.load_model()` first."
            )

        messages = self.memory.to_messages()
        # Add current user request if not already present
        if not messages or messages[-1].get("content") != req:
            messages.append({"role": "user", "content": req})

        try:
            if hasattr(self.model, "chat_with_stats"):
                raw_response, stats = self.model.chat_with_stats(messages)
                state.metadata["model_stats"] = stats
            else:
                raw_response = self.model.chat(messages)
        except Exception as exc:
            return f"Inference error: {exc}"

        return self._apply_identity_guardrails(req, raw_response)

    def _synthesize_mixed_response(self, req: str, observation: Optional[str], state: AgentState) -> str:
        """Synthesize tool observations with LLM explanation."""
        if not observation:
            return "Task completed."
        if not getattr(self.model, "is_loaded", False):
            return f"Multi-step task completed. Observation:\n{observation}"
        obs_summary = f"[Observation from tools]\n{observation}\n\nTask request: {req}"
        return self._run_conversation(obs_summary, state)

    # ------------------------------------------------------------------
    # Guardrails & Extraction Helpers
    # ------------------------------------------------------------------

    def _apply_identity_guardrails(self, user_input: str, response: str) -> str:
        """Guarantee no hallucinated employer, human hobbies, or runaway greeting monologue."""
        cleaned = response.strip()
        lower_q = user_input.lower().strip()
        lower_r = cleaned.lower()

        # For simple greetings ('hi', 'hello', etc.), return concise greeting
        if re.match(r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|sup)[\s\.\!\?]*$", lower_q):
            cleaned = cleaned.split("\n\n")[0].strip()
            cleaned = cleaned.split("\n")[0].strip()
            return cleaned

        # Identity questions: "who are you", "what are you"
        if re.search(r"\b(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name)\b", lower_q):
            if any(phrase in lower_r for phrase in ["work for", "employed by", "founded by", "company called", "job search", "hobbies"]):
                return "I am AXEN, a local AI assistant running on your computer."

        # Company / employer questions
        if re.search(r"\b(?:what|which)\s+company\s+(?:do\s+you|are\s+you)\b|\bwho\s+do\s+you\s+work\s+for\b|\byour\s+employer\b", lower_q):
            if any(phrase in lower_r for phrase in ["i work for", "employed by", "founded by", "company called", "joined the company"]):
                return "I don't work for a company. I'm AXEN, a local AI assistant running on your computer."

        # Hobbies questions
        if re.search(r"\b(?:do\s+you\s+have|what\s+are\s+your)\s+hobbies\b", lower_q):
            if any(phrase in lower_r for phrase in ["i like to", "i enjoy playing", "i love playing", "my hobbies are", "guitar", "hiking", "watching movies"]):
                return "I don't have personal hobbies. I'm an AI assistant running locally on your computer."

        return cleaned

    def _extract_math_expression(self, text: str) -> Optional[str]:
        """Extract clean arithmetic expression."""
        t = text
        # If text has "calculate <expr>" or "compute <expr>", isolate the target
        calc_match = re.search(r"\b(?:calculate|compute|what\s+is)\s+(.+)", t, re.I)
        if calc_match:
            candidate = calc_match.group(1).strip()
            candidate = re.split(r"\b(?:and\s+then|and\s+explain|and\s+show)\b", candidate, flags=re.I)[0].strip()
            t = candidate

        t = re.sub(r"\bmultiplied\s+by\b", "*", t, flags=re.I)
        t = re.sub(r"\btimes\b", "*", t, flags=re.I)
        t = re.sub(r"\bdivided\s+by\b", "/", t, flags=re.I)
        t = re.sub(r"\bplus\b", "+", t, flags=re.I)
        t = re.sub(r"\bminus\b", "-", t, flags=re.I)
        t = re.sub(r"\bto\s+the\s+power\s+of\b", "**", t, flags=re.I)

        m = re.search(r"\(?\s*\d[\d\s\+\-\*\/\(\)\^\.%]*\d\)?", t)
        if m:
            c = m.group(0).strip()
            if re.search(r"\d.*[+\-*/^%]\s*.*\d|\d.*\*\*.*\d", c):
                return c.replace("^", "**")
        return None

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from markdown or keywords."""
        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?:run|execute):?\s*\n((?:[ \t].+\n?)+)", text, re.I)
        if m:
            return m.group(1).strip()
        # Single line print statement like "print(2 + 3)"
        m = re.search(r"\b(print\s*\(.+?\))", text, re.I)
        if m:
            return m.group(1).strip()
        return None
