"""Capability-contained Python arithmetic practice. Never calls eval/exec.

This is a deliberately small interpreted language, not a general Python sandbox.
Only integers, local names, arithmetic, bounded range loops and returns exist.
Host objects, imports, attributes, strings, I/O and arbitrary calls cannot be
represented. General Python experiments require a separate OS container.
"""
from __future__ import annotations

import ast
import time


class PracticeRejected(ValueError):
    pass


class _Returned(Exception):
    def __init__(self, value):
        self.value = value


class ArithmeticSandbox:
    containment = "interpreted arithmetic Python subset; no host capabilities"

    def run(self, code: str, arguments: list[int], *, cancel=lambda: False):
        if len(code) > 4096 or len(arguments) > 8:
            raise PracticeRejected("Input budget exceeded")
        try:
            tree = ast.parse(code)
        except (SyntaxError, RecursionError) as exc:
            raise PracticeRejected("Invalid or excessively nested source") from exc
        nodes = list(ast.walk(tree))
        permitted = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                     ast.Assign, ast.AugAssign, ast.For, ast.Name, ast.Load, ast.Store,
                     ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
                     ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Call)
        if len(nodes) > 256 or any(not isinstance(n, permitted) for n in nodes):
            raise PracticeRejected("Unsupported syntax: exercise requires containment backend")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise PracticeRejected("One arithmetic function required")
        fn = tree.body[0]
        if (fn.decorator_list or fn.returns or fn.args.defaults or fn.args.kw_defaults
                or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs
                or any(a.annotation for a in fn.args.args)):
            raise PracticeRejected("Plain positional arguments only")
        if len(fn.args.args) != len(arguments):
            raise PracticeRejected("Argument count mismatch")
        for n in nodes:
            if isinstance(n, ast.Constant) and type(n.value) is not int:
                raise PracticeRejected("Only integer constants exist")
            if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name)
                    or n.func.id != "range" or n.keywords or not 1 <= len(n.args) <= 3):
                raise PracticeRejected("Only bounded range calls exist")
            if isinstance(n, ast.Name) and n.id.startswith("_"):
                raise PracticeRejected("Private names forbidden")
        env = {a.arg: self._integer(v) for a, v in zip(fn.args.args, arguments)}
        fuel = 10000
        deadline = time.monotonic() + 1

        def tick():
            nonlocal fuel
            fuel -= 1
            if fuel <= 0 or time.monotonic() > deadline or cancel():
                raise PracticeRejected("Cancelled or execution budget exhausted")

        def binary(op, left, right):
            if isinstance(op, ast.Add): value = left + right
            elif isinstance(op, ast.Sub): value = left - right
            elif isinstance(op, ast.Mult): value = left * right
            elif isinstance(op, ast.FloorDiv): value = left // right
            elif isinstance(op, ast.Mod): value = left % right
            else: raise PracticeRejected("Unsupported operator")
            return self._integer(value)

        def expression(n):
            tick()
            if isinstance(n, ast.Constant): return self._integer(n.value)
            if isinstance(n, ast.Name): return env[n.id]
            if isinstance(n, ast.BinOp): return binary(n.op, expression(n.left), expression(n.right))
            if isinstance(n, ast.UnaryOp):
                return self._integer(-expression(n.operand) if isinstance(n.op, ast.USub)
                                     else expression(n.operand))
            if isinstance(n, ast.Call):
                values = [expression(a) for a in n.args]
                r = range(*values)
                if len(r) > 1000: raise PracticeRejected("Range budget exceeded")
                return r
            raise PracticeRejected("Unsupported expression")

        def assign(n, value):
            if not isinstance(n, ast.Name) or n.id == "range":
                raise PracticeRejected("Only local integer assignment exists")
            env[n.id] = self._integer(value)

        def block(statements):
            for n in statements:
                tick()
                if isinstance(n, ast.Return): raise _Returned(self._integer(expression(n.value)))
                elif isinstance(n, ast.Assign):
                    value = expression(n.value)
                    for target in n.targets: assign(target, value)
                elif isinstance(n, ast.AugAssign):
                    assign(n.target, binary(n.op, expression(n.target), expression(n.value)))
                elif isinstance(n, ast.For):
                    r = expression(n.iter)
                    if type(r) is not range or n.orelse:
                        raise PracticeRejected("Only range loops without else exist")
                    for value in r:
                        tick(); assign(n.target, value); block(n.body)
                else: raise PracticeRejected("Unsupported statement")
        try:
            block(fn.body)
        except _Returned as result:
            return result.value
        raise PracticeRejected("Function did not return an integer")

    @staticmethod
    def _integer(value):
        if type(value) is not int or abs(value) > 10**12:
            raise PracticeRejected("Integer budget exceeded")
        return value


# A fixed exercise fixture, explicitly not AXEN's measured general ability.
DEBUGGING_SOURCE = "def inclusive_sum(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
TRAIN_CASES = [(0, 0), (1, 1), (4, 10), (6, 21)]
HELD_OUT_CASES = [(2, 3), (9, 45), (13, 91)]


class SkillPracticeAgent:
    def __init__(self, sandbox=None):
        self.sandbox = sandbox or ArithmeticSandbox()

    def evaluate(self, source, cases, *, cancel=lambda: False):
        started = time.perf_counter()
        results = []
        for arg, expected in cases:
            if cancel(): raise PracticeRejected("Cancelled")
            try:
                actual = self.sandbox.run(source, [arg], cancel=cancel)
                results.append({"input": arg, "expected": expected, "actual": actual,
                                "passed": actual == expected})
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                results.append({"input": arg, "passed": False, "error": str(exc)})
        return {"passed": sum(r["passed"] for r in results), "total": len(results),
                "cases": results, "elapsed_seconds": time.perf_counter() - started,
                "containment": self.sandbox.containment}

    def propose(self, source, concept):
        """A bounded rule-based repair derived from the supported lesson.

        Does not claim to be LLM reasoning. Held-out cases are never inputs.
        """
        if concept != "range_stop_exclusive":
            raise PracticeRejected("No supported verified repair concept")
        tree = ast.parse(source)
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "range" and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)):
                node.args[0] = ast.BinOp(node.args[0], ast.Add(), ast.Constant(1))
                changed = True
        if not changed: raise PracticeRejected("No eligible boundary repair")
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
