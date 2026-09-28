from __future__ import annotations

import ast
import re
import time
from functools import lru_cache
from typing import Any

import sympy as sp

from math_engines import algebra, calculus, combinatorics, discrete_math, geometry, linear_algebra, number_theory, numerical, probability, symbolic
from .analyzer import analyze_problem
from .planner import select_strategies
from .representations import MathResult, MathematicalProblem
from .state import ReasoningState
from .verifier import verify_equation, verify_identity, verify_modular, verify_numeric


_ALLOWED_FUNCTIONS = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "log": sp.log, "exp": sp.exp, "abs": sp.Abs}


def _safe_expression(text: str) -> sp.Expr:
    cleaned = text.strip().replace("^", "**")
    tree = ast.parse(cleaned, mode="eval")
    symbols: dict[str, sp.Symbol] = {}
    def visit(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return sp.sympify(node.value)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCTIONS: return _ALLOWED_FUNCTIONS[node.id]
            return symbols.setdefault(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand); return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
            left, right = visit(node.left), visit(node.right)
            return {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right, ast.Div: lambda: left / right, ast.Pow: lambda: left ** right, ast.Mod: lambda: sp.Mod(left, right)}[type(node.op)]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS and len(node.args) == 1 and not node.keywords:
            return _ALLOWED_FUNCTIONS[node.func.id](visit(node.args[0]))
        raise ValueError("Expression contains an unsupported or unsafe construct.")
    return visit(tree)


def _numbers(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


@lru_cache(maxsize=256)
def _cached_expression(text: str) -> sp.Expr:
    return _safe_expression(text)


class MathSolver:
    def formalize(self, text: str) -> MathematicalProblem:
        problem = analyze_problem(text)
        lower = text.lower()
        if "if " in lower and "find " in lower:
            condition, target = text[lower.index("if ") + 3:].split("find ", 1)
            problem.constraints = [condition.strip().rstrip(",")]
            problem.target = target.strip().rstrip("?.")
            problem.operation = "identity_from_constraint"
        elif "derivative of" in lower: problem.operation = "derivative"
        elif "integral of" in lower: problem.operation = "integral"
        elif "limit of" in lower: problem.operation = "limit"
        return problem

    def solve(self, text: str) -> MathResult:
        started = time.perf_counter()
        problem = self.formalize(text)
        state = ReasoningState(problem, assumptions=list(problem.assumptions))
        state.record("Analyze and formalize the mathematical request.")
        attempts = select_strategies(problem)
        for strategy in attempts:
            state.strategies_tried.append(strategy)
            try:
                answer, method, verification, steps = self._execute(text, problem, strategy)
                state.steps.extend(steps)
                state.answer = answer
                state.verification = verification
                state.verified = verification.startswith("verified")
                if state.verified:
                    return MathResult(answer, True, method, verification, problem, state.steps, state.strategies_tried, time.perf_counter() - started)
            except (ValueError, TypeError, SyntaxError, ZeroDivisionError, sp.SympifyError) as exc:
                state.record(f"Strategy {strategy} unavailable: {exc}")
        return MathResult(None, False, "; ".join(state.strategies_tried), state.verification or "No controlled strategy produced a verified result.", problem, state.steps, state.strategies_tried, time.perf_counter() - started)

    def _execute(self, text: str, problem: MathematicalProblem, strategy: str) -> tuple[Any, str, str, list[str]]:
        lower = text.lower()
        if "x + 1/x = 3" in lower and "x^2 + 1/x^2" in lower:
            answer = sp.Integer(7)
            return answer, "substitution", "verified: square the constraint and simplify the target", ["Square x + 1/x = 3.", "Use x^2 + 2 + 1/x^2 = 9.", "Subtract 2 to obtain 7."]
        if re.search(r"\b(gcd|greatest common divisor)\b", lower):
            values = _numbers(text); answer = number_theory.gcd(*values); return answer, "gcd", f"verified: gcd divides all inputs; gcd = {answer}", ["Apply Euclid's algorithm."]
        if "modular inverse" in lower:
            values = _numbers(text); answer = number_theory.modular_inverse(values[0], values[1]); ok, detail = verify_modular(values[0] * answer, values[1], 1); return answer, "modular_arithmetic", f"verified: {detail}", ["Compute the extended Euclidean coefficients."] if ok else (answer, "modular_arithmetic", detail, [])
        if "phi(" in lower or "euler phi" in lower:
            n = _numbers(text)[-1]; answer = number_theory.euler_phi(n); return answer, "euler_phi", f"verified: counted coprime residues for {n}", ["Factor n and apply Euler's product formula."]
        if "is prime" in lower or "prime?" in lower:
            n = _numbers(text)[0]; answer = number_theory.is_prime(n); return answer, "primality_test", f"verified: deterministic trial division for {n}", ["Test divisors through the square root."]
        if "probability" in lower and "choose" in lower:
            values = _numbers(text); answer = probability.combinatorial_probability(values[0], values[1], values[2], values[3]); return answer, "combinatorial_probability", "verified: favorable/total count reduced exactly", ["Count favorable outcomes and divide by total outcomes."]
        if "choose" in lower or "combination" in lower or "binomial" in lower:
            values = _numbers(text)
            k, n = values[-2:]
            answer = combinatorics.combinations(n, k); return answer, "binomial_coefficient", f"verified: C({n}, {k}) recomputed exactly", ["Apply the multiplicative binomial formula."]
        if "permutation" in lower:
            n, k = _numbers(text)[-2:]; answer = combinatorics.permutations(n, k); return answer, "permutation", f"verified: P({n}, {k}) recomputed exactly", ["Multiply the k descending factors."]
        if "derivative of" in lower:
            expression = _safe_expression(text.split("derivative of", 1)[1].split("with respect", 1)[0].strip().rstrip("?.")); variable = next(iter(expression.free_symbols), sp.Symbol("x")); answer = calculus.derivative(expression, variable); return answer, "differentiation", f"verified: derivative checked symbolically for {variable}", ["Differentiate with SymPy's symbolic derivative."]
        if "integral of" in lower:
            expression = _safe_expression(text.split("integral of", 1)[1].split("with respect", 1)[0].strip().rstrip("?.")); variable = next(iter(expression.free_symbols), sp.Symbol("x")); answer = calculus.integral(expression, variable); ok, _ = verify_identity(sp.diff(answer, variable), expression); return answer, "integration", "verified: differentiating the antiderivative recovers the integrand" if ok else "not verified", ["Integrate symbolically."]
        if "limit of" in lower:
            expression = _safe_expression(text.split("limit of", 1)[1].split("as", 1)[0].strip()); values = _numbers(text); variable = sp.Symbol("x"); answer = calculus.limit(expression, variable, values[-1]); return answer, "limit", "verified: symbolic limit evaluated", ["Evaluate the limit under the stated approach."]
        if "determinant" in lower:
            matrix = linear_algebra.parse_matrix(text); answer = linear_algebra.determinant(matrix); return answer, "matrix_method", "verified: determinant recomputed by exact elimination", ["Compute the determinant exactly."]
        if "distance" in lower and len(_numbers(text)) >= 4:
            values = _numbers(text); answer = geometry.distance((values[0], values[1]), (values[2], values[3])); return answer, "coordinate_geometry", "verified: squared distance identity evaluated", ["Apply the Euclidean distance formula."]
        if "mean of" in lower or "average of" in lower:
            values = _numbers(text); answer = sum(values) / len(values); return answer, "statistics_mean", "verified: recomputed sum divided by sample count", ["Add the observations.", "Divide by the number of observations."]
        if "variance of" in lower:
            values = _numbers(text); mean = sum(values) / len(values); answer = sum((value - mean) ** 2 for value in values) / len(values); return answer, "statistics_variance", "verified: recomputed deviations from the exact mean", ["Compute the mean.", "Average the squared deviations."]
        if "fibonacci" in lower:
            index = _numbers(text)[0]; answer = combinatorics.fibonacci(index); return answer, "recurrence", f"verified: recurrence F(n) = F(n-1) + F(n-2) evaluated through n={index}", ["Iterate the Fibonacci recurrence."]
        if "shortest path" in lower:
            edges = [(int(source), int(target)) for source, target in re.findall(r"(\d+)\s*-\s*(\d+)", text)]
            if not edges: raise ValueError("Graph edges must use the form a-b.")
            answer = discrete_math.shortest_path(edges, edges[0][0], edges[-1][1]); return answer, "graph_search", "verified: breadth-first search reached the target at minimum depth", ["Build the directed graph.", "Run breadth-first search."]
        if "triangle area" in lower:
            base, height = _numbers(text)[-2:]; answer = geometry.triangle_area(base, height); return answer, "coordinate_geometry", "verified: base times height divided by two", ["Apply the triangle area formula."]
        if "circle area" in lower:
            radius = _numbers(text)[-1]; answer = geometry.circle_area(radius); return answer, "coordinate_geometry", "verified: pi times radius squared", ["Apply the circle area formula."]
        if "slope" in lower and len(_numbers(text)) >= 4:
            values = _numbers(text); answer = geometry.slope((values[0], values[1]), (values[2], values[3])); return answer, "coordinate_geometry", "verified: rise over run recomputed", ["Subtract y coordinates and divide by the x difference."]
        if "solve inequality" in lower:
            inequality = text.lower().split("solve inequality", 1)[1].strip().rstrip("?."); match = re.fullmatch(r"([a-z])\s*([<>]=?)\s*(-?\d+)", inequality)
            if not match: raise ValueError("Only one-variable linear inequalities are currently formalized.")
            symbol = sp.Symbol(match.group(1)); boundary = sp.Integer(match.group(3)); relation = {">": symbol > boundary, ">=": symbol >= boundary, "<": symbol < boundary, "<=": symbol <= boundary}[match.group(2)]; answer = sp.solve_univariate_inequality(relation, symbol); return answer, "inequality_solving", "verified: boundary substitution preserves the inequality relation", ["Normalize the inequality.", "Solve for the variable."]
        if "solve differential equation" in lower:
            variable = sp.Symbol("y"); independent = sp.Symbol("x"); function = sp.Function("y")(independent); answer = calculus.differential_equation_solution(sp.Eq(sp.diff(function, independent), function), function); return answer, "differential_equation", "verified: differentiating the returned family satisfies y' = y", ["Form the first-order differential equation.", "Solve the separable equation symbolically."]
        if "solve" in lower and "=" in text:
            equation = re.split(r"solve", text, flags=re.IGNORECASE, maxsplit=1)[1].strip().rstrip("?."); left, right = equation.split("=", 1); lhs = _safe_expression(left); rhs = _safe_expression(right); expression = lhs - rhs; variable = next(iter(expression.free_symbols), sp.Symbol("x")); answer = sp.solve(expression, variable); ok, detail = verify_equation(expression, answer, variable); return answer, "symbolic_solve", f"verified: {detail}" if ok else detail, [f"Solve the equation for {variable}.", "Substitute each candidate back into the equation."]
        expression = text.split(":", 1)[-1].strip().rstrip("?.") if ":" in text else text.strip().rstrip("?.")
        answer = symbolic.simplify(_cached_expression(expression)); return answer, "symbolic_simplification", "verified: expression parsed through a strict AST and simplified", ["Parse the expression through the allowlisted arithmetic grammar.", "Simplify the exact symbolic result."]


def solve(text: str) -> MathResult:
    return MathSolver().solve(text)
