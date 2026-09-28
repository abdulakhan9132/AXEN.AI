from __future__ import annotations

from dataclasses import dataclass

from .problem_types import ProblemCategory


DOMAIN_ALIASES = {
    "sequences": ProblemCategory.SEQUENCES_SERIES.value,
    "series": ProblemCategory.SEQUENCES_SERIES.value,
    "discrete": ProblemCategory.DISCRETE_MATH.value,
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    domains: frozenset[str]
    priority: int


STRATEGY_REGISTRY: tuple[StrategySpec, ...] = (
    StrategySpec("substitution", frozenset({"algebra", "equations", "calculus"}), 10),
    StrategySpec("factorization", frozenset({"algebra", "equations", "number theory"}), 20),
    StrategySpec("expansion", frozenset({"algebra"}), 30),
    StrategySpec("completing_square", frozenset({"algebra", "equations", "optimization"}), 40),
    StrategySpec("polynomial_identity", frozenset({"algebra", "equations"}), 5),
    StrategySpec("symmetry", frozenset({"algebra", "combinatorics", "geometry"}), 50),
    StrategySpec("change_of_variable", frozenset({"algebra", "calculus"}), 60),
    StrategySpec("modular_arithmetic", frozenset({"number theory"}), 5),
    StrategySpec("gcd", frozenset({"number theory"}), 10),
    StrategySpec("contradiction", frozenset({"number theory", "algebra"}), 70),
    StrategySpec("parity", frozenset({"number theory", "discrete mathematics"}), 80),
    StrategySpec("divisibility", frozenset({"number theory"}), 20),
    StrategySpec("prime_factorization", frozenset({"number theory"}), 30),
    StrategySpec("chinese_remainder_theorem", frozenset({"number theory"}), 40),
    StrategySpec("induction", frozenset({"number theory", "sequences/series", "discrete mathematics"}), 60),
    StrategySpec("differentiation", frozenset({"calculus", "optimization"}), 5),
    StrategySpec("integration_by_parts", frozenset({"calculus"}), 30),
    StrategySpec("partial_fractions", frozenset({"calculus", "algebra"}), 40),
    StrategySpec("limit_algebra", frozenset({"calculus"}), 10),
    StrategySpec("taylor_series", frozenset({"calculus", "sequences/series"}), 50),
    StrategySpec("optimization", frozenset({"calculus", "optimization"}), 20),
    StrategySpec("counting", frozenset({"combinatorics", "probability"}), 5),
    StrategySpec("inclusion_exclusion", frozenset({"combinatorics", "probability"}), 20),
    StrategySpec("recurrence", frozenset({"combinatorics", "sequences/series", "discrete mathematics"}), 10),
    StrategySpec("dynamic_programming", frozenset({"combinatorics", "discrete mathematics", "graph theory"}), 30),
    StrategySpec("pigeonhole", frozenset({"combinatorics", "discrete mathematics"}), 40),
    StrategySpec("generating_function", frozenset({"combinatorics", "sequences/series"}), 50),
    StrategySpec("conditional_probability", frozenset({"probability"}), 10),
    StrategySpec("bayes", frozenset({"probability"}), 20),
    StrategySpec("expectation", frozenset({"probability", "statistics"}), 30),
    StrategySpec("independence", frozenset({"probability"}), 40),
    StrategySpec("gaussian_elimination", frozenset({"linear algebra"}), 5),
    StrategySpec("matrix_method", frozenset({"linear algebra"}), 10),
    StrategySpec("eigenvalue_method", frozenset({"linear algebra"}), 20),
    StrategySpec("vector_method", frozenset({"linear algebra", "geometry"}), 30),
    StrategySpec("coordinate_geometry", frozenset({"geometry"}), 5),
    StrategySpec("vector_geometry", frozenset({"geometry", "linear algebra"}), 20),
    StrategySpec("analytic_geometry", frozenset({"geometry"}), 10),
    StrategySpec("transformation", frozenset({"geometry"}), 30),
    StrategySpec("algebraic_geometry", frozenset({"geometry", "algebra"}), 40),
    StrategySpec("numerical_method", frozenset({"arithmetic", "calculus", "optimization", "equations"}), 100),
)

_REGISTRY = {spec.name: spec for spec in STRATEGY_REGISTRY}


def canonical_domain(domain: str) -> str:
    return DOMAIN_ALIASES.get(domain.strip().lower(), domain.strip().lower())


def supported_strategies() -> list[dict[str, object]]:
    return [{"name": spec.name, "domains": sorted(spec.domains), "priority": spec.priority} for spec in STRATEGY_REGISTRY]


def validate_candidate_strategies(domains: list[str], candidates: list[str]) -> list[str]:
    canonical_domains = {canonical_domain(domain) for domain in domains}
    if not canonical_domains or not canonical_domains.issubset({category.value for category in ProblemCategory}):
        raise ValueError("Strategy selection contains an unsupported domain.")
    if not isinstance(candidates, list) or not all(isinstance(candidate, str) for candidate in candidates):
        raise ValueError("candidate_strategies must be a list of strategy names.")
    selected: list[str] = []
    for candidate in candidates:
        if candidate not in _REGISTRY:
            raise ValueError(f"Unsupported strategy: {candidate!r}.")
        if not (_REGISTRY[candidate].domains & canonical_domains):
            raise ValueError(f"Strategy {candidate!r} is incompatible with domains {sorted(canonical_domains)}.")
        if candidate not in selected:
            selected.append(candidate)
    return selected


def strategies_for_domains(domains: list[str]) -> list[str]:
    canonical_domains = {canonical_domain(domain) for domain in domains}
    return [spec.name for spec in sorted(STRATEGY_REGISTRY, key=lambda item: item.priority) if spec.domains & canonical_domains]
