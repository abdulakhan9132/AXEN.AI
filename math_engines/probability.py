from __future__ import annotations

import math


def conditional_probability(joint: float, condition: float) -> float:
    if condition == 0: raise ValueError("Condition probability cannot be zero.")
    return joint / condition

def bayes(prior: float, likelihood: float, evidence: float) -> float:
    if evidence == 0: raise ValueError("Evidence probability cannot be zero.")
    return prior * likelihood / evidence

def expected(values, probabilities): return sum(value * probability for value, probability in zip(values, probabilities))
def variance(values, probabilities):
    mean = expected(values, probabilities)
    return sum(probability * (value - mean) ** 2 for value, probability in zip(values, probabilities))
def combinatorial_probability(favorable: int, total_favorable: int, total: int, total_count: int) -> float:
    return math.comb(total_favorable, favorable) * math.comb(total - total_favorable, total_count - favorable) / math.comb(total, total_count)
