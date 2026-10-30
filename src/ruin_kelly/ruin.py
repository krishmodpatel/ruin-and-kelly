"""Closed-form gambler's ruin for equal wins and losses (payoff = 1).

The account moves one step up with probability p and one step down with
probability q = 1 - p. It starts `start` steps above the loss limit and passes
`total` steps above it. Derivations are in paper/ruin_kelly.tex, section 2.
"""

from __future__ import annotations

import math

# Within this distance of a fair game the general formulas lose precision to
# cancellation, and the fair-game formulas are accurate to about 1e-6 relative.
FAIR_TOL = 1e-7


def _check(p: float, start: int, total: int) -> None:
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    if not 0 <= start <= total or total < 1:
        raise ValueError("need 0 <= start <= total and total >= 1")


def pass_probability(p: float, start: int, total: int) -> float:
    """P(reach `total` before 0) = (1 - rho^start) / (1 - rho^total), rho = q / p."""
    _check(p, start, total)
    q = 1 - p
    if abs(p - q) < FAIR_TOL:
        return start / total
    rho = q / p
    if rho < 1:
        return (1 - rho**start) / (1 - rho**total)
    # rho > 1: divide through by rho^total so no power overflows
    inv = 1 / rho
    return (inv ** (total - start) - inv**total) / (1 - inv**total)


def expected_steps(p: float, start: int, total: int) -> float:
    """Expected number of steps until the account passes or fails."""
    _check(p, start, total)
    q = 1 - p
    if abs(p - q) < FAIR_TOL:
        return start * (total - start)
    return (start - total * pass_probability(p, start, total)) / (q - p)
