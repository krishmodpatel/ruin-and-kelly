"""The trading model shared by every method.

Each trade risks a fixed fraction `risk` of the *initial* balance: a loss costs
`risk`, a win pays `payoff * risk`, and trades win independently with probability
`win_rate`. Profit and loss (P&L) is measured as a fraction of the initial balance,
as prop firms do. The account passes when P&L reaches `target` and fails when it
reaches `-max_loss`, or when one day's P&L reaches `-daily_loss`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

EPS = 1e-9  # guards integer conversions of values like 0.10 / 0.0004 against rounding


@dataclass(frozen=True)
class Edge:
    win_rate: float  # p
    payoff: float    # b = average win / average loss
    risk: float      # fraction of the initial balance lost on a losing trade

    def __post_init__(self):
        if not 0 < self.win_rate < 1:
            raise ValueError("win_rate must be strictly between 0 and 1")
        if self.payoff <= 0 or self.risk <= 0:
            raise ValueError("payoff and risk must be positive")

    @property
    def expectancy(self) -> float:
        """Expected P&L per trade, in units of `risk`: p*b - (1 - p)."""
        return self.win_rate * self.payoff - (1 - self.win_rate)


@dataclass(frozen=True)
class Rules:
    target: float = 0.10
    max_loss: float = 0.10                # static: measured from the initial balance
    daily_loss: float | None = 0.05       # measured from the balance at the start of each day
    trades_per_day: int = 3
    stop_before_daily_breach: bool = False  # skip a trade whose loss would breach the daily limit

    def __post_init__(self):
        if self.target <= 0 or self.max_loss <= 0 or self.trades_per_day < 1:
            raise ValueError("target, max_loss and trades_per_day must be positive")
        if self.daily_loss is not None and self.daily_loss <= 0:
            raise ValueError("daily_loss must be positive or None")


@dataclass(frozen=True)
class Lattice:
    """Integer version of the model: P&L counted in units of `unit`.

    The payoff is written exactly as a fraction win_units / loss_units, so every P&L
    the account can reach is a whole number of units.
    """
    unit: float
    win_units: int
    loss_units: int
    pass_at: int            # P&L >= pass_at units passes
    fail_at: int            # P&L <= fail_at units fails (negative)
    daily_fail_at: int | None
    payoff_used: float      # win_units / loss_units


def make_lattice(edge: Edge, rules: Rules, max_denominator: int = 200) -> Lattice:
    # Rounding the payoff is not safe: moving it by 0.00001 can decide whether a run
    # of wins lands exactly on the target, so an inexact fraction is refused.
    ratio = Fraction(edge.payoff).limit_denominator(max_denominator)
    if abs(float(ratio) - edge.payoff) > 1e-12 * edge.payoff:
        raise ValueError(f"payoff {edge.payoff} is not a fraction with denominator "
                         f"<= {max_denominator}; round it (e.g. to 2 decimals) first")
    unit = edge.risk / ratio.denominator
    daily = None
    if rules.daily_loss is not None:
        daily = math.floor(-rules.daily_loss / unit + EPS)
    return Lattice(
        unit=unit,
        win_units=ratio.numerator,
        loss_units=ratio.denominator,
        pass_at=math.ceil(rules.target / unit - EPS),
        fail_at=math.floor(-rules.max_loss / unit + EPS),
        daily_fail_at=daily,
        payoff_used=float(ratio),
    )
