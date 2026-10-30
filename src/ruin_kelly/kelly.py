"""Kelly criterion: the stake that maximises long-run growth when profits compound.

Here a trade risks a fraction f of the *current* balance (unlike the prop-firm
model, which risks a fraction of the initial balance). A win multiplies the balance
by 1 + b f and a loss by 1 - f, so the expected log growth per trade is

    g(f) = p ln(1 + b f) + q ln(1 - f),

maximised at f* = p - q / b (derivation in paper/ruin_kelly.tex, section 4).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def growth(f, p: float, b: float):
    """Expected log growth per trade. -inf when f >= 1 (one loss wipes out the account)."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = p * np.log1p(b * f) + (1 - p) * np.log1p(-f)
    return np.where(f >= 1, -np.inf, g)


def kelly_fraction(p: float, b: float) -> float:
    """f* = p - q / b. Negative means the bet has no edge: do not trade."""
    return p - (1 - p) / b


def plug_in_growth(p_true: float, b: float, n_trades: int, multiplier: float,
                   max_fraction: float = 0.99) -> dict[str, float]:
    """What happens if you estimate p from n trades and then bet multiplier x Kelly.

    The estimate p_hat = wins / n is binomial, so the expectation over all possible
    samples is an exact finite sum. Negative stakes are set to 0 (no trading) and
    stakes are capped at `max_fraction`. Without the cap, full Kelly would stake 100%
    after a sample of all wins and the expected growth would be minus infinity. The
    payoff b is treated as known, so this understates the real risk.
    """
    wins = np.arange(n_trades + 1)
    weights = stats.binom.pmf(wins, n_trades, p_true)
    stakes = np.clip(multiplier * (wins / n_trades - (1 - wins / n_trades) / b), 0, max_fraction)
    f_star = kelly_fraction(p_true, b)
    return {
        "expected_growth": float(weights @ growth(stakes, p_true, b)),
        "p_losing_growth": float(weights @ (growth(stakes, p_true, b) < 0)),
        "p_over_twice_kelly": float(weights @ (stakes > 2 * f_star)) if f_star > 0 else float("nan"),
    }


def drawdown_probability(f: float, p: float, b: float, depth: float, n_trades: int,
                         n_paths: int, rng: np.random.Generator) -> float:
    """Share of simulated paths whose balance falls `depth` below its running peak."""
    log_win, log_loss = np.log1p(b * f), np.log1p(-f)
    steps = np.where(rng.random((n_paths, n_trades)) < p, log_win, log_loss)
    log_equity = np.cumsum(steps, axis=1)
    peak = np.maximum.accumulate(np.maximum(log_equity, 0.0), axis=1)
    return float((log_equity - peak <= np.log1p(-depth)).any(axis=1).mean())
