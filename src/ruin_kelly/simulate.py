"""Monte Carlo simulation of the same rules, used to check the exact methods.

It works with the payoff directly rather than on the integer lattice, so it is an
independent check of the chain's bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .model import EPS, Edge, Rules


def simulate(edge: Edge, rules: Rules, n_accounts: int, rng: np.random.Generator,
             max_days: int = 100_000) -> dict[str, float]:
    """Share of accounts ending in each outcome; `still_running` if max_days was reached."""
    pnl = np.zeros(n_accounts)
    alive = np.ones(n_accounts, dtype=bool)
    counts = dict.fromkeys(("pass", "fail_overall", "fail_daily"), 0)
    days = np.zeros(n_accounts)
    win, loss = edge.payoff * edge.risk, edge.risk

    for _ in range(max_days):
        if not alive.any():
            break
        day_start = pnl.copy()
        days[alive] += 1
        for _ in range(rules.trades_per_day):
            trading = alive.copy()
            if rules.stop_before_daily_breach and rules.daily_loss is not None:
                trading &= (pnl - day_start) - loss > -rules.daily_loss + EPS
            wins = rng.random(n_accounts) < edge.win_rate
            pnl = np.where(trading, pnl + np.where(wins, win, -loss), pnl)
            passed = trading & (pnl >= rules.target - EPS)
            failed = trading & (pnl <= -rules.max_loss + EPS)
            daily = trading & ~passed & ~failed
            if rules.daily_loss is not None:
                daily &= pnl - day_start <= -rules.daily_loss + EPS
            else:
                daily[:] = False
            counts["pass"] += passed.sum()
            counts["fail_overall"] += failed.sum()
            counts["fail_daily"] += daily.sum()
            alive &= ~(passed | failed | daily)

    result = {name: count / n_accounts for name, count in counts.items()}
    result["still_running"] = alive.mean()
    result["mean_days"] = float(days[~alive].mean()) if (~alive).any() else float("nan")
    return result
