"""Every experiment in the write-up. Each returns a DataFrame; __main__ saves them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import chain, kelly, ruin
from .model import Edge, Rules
from .simulate import simulate

# Krish's self-reported record: 5 wins in 12 trades, average win 2.28 x average loss.
RECORD_WINS, RECORD_TRADES, RECORD_PAYOFF = 5, 12, 2.28
RECORD_P = RECORD_WINS / RECORD_TRADES
RECORD_CI = (0.193, 0.680)  # 95% Wilson interval, as computed in the Skill-or-Luck project
CHALLENGE = Rules(target=0.10, max_loss=0.10, daily_loss=0.05, trades_per_day=3)
SEED = 2027


def validation(n_accounts: int) -> pd.DataFrame:
    """Equal wins and losses: closed form vs Markov chain vs simulation."""
    rows = []
    rng = np.random.default_rng(SEED)
    rules = Rules(daily_loss=None, trades_per_day=1)
    for risk in (0.01, 0.02):
        start = round(rules.max_loss / risk)
        total = start + round(rules.target / risk)
        for p in (0.40, 0.45, 0.50, 0.55):
            edge = Edge(p, 1.0, risk)
            exact = chain.solve(chain.build_chain(edge, rules))
            mc = simulate(edge, rules, n_accounts, rng)
            se = np.sqrt(mc["pass"] * (1 - mc["pass"]) / n_accounts)
            rows.append({
                "risk": risk, "win_rate": p,
                "formula_pass": ruin.pass_probability(p, start, total),
                "chain_pass": exact["pass"], "simulated_pass": mc["pass"], "simulation_se": se,
                "formula_trades": ruin.expected_steps(p, start, total),
                "chain_trades": exact["expected_days"], "simulated_trades": mc["mean_days"],
            })
    return pd.DataFrame(rows)


def challenge_by_risk(win_rate: float, payoff: float, rules: Rules, n_accounts: int,
                      horizon_days: int = 60) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 1)
    for risk in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        edge = Edge(win_rate, payoff, risk)
        ch = chain.build_chain(edge, rules)
        try:
            exact = chain.solve(ch)
        except ValueError:  # the rules never allow a trade at this risk
            continue
        timed = chain.within_days(ch, horizon_days)
        mc = simulate(edge, rules, n_accounts, rng)
        rows.append({
            "risk": risk, "pass": exact["pass"], "fail_overall": exact["fail_overall"],
            "fail_daily": exact["fail_daily"], "expected_days": exact["expected_days"],
            f"pass_within_{horizon_days}_days": timed["pass"], "simulated_pass": mc["pass"],
            "payoff_used": ch.lattice.payoff_used,
        })
    return pd.DataFrame(rows)


def pass_vs_win_rate(payoff: float, risk: float, rules: Rules) -> pd.DataFrame:
    """How much the answer depends on a win rate that 12 trades cannot pin down."""
    rows = []
    for p in np.round(np.arange(0.15, 0.71, 0.01), 2):
        exact = chain.solve(chain.build_chain(Edge(float(p), payoff, risk), rules))
        rows.append({"win_rate": p, "pass": exact["pass"], "expected_days": exact["expected_days"],
                     "expectancy_R": p * payoff - (1 - p)})
    return pd.DataFrame(rows)


def heatmap(rules: Rules, axis: str) -> pd.DataFrame:
    """Pass probability over win rate and either payoff (risk 1%) or risk (payoff 2.28)."""
    win_rates = np.round(np.arange(0.20, 0.71, 0.025), 3)
    if axis == "payoff":
        others, make = np.round(np.arange(0.5, 3.01, 0.125), 3), lambda p, x: Edge(p, x, 0.01)
    else:
        others, make = np.array([0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]), \
            lambda p, x: Edge(p, RECORD_PAYOFF, x)
    rows = [{"win_rate": p, axis: x, "pass": chain.solve(chain.build_chain(make(float(p), float(x)), rules))["pass"]}
            for p in win_rates for x in others]
    return pd.DataFrame(rows)


def kelly_curves(payoff: float = RECORD_PAYOFF) -> pd.DataFrame:
    rows = []
    for label, p in (("CI low", RECORD_CI[0]), ("record", RECORD_P), ("CI high", RECORD_CI[1])):
        f_star = kelly.kelly_fraction(p, payoff)
        row = {"case": label, "win_rate": p, "kelly_fraction": f_star}
        if f_star > 0:
            g_star = float(kelly.growth(f_star, p, payoff))
            row["growth_full"] = g_star
            for c in (0.25, 0.5, 2.0):
                g = float(kelly.growth(c * f_star, p, payoff))
                # -inf means the stake is 100% or more: one loss ends the account
                row[f"growth_share_{c:g}x"] = g / g_star if np.isfinite(g) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def kelly_drawdowns(p: float, payoff: float, n_trades: int = 500, n_paths: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    f_star = kelly.kelly_fraction(p, payoff)
    rows = []
    for c in (0.25, 0.5, 1.0, 2.0):
        row = {"multiplier": c, "fraction": c * f_star, "growth": float(kelly.growth(c * f_star, p, payoff))}
        for depth in (0.2, 0.5, 0.8):
            row[f"p_drawdown_{int(depth * 100)}pct"] = kelly.drawdown_probability(
                c * f_star, p, payoff, depth, n_trades, n_paths, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def estimation_error(p_true: float = RECORD_P, payoff: float = RECORD_PAYOFF) -> pd.DataFrame:
    """Best Kelly multiplier when p is estimated from n trades and plugged into the formula."""
    multipliers = np.round(np.arange(0.0, 1.51, 0.05), 2)
    g_star = float(kelly.growth(kelly.kelly_fraction(p_true, payoff), p_true, payoff))
    rows = []
    for n in (12, 30, 100, 300, 1000):
        results = {c: kelly.plug_in_growth(p_true, payoff, n, c) for c in multipliers}
        best = max(results, key=lambda c: results[c]["expected_growth"])
        rows.append({
            "n_trades": n, "best_multiplier": best,
            "growth_share_at_best": results[best]["expected_growth"] / g_star,
            "growth_share_full_kelly": results[1.0]["expected_growth"] / g_star,
            "p_losing_growth_full_kelly": results[1.0]["p_losing_growth"],
            "p_over_twice_kelly_full": results[1.0]["p_over_twice_kelly"],
        })
    return pd.DataFrame(rows)
