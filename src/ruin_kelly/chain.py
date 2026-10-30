"""Exact answers from an absorbing Markov chain with one step per trading day.

The state is the account's P&L at the start of a day, in lattice units. A day's
trades are handled by listing every win/loss sequence (2^k for k trades a day),
which keeps the daily loss limit exact. Absorption probabilities and expected
times then come from solving (I - Q) x = b, where Q holds the transitions between
non-absorbed states (the fundamental-matrix method).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .model import Edge, Lattice, Rules, make_lattice

OUTCOMES = ("pass", "fail_overall", "fail_daily")


@dataclass
class DayChain:
    lattice: Lattice
    states: np.ndarray          # P&L (units) of each non-absorbed state
    Q: sparse.csr_matrix        # Q[i, j] = P(start day in i, end day in j, not absorbed)
    absorb: dict[str, np.ndarray]  # P(absorbed into each outcome during one day | start in i)
    start_index: int            # index of P&L = 0


def build_chain(edge: Edge, rules: Rules, max_denominator: int = 200) -> DayChain:
    lat = make_lattice(edge, rules, max_denominator)
    states = np.arange(lat.fail_at + 1, lat.pass_at)
    n = len(states)
    p, q = edge.win_rate, 1 - edge.win_rate
    rows, cols, vals = [], [], []
    absorb = {name: np.zeros(n) for name in OUTCOMES}

    for sequence in itertools.product((True, False), repeat=rules.trades_per_day):
        wins = sum(sequence)
        prob = p**wins * q ** (len(sequence) - wins)
        pnl = states.copy()
        alive = np.ones(n, dtype=bool)
        for win in sequence:
            trading = alive.copy()
            if rules.stop_before_daily_breach and lat.daily_fail_at is not None:
                trading &= (pnl - states) - lat.loss_units > lat.daily_fail_at
            pnl = np.where(trading, pnl + (lat.win_units if win else -lat.loss_units), pnl)
            # checked in this order: a trade that passes cannot also fail
            hit = {"pass": trading & (pnl >= lat.pass_at),
                   "fail_overall": trading & (pnl <= lat.fail_at)}
            daily_breach = (pnl - states <= lat.daily_fail_at) if lat.daily_fail_at is not None \
                else np.zeros(n, dtype=bool)
            hit["fail_daily"] = trading & ~hit["pass"] & ~hit["fail_overall"] & daily_breach
            for name, mask in hit.items():
                absorb[name][mask] += prob
                alive &= ~mask
        rows.append(np.flatnonzero(alive))
        cols.append(pnl[alive] - states[0])
        vals.append(np.full(alive.sum(), prob))

    Q = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
    return DayChain(lat, states, Q, absorb, int(np.flatnonzero(states == 0)[0]))


def solve(chain: DayChain) -> dict[str, float]:
    """Probability of each outcome and expected trading days, with no time limit."""
    # From every state a run of wins passes, so absorption is reachable unless some
    # state can never trade. Whether a trade is allowed does not depend on the state,
    # so that case shows up as a state that stays put for certain (e.g. the daily
    # guard with risk >= daily_loss).
    if np.isclose(chain.Q.diagonal(), 1.0).any():
        raise ValueError("some states never change, so the account never finishes")
    A = (sparse.identity(len(chain.states), format="csc") - chain.Q).tocsc()
    result = {name: float(spsolve(A, chain.absorb[name])[chain.start_index]) for name in OUTCOMES}
    result["expected_days"] = float(spsolve(A, np.ones(len(chain.states)))[chain.start_index])
    return result


def within_days(chain: DayChain, days: int) -> dict[str, float]:
    """Probability of each outcome within `days` trading days (the rest are still running)."""
    dist = np.zeros(len(chain.states))
    dist[chain.start_index] = 1.0
    QT = chain.Q.T.tocsr()
    totals = dict.fromkeys(OUTCOMES, 0.0)
    for _ in range(days):
        for name in OUTCOMES:
            totals[name] += float(dist @ chain.absorb[name])
        dist = QT @ dist
    totals["still_running"] = float(dist.sum())
    return totals
