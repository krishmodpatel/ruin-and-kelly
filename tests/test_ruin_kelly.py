import numpy as np
import pytest
from scipy import optimize

from ruin_kelly import chain, kelly, ruin
from ruin_kelly.model import Edge, Rules, make_lattice
from ruin_kelly.simulate import simulate

NO_DAILY = Rules(daily_loss=None, trades_per_day=1)


# --- closed-form gambler's ruin ---------------------------------------------------

def test_fair_game_values():
    assert ruin.pass_probability(0.5, 3, 10) == pytest.approx(0.3)
    assert ruin.expected_steps(0.5, 3, 10) == pytest.approx(21)


@pytest.mark.parametrize("p", [0.3, 0.45, 0.55, 0.7])
def test_closed_form_satisfies_the_recurrence(p):
    """P_i = p P_{i+1} + q P_{i-1} and D_i = 1 + p D_{i+1} + q D_{i-1}, with the boundaries."""
    total = 12
    P = [ruin.pass_probability(p, i, total) for i in range(total + 1)]
    D = [ruin.expected_steps(p, i, total) for i in range(total + 1)]
    assert P[0] == pytest.approx(0) and P[-1] == pytest.approx(1)
    assert D[0] == pytest.approx(0) and D[-1] == pytest.approx(0)
    for i in range(1, total):
        assert P[i] == pytest.approx(p * P[i + 1] + (1 - p) * P[i - 1])
        assert D[i] == pytest.approx(1 + p * D[i + 1] + (1 - p) * D[i - 1])


def test_no_overflow_for_long_games():
    assert 0 <= ruin.pass_probability(0.3, 2000, 4000) < 1e-300
    assert ruin.pass_probability(0.7, 2000, 4000) == pytest.approx(1)


def test_symmetry():
    assert ruin.pass_probability(0.6, 4, 10) == pytest.approx(1 - ruin.pass_probability(0.4, 6, 10))


# --- Markov chain ------------------------------------------------------------------

@pytest.mark.parametrize("p,risk", [(0.42, 0.01), (0.5, 0.02), (0.58, 0.025)])
def test_chain_matches_closed_form(p, risk):
    exact = chain.solve(chain.build_chain(Edge(p, 1.0, risk), NO_DAILY))
    start = round(0.10 / risk)
    assert exact["pass"] == pytest.approx(ruin.pass_probability(p, start, 2 * start))
    assert exact["expected_days"] == pytest.approx(ruin.expected_steps(p, start, 2 * start))


def test_outcome_probabilities_sum_to_one():
    for rules in (Rules(), Rules(stop_before_daily_breach=True), Rules(trades_per_day=5)):
        exact = chain.solve(chain.build_chain(Edge(0.4, 2.28, 0.02), rules))
        assert sum(exact[k] for k in chain.OUTCOMES) == pytest.approx(1)


def test_several_trades_per_day_equals_one_trade_per_day_without_daily_limit():
    edge = Edge(0.45, 1.5, 0.01)
    one = chain.solve(chain.build_chain(edge, NO_DAILY))
    three = chain.solve(chain.build_chain(edge, Rules(daily_loss=None, trades_per_day=3)))
    assert three["pass"] == pytest.approx(one["pass"])


def test_daily_limit_by_hand():
    """Risk 5%, payoff 3: the first trade decides. A win (+15%) passes, a loss (-5%) breaches the daily limit."""
    exact = chain.solve(chain.build_chain(Edge(0.3, 3.0, 0.05), Rules()))
    assert exact["pass"] == pytest.approx(0.3)
    assert exact["fail_daily"] == pytest.approx(0.7)


def test_daily_limit_only_binds_when_a_day_can_lose_5_percent():
    edge_small, edge_big = Edge(0.42, 2.28, 0.015), Edge(0.42, 2.28, 0.02)
    assert chain.solve(chain.build_chain(edge_small, Rules()))["fail_daily"] == 0
    assert chain.solve(chain.build_chain(edge_big, Rules()))["fail_daily"] > 0


def test_time_limited_converges_to_unlimited():
    ch = chain.build_chain(Edge(5 / 12, 2.28, 0.01), Rules())
    assert chain.within_days(ch, 2000)["pass"] == pytest.approx(chain.solve(ch)["pass"], abs=1e-9)
    assert chain.within_days(ch, 5)["pass"] < chain.solve(ch)["pass"]


def test_guard_that_blocks_every_trade_is_reported():
    with pytest.raises(ValueError):
        chain.solve(chain.build_chain(Edge(0.4, 2.0, 0.05), Rules(stop_before_daily_breach=True)))


def test_lattice_rounding():
    lat = make_lattice(Edge(0.4, 2.28, 0.01), Rules())
    assert (lat.win_units, lat.loss_units) == (57, 25)
    assert lat.unit * lat.pass_at == pytest.approx(0.10)
    assert lat.unit * lat.fail_at == pytest.approx(-0.10)


@pytest.mark.parametrize("rules", [Rules(), Rules(stop_before_daily_breach=True)])
def test_simulation_agrees_with_chain(rules):
    edge = Edge(0.4, 2.28, 0.02)
    exact = chain.solve(chain.build_chain(edge, rules))
    mc = simulate(edge, rules, 100_000, np.random.default_rng(0))
    for outcome in chain.OUTCOMES:
        se = np.sqrt(exact[outcome] * (1 - exact[outcome]) / 100_000)
        assert abs(mc[outcome] - exact[outcome]) < 4 * se + 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Edge(1.0, 2.0, 0.01)
    with pytest.raises(ValueError):
        Rules(trades_per_day=0)
    with pytest.raises(ValueError):
        ruin.pass_probability(0.5, 11, 10)


# --- Kelly -------------------------------------------------------------------------

@pytest.mark.parametrize("p,b", [(5 / 12, 2.28), (0.55, 1.0), (0.68, 2.28)])
def test_kelly_fraction_maximises_growth(p, b):
    best = optimize.minimize_scalar(lambda f: -kelly.growth(f, p, b), bounds=(0, 0.99), method="bounded")
    assert kelly.kelly_fraction(p, b) == pytest.approx(best.x, abs=1e-5)


def test_kelly_values_for_the_record():
    assert kelly.kelly_fraction(5 / 12, 2.28) == pytest.approx(0.1608, abs=1e-4)
    assert kelly.kelly_fraction(0.19, 2.28) < 0


def test_twice_kelly_growth_is_about_zero_for_small_edges():
    p, b = 0.52, 1.0
    f = kelly.kelly_fraction(p, b)
    assert abs(kelly.growth(2 * f, p, b)) < 0.05 * kelly.growth(f, p, b)


def test_growth_is_minus_infinity_at_full_stake():
    assert kelly.growth(1.0, 0.6, 2.0) == -np.inf


def test_plug_in_with_huge_sample_recovers_full_growth():
    p, b = 5 / 12, 2.28
    g_star = kelly.growth(kelly.kelly_fraction(p, b), p, b)
    assert kelly.plug_in_growth(p, b, 100_000, 1.0)["expected_growth"] == pytest.approx(g_star, rel=1e-3)


def test_drawdown_probability_edges():
    rng = np.random.default_rng(1)
    assert kelly.drawdown_probability(0.1, 0.5, 2.0, 0.5, 200, 2000, rng) <= 1
    # one loss at a 60% stake is already a 60% drawdown
    assert kelly.drawdown_probability(0.6, 0.01, 2.0, 0.5, 5, 2000, rng) > 0.99


def test_inexact_payoff_is_refused():
    with pytest.raises(ValueError):
        make_lattice(Edge(0.4, 1.99999, 0.02), Rules())
    assert make_lattice(Edge(0.4, 2.2837, 0.01), Rules(), max_denominator=10_000).win_units == 22837


def test_precision_near_a_fair_game():
    assert ruin.expected_steps(0.5 + 1e-9, 50, 100) == pytest.approx(2500, rel=1e-6)
    assert ruin.expected_steps(0.5 + 1e-4, 50, 100) == pytest.approx(2500, rel=1e-3)
