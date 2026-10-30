# Ruin, drawdowns and the Kelly criterion

How likely is a trader to pass a funded-account ("prop-firm") challenge before
breaching its loss limits, and how much should a trader risk per trade? This project
answers both questions with probability theory: a hand derivation of the gambler's
ruin, an exact Markov-chain model of a challenge with daily and overall loss limits,
and a derivation of the Kelly criterion. Every result is checked independently, by
simulation or numerical optimisation.

The full write-up, with proofs, is [`paper/ruin_kelly.pdf`](paper/ruin_kelly.pdf).
All numbers below come from [`results/RESULTS.md`](results/RESULTS.md).

## Model

Each trade wins with probability *p*. A loss costs a fixed share *r* of the initial
balance; a win pays *b × r*, where *b* is the payoff ratio (average win / average
loss). The challenge passes at +10% and fails at −10% overall or at −5% within one day
(measured from the day's starting balance). The trader makes 3 trades a day.

The example edge is a self-reported record of **5 wins in 12 trades with an average
win 2.28 times the average loss**. Its 95% interval for the true win rate is 19% to
68%, so every result that uses it is conditional on an uncertain input.

## Methods

**1. Gambler's ruin (equal wins and losses).** Conditioning on the first trade gives
the recurrence *P(i) = p·P(i+1) + q·P(i−1)*. Its characteristic equation
*pλ² − λ + q = 0* has roots 1 and *ρ = q/p*, which gives

    P(pass) = (1 − ρ^i) / (1 − ρ^N)        expected trades = (i − N·P(pass)) / (q − p)

where the account starts *i* steps above the loss limit and passes *N* steps above it.

**2. Absorbing Markov chain (unequal wins and losses, daily limit).** The payoff is
written as a fraction *w/m* (2.28 = 57/25), so P&L moves on a whole-number grid. One
step of the chain is one trading day; all 2³ win/loss sequences of a day are followed
trade by trade so that the daily limit is applied exactly. Pass and fail probabilities
solve *(I − Q)h = R*, and expected days solve *(I − Q)t = 1*, where *Q* holds the
transitions between non-absorbed states (the fundamental-matrix method).

**3. Monte Carlo simulation** of 200,000 accounts per case, using the payoff as given
(no grid), checks methods 1 and 2.

**4. Kelly criterion.** Risking a fraction *f* of the *current* balance, long-run growth
per trade is *g(f) = p·ln(1 + bf) + q·ln(1 − f)*. Setting *g′(f) = 0* gives
*f\* = p − q/b*; *g″ < 0* confirms it is the maximum.

**5. Kelly with an estimated win rate.** If *p* is estimated from *n* trades and
plugged into the formula, the expected true growth is an exact binomial sum. This shows
how much of the Kelly stake to bet when the estimate is noisy.

## Results

**The three methods agree.** For equal wins and losses the chain matches the closed
form to at least six decimal places, and the simulation agrees within sampling error
(for example, *p* = 0.45 at 1% risk: formula 0.118501, chain 0.118501, simulation
0.1189 ± 0.0007).

**Challenge outcomes at the record's edge** (exact, 3 trades a day):

| Risk per trade | Pass | Fail (overall) | Fail (daily) | Expected days | Pass within 60 days |
|---|---|---|---|---|---|
| 0.25% | 100.0% | 0.0% | 0.0% | 37.5 | 91.6% |
| 0.5% | 99.8% | 0.2% | 0.0% | 19.3 | 99.2% |
| 1% | 96.1% | 3.9% | 0.0% | 9.5 | 96.1% |
| 1.5% | 89.6% | 10.4% | 0.0% | 5.8 | 89.6% |
| 2% | 53.0% | 0.0% | 47.0% | 2.4 | 53.0% |
| 3% | 52.6% | 0.0% | 47.4% | 1.4 | 52.6% |

- **The daily limit takes over between 1.5% and 2% risk.** Three losses reach the 5%
  daily limit once risk is 1.67% or more; at 2% they cost 6%, and the pass probability
  falls from 90% (at 1.5%) to 53%. Skipping any trade
  whose loss would breach the daily limit lifts it back to 83%.
- **With a positive edge, smaller stakes pass more often but more slowly.** With a
  60-day limit, 0.5% risk does best of the sizes tested.
- **With a negative edge, larger stakes tend to pass more often**, in line with the
  intuition behind "bold play": at a 25% win rate, pass probability rises from 0.1% at
  0.25% risk to 25% at 5% risk, though not smoothly (15% at 2%).

**But the record cannot tell which case applies.** Across the 95% interval for the
true win rate, the pass probability at 1% risk runs from 2% to almost 100%; the
break-even win rate (30.5%) is inside the interval.

![Pass probability against win rate](results/pass_vs_win_rate.png)

![Pass probability by win rate and risk](results/heatmap_risk.png)

**Kelly.** For the record, *f\** = 16.1% of the balance per trade; across the win-rate
interval it runs from −16.1% (do not trade) to 54.0%. Half Kelly keeps 76% of the growth
of full Kelly; twice Kelly keeps 10%. Over 500 simulated trades, the balance fell 50%
below its peak on 24% of paths at quarter Kelly and on 99% at half Kelly.

![Kelly growth curves](results/kelly_growth.png)

**Estimated win rate.** Betting full Kelly on a win rate estimated from 12 trades gives
*negative* expected growth (−9% of what a trader who knew *p* would get, with stakes
capped at 99%; without the cap it is minus infinity, because 12 wins out of 12 would
mean staking everything). The best
multiple of the estimated stake is 0.5 with 12 trades, 0.85 with 100, and about 1 only
after 1,000.

![Best Kelly multiple against sample size](results/kelly_estimation_error.png)

## Reproducing

Requires Python 3.10 or later. Run commands from the repository folder.

```bash
git clone https://github.com/krishmodpatel/ruin-and-kelly.git
cd ruin-and-kelly
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m pytest                    # 30 unit tests
python -m ruin_kelly                # all experiments -> results/ (about 1 minute)
```

To rebuild the paper, open `paper/ruin_kelly.tex` in Overleaf (upload the `results`
PNGs alongside it) or run `latexmk -pdf ruin_kelly.tex` in the `paper` folder.
Simulations use fixed seeds, so results are identical on every run.

## Project layout

```
src/ruin_kelly/
  model.py        the trading model and its whole-number lattice
  ruin.py         closed-form gambler's ruin
  chain.py        absorbing Markov chain with a daily loss limit
  simulate.py     Monte Carlo check
  kelly.py        Kelly growth, the plug-in experiment, drawdowns
  experiments.py  every experiment in the write-up
  plots.py        figures
  __main__.py     runs everything and writes results/RESULTS.md
tests/            unit tests
paper/            LaTeX source and PDF
results/          tables (CSV), figures and RESULTS.md
```

## Limitations

- Trades are independent and every win and every loss has the same size. Real trades
  vary in size and cluster in time, which makes ruin more likely than shown.
- The 12-trade record is self-reported and unverified.
- Prop-firm rules vary: this model uses a static overall limit, a daily limit from the
  start-of-day balance, counts touching a limit as a breach, and passes the moment the
  target is reached. Trailing drawdowns or minimum trading days would change the numbers.
- The Kelly analysis ignores trading costs and treats the payoff ratio as known.

## References

- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*, Vol. 1, Ch. XIV.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35(4).
- Kemeny, J. G. and Snell, J. L. (1960). *Finite Markov Chains*.
- Dubins, L. E. and Savage, L. J. (1965). *How to Gamble If You Must*.
- MacLean, L. C., Thorp, E. O. and Ziemba, W. T. (eds.) (2011). *The Kelly Capital Growth Investment Criterion*.

## How this was built

The code, derivations, paper and documentation of this first version were produced
on 30 October 2026 by an AI assistant (Claude) at my request, checked by running the
tests and all experiments, and reviewed by a second, separate AI review pass. The
trading record used as input is my own. Work I do on it myself is recorded with its
date in [DEVLOG.md](DEVLOG.md).

## Licence

MIT. See [LICENSE](LICENSE).
