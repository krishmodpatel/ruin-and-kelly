# Development log

The project is mathematical: there is no market data, and the only input is a
self-reported trading record (5 wins in 12 trades, payoff ratio 2.28). Dates below are
when the work was done.

## 2026-10-30 — first version

Written with the help of an AI assistant (Claude) in one session, in a Linux
environment with Python 3.13.

**Decisions**
- Challenge rules: 10% target, 10% static overall limit, 5% daily limit from the
  start-of-day balance, 3 trades a day. Touching a limit counts as a breach; the
  account passes as soon as it reaches the target.
- Risk per trade is a share of the *initial* balance (as prop firms measure limits);
  the Kelly section uses a share of the *current* balance (compounding).
- Payoff ratios are rounded to a fraction with denominator at most 50 so P&L lives on
  a whole-number grid; 2.28 = 57/25 exactly.
- 200,000 simulated accounts per case; seed 2027.

**Problems found while building**
- With "skip trades that would breach the daily limit" and a 5% risk per trade, no
  trade is ever allowed, the chain never ends, and the linear system is singular. The
  solver now detects this and reports it; that row is left out of the results.
- The first check for this was wrong (it flagged ordinary states whose day cannot end
  in absorption) and was replaced by a check for states that stay put for certain.

**Verified**
- Unit tests pass, including checks that the closed form satisfies its recurrence,
  that the chain matches the closed form, that simulation matches the chain within
  four standard errors, and that f* maximises g numerically.
- `python -m ruin_kelly` runs in about one minute; the paper compiles with latexmk.

**Changes after an independent review (same day)**

A separate AI review pass, which had not seen the project being built, recomputed the
main numbers with its own code (exact fractions, one trade per step, no lattice) and
they all matched. It reported no wrong results. Fixed:
- *Authorship note.* The paper's note now says plainly that the first version was
  produced by an AI assistant and reviewed by a second AI pass, not by a person.
- *Payoff rounding.* Rounding the payoff to a nearby fraction could silently change
  results (1.99999 rounded to 2 moved a pass probability from 0.70 to 0.72). Payoffs
  that are not exact fractions are now refused, with a test.
- *Plug-in Kelly.* The 99% stake cap is now stated: without it, full Kelly on an
  estimate has expected growth of minus infinity (a sample of all wins stakes 100%).
- *Overclaims in the paper.* "Bold play" is now described as consistent with Dubins
  and Savage's intuition rather than their theorem; the long-run optimality of Kelly is
  attributed to Breiman (1961); the daily limit binds from 1.67% risk, not "first at
  2%"; full Kelly is already reasonable by 100 trades, not "several hundred".
- *Smaller items.* Notation clashes removed (daily limit L_d, fundamental matrix M),
  second moment correctly named, existence and uniqueness arguments added to the
  proofs, precision near a fair game improved, the singular-state check documented.
