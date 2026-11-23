# Research error: design

Status: **approved (2026-11-22). Step 1 built (order book, see "Step 1 result"); steps 2-3 not started.**

## Problem

An estimate is the truth plus one error fixed for the whole market. So it moves exactly with the truth
whatever it cost, and a trader that remembers its first estimate can remove much of the error (all of
it in the current market, where every event opens at 50%). Paying for research must buy both the level
and the moves, and nothing a trader can work out from its own past may improve on what it bought.

## Approaches

| approach | error removed from memory, at tick 5 / 20 / 100 / 180 | verdict |
|---|---|---|
| today: fixed error | 36% / 35% / 29% / 11% | broken |
| error drifts, half-life 20 ticks | 31% / 16% / 2% / 0% | still leaks; drifting faster can leak more (27% at tick 20 from history alone at 1 tick) |
| **noisy looks, best guess (proposed)** | 0% / 0% / 0% / 0% | leak-free by construction |

Share of squared error removed by a fitted attacker that sees the trader's whole history, the public
opening and the news times. Tested out of sample on 30,000 markets; a throwaway test, with no trading.

## The rule

1. **A look.** Each tick a researcher sees the hidden lead as it was 2 ticks earlier, plus fresh normal
   noise. Half the noise variance is shared by everyone in the market, as now.
2. **The estimate.** The exact chance of YES given all its looks so far, the public opening and the
   public news times. Given the news times the lead is linear and normal, so a Kalman filter is exact:
   `p = sum over k of Binom(k; left, 0.02) * Phi(m / sqrt(P + left + 25k))`, with m and P the filter's
   mean and variance of the lead now.
3. **The price of research.** The look's s.d. is `3.73 * (0.25 / spend) * exp(2z)` lead points. 3.73
   gives a $0.25 forecaster today's average error (0.057). The form keeps the approved cost curve and
   the effect of luck z. No spend still means no research trading.
4. **The public opening** becomes the exact chance given a public look at the start (s.d. 3.07 points,
   which keeps its 0.3 of log-odds noise). Today's is the truth shifted by noise, which isn't a best
   guess.
5. **News traders** get one look at a jump, at once, with the same noise. They shrink it by how
   reliable it is and apply it to the level implied by the last price. They never see the true change.
6. **Unchanged:** the truth, scoring, insiders, retail and makers. Makers get no research in this change.

## What research buys

| spend a market | error today | error proposed | share of the truth's 10-tick moves seen: today | proposed | jump learned at once |
|---|---|---|---|---|---|
| none | - | 0.246 | - | 0% | - |
| $0.025 | 0.132 | 0.140 | 67% | 3% | 2% |
| $0.0625 | 0.091 | 0.096 | 68% | 11% | 10% |
| $0.25 | 0.057 | 0.057 | 68% | 40% | 64% |
| $1 | 0.044 | 0.043 | 68% | 62% | 97% |
| $2.50 | 0.041 | 0.040 | 68% | 66% | 99% |

Error is the root mean square distance from the true probability over all ticks.

## What it changes, knowingly

- Forecasters are better early (error 0.048 against 0.059 in the first 10 ticks) and worse late
  (0.090 against 0.067 in the last 20), since the same doubt about the lead matters more near the end.
- A crowd of 15 is slightly less accurate (0.053 against 0.050).
- Either can move maker profit and the price gap in either direction. No economics was run while
  choosing this, so that it couldn't be tuned to a result.
- It assumes a researcher knows how good its own research is.
- A trader can still improve on its research with prices and other wallets. That is real information.

## Checks, written before the code

1. The fitted attacker removes under 2% of squared error at 5, 20, 100 and 180 ticks, with a spread
   start and with a known 50% start.
2. Estimates and the opening match outcomes within 4 standard errors at 200, 100, 10 and 1 ticks left.
3. No research equals the public view. Error falls, and moves seen rise, at every step up in spend.
4. No future information: changing the lead from tick s changes no forecaster before s + 2, and no
   news trader before the jump has landed.
5. YES and NO mirror exactly, expected profit equals real profit, money balances, and the order book
   stays under 5x slower (the filters add an estimated 0.8 s a generation before optimising).
6. Planted bugs (no delay, fixed error, noise left out of P, unshrunk jumps) each fail a check.

## Step 1 result

Built in `arena/events.py` (`chance`, the opening) and `arena/ecology.py` (`estimates`, `jump_move`).

- All checks pass: 18 event, 17 book and 36 role checks, 7 of them new.
- Six planted bugs were each caught: no delay, a fixed error, noise left out of the doubt, unshrunk
  jumps, news traders reading the true move, and unshared noise.
- The first planting found that no check caught the last two, so two checks were added.
- One check was corrected: the opening is 0.3 of log-odds off on 10-90c events, where it was set; it is
  0.32 over all events.
- The current market's 36 reference results are unchanged.
- Speed: 1.39x the current market (gate limit 5x; it was 0.69x).

Not a gate (gate 2's setting, 6 seeds, 37000-37005, a market):

| | before | after |
|---|---|---|
| makers | -34.0 +- 2.5 | -16.9 +- 4.0 |
| news traders | +36.3 +- 2.0 | +25.2 +- 1.3 |
| forecasters | +93.0 +- 5.2 | +87.8 +- 5.1 |
| retail | -76.3 +- 0.7 | -79.7 +- 1.5 |
| price gap | 0.078 | 0.083 |

## Steps, each needing its own yes

1. **Order book:** build the rule test-first, rerun the checks of stages 1-3, stop and report
   (about an hour). No search and no gate.
2. **Decide on gate 3:** whether makers need research at all, and what to do about retail's limit and
   sizes, cartels and time left (the audit's open list). Then seal one gate on unused seeds.
3. **Current market:** the same rule with a known 50% start and the exact probability, which also
   fixes the README's other known issue. Forecasters would then see news times, and the Kalman trader
   needs a new job, since there would be no fixed error to learn. Phases 2-4 rerun (about 8 hours).
