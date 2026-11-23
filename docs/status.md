# Where this stands (2026-11-23)

A note to pick the work up from. Details live in [order-book-design.md](order-book-design.md) and
[research-error-design.md](research-error-design.md).

## Done

1. **Phases 1-4** on the current (LMSR) market: evolution, hunting, PPO, edge decay. Results in the
   README.
2. **Order-book rebuild, stages 1-2:** `arena/events.py` (exact probability) and `arena/book.py`
   (matching engine) passed their gates.
3. **Stage 3 (`arena/ecology.py`, the trader roles):** failed both gates. The audit then found the
   second verdict overstated (settings compared at unequal retail) and fixed that, free research,
   by-market standard errors and maker profit before research.
4. **Research fix, step 1 (order book only):** research is now noisy looks at the hidden lead, and the
   estimate is the exact best guess from them, so nothing in a trader's own past improves on it.
   All checks pass (18 event, 17 book, 36 role); six planted bugs are each caught; the current
   market's reference results are unchanged.
5. **Retail measured from real trade data** (`arena/retail_data.py`), with the limits of what it means
   written down in the audit.

## To do, in order

1. **Step 2, decide what gate 3 needs** (audit's open list):
   - Do makers need research at all?
   - Retail's price limit and order sizes (fill-only data, a shallow book, accounts running out of
     money).
   - Maker cartels: cap the spread and require that undercutting doesn't pay.
   - Let makers act on the time left (public, and where they lose).
   - Then seal one gate on unused seeds.
2. **Smaller open items:** institutions fill only 4-7% of capital; log-odds stop at +-6; the blind gap
   benchmark is weak; capital, institution-share, fee and longshot sensitivities not run.
3. **Step 3, the current market:** the same research rule with a known 50% start and the exact
   probability. That fixes both README known issues. The Kalman trader needs a new job. Phases 2-4
   rerun, about 8 hours.
4. **Stages 4-5** of the order book: not started, and blocked on 1.

## Rules for this work

- Seeds already used: 1000s, 2000s, 5000-7999, 9000s, 10000-15001, 30000-37005. Gates use new ones.
- `python -m arena.evolve` runs 1000 generations and overwrites `runs/pop.npz`; don't run it just for
  its checks. `python -m arena.ecology search` overwrites `runs/stage3_setting.npz`.
- Checks: `python -m arena.events`, `python -m arena.book`, `python -m arena.ecology`.
- Data: Kalshi's API isn't used (its terms don't allow it). Recheck Polymarket's terms before any
  commercial use. Downloaded data stays in `data/`, which isn't committed.
- Not-a-gate numbers (gate 2's setting, seeds 37000-37005, a market): makers -16.9 +- 4.0, news
  traders +25.2, forecasters +87.8, retail -79.7, price gap 0.083.
