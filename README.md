# Prediction Arena

A population of trading bots competing on simulated prediction markets. Bots trade, go bust and
reproduce; Bayesian and reinforcement-learning agents are dropped in to see whether they can keep
an edge against a population that evolves around them.

## The market (`market.py`)

- **Truth.** Each market is a binary event. A hidden "score lead" does a random walk with rare news
  jumps, and the event happens if the lead is positive at the end. The true probability is the
  chance the lead holds given the time left: `p = Phi(lead / sqrt(var * ticks_left))`, so it drifts
  toward 0 or 1 as the market nears resolution.
- **Market maker.** An LMSR (logarithmic market scoring rule) market maker takes the net order of
  all bots each tick. Its worst-case loss per market is capped at `b * ln 2`.
- **Uniform-price batch auction with limit orders.** Each tick, all orders clear at one price and
  everyone trades at that price. Price-sensitive bots say how many shares they'd hold at any price,
  and the clearing price is solved exactly. Without batching, bots reacting to the old price all
  piled in together and the price swung between ~0 and ~1 every tick.
- **Full collateral, as on real prediction markets.** A long position must be paid for in cash and a
  short must keep $1 per share in cash, fees included, so no bot can ever lose more than its budget.
  An early version without this let bots run up debt (one lost 240 on a budget of 100). A built-in
  check has reckless bots trade at random at fees of 0, 1c and 5c and confirms nobody loses more
  than they had.
- **Speed.** All vectorised in numpy. The clearing price takes ~5-7 evaluations of everyone's orders
  per tick (a guarded Newton/secant search that gives the same answer as 30 rounds of bisection),
  so one generation (400 markets x 200 ticks, 150 bots) takes ~1.5-4 s on a laptop.

**A loophole found by the RL agent.** Earlier versions filled everyone at the market maker's
*average* price over the batch, while bots chose their orders at the *clearing* price. Whoever
traded with the net flow got a better price than they'd planned for, so a big trader could push the
clearing price past the truth and buy cheaply from the bots selling into it. The PPO agent (Phase 4)
found this within 2 training iterations and earned 17x a forecaster while prices barely moved. The
market now uses a single uniform price, as frequent batch auctions do, and every result below was
re-run on the fixed market. The fix changed the evolved ecology (Phase 2).

## Phase 1: does the market work?

50 random "punters" plus one test oracle that sees the true probability, over 100 markets. The
oracle is a deliberately unrealistic test instrument: if even a bot that knows the truth couldn't
pull the price to it, the market would be broken. It isn't used after Phase 1.

| | per market |
|---|---|
| Oracle profit | +14.0 |
| Punter profit | -0.4 each |
| Market maker P&L | +5.0 (worst -28.4, cap -34.7) |
| Price-to-truth gap | 0.116 with oracle vs 0.317 punters only |
| Calibration of the true probability | worst bin off by 1.7% (20k markets) |

![Phase 1: price vs true probability](phase1.png)

## Phase 2: an evolving population (`evolve.py`)

- **Who trades:** 100 evolving strategy bots, plus 50 punters who don't evolve (they lose money, so
  evolution would wipe them out, and with them every other bot's profit).
- **Strategies (the genome):** trend followers hold `alpha x (log-odds move over the last L ticks)`
  (alpha > 0 momentum, < 0 contrarian); forecasters each have their own noisy estimate of the truth
  (error size sigma) and hold `alpha x (estimate - price)`. Half of every forecaster's error is shared
  with all the others in that market (same data, same models), so the crowd can't simply average its
  mistakes away. Accuracy costs `0.05 / sigma^2` per market, so forecasters choose how much to buy.
- **Each generation:** 400 fresh markets; fitness is total profit; the bottom 20 die and are replaced
  by mutated copies of tournament winners (best of 3); 5% of children switch strategy type; bot size
  is capped at 100. Results are from 3 seeds x 1000 generations.

Things that had to be fixed before the results meant anything (each mirrors a real market effect):

1. **Selection was half luck.** Over 100 markets, a bot's rank on one set of markets agreed with its
   rank on another at only 0.50; at 400 markets it's 0.91.
2. **Insiders starved everyone.** Two bots seeing 10 ticks ahead took most of the profit and killed
   the forecasters (adverse selection). They're out for now and return as a stress-test scenario.
3. **Too little "food".** The only free money is what punters lose; with small punters, the best
   strategy was to barely trade.
4. **Unbounded size made fake bubbles.** Bot size drifted from 13 to ~1000. Size is now capped.
5. **Independent errors made the market unrealistically perfect.** 50 forecasters with independent
   errors average them out almost completely; real traders' errors are correlated.
6. **Bots could lose money they didn't have** (no collateral; see above).
7. **Average-price fills were a free option** for whoever traded with the crowd (see above).

### Result: boom and bust around a small core of forecasters

All three seeds end up in the same kind of ecology (averages over generations 500-999):

![Phase 2, seed 0](phase2.png)

- **Forecasters are ~23-24% of the population**, at size ~90 (near the cap) and accuracy
  sigma = 0.30-0.31. They're the only bots that make money: +0.89, +1.03 and +0.02 per market after
  paying for accuracy (seed 2 has rare crash generations, the worst at -12.4 per market).
- **The other ~76% are trend followers that have evolved to almost stop trading** (size 0.1-1.5) and
  earn about 0. (Their "momentum" or "contrarian" label in the charts is just the sign of a
  near-zero size.) They survive because selection removes the 20 worst bots each generation: a bot
  earning exactly zero is never among them, while forecasters' profits are noisy enough that the
  unlucky ones often are. Truncation selection punishes risk.
- **Boom and bust.** The fewer forecasters there are, the more each one earns (correlation -0.35 to
  -0.57 between their share and their profit). When they get scarce the price drifts from the truth
  (the gap spikes to 0.19-0.42 in the worst generations), the survivors make big profits (up to +34
  per market in one generation), and they multiply again.
- Punters lose ~0.4 per market.

Before the loophole was fixed, one seed grew a momentum herd that caused bubbles. That was the
loophole at work: trading with the crowd was subsidised. With a uniform price, market orders pay
their full price impact, trend-following doesn't pay, and no seed forms a herd.

## Phase 3: measuring the market (`measure.py`)

Because the simulator knows the true probability, every measure compares the price with the truth
directly, and with a "perfect bots" benchmark on the same markets. Each measure is checked in
`python measure.py` on fake markets with a known answer (perfect, timid, slow).

![Phase 3: market quality over generations, seed 0](phase3.png)

Averages over generations 500-999:

| | perfect bots | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| price-to-truth gap | 0.020 | 0.045 | 0.046 | 0.044 |
| calibration error | 0.000 | 0.011 | 0.011 | 0.010 |
| share of news priced in after 1 tick | 1.00 | 0.96 | 0.96 | 0.96 |
| favourite-longshot slope (1 = unbiased) | 1.01 | 1.02 | 1.02 | 1.02 |
| predictability (lag-1 autocorr of returns) | 0.04 | 0.05 | 0.03 | 0.05 |

The evolved market is about twice as far from the truth as perfect bots, but well calibrated and
unbiased.

**What didn't emerge:** fat tails and volatility clustering match the perfect-bot benchmark, so they
come from the news jumps built into the truth, not from the bots. There's no favourite-longshot bias
either: punters here trade at random, while real punters overbet longshots, and adding that
preference is the obvious next experiment.

### Fees: even 0.2c makes prices noticeably worse

Same setup with a fee per share traded, seeds 0 and 2, 300 generations each, averages over
generations 150-299 and over the two seeds (they tell the same story):

| | no fee | 0.2c fee | 1c fee |
|---|---|---|---|
| price-to-truth gap | 0.044 | 0.074 | 0.093 |
| calibration error | 0.009 | 0.032 | 0.040 |
| share of news priced in after 1 tick | 0.97 | 0.90 | 0.89 |
| forecasters (share, size, error sigma) | 23%, ~88, 0.32 | 20%, 61-83, 0.33 | 98%, 2-3, 1.0 |
| forecaster profit per market (after info cost) | +0.53 | +2.4 | ~0 |
| punter loss per market (mostly fees at 1c) | 0.40 | 2.6 | 8.8 |

- **0.2c:** fewer or smaller forecasters, each earning more (less competition), and prices 70% less
  accurate.
- **1c:** a forecaster's edge per share is smaller than the fee, so evolution shrinks forecasters to
  size 2-3 and lets them stop paying for accuracy. Nobody corrects mispricings, so the gap doubles,
  and the punters' losses go to the exchange, not to traders.

![1c fee, seed 0](phase2_fee0.01_g300.png)

## Phase 4: learning agents

Every agent below is dropped into the final evolved population of each seed (generation 999) and
compared with an identical forecaster on the same 2000 markets.

**Measuring profit.** Each market resolves to 0 or 1, and a 90-share position turns that coin flip
into about +-12 of luck per market, which buried every difference. Because the simulator knows the
true probability at every tick, each trade is valued at the truth when it was made instead. That
gives the same average with ~10x smaller error bars (the bookkeeping is checked against real profit
in every run).

### A Bayesian trader (`kalman.py`)

The Kalman trader gets exactly what a forecaster gets: its own estimate of the true probability
(with an error fixed for the market, half of it shared with everyone) and the market price. It also
learns its own error: every tick it compares its last estimate with the price that cleared on it,
and a Kalman filter turns those comparisons into a best guess of its error. One setting, tau, says
how noisy it thinks the price is: tau = inf is a plain forecaster, small tau trusts the price.

![Kalman trader vs forecaster](phase4_kalman.png)

| per market (seeds 0, 1, 2) | forecaster | Kalman (tau 0.1-0.3) |
|---|---|---|
| expected profit | +0.51, +0.52, +0.49 | +0.24 to +0.26 (about half) |
| risk (s.d. of profit) | 11.7, 8.9, 7.9 | 4.0, 3.4, 3.0 |
| profit per unit of risk | 0.044, 0.058, 0.063 | 0.064, 0.075, 0.087 (1.3-1.5x) |

- **It does learn its error.** Its estimate ends up 25% closer to the truth (seed 0), and what it learns is
  the private half of its error (correlation 0.68, close to sqrt(0.5)); the shared half can't be
  learned from the price, because the price carries the same mistake.
- **But it learns some of the market's mistakes as its own.** Punters' accumulated positions push
  the price away from the truth, and a forecaster profits by leaning against that. The Kalman trader
  can't tell that distortion from its own error: the mistake in its learned error correlates -0.36
  with the punters' net position (its true error: -0.05). So it "corrects" itself into agreement
  with the distorted price, holds a third of the position, and earns half as much, mostly late in
  the market when the distortion has built up.

### Visible wallets (`wallets.py`)

On Polymarket every position is public under a pseudonymous wallet. The simulator works the same
way: every bot can see every wallet's position after the last tick, and a public leaderboard of past
profit, but never a wallet's strategy. Does that ledger hold exploitable information? A wallet
watcher copies (or fades) the average position of the top (or bottom) wallets on a leaderboard
built from 2000 earlier markets:

| watcher (2000 test markets, seeds 0, 1, 2) | expected profit per market | risk |
|---|---|---|
| copy the top wallet by **profit** (always a lucky punter) | -0.46, -0.50, -0.53 | 24 |
| copy the top wallet by **profit per unit of risk** (a forecaster) | +0.81, +0.76, +0.39 | 5-11 |
| copy the top 10 by profit per unit of risk (all forecasters) | +0.63, +0.53, +0.58 | 5-7 |
| fade the bottom 10 by profit (mostly punters) | +0.28, +0.30, +0.38 | 7-8 |

- **Copying luck loses.** Punters hold big random positions, so over 2000 markets the luckiest one
  out-earns every forecaster, and copying it loses money. Ranking by profit per unit of risk finds
  the skilled wallets.
- **Copiers free-ride.** A watcher with no information of its own, a tick behind, earns as much as a
  forecaster (often more) without paying for accuracy: transparency lets copiers live off informed
  traders.
- **Fading the punters pays**: their accumulated positions keep the price slightly off.

### A reinforcement-learning trader (`ppo.py`)

- **Same information** as the forecaster and Kalman trader: its private estimate, the price, its own
  position, and the public ledger (positions and last trades of the top and bottom 10 wallets).
- **Its actions** each tick: a shift to its valuation and a multiplier on its size (capped at 100,
  like every bot). With both at 0 it is exactly a forecaster, and training starts there.
- **Reward:** each trade valued at the true probability when it was made (the network never sees
  the truth). PPO as in CleanRL; each iteration is 400 markets against the evolved population;
  150 iterations (~10 min on a laptop CPU).
- **Checks:** with its last layer zeroed it reproduces the forecaster's profit exactly; the first PPO
  update checks that the recorded actions and log-probabilities line up.

![PPO training](phase4_ppo.png)

| per market (seeds 0, 1, 2) | expected profit | risk | profit per unit of risk |
|---|---|---|---|
| forecaster (size ~90) | +0.51, +0.52, +0.49 | 7.9-11.7 | 0.044-0.063 |
| best plain forecaster with no size cap (size ~400) | +0.99, +1.33, +1.43 | 33-34 | ~0.03-0.04 |
| PPO | **+1.92, +1.90, +2.12** | 17.7-23.7 | **0.081-0.120** |

- **PPO earns ~4x the forecaster**, and still more than any plain forecaster with the size cap
  removed, at lower risk than that one. Trading bigger explains a third to 60% of its edge.
- **What it learned** (its actions regressed on its inputs): it raises its valuation when it already
  holds a position (which lets it trade bigger than the cap), subtracts its average past
  disagreement with the price (a Kalman-style correction), leans toward what the top wallets hold,
  and trades bigger late in the market.
- Training was still improving at 150 iterations.

### Edge decay: does the population adapt to the agent? (`decay.py`)

The trained agent joins its evolving population (it never reproduces or dies) for 200 more
generations; the population evolves on profit as usual, so whatever the agent takes becomes
selection pressure. Three versions: a plain forecaster (control), the agent frozen, and the agent
still learning (one PPO update per generation).

![Edge decay](phase4_decay.png)

| expected profit per market, first -> last 25 generations | forecaster | PPO frozen | PPO learning |
|---|---|---|---|
| seed 0 | +1.16 -> +1.03 | +2.70 -> +2.76 | +2.39 -> +2.17 |
| seed 1 | +0.99 -> +0.98 | +3.03 -> +2.22 | +2.37 -> +2.81 |
| seed 2 | +0.76 -> +0.65 | +2.09 -> +2.02 | +1.94 -> +2.03 |

- **The edge doesn't decay.** The frozen agent keeps earning 2-3x the forecaster for 200
  generations, so continued learning has nothing to fix.
- **The population barely notices it.** Forecaster share and market accuracy are the same with or
  without the agent. No evolving species can single out one wallet, so the agent's cut never becomes
  something selection can target; all three traders simply rise and fall with the forecasters'
  boom-and-bust cycle.
- That's the motivation for the next experiment: an evolving "wallet watcher" species that can copy
  or fade specific wallets, giving the population a way to hunt the agent.

## Run it

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # macOS/Linux: .venv/bin/pip
.venv/Scripts/python market.py      # Phase 1 checks + chart
.venv/Scripts/python measure.py     # Phase 3 measure checks
.venv/Scripts/python evolve.py      # Phase 2+3, 1000 generations (~30-40 min); --seed, --fee, --gens
                                    # also saves the final population (pop*.npz) for Phase 4
.venv/Scripts/python kalman.py      # Kalman checks + experiment (~10 min)
.venv/Scripts/python wallets.py     # wallet checks + watcher experiment (~20 min)
.venv/Scripts/python ppo.py train pop.npz   # PPO checks + training (~10 min); then `ppo.py report`
.venv/Scripts/python decay.py run pop.npz   # edge decay (~30 min); then `decay.py report`
```

## Roadmap

1. ~~Market core~~
2. ~~Evolving bot population~~
3. ~~Measurement~~
4. ~~Learning agents: Kalman trader, visible wallets, PPO agent, edge decay~~; next, an evolving
   wallet-watcher species that can target the agent
5. Stress test: plug in your own bot and see how fast an adaptive population erodes its edge
6. Animated arena
