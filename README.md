# Prediction Arena

A population of trading bots competing on simulated prediction markets. Bots trade, go bust and
reproduce; reinforcement-learning and Bayesian agents are dropped in to see whether they can keep
an edge against a population that evolves to exploit them.

## The market

- **Truth.** Each market is a binary event. A hidden "score lead" does a random walk with rare news
  jumps, and the event happens if the lead is positive at the end. The true probability is the
  chance the lead holds given the time left: `p = Phi(lead / sqrt(var * ticks_left))`, so it drifts
  toward 0 or 1 as the market nears resolution.
- **Market maker.** An LMSR (logarithmic market scoring rule) market maker fills the net order of
  all bots each tick, batch-auction style, so everyone pays the same average price. Its worst-case
  loss per market is capped at `b * ln 2`.
- **Speed.** A run of 100 markets with 51 bots takes ~0.03 s (everything is vectorised in numpy).

## Run it

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # macOS/Linux: .venv/bin/pip
.venv/Scripts/python market.py
```

## Phase 1 results

50 random "punters" plus one oracle bot that sees the true probability, over 100 markets:

| | per market |
|---|---|
| Oracle profit | +35.1 |
| Punter profit | -0.5 each |
| Market maker P&L | -11.3 (worst -34.4, cap -34.7) |
| Price-to-truth gap | 0.108 with oracle vs 0.309 punters only |
| Calibration of the true probability | worst bin off by 1.7% (20k markets) |

![Phase 1: price vs true probability](phase1.png)

The oracle pulls the price toward the truth until it hits its position limit; after that, punter
noise drags the price away (market 1 and market 2). One informed trader with limited capital can't
hold the price on its own.

## Roadmap

1. ~~Market core~~ (done)
2. Evolving bot population: punters, momentum, contrarian, forecasters and insiders, with selection and mutation
3. Measurement: calibration, favourite-longshot bias, overreaction to news, price discovery speed
4. Learning agents: particle-filter Bayesian trader, PPO agent, edge-decay experiment
5. Stress test: plug in your own bot and see how fast an adaptive population erodes its edge
6. Animated arena
