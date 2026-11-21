"""Phase 2: an evolving population of trading bots.

100 strategy bots evolve; 50 punters are fixed (they would otherwise die out and take everyone's
edge with them). Each generation is 400 fresh markets; fitness is total profit.

Run `python -m arena.evolve` for the Phase 2 run, check and chart.
"""
import argparse
import time
from collections import defaultdict

import numpy as np

from arena import FIGURES, RUNS
from arena.market import lo, simulate
from arena.measure import SCALARS, dashboard, measure, perfect

TREND, FORECAST = 0, 1
# Insiders (N_INS) are off: with even 2 of them they took most of the profit and starved the
# ecology. They come back later as a stress-test scenario.
N_EVO, N_PUNT, N_INS = 100, 50, 0
N_MARKETS = 400     # at 100, a bot's fitness was ~half luck (rank agreement 0.5 vs 0.91 at 400)
PUNTER_SIZE = 5.0   # punters' random trade size: the "food" the strategy bots compete for
INSIDER_LEAD = 10  # insiders see the true probability this many ticks ahead
INSIDER_SIZE = 200  # insider holds this many shares per unit of log-odds mispricing
INFO_COST = 0.05    # forecaster fee per market = INFO_COST / sigma**2 (accuracy isn't free)
N_DIE = 20          # bottom 20% replaced each generation
TOURNAMENT = 3
SWITCH_PROB = 0.05  # chance a child switches between trend follower and forecaster
ERR_CORR = 0.5      # share of forecaster error variance that's common to everyone in a market
                    # (same data, same models); with 0, 50 independent errors average out and
                    # the market is unrealistically close to perfect
FEE = 0.0           # per share traded; a 1c fee (`python -m arena.evolve --fee 0.01`) left forecasters ~zero profit
ALPHA_MAX = 100     # size cap; uncapped, forecaster size drifted from 13 to ~1000 and switched
                    # giants turned into momentum bots that caused fake bubbles


def random_population(rng):
    kind = rng.integers(0, 2, N_EVO)
    # Trend bots: + momentum, - contrarian. They send market orders that all land in the same
    # batch, so they start small; at size 20 their combined orders whipped the price every tick.
    alpha = np.where(kind == FORECAST, np.abs(rng.normal(0, 20, N_EVO)), rng.normal(0, 2, N_EVO))
    return dict(kind=kind, alpha=alpha,
                lag=rng.integers(1, 31, N_EVO),                          # trend lookback, ticks
                sigma=np.exp(rng.uniform(np.log(0.1), np.log(2), N_EVO)))  # forecaster error, log-odds


def make_orders(pop, err):
    """Order function for simulate(): evolving bots, then punters, then insiders.

    Strategy bots and insiders hold a target position and trade the difference:
    trend bots target alpha x (recent log-odds move) as a market order; forecasters and insiders
    target size x (estimate - clearing log-odds), so they never chase the price past their estimate.
    """
    is_trend = pop["kind"] == TREND
    ev, ins = slice(0, N_EVO), slice(N_EVO + N_PUNT, None)

    def fn(t, price_hist, truth, shares, rng):
        now = lo(price_hist[-1])                                  # (M,)
        past = lo(price_hist[np.maximum(t - pop["lag"], 0)]).T    # (M, N_EVO), each bot's own lookback
        d, a, e = np.zeros(shares.shape), np.zeros(shares.shape), np.zeros(shares.shape)
        d[:, ev] = np.where(is_trend, pop["alpha"] * (now[:, None] - past), 0) - shares[:, ev]
        a[:, ev] = np.where(is_trend, 0, np.abs(pop["alpha"]))
        e[:, ev] = lo(truth[t])[:, None] + err
        d[:, N_EVO:N_EVO + N_PUNT] = PUNTER_SIZE * rng.standard_normal((len(now), N_PUNT))
        d[:, ins] = -shares[:, ins]
        a[:, ins] = INSIDER_SIZE
        e[:, ins] = lo(truth[min(t + INSIDER_LEAD, len(truth) - 1)])[:, None]
        return d, a, e
    return fn


def labels(pop):
    """0 momentum, 1 contrarian, 2 forecaster."""
    return np.where(pop["kind"] == FORECAST, 2, np.where(pop["alpha"] >= 0, 0, 1))


def next_generation(pop, fitness, rng):
    order = np.argsort(fitness)
    dead, alive = order[:N_DIE], order[N_DIE:]
    picks = rng.choice(alive, (N_DIE, TOURNAMENT))
    parents = picks[np.arange(N_DIE), np.argmax(fitness[picks], 1)]
    pop = {k: v.copy() for k, v in pop.items()}
    for k in pop:
        pop[k][dead] = pop[k][parents]
    # Mutate the children only.
    pop["alpha"][dead] = np.clip(pop["alpha"][dead] + rng.normal(0, 0.2 * np.abs(pop["alpha"][dead]) + 1),
                                 -ALPHA_MAX, ALPHA_MAX)
    pop["lag"][dead] = np.clip(pop["lag"][dead] + rng.integers(-3, 4, N_DIE), 1, 50)
    pop["sigma"][dead] = np.clip(pop["sigma"][dead] * np.exp(rng.normal(0, 0.2, N_DIE)), 0.02, 5)
    flip = dead[rng.random(N_DIE) < SWITCH_PROB]
    pop["kind"][flip] = 1 - pop["kind"][flip]
    return pop


def drop_in(pop, make_agent, n_markets, seed, sigma=None, size=None):
    """Run an evolved population with one extra trader in the last column, and score it.

    make_agent(base_fn, err, sigma, size) returns the full order function. err (M, 1) is an
    estimate error like a forecaster's (shared part in common with the population), for the median
    forecaster's accuracy sigma; size is the median forecaster's size (either can be fixed by
    passing it). Agents that need neither ignore them. Returns the trader's expected profit per
    market and the simulation (with r["trades"], its trades per tick): each trade is
    valued at the true probability when it was made instead of at the final 0/1 outcome, which has
    the same average but removes the luck of how each market resolved (most of the noise).
    """
    fc = pop["kind"] == FORECAST
    sigma = np.median(pop["sigma"][fc]) if sigma is None else sigma
    size = np.median(np.abs(pop["alpha"][fc])) if size is None else size
    rng = np.random.default_rng(seed)
    z = np.sqrt(ERR_CORR) * rng.standard_normal((n_markets, 1)) \
        + np.sqrt(1 - ERR_CORR) * rng.standard_normal((n_markets, N_EVO + 1))
    fn = make_agent(make_orders(pop, z[:, :N_EVO] * pop["sigma"]), z[:, N_EVO:] * sigma, sigma, size)
    pos = []

    def recorded(t, price_hist, truth, shares, rng):
        pos.append(shares[:, -1].copy())  # its position before this tick's trade
        return fn(t, price_hist, truth, shares, rng)
    r = simulate(recorded, N_EVO + N_PUNT + 1, n_markets=n_markets, seed=seed)
    trades = np.diff(np.vstack(pos + [r["shares"][:, -1]]), axis=0)  # (T, M)
    # Bookkeeping check (no fees here): valuing trades at the outcome gives the real profit.
    assert np.allclose((trades * (r["outcome"] - r["fills"])).sum(0), r["profit"][:, -1])
    r["trades"] = trades
    return (trades * (r["truth"][:-1] - r["fills"])).sum(0), r


def evolve(generations=300, seed=0, fee=FEE, progress=False):
    rng = np.random.default_rng(seed)
    pop = random_population(rng)
    hist = defaultdict(list)
    start = time.time()
    for g in range(generations):
        if progress and g and g % 50 == 0:
            el = time.time() - start
            print(f"gen {g}/{generations}  {el / 60:.0f} min elapsed, ~{el / g * (generations - g) / 60:.0f} min left", flush=True)
        # Each forecaster's fixed error per market: a shared part (same for everyone) plus its own part.
        z = np.sqrt(ERR_CORR) * rng.standard_normal((N_MARKETS, 1)) + np.sqrt(1 - ERR_CORR) * rng.standard_normal((N_MARKETS, N_EVO))
        err = z * pop["sigma"]
        r = simulate(make_orders(pop, err), N_EVO + N_PUNT + N_INS, n_markets=N_MARKETS, fee=fee, seed=[seed, g])
        m, ref = measure(r), measure(perfect(r))  # Phase 3 measures, plus the perfect-bot benchmark
        for k in SCALARS + ["news_response"]:
            hist[k].append(m[k])
            hist["ref_" + k].append(ref[k])
        profit = r["profit"].sum(0)
        fitness = profit[:N_EVO] - (pop["kind"] == FORECAST) * N_MARKETS * INFO_COST / pop["sigma"] ** 2

        lab = labels(pop)
        hist["share"].append(np.bincount(lab, minlength=3) / N_EVO)
        hist["fitness"].append([fitness[lab == k].mean() / N_MARKETS if (lab == k).any() else np.nan for k in range(3)])
        hist["alpha"].append([np.median(np.abs(pop["alpha"][lab == k])) if (lab == k).any() else np.nan for k in range(3)])
        hist["lag"].append(np.median(pop["lag"][lab < 2]) if (lab < 2).any() else np.nan)
        hist["sigma"].append(np.median(pop["sigma"][lab == 2]) if (lab == 2).any() else np.nan)
        hist["insider"].append(profit[N_EVO + N_PUNT:].mean() / N_MARKETS if N_INS else 0.0)
        hist["punter"].append(profit[N_EVO:N_EVO + N_PUNT].mean() / N_MARKETS)

        pop = next_generation(pop, fitness, rng)
    return pop, {k: np.array(v) for k, v in hist.items()}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same seed, same result.
    a, b = evolve(5)[1], evolve(5)[1]
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a), "not deterministic"

    # The default run is the main setup; other fees/seeds/lengths get suffixed output files.
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=1000)
    args = ap.parse_args()
    tag = (f"_fee{args.fee:g}" if args.fee != FEE else "") + (f"_seed{args.seed}" if args.seed else "") \
        + (f"_g{args.gens}" if args.gens != 1000 else "")
    pop, h = evolve(args.gens, seed=args.seed, fee=args.fee, progress=True)
    assert np.allclose(h["share"].sum(1), 1)
    np.savez(RUNS / f"phase2{tag}.npz", **h)  # per-generation history, for analysis
    np.savez(RUNS / f"pop{tag}.npz", **pop)    # final population, for dropping new agents into
    dashboard(h, FIGURES / f"phase3{tag}.png")

    names = ["momentum", "contrarian", "forecaster"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for k in range(3):
        print(f"{names[k]:>11}: share {h['share'][0, k]:.2f} -> {h['share'][-1, k]:.2f}, "
              f"profit/market (last 50 gens) {np.nanmean(h['fitness'][-50:, k]):6.2f}")
    print(f"forecaster error sigma: {h['sigma'][0]:.2f} -> {h['sigma'][-1]:.2f}")
    print(f"trend lookback:         {h['lag'][0]:.0f} -> {h['lag'][-1]:.0f}")
    print(f"price-truth gap:        {h['gap'][:20].mean():.3f} -> {h['gap'][-20:].mean():.3f}")
    print(f"insider / punter profit per market (last 50): {h['insider'][-50:].mean():.2f} / {h['punter'][-50:].mean():.2f}")
    print("Phase 2 checks passed")

    gens = np.arange(len(h["gap"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax[0, 0].stackplot(gens, h["share"].T, labels=names, colors=colors, alpha=0.8)
    ax[0, 0].set_title("population share")
    ax[0, 0].legend(loc="upper left", fontsize=8)
    for k in range(3):
        ax[0, 1].plot(gens, h["fitness"][:, k], color=colors[k], label=names[k])
    ax[0, 1].axhline(0, color="grey", lw=0.5)
    ax[0, 1].set_title("mean profit per market (after info cost)")
    ax[1, 0].plot(gens, h["sigma"], color="tab:blue", label="forecaster error sigma")
    ax[1, 0].plot(gens, h["lag"] / 10, color="tab:red", label="trend lookback / 10")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("evolved traits (median)")
    ax[1, 1].plot(gens, h["gap"], color="black")
    ax[1, 1].set_title("price-to-truth gap")
    for a_ in ax[1]:
        a_.set_xlabel("generation")
    fig.tight_layout()
    fig.savefig(FIGURES / f"phase2{tag}.png", dpi=120)
    print(f"charts saved to figures/phase2{tag}.png and figures/phase3{tag}.png; data in runs/")
