"""Phase 2: an evolving population of trading bots.

100 strategy bots evolve; 50 punters are fixed (they would otherwise die out and take everyone's
edge with them). Each generation is 400 fresh markets; fitness is total profit.

With --watchers a third species can evolve: the wallet watcher. Every wallet's positions are
public (as on Polymarket), and so is last generation's leaderboard. A watcher picks a spot on that
leaderboard and trades like a trend bot, but on the wallet in that spot instead of on the price:
it holds alpha x (that wallet's position now - its position lag ticks ago). alpha > 0 copies the
wallet, alpha < 0 fades it; a short lag follows its latest trades, a long one copies its whole
position.

Run `python -m arena.evolve` for the Phase 2 run, check and chart (`--watchers` for the run with
wallet watchers).
"""
import argparse
import time
from collections import defaultdict

import numpy as np

from arena import FIGURES, RUNS
from arena.market import lo, simulate
from arena.measure import SCALARS, dashboard, measure, perfect

TREND, FORECAST, WATCH = 0, 1, 2
NAMES = ["momentum", "contrarian", "forecaster", "copier", "fader"]  # what labels() returns
# Insiders (N_INS) are off: with even 2 of them they took most of the profit and starved the
# ecology. They come back later as a stress-test scenario.
N_EVO, N_PUNT, N_INS = 100, 50, 0
WALLETS = N_EVO + N_PUNT  # wallets on the leaderboard; a dropped-in trader is the next column
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
RANK_STEP = 0.05    # mutation s.d. of a watcher's leaderboard spot (0 = top, 1 = bottom)
WATCH_LAG_MAX = 200  # a watcher may look back a whole market (200 ticks), i.e. copy the whole position


def random_population(rng, watchers=False):
    kind = rng.integers(0, 3 if watchers else 2, N_EVO)
    # Trend bots: + momentum, - contrarian. They send market orders that all land in the same
    # batch, so they start small; at size 20 their combined orders whipped the price every tick.
    # Watchers start the same way: alpha ~1 holds about what the watched wallet holds.
    alpha = np.where(kind == FORECAST, np.abs(rng.normal(0, 20, N_EVO)), rng.normal(0, 2, N_EVO))
    pop = dict(kind=kind, alpha=alpha,
               lag=rng.integers(1, 31, N_EVO),                          # trend/watcher lookback, ticks
               sigma=np.exp(rng.uniform(np.log(0.1), np.log(2), N_EVO)))  # forecaster error, log-odds
    if watchers:
        pop.update(rank=rng.uniform(0, 1, N_EVO),      # leaderboard spot: 0 top, 1 bottom
                   board=rng.integers(0, 2, N_EVO),    # ranked by 0: profit, 1: profit per unit of risk
                   age=np.zeros(N_EVO, int),           # generations lived (0 = no record of its own yet)
                   target=np.full(N_EVO, -1))          # wallet watched this generation (-1: none yet)
    return pop


def targets(pop, profit):
    """The wallet each watcher watches next generation: the one in its spot on the leaderboard.

    profit (M, W): last generation's profit per market of every wallet (strategy bots, punters,
    then any dropped-in trader), all public on chain. A wallet is ranked only on a record of its
    own, so a strategy bot born this generation is left off (its column's record belongs to the bot
    it replaced); and no watcher watches itself. Returns (N_EVO,) wallet columns, -1 for non-watchers.

    With a "lock" gene (arena.decay --lock), a locked watcher keeps the wallet it already watches
    until that wallet's owner dies, wherever it moves on the leaderboard. Children inherit the wallet.
    """
    has_record = np.r_[pop["age"] > 0, np.ones(profit.shape[1] - N_EVO, bool)]
    boards = [profit.mean(0), profit.mean(0) / (profit.std(0) + 1e-9)]  # board 0 profit, 1 profit per risk
    ranked = [[c for c in np.argsort(-b, kind="stable") if has_record[c]] for b in boards]  # best first
    out = np.full(N_EVO, -1)
    for j in np.flatnonzero(pop["kind"] == WATCH):
        held = pop["target"][j]
        if "lock" in pop and pop["lock"][j] and held >= 0 and held != j and has_record[held]:
            out[j] = held
            continue
        others = [c for c in ranked[pop["board"][j]] if c != j]
        out[j] = others[round(pop["rank"][j] * (len(others) - 1))]
    return out


def make_orders(pop, err):
    """Order function for simulate(): evolving bots, then punters, then insiders.

    Strategy bots and insiders hold a target position and trade the difference:
    trend bots target alpha x (recent log-odds move) as a market order; forecasters and insiders
    target size x (estimate - clearing log-odds), so they never chase the price past their estimate;
    watchers target alpha x (watched wallet's position now - lag ticks ago) as a market order.
    """
    is_trend = pop["kind"] == TREND
    ev, ins = slice(0, N_EVO), slice(N_EVO + N_PUNT, None)
    w = np.flatnonzero(pop["kind"] == WATCH)
    if len(w):
        tgt = pop["target"][w]
        cols, inv = np.unique(np.maximum(tgt, 0), return_inverse=True)  # each watched wallet stored once
    seen = {}  # positions of the watched wallets at every tick so far, as seen on chain

    def fn(t, price_hist, truth, shares, rng):
        now = lo(price_hist[-1])                                  # (M,)
        past = lo(price_hist[np.maximum(t - pop["lag"], 0)]).T    # (M, N_EVO), each bot's own lookback
        d, a, e = np.zeros(shares.shape), np.zeros(shares.shape), np.zeros(shares.shape)
        d[:, ev] = np.where(is_trend, pop["alpha"] * (now[:, None] - past), 0) - shares[:, ev]
        a[:, ev] = np.where(pop["kind"] == FORECAST, np.abs(pop["alpha"]), 0)
        e[:, ev] = lo(truth[t])[:, None] + err
        if len(w):
            if t == 0:
                seen["pos"] = np.zeros((len(truth), len(now), len(cols)))
            seen["pos"][t] = shares[:, cols]  # after last tick's trades (all 0 at t = 0)
            move = seen["pos"][t][:, inv] - seen["pos"][np.maximum(t - pop["lag"][w], 0), :, inv].T  # (M, len(w))
            d[:, w] = np.where(tgt >= 0, pop["alpha"][w] * move, 0) - shares[:, w]  # no target yet: hold 0
        d[:, N_EVO:N_EVO + N_PUNT] = PUNTER_SIZE * rng.standard_normal((len(now), N_PUNT))
        d[:, ins] = -shares[:, ins]
        a[:, ins] = INSIDER_SIZE
        e[:, ins] = lo(truth[min(t + INSIDER_LEAD, len(truth) - 1)])[:, None]
        return d, a, e
    return fn


def labels(pop):
    """Index into NAMES: 0 momentum, 1 contrarian, 2 forecaster, 3 copier, 4 fader."""
    up = pop["alpha"] >= 0
    return np.select([pop["kind"] == FORECAST, pop["kind"] == WATCH], [2, np.where(up, 3, 4)], np.where(up, 0, 1))


def next_generation(pop, fitness, rng, watch_step=1.0):
    if "rank" in pop:
        # Watchers often end a generation on exactly 0 (watching wallets that hold nothing), and
        # argsort orders ties by column, so the low columns would always die first and the high
        # ones never. Break ties at random. (Without watchers no two profits are ever equal.)
        order = np.lexsort((rng.random(N_EVO), fitness))
    else:
        order = np.argsort(fitness)
    dead, alive = order[:N_DIE], order[N_DIE:]
    picks = rng.choice(alive, (N_DIE, TOURNAMENT))
    parents = picks[np.arange(N_DIE), np.argmax(fitness[picks], 1)]
    pop = {k: v.copy() for k, v in pop.items()}
    for k in pop:
        pop[k][dead] = pop[k][parents]
    lag = pop["lag"][dead].copy()
    # Mutate the children only.
    step = np.where(pop["kind"][dead] == WATCH, watch_step, 1)  # absolute part of the size mutation
    pop["alpha"][dead] = np.clip(pop["alpha"][dead] + rng.normal(0, 0.2 * np.abs(pop["alpha"][dead]) + step),
                                 -ALPHA_MAX, ALPHA_MAX)
    lag_step = rng.integers(-3, 4, N_DIE)
    pop["lag"][dead] = np.clip(lag + lag_step, 1, 50)
    pop["sigma"][dead] = np.clip(pop["sigma"][dead] * np.exp(rng.normal(0, 0.2, N_DIE)), 0.02, 5)
    flip = dead[rng.random(N_DIE) < SWITCH_PROB]
    if "rank" not in pop:  # two species: trend bots and forecasters
        pop["kind"][flip] = 1 - pop["kind"][flip]
        return pop

    # With watchers: a switching child becomes one of the other two species.
    pop["kind"][flip] = (pop["kind"][flip] + rng.integers(1, 3, len(flip))) % 3
    # ...at its new species' usual size, drawn as in random_population. A forecaster's size (~100)
    # carried into a watcher copies at 100x: it maxes out on any move, loses ~40/market and dies,
    # after feeding everyone else for a generation.
    pop["alpha"][flip] = np.where(pop["kind"][flip] == FORECAST, np.abs(rng.normal(0, 20, len(flip))),
                                  rng.normal(0, 2, len(flip)))
    # A watcher's lookback runs up to a whole market, so it mutates in proportion as well as by
    # +-3 ticks (by steps alone it would take hundreds of generations to get from 20 to 200).
    w = pop["kind"][dead] == WATCH
    grown = np.rint(lag * np.exp(rng.normal(0, 0.2, N_DIE))).astype(int) + lag_step
    pop["lag"][dead[w]] = np.clip(grown[w], 1, WATCH_LAG_MAX)
    pop["rank"][dead] = np.clip(pop["rank"][dead] + rng.normal(0, RANK_STEP, N_DIE), 0, 1)
    swap = dead[rng.random(N_DIE) < SWITCH_PROB]
    pop["board"][swap] = 1 - pop["board"][swap]
    if "lock" in pop:
        swap = dead[rng.random(N_DIE) < SWITCH_PROB]
        pop["lock"][swap] = 1 - pop["lock"][swap]
    pop["age"] += 1
    pop["age"][dead] = 0
    return pop


def drop_in(pop, make_agent, n_markets, seed, sigma=None, size=None):
    """Run an evolved population with one extra trader in the last column, and score it.

    make_agent(base_fn, err, sigma, size) returns the full order function. err (M, 1) is an
    estimate error like a forecaster's (shared part in common with the population), for the median
    forecaster's accuracy sigma; size is the median forecaster's size (either can be fixed by
    passing it). Agents that need neither ignore them. Returns the trader's expected profit per
    market and the simulation (with r["trades"], its trades per tick): each trade is
    valued at the true probability when it was made instead of at the final 0/1 outcome, which has
    about the same average (see "Known issues" in the README) but removes the luck of how each
    market resolved (most of the noise).
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


def evolve(generations=300, seed=0, fee=FEE, progress=False, watchers=False):
    rng = np.random.default_rng(seed)
    pop = random_population(rng, watchers)
    n_lab = len(NAMES) if watchers else 3
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
        hist["share"].append(np.bincount(lab, minlength=n_lab) / N_EVO)
        hist["fitness"].append([fitness[lab == k].mean() / N_MARKETS if (lab == k).any() else np.nan for k in range(n_lab)])
        hist["alpha"].append([np.median(np.abs(pop["alpha"][lab == k])) if (lab == k).any() else np.nan for k in range(n_lab)])
        hist["lag"].append(np.median(pop["lag"][lab < 2]) if (lab < 2).any() else np.nan)
        hist["sigma"].append(np.median(pop["sigma"][lab == 2]) if (lab == 2).any() else np.nan)
        hist["insider"].append(profit[N_EVO + N_PUNT:].mean() / N_MARKETS if N_INS else 0.0)
        hist["punter"].append(profit[N_EVO:N_EVO + N_PUNT].mean() / N_MARKETS)
        if watchers:
            for k, v in watch_stats(pop, lab).items():
                hist[k].append(v)

        pop = next_generation(pop, fitness, rng)
        if watchers:  # this generation's results are next generation's leaderboard
            pop["target"] = targets(pop, r["profit"][:, :WALLETS])
    return pop, {k: np.array(v) for k, v in hist.items()}


WATCHED = NAMES + ["punter", "dropped-in trader"]


def watch_stats(pop, lab):
    """What the watchers are doing: share of them watching each kind of wallet (WATCHED; a
    dropped-in trader is column WALLETS), their median lookback, the share ranking by profit per
    risk, and their median leaderboard spot."""
    w = pop["kind"] == WATCH
    tgt = pop["target"][w & (pop["target"] >= 0)]
    kind_of = np.r_[lab, np.full(N_PUNT, len(NAMES)), len(NAMES) + 1]
    return {"watched": np.bincount(kind_of[tgt], minlength=len(WATCHED)) / max(len(tgt), 1),
            "watch_lag": np.median(pop["lag"][w]) if w.any() else np.nan,
            "watch_ppr": pop["board"][w].mean() if w.any() else np.nan,
            "watch_rank": np.median(pop["rank"][w]) if w.any() else np.nan}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same seed, same result.
    for w in [False, True]:
        a, b = evolve(5, watchers=w)[1], evolve(5, watchers=w)[1]
        assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a), "not deterministic"

    # Watcher check 1: a watcher holds alpha x (watched wallet's position now - lag ticks ago).
    # All other bots idle (forecasters of size 0); the watchers watch the first punter's wallet.
    # Short markets (40 ticks) so no position gets near a limit and every order fills in full.
    pop = random_population(np.random.default_rng(0), watchers=True)
    pop["kind"][:], pop["alpha"][:] = FORECAST, 0
    tests = [(0.5, 5), (-1.0, 1), (1.0, 300)]  # (alpha, lag); a lag longer than the market copies the whole position
    for j, (al, lg) in enumerate(tests):
        pop["kind"][j], pop["alpha"][j], pop["lag"][j], pop["target"][j] = WATCH, al, lg, N_EVO
    pop["kind"][3], pop["alpha"][3], pop["target"][3] = WATCH, 1.0, -1  # no leaderboard yet: holds nothing
    fn, seen = make_orders(pop, np.zeros((50, N_EVO))), []

    def recorded(t, price_hist, truth, shares, rng):
        seen.append(shares.copy())
        return fn(t, price_hist, truth, shares, rng)
    r = simulate(recorded, WALLETS, n_markets=50, T=40, seed=0)
    pos, ticks = np.array(seen + [r["shares"]]), np.arange(40)  # pos[t]: positions before tick t's trades
    assert np.abs(pos[:, :, N_EVO]).max() > 10, "the watched wallet should move"
    for j, (al, lg) in enumerate(tests):
        want = al * (pos[ticks, :, N_EVO] - pos[np.maximum(ticks - lg, 0), :, N_EVO])
        assert np.allclose(pos[ticks + 1, :, j], want), f"watcher {j} doesn't follow the wallet"
    assert (pos[:, :, 3] == 0).all(), "a watcher with no target should hold nothing"

    # Watcher check 2: the leaderboard. Wallet 5 has the most profit, wallet WALLETS-2 the best
    # profit per unit of risk (tiny s.d.), wallet 0 the least profit.
    pop = random_population(np.random.default_rng(1), watchers=True)
    pop["kind"][:], pop["age"][:] = TREND, 1
    z = np.random.default_rng(2).standard_normal(10)
    z = (z - z.mean()) / z.std()
    mean, sd = np.linspace(-1, 1, WALLETS), np.ones(WALLETS)
    mean[5], sd[WALLETS - 2] = 5, 0.01
    profit = mean + sd * z[:, None]
    for j, (board, rank) in enumerate([(0, 1), (0, 0), (1, 0)]):
        pop["kind"][j], pop["board"][j], pop["rank"][j] = WATCH, board, rank
    got = targets(pop, profit)
    # Watcher 0 wants the bottom wallet (0), which is itself, so it gets the next one up.
    assert list(got[:3]) == [1, 5, WALLETS - 2] and (got[3:] == -1).all(), got[:3]
    pop["age"][5] = 0  # wallet 5 just changed hands: its record belongs to the bot that died
    assert targets(pop, profit)[1] == WALLETS - 1
    # A dropped-in trader (column WALLETS) is on the leaderboard too.
    assert targets(pop, np.c_[profit, np.full(10, 9.0)])[1] == WALLETS

    # Lock-on: a locked watcher keeps its wallet when that wallet's spot changes; it picks again by
    # spot only when the owner dies, when it has no wallet yet, or when the wallet is its own.
    locked = {**{k: v.copy() for k, v in pop.items()}, "lock": np.zeros(N_EVO, int)}
    locked["lock"][1], locked["target"][1] = 1, 7
    assert targets(locked, profit)[1] == 7, "a locked watcher lost its wallet"
    locked["age"][7] = 0
    assert targets(locked, profit)[1] == WALLETS - 1, "a locked watcher followed a dead wallet's column"
    for held in [-1, 1]:
        locked["target"][1] = held
        assert targets(locked, profit)[1] == WALLETS - 1, "a locked watcher should have picked by spot"
    last_place = np.c_[profit, np.full(10, -9.0)]  # a dropped-in trader at the bottom of the board
    locked["kind"][:], locked["lock"][:], locked["target"][:] = WATCH, 1, WALLETS
    assert (targets(locked, last_place) == WALLETS).all(), "lock-on should hold whatever the spot"
    child = next_generation(locked, np.arange(N_EVO, dtype=float), np.random.default_rng(4))
    assert (targets(child, last_place) == WALLETS).sum() > 90, "children should inherit the wallet"

    # Watcher check 3: when many bots tie on exactly 0, which of them die doesn't depend on their
    # column. 10 losers, 70 on zero, 20 winners: 10 of the zeros die each generation.
    rng = np.random.default_rng(3)
    deaths = np.zeros(N_EVO)
    for _ in range(400):
        f = np.zeros(N_EVO)
        f[:10], f[80:] = -1, 1
        deaths += next_generation(pop, f, rng)["age"] == 0
    zero = deaths[10:80] / 400
    assert (deaths[:10] == 400).all() and (deaths[80:] == 0).all()
    assert abs(zero[:35].mean() - zero[35:].mean()) < 0.03 and abs(zero.mean() - 10 / 70) < 0.01, zero.round(2)

    # Watcher check 4: a child that switches species starts at its new species' size. All parents
    # are forecasters of size 100; children that became trend bots or watchers must be small.
    pop["kind"][:], pop["alpha"][:] = FORECAST, ALPHA_MAX
    switched = []
    for _ in range(200):
        child = next_generation(pop, np.arange(N_EVO, dtype=float), rng)
        switched += list(child["alpha"][child["kind"] != FORECAST])
    assert len(switched) > 50 and np.abs(switched).max() < 15, (len(switched), np.abs(switched).max())
    # A fine mutation step keeps a quiet watcher's children quiet; the standard step doesn't.
    quiet = random_population(np.random.default_rng(2), watchers=True)
    quiet["kind"][:], quiet["alpha"][:] = WATCH, 0.25
    for s, stays in [(0.02, True), (1.0, False)]:
        kids = next_generation(quiet, np.arange(N_EVO, dtype=float), np.random.default_rng(3), watch_step=s)
        born = (kids["age"] == 0) & (kids["kind"] == WATCH)
        assert (np.abs(kids["alpha"][born]).max() < 0.6) == stays, f"watch_step {s} has no effect"

    print("watcher checks passed")

    # The default run is the main setup; other fees/seeds/lengths get suffixed output files.
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--watchers", action="store_true", help="wallet watchers can evolve too")
    args = ap.parse_args()
    tag = (f"_fee{args.fee:g}" if args.fee != FEE else "") + ("_watch" if args.watchers else "") \
        + (f"_seed{args.seed}" if args.seed else "") + (f"_g{args.gens}" if args.gens != 1000 else "")
    pop, h = evolve(args.gens, seed=args.seed, fee=args.fee, progress=True, watchers=args.watchers)
    assert np.allclose(h["share"].sum(1), 1)
    np.savez(RUNS / f"phase2{tag}.npz", **h)  # per-generation history, for analysis
    np.savez(RUNS / f"pop{tag}.npz", **pop)    # final population, for dropping new agents into
    dashboard(h, FIGURES / f"phase3{tag}.png")

    n_lab = h["share"].shape[1]
    names = NAMES[:n_lab]
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"][:n_lab]
    for k in range(n_lab):
        print(f"{names[k]:>11}: share {h['share'][0, k]:.2f} -> {h['share'][-1, k]:.2f}, "
              f"profit/market (last 50 gens) {np.nanmean(h['fitness'][-50:, k]):6.2f}")
    print(f"forecaster error sigma: {h['sigma'][0]:.2f} -> {h['sigma'][-1]:.2f}")
    print(f"trend lookback:         {h['lag'][0]:.0f} -> {h['lag'][-1]:.0f}")
    print(f"price-truth gap:        {h['gap'][:20].mean():.3f} -> {h['gap'][-20:].mean():.3f}")
    print(f"insider / punter profit per market (last 50): {h['insider'][-50:].mean():.2f} / {h['punter'][-50:].mean():.2f}")
    if args.watchers and (h["watched"][-50:].sum(1) > 0).any():
        wt = h["watched"][-50:]
        wt = wt[wt.sum(1) > 0]  # generations with any watchers
        print("watchers watch (last 50 gens): " + ", ".join(f"{n} {s:.2f}" for n, s in zip(WATCHED[:-1], wt.mean(0))))
        print(f"watchers (median, last 50 gens): lookback {np.nanmedian(h['watch_lag'][-50:]):.0f} ticks, "
              f"share ranking by profit per risk {np.nanmean(h['watch_ppr'][-50:]):.2f}, "
              f"leaderboard spot {np.nanmedian(h['watch_rank'][-50:]):.2f} (0 = top)")
    print("Phase 2 checks passed")

    gens = np.arange(len(h["gap"]))
    rows = 3 if args.watchers else 2
    fig, ax = plt.subplots(rows, 2, figsize=(11, 3.5 * rows), sharex=True)
    ax[0, 0].stackplot(gens, h["share"].T, labels=names, colors=colors, alpha=0.8)
    ax[0, 0].set_title("population share")
    ax[0, 0].legend(loc="upper left", fontsize=8)
    for k in range(n_lab):
        ax[0, 1].plot(gens, h["fitness"][:, k], color=colors[k], label=names[k])
    ax[0, 1].axhline(0, color="grey", lw=0.5)
    ax[0, 1].set_title("mean profit per market (after info cost)")
    ax[1, 0].plot(gens, h["sigma"], color="tab:blue", label="forecaster error sigma")
    ax[1, 0].plot(gens, h["lag"] / 10, color="tab:red", label="trend lookback / 10")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("evolved traits (median)")
    ax[1, 1].plot(gens, h["gap"], color="black")
    ax[1, 1].set_title("price-to-truth gap")
    if args.watchers:
        ax[2, 0].stackplot(gens, h["watched"][:, :-1].T, labels=WATCHED[:-1], colors=colors + ["tab:gray"], alpha=0.8)
        ax[2, 0].legend(loc="upper left", fontsize=7)
        ax[2, 0].set_title("what the watchers watch (share of watchers)")
        ax[2, 1].plot(gens, h["watch_lag"] / WATCH_LAG_MAX, label=f"lookback / {WATCH_LAG_MAX} ticks")
        ax[2, 1].plot(gens, h["watch_ppr"], label="share ranking by profit per risk")
        ax[2, 1].plot(gens, h["watch_rank"], label="leaderboard spot (0 top, 1 bottom)")
        ax[2, 1].set_ylim(-0.02, 1.02)
        ax[2, 1].legend(fontsize=7)
        ax[2, 1].set_title("watcher traits (median)")
    for a_ in ax[-1]:
        a_.set_xlabel("generation")
    fig.tight_layout()
    fig.savefig(FIGURES / f"phase2{tag}.png", dpi=120)
    print(f"charts saved to figures/phase2{tag}.png and figures/phase3{tag}.png; data in runs/")
