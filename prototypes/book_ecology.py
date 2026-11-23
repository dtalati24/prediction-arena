"""PROTOTYPE, throwaway: the simplest ecology on the order book, and the go/no-go measurements.

150 bots, as in the current market:
  25 forecasters  size 90, error sigma 0.31 (as evolved). Each posts one limit order halfway between
                  the price and where it would stop trading, so it takes or provides liquidity.
  10 makers       bid and offer around the last traded price, shaded against their inventory.
  50 retail       random market orders (s.d. 5) that won't pay more than 5c from the mid.
  65 trend bots   small momentum/contrarian market orders, with a 2c price limit.
"""
import time

import numpy as np
from scipy.special import expit, ndtr

from arena import RUNS
from arena.evolve import ERR_CORR, INFO_COST, N_EVO, WALLETS, make_orders
from arena.market import lo, simulate
from arena.measure import measure
from prototypes.order_book import Books, shuffled

N_FC, N_MM, N_RETAIL, N_TREND = 25, 10, 50, 65
N, T = N_FC + N_MM + N_RETAIL + N_TREND, 200
FC, MM, RET, TR = np.split(np.arange(N), np.cumsum([N_FC, N_MM, N_RETAIL]))
FC, MM, RET, TR = (slice(x[0], x[-1] + 1) for x in (FC, MM, RET, TR))


def true_paths(M, rng):
    """The same truth process, from the same random draws, as arena.market.simulate."""
    steps = rng.standard_normal((T, M))
    news = rng.random((T, M)) < 0.02
    steps += news * 5.0 * rng.standard_normal((T, M))
    lead = np.vstack([np.zeros(M), steps.cumsum(0)])
    left = np.arange(T, 0, -1)[:, None]
    return np.vstack([ndtr(lead[:-1] / np.sqrt(1.5 * left)), lead[-1] > 0]), news


def run(M=400, seed=0, forecasters=True, makers=True, half=(1, 3), skew=0.05, quote=20,
        active=1.0, n_fc=N_FC):
    """half: range of the makers' half-spreads (cents); skew: cents of shading per share held;
    active: chance a retail bot trades on a tick; n_fc: forecasters switched on."""
    truth, news = true_paths(M, np.random.default_rng(seed))
    rng = np.random.default_rng([seed, 1])
    err = 0.31 * (np.sqrt(ERR_CORR) * rng.standard_normal((M, 1))
                  + np.sqrt(1 - ERR_CORR) * rng.standard_normal((M, N_FC)))
    h = np.linspace(*half, N_MM).round()
    alpha, lag = rng.normal(0, 1, N_TREND), rng.integers(1, 31, N_TREND)
    ioc = np.zeros(N, bool)
    ioc[RET] = ioc[TR] = True

    books, last = Books(M, N), np.full(M, 50.0)
    mids = np.full((T + 1, M), 50.0)
    logs = np.zeros((T + 1, M))
    spread, two_sided, volume = [], [], 0
    expected = np.zeros((M, N))  # each trade valued at the truth, as in evolve.drop_in
    for t in range(T):
        sh = books.shares.copy()
        before = books.cash.copy()
        mid, now = mids[t][:, None], logs[t][:, None]
        bid, ask = np.ones((M, N), np.int64), np.full((M, N), 99, np.int64)
        bid_n, ask_n = np.zeros((M, N), np.int64), np.zeros((M, N), np.int64)

        if forecasters:
            stop = lo(truth[t])[:, None] + err - sh[:, FC] / 90  # where it would stop trading
            p = 100 * expit((now + stop) / 2)
            n = np.rint(90 * (stop - now) / 2).astype(np.int64)
            bid[:, FC], bid_n[:, FC] = np.clip(np.floor(p), 1, 99), np.maximum(n, 0)
            ask[:, FC], ask_n[:, FC] = np.clip(np.ceil(p), 1, 99), np.maximum(-n, 0)
            bid_n[:, FC.start + n_fc:FC.stop] = ask_n[:, FC.start + n_fc:FC.stop] = 0
        if makers:
            centre = last[:, None] - skew * sh[:, MM]
            b = np.clip(np.floor(centre - h), 1, 98)
            bid[:, MM], ask[:, MM] = b, np.clip(np.ceil(centre + h), b + 1, 99)
            bid_n[:, MM] = ask_n[:, MM] = quote

        past = logs[np.maximum(t - lag, 0)].T
        retail = np.rint(5 * rng.standard_normal((M, N_RETAIL))).astype(np.int64)
        if active < 1:
            retail *= rng.random(retail.shape) < active
        trend = np.rint(alpha * (now - past) - sh[:, TR]).astype(np.int64)
        for who, n, cap in ((RET, retail, 5), (TR, trend, 2)):
            bid[:, who], bid_n[:, who] = np.clip(np.ceil(mid + cap), 1, 99), np.maximum(n, 0)
            ask[:, who], ask_n[:, who] = np.clip(np.floor(mid - cap), 1, 99), np.maximum(-n, 0)

        vol, traded = books.step(bid, bid_n, ask, ask_n, ioc, shuffled(rng.random((M, N))))
        traded_at = vol > 0
        last[traded_at] = traded[traded_at] / vol[traded_at]
        b, a = books.best.T
        both = (b > 0) & (a < 100)
        mids[t + 1] = np.where(both, (b + a) / 2, last)
        logs[t + 1] = lo(mids[t + 1] / 100)
        spread.append(np.where(both, a - b, np.nan))
        two_sided.append(both)
        volume += vol.sum()
        expected += (books.cash - before) / 100 + (books.shares - sh) * truth[t][:, None]

    assert (books.cash.sum(1) == books.cash.shape[1] * 100_00).all(), "money not conserved"
    profit = (books.cash + 100 * books.shares * truth[T][:, None]) / 100 - 100
    assert np.allclose(profit.sum(1), 0) and profit.min() >= -100, "settlement is off"
    return dict(prices=mids / 100, truth=truth, news=news, profit=profit, expected=expected,
                spread=np.array(spread), two_sided=np.mean(two_sided), volume=volume / M)


def current(M=400, seed=0):
    """The market we have now, with its evolved population, on the same true paths."""
    pop = dict(np.load(RUNS / "pop.npz"))
    rng = np.random.default_rng([seed, 1])
    err = pop["sigma"] * (np.sqrt(ERR_CORR) * rng.standard_normal((M, 1))
                          + np.sqrt(1 - ERR_CORR) * rng.standard_normal((M, N_EVO)))
    return simulate(make_orders(pop, err), WALLETS, n_markets=M, seed=seed)


def describe(name, r, seconds):
    m = measure(r)
    inner = slice(10, 191)  # away from the opening and the last ticks
    print(f"{name:30s} {seconds:5.2f}s | gap {m['gap']:.3f} | calibration {m['calibration']:.3f} "
          f"| slope {m['extremity_slope']:.2f} | news priced in after 1/5/20 ticks "
          f"{'/'.join(f'{x:.2f}' for x in m['news_response'][[0, 4, 19]])} | return autocorr {m['return_autocorr']:+.2f}")
    if "spread" not in r:
        return
    kinds = [("forecaster", FC, INFO_COST / 0.31**2), ("maker", MM, 0), ("retail", RET, 0), ("trend", TR, 0)]
    print(f"{'':30s} spread median {np.nanmedian(r['spread'][inner]):.0f}c | two-sided {r['two_sided']:.0%} "
          f"| {r['volume']:,.0f} contracts/market | per bot, per market: "
          + ", ".join(f"{k} {r['expected'][:, s].mean() - c:+.2f}" for k, s, c in kinds))


def bench(M):
    run(4)  # compile first
    assert np.array_equal(run(20, 3)["truth"], current(20, 3)["truth"]), "truth paths differ"
    assert np.array_equal(run(20, 3)["profit"], run(20, 3)["profit"]), "not deterministic"
    times = {"current": [], "book": []}
    for seed in range(3):  # interleaved, so both see the same machine load
        for name, f in (("current", current), ("book", run)):
            t0 = time.perf_counter()
            r = f(M, seed)
            times[name].append(time.perf_counter() - t0)
            describe(f"{name} seed {seed}", r, times[name][-1])
    print(f"slowdown (median, {M} markets): {np.median(times['book']) / np.median(times['current']):.2f}x")
    for name, kw in [("no forecasters", dict(forecasters=False)), ("no makers", dict(makers=False)),
                     ("tight makers (1c)", dict(half=(1, 1))), ("wide makers (3-6c)", dict(half=(3, 6))),
                     ("no inventory shading", dict(skew=0.0)), ("big quotes (50)", dict(quote=50)),
                     ("retail trades 1 tick in 4", dict(active=0.25)), ("retail trades 1 tick in 20", dict(active=0.05)),
                     ("retail trades 1 tick in 100", dict(active=0.01)),
                     ("12 forecasters", dict(n_fc=12)), ("5 forecasters", dict(n_fc=5)),
                     ("1 tick in 20, 5 forecasters", dict(active=0.05, n_fc=5))]:
        t0 = time.perf_counter()
        r = run(M, 0, **kw)
        describe(name, r, time.perf_counter() - t0)

    # Heaviest load: all 150 bots quote both sides every tick, crossing each other.
    books, rng, spent = Books(M, N), np.random.default_rng(9), 0.0
    for _ in range(T):
        centre = rng.integers(10, 91, (M, 1)) + rng.integers(-4, 5, (M, N))
        order, n = shuffled(rng.random((M, N))), np.full((M, N), 30)
        t0 = time.perf_counter()
        books.step(centre - 1, n, centre + 1, n, np.zeros(N, bool), order)
        spent += time.perf_counter() - t0
    print(f"heaviest load, engine only: {spent:.2f}s ({spent / np.median(times['current']):.2f}x a current generation)")
