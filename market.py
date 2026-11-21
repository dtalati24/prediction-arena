"""Phase 1: the market.

Each market is a binary event. A hidden "score lead" does a random walk (plus rare news jumps);
the event happens if the lead is positive at the end. An LMSR market maker trades with the bots.
All markets run side by side as numpy arrays.

Run `python market.py` for the Phase 1 check and chart.
"""
import numpy as np
from scipy.special import expit, logit, ndtr

GRID = 40 / 2**30  # clearing-price precision in log-odds (30 halvings of [-20, 20])


def lo(p):
    """Log-odds, clipped so prices/truth at 0 or 1 stay finite."""
    return np.clip(logit(p), -6, 6)


def simulate(order_fn, n_bots, n_markets=100, T=200, b=50.0, sigma=1.0,
             news_prob=0.02, news_size=5.0, budget=100.0, limit=100.0, fee=0.0, seed=0):
    """Run n_markets markets for T ticks.

    order_fn(t, price_hist, truth, shares, rng) returns (d, a, e), each (n_markets, n_bots).
    A bot's order in YES shares (+ buy, - sell) is d + a * (e - L), where L is the clearing
    log-odds of this tick's batch auction: d is a plain market order, and a > 0 makes the bot
    price-sensitive (like a limit order) around its valuation e in log-odds.
    price_hist is (t+1, n_markets); truth is the whole (T+1, n_markets) true-probability path,
    so honest bots should only read truth[:t+1].
    """
    rng = np.random.default_rng(seed)
    M, N = n_markets, n_bots

    # The true-probability path comes first: trading never moves it, and insiders peek ahead.
    # The lead does a random walk with news jumps; truth = chance it stays positive with the time left.
    var = sigma**2 + news_prob * news_size**2  # per-tick variance of the lead, news folded in
    steps = sigma * rng.standard_normal((T, M))
    news = rng.random((T, M)) < news_prob  # news[t] moves the lead between tick t and t+1
    steps += news * news_size * rng.standard_normal((T, M))
    lead = np.vstack([np.zeros(M), steps.cumsum(0)])
    left = np.arange(T, 0, -1)[:, None]
    # Note: normal approximation ignores the jumps' fat tails; fine while jumps are rare
    truth = np.vstack([ndtr(lead[:-1] / np.sqrt(var * left)), lead[-1] > 0])

    x = np.zeros(M)  # net YES shares the LMSR has sold; its price is expit(x / b)
    cash = np.full((M, N), budget)
    shares = np.zeros((M, N))
    mm_cash = np.zeros(M)
    prices = np.empty((T + 1, M))
    prices[0] = 0.5
    fills = np.empty((T, M))  # the price each tick's batch traded at

    for t in range(T):
        d, a, e = order_fn(t, prices[:t + 1], truth, shares, rng)

        def orders(L):
            """Each bot's order if the batch clears at log-odds L, within its limits.

            Full collateral, as on real prediction markets: a long must be paid for in cash, and a
            short (YES sold) must keep $1 per share in cash, so no bot can lose more than its budget.
            Everyone trades at the clearing price, so that's what buyers pay and sellers receive
            (plus or minus the fee).
            """
            pc = expit(L)[:, None]
            buy, sell = pc + fee, pc - fee
            hi = np.minimum(limit, shares + cash / buy)
            lo_ = np.maximum(-limit, -(cash + shares * sell) / (1 - sell))
            # If a bot is ever short of collateral (rounding), it may be forced to close its
            # position, never pushed into a bigger opposite one.
            hi, lo_ = np.maximum(hi, np.minimum(shares, 0)), np.minimum(lo_, np.maximum(shares, 0))
            target = shares + d + a * (e - L[:, None])
            return np.clip(target, lo_, hi) - shares, (target > lo_) & (target < hi)

        # Clearing price: LMSR log-odds after the trade is (x + net order) / b, and orders are
        # judged at that same price. Net order falls as L rises, so solve b*L = x + net order(L).
        # The answer is the middle of the step, on a grid of 2^30 steps over [-20, 20], that holds
        # the root: exactly what 30 rounds of bisection give. (At 2^20 the clearing price and the
        # LMSR's price afterwards differed by ~0.001 log-odds, worth ~0.03 per market to a
        # forecaster.) Guesses only pick which grid point to
        # test, so they change the speed, never the answer. Each guess is a Newton step (slope =
        # b + total size of the bots not at a limit), or, if that lands outside the bracket or the
        # last step didn't halve it, false position with the Illinois fix (Newton alone stalled
        # when many bots hit their limits).
        # ~5-7 evaluations instead of 31; after 12 tries, plain bisection takes over as a fallback.
        k_lo, k_hi = np.zeros(M, np.int64), np.full(M, 2**30)
        f_lo, f_hi = np.full(M, np.nan), np.full(M, np.nan)  # net - b*L at the bracket ends, once tested
        last = np.zeros(M)                                   # which end moved last: +1 low, -1 high
        width, halved = k_hi - k_lo, np.ones(M, bool)        # did the last step halve the bracket?
        L = x / b
        for i in range(50):
            open_ = k_hi - k_lo > 1
            if not open_.any():
                break
            L_lo, L_hi = -20 + k_lo * GRID, -20 + k_hi * GRID
            newton_ok = np.isnan(f_lo + f_hi) | ((L > L_lo) & (L < L_hi) & halved)
            L = np.where(newton_ok, L, L_lo + f_lo * (L_hi - L_lo) / (f_lo - f_hi))
            k = np.floor((np.clip(L, -21, 21) + 20) / GRID).astype(np.int64) if i < 12 else (k_lo + k_hi) // 2
            k = np.clip(k, k_lo + 1, k_hi - 1)
            Lk = -20 + k * GRID
            o, free = orders(Lk)
            net = x + o.sum(1)
            up = net > b * Lk
            f = net - b * Lk
            # Illinois: if the same end moves twice in a row, halve the other end's value.
            f_hi = np.where(open_ & up & (last == 1), f_hi / 2, f_hi)
            f_lo = np.where(open_ & ~up & (last == -1), f_lo / 2, f_lo)
            k_lo, f_lo = np.where(open_ & up, k, k_lo), np.where(open_ & up, f, f_lo)
            k_hi, f_hi = np.where(open_ & ~up, k, k_hi), np.where(open_ & ~up, f, f_hi)
            last = np.where(open_, np.where(up, 1, -1), last)
            halved = 2 * (k_hi - k_lo) <= width
            width = k_hi - k_lo
            L = Lk + f / (b + (a * free).sum(1))
        assert (k_hi - k_lo == 1).all()
        L = ((-20 + k_lo * GRID) + (-20 + k_hi * GRID)) / 2
        d = orders(L)[0]

        # Uniform-price batch auction (as in frequent batch auctions): every order in the batch
        # trades at the one clearing price, the price its limit was judged at. The LMSR takes the
        # net order at that price. (An earlier version filled everyone at the LMSR's average price
        # over the batch, which is better than the clearing price for whoever trades with the net
        # flow: a large trader could push the clearing price past the truth and buy cheaply from
        # bots selling into it. The RL agent found that within 2 training iterations.)
        D = d.sum(1)
        fill = fills[t] = expit(L)
        # Fee per share traded goes to the exchange; like real exchanges, it never exceeds the
        # contract's price (or 1 - price), so selling never costs more than it brings in.
        fee_ps = np.minimum(fee, np.minimum(fill, 1 - fill))[:, None]
        cash -= d * fill[:, None] + fee_ps * np.abs(d)
        shares += d
        mm_cash += D * fill
        x += D
        prices[t + 1] = expit(x / b)

    outcome = truth[T]
    cash += shares * outcome[:, None]  # each YES share pays 1 or 0
    return dict(prices=prices, truth=truth, outcome=outcome, news=news, fills=fills, shares=shares,
                profit=cash - budget, mm_pnl=mm_cash - x * outcome)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, b = 51, 50.0  # bot 0 is the oracle, bots 1..50 are punters

    def orders(oracle):
        def fn(t, price_hist, truth, shares, rng):
            d = rng.standard_normal(shares.shape)  # punters: random buys/sells, ~1 share
            a, e = np.zeros(shares.shape), np.zeros(shares.shape)
            d[:, 0] = 0
            if oracle:  # oracle holds 200 shares per unit of log-odds the price is below the truth
                d[:, 0], a[:, 0], e[:, 0] = -shares[:, 0], 200, lo(truth[t])
            return d, a, e
        return fn

    r = simulate(orders(True), N, b=b)
    r0 = simulate(orders(False), N, b=b)
    err = np.abs(r["prices"] - r["truth"]).mean()
    err0 = np.abs(r0["prices"] - r0["truth"]).mean()

    # Is the true probability honest? Bin it mid-market and compare with how often the event happened.
    c = simulate(lambda t, ph, tp, sh, rng: (np.zeros(sh.shape),) * 3, 1, n_markets=20000, seed=1)
    mid, out = c["truth"][100], c["outcome"]
    bins = np.minimum((mid * 10).astype(int), 9)
    calib = max(abs(out[bins == k].mean() - mid[bins == k].mean()) for k in range(10))

    # Collateral: no bot may lose more than its budget, however recklessly it trades and whatever
    # the fee. 20 bots jump to random positions of up to +-100 shares each tick, 10 place random
    # limit orders, the rest are large punters.
    def flippers(t, price_hist, truth, shares, rng):
        M = len(shares)
        d, a, e = 20 * rng.standard_normal(shares.shape), np.zeros(shares.shape), np.zeros(shares.shape)
        d[:, :20] = rng.uniform(-100, 100, (M, 20)) - shares[:, :20]
        a[:, 20:30], e[:, 20:30] = rng.uniform(0, 300, (M, 10)), rng.normal(0, 3, (M, 10))
        return d, a, e
    worst = min(simulate(flippers, 50, n_markets=400, fee=fee, seed=s)["profit"].min()
                for fee in [0, 0.01, 0.05] for s in range(2))

    print(f"worst loss, reckless bots: {worst:7.2f}   (budget 100)")
    print(f"oracle profit / market:   {r['profit'][:, 0].mean():7.2f}")
    print(f"punter profit / market:   {r['profit'][:, 1:].mean():7.2f}")
    print(f"LMSR pnl / market:        {r['mm_pnl'].mean():7.2f}   (worst {r['mm_pnl'].min():.2f}, cap {-b * np.log(2):.2f})")
    print(f"price vs truth gap:       {err:.3f} with oracle, {err0:.3f} punters only")
    print(f"truth calibration error:  {calib:.3f} (worst of 10 bins, 20k markets)")

    # Tolerance 0.01: the clearing price is exact to ~2e-8 log-odds; rounding could leave a bot a
    # tiny fraction of a cent past its budget, never more.
    assert np.isfinite(worst) and worst >= -100 - 0.01, "a bot lost more than its budget (positions not fully collateralised)"
    assert np.allclose(r["profit"].sum(1) + r["mm_pnl"], 0), "money not conserved"
    assert r["mm_pnl"].min() >= -b * np.log(2) - 1e-9, "LMSR lost more than its cap"
    assert r["profit"][:, 0].mean() > 0 > r["profit"][:, 1:].mean(), "oracle should win, punters lose"
    assert err < err0, "oracle should pull the price toward the truth"
    assert set(np.unique(r["truth"][-1])) <= {0.0, 1.0} and calib < 0.04
    print("Phase 1 checks passed")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, m in zip(axes, range(3)):
        ax.plot(r["truth"][:, m], label="true probability", color="black")
        ax.plot(r["prices"][:, m], label="price (with oracle)", color="tab:blue")
        ax.plot(r0["prices"][:, m], label="price (punters only)", color="tab:orange", alpha=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"market {m}")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("tick")
    fig.tight_layout()
    fig.savefig("phase1.png", dpi=120)
    print("chart saved to phase1.png")
