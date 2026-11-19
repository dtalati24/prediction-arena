"""Phase 1: the market.

Each market is a binary event. A hidden "score lead" does a random walk (plus rare news jumps);
the event happens if the lead is positive at the end. An LMSR market maker trades with the bots.
All markets run side by side as numpy arrays.

Run `python market.py` for the Phase 1 check and chart.
"""
import numpy as np
from scipy.special import expit, ndtr


def simulate(order_fn, n_bots, n_markets=100, T=200, b=50.0, sigma=1.0,
             news_prob=0.02, news_size=5.0, budget=100.0, limit=100.0, seed=0):
    """Run n_markets markets for T ticks.

    order_fn(t, price_hist, true_p, shares, rng) returns a (n_markets, n_bots) array of orders
    in YES shares (+ buy, - sell). price_hist is (t+1, n_markets); true_p is (n_markets,).
    """
    rng = np.random.default_rng(seed)
    M, N = n_markets, n_bots
    var = sigma**2 + news_prob * news_size**2  # per-tick variance of the lead, news folded in
    lead = np.zeros(M)
    x = np.zeros(M)  # net YES shares the LMSR has sold; its price is expit(x / b)
    cash = np.full((M, N), budget)
    shares = np.zeros((M, N))
    mm_cash = np.zeros(M)
    prices = np.empty((T + 1, M))
    truth = np.empty((T + 1, M))
    prices[0] = truth[0] = 0.5

    for t in range(T):
        d = order_fn(t, prices[:t + 1], truth[t], shares, rng)
        d = np.clip(shares + d, -limit, limit) - shares       # position limit
        d[cash + shares * prices[t][:, None] <= 0] = 0          # bust bots sit out

        # Batch auction: LMSR fills the net order, everyone pays the same average price.
        D = d.sum(1)
        cost = b * (np.logaddexp(0, (x + D) / b) - np.logaddexp(0, x / b))
        fill = np.where(D != 0, cost / np.where(D != 0, D, 1), prices[t])
        cash -= d * fill[:, None]
        shares += d
        mm_cash += cost
        x += D
        prices[t + 1] = expit(x / b)

        # The lead moves; true probability = chance it stays positive with the time left.
        lead += sigma * rng.standard_normal(M) + (rng.random(M) < news_prob) * news_size * rng.standard_normal(M)
        left = T - t - 1
        # Note: normal approximation ignores the jumps' fat tails; fine while jumps are rare
        truth[t + 1] = ndtr(lead / np.sqrt(var * left)) if left else (lead > 0)

    outcome = truth[T]
    cash += shares * outcome[:, None]  # each YES share pays 1 or 0
    return dict(prices=prices, truth=truth, outcome=outcome,
                profit=cash - budget, mm_pnl=mm_cash - x * outcome)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, b = 51, 50.0  # bot 0 is the oracle, bots 1..50 are punters

    def orders(oracle):
        def fn(t, price_hist, true_p, shares, rng):
            d = rng.standard_normal(shares.shape)  # punters: random buys/sells, ~1 share
            d[:, 0] = 200 * (true_p - price_hist[-1]) if oracle else 0  # oracle trades toward the truth
            return d
        return fn

    r = simulate(orders(True), N, b=b)
    r0 = simulate(orders(False), N, b=b)
    err = np.abs(r["prices"] - r["truth"]).mean()
    err0 = np.abs(r0["prices"] - r0["truth"]).mean()

    # Is the true probability honest? Bin it mid-market and compare with how often the event happened.
    c = simulate(lambda t, ph, tp, sh, rng: np.zeros(sh.shape), 1, n_markets=20000, seed=1)
    mid, out = c["truth"][100], c["outcome"]
    bins = np.minimum((mid * 10).astype(int), 9)
    calib = max(abs(out[bins == k].mean() - mid[bins == k].mean()) for k in range(10))

    print(f"oracle profit / market:   {r['profit'][:, 0].mean():7.2f}")
    print(f"punter profit / market:   {r['profit'][:, 1:].mean():7.2f}")
    print(f"LMSR pnl / market:        {r['mm_pnl'].mean():7.2f}   (worst {r['mm_pnl'].min():.2f}, cap {-b * np.log(2):.2f})")
    print(f"price vs truth gap:       {err:.3f} with oracle, {err0:.3f} punters only")
    print(f"truth calibration error:  {calib:.3f} (worst of 10 bins, 20k markets)")

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
