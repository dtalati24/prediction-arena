"""Phase 4, step 2: a Bayesian (Kalman-filter) trader.

It gets exactly what a forecaster gets: a private estimate of the true log-odds whose error is
fixed for the market (half of it shared with every other forecaster), and the market price.
A forecaster trusts its estimate. The Kalman trader also learns its own error, by comparing each
estimate with the price that cleared on it:

    estimate  s_t = x_t + eps        eps ~ N(0, sigma^2), fixed for the market
    price     q_t = x_t + eta_t      eta_t ~ N(0, tau^2), treated as independent each tick
    so        s_{t-1} - q_t = eps - eta_t

That's a Kalman filter for the constant eps: after n comparisons its best guess is
eps_hat = P * sum(s - q) / tau^2 with uncertainty P = 1 / (1/sigma^2 + n/tau^2). It then trades
exactly like a forecaster, on s_t - eps_hat. tau (how noisy it thinks the price is) is the only
new parameter: tau = inf is a plain forecaster, tau -> 0 copies the price.

Run `python -m arena.kalman` for the checks and the Kalman-vs-forecaster experiment (needs the
runs/pop*.npz files written by arena.evolve).
"""
import numpy as np

from arena import FIGURES, RUNS
from arena.evolve import drop_in
from arena.market import lo, simulate


def bias_estimate(diff_sum, n, sigma, tau):
    """Posterior mean of eps after n comparisons summing to diff_sum (prior N(0, sigma^2))."""
    return diff_sum / tau**2 / (1 / sigma**2 + n / tau**2)


def with_kalman(base_fn, err, size, sigma, tau):
    """Order function: base_fn's bots, plus Kalman traders in the last err.shape[1] columns.

    err (M, K): each trader's estimate error per market; size, sigma, tau: scalars or (K,).
    fn.state["sum"] / fn.state["n"] hold the latest comparisons, for checks.
    """
    K = err.shape[1]
    st = {}

    def fn(t, price_hist, truth, shares, rng):
        d, a, e = base_fn(t, price_hist, truth, shares, rng)
        s = lo(truth[t])[:, None] + err
        if t == 0:
            st["sum"] = np.zeros(err.shape)
        else:  # last tick's estimate vs the price that cleared on it
            st["sum"] += st["prev"] - lo(price_hist[t])[:, None]
        st["prev"], st["n"] = s, t
        d[:, -K:], a[:, -K:] = -shares[:, -K:], size
        e[:, -K:] = s - bias_estimate(st["sum"], t, sigma, tau)
        return d, a, e
    fn.state = st
    return fn


def trial(pop, tau, n_markets, seed):
    """One Kalman trader with the median forecaster's accuracy and size, dropped into pop.
    Returns its expected profit per market and the simulation (see evolve.drop_in)."""
    return drop_in(pop, lambda base, err, sigma, size: with_kalman(base, err, size, sigma, tau), n_markets, seed)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Check 1: the filter is the Bayes-optimal guess. With eta really independent, its average
    # squared error after n comparisons should equal its own uncertainty P.
    rng = np.random.default_rng(0)
    sigma, tau, n = 0.35, 0.5, 30
    eps = sigma * rng.standard_normal(100_000)
    diff_sum = n * eps - tau * np.sqrt(n) * rng.standard_normal(100_000)
    mse = np.mean((bias_estimate(diff_sum, n, sigma, tau) - eps) ** 2)
    assert abs(mse / (1 / (1 / sigma**2 + n / tau**2)) - 1) < 0.02, "filter is not Bayes-optimal"

    # Check 2: in a market pinned to the truth (a huge oracle with unlimited money), the price has
    # no error, so a tiny Kalman trader must learn its own error almost exactly. Comparing the
    # wrong ticks (this tick's estimate vs last tick's price) would fail this.
    def oracle(t, ph, truth, shares, rng):
        d, a, e = -shares, np.zeros(shares.shape), np.zeros(shares.shape)
        a[:, 0], e[:, 0] = 1e5, lo(truth[t])
        return d, a, e
    eps = 0.35 * rng.standard_normal((500, 1))
    fn = with_kalman(oracle, eps, 0.01, 0.35, 0.05)
    simulate(fn, 2, n_markets=500, budget=1e9, limit=1e9, seed=3)
    eps_hat = bias_estimate(fn.state["sum"], fn.state["n"], 0.35, 0.05)
    assert np.abs(eps_hat - eps).mean() < 0.005, np.abs(eps_hat - eps).mean()
    print("Kalman checks passed")

    # Experiment: one Kalman trader in each evolved population, for a range of tau. Same seed =
    # same markets, punters and errors for every tau, so differences are paired.
    pops = {f.stem: dict(np.load(f)) for f in sorted(RUNS.glob("pop*.npz"))
            if "fee" not in f.stem and "watch" not in f.stem}  # no fee, no wallet watchers
    assert pops, "run arena.evolve first (it saves the final population)"
    taus = [0.1, 0.3, 1.0, 3.0, 10.0]
    M = 2000
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (f, pop), c in zip(pops.items(), ["tab:blue", "tab:orange", "tab:green"]):
        base, r = trial(pop, np.inf, M, seed=7)
        risk0 = r["profit"][:, -1].std()
        price_err = np.std(lo(r["prices"][1:-1]) - lo(r["truth"][:-2]))
        print(f"== {f}: price error in log-odds {price_err:.2f}")
        print(f"   forecaster:  expected profit {base.mean():+.2f} +- {base.std() / np.sqrt(M):.2f}/market "
              f"(real {r['profit'][:, -1].mean():+.2f}), risk {risk0:.1f}, profit/risk {base.mean() / risk0:.3f}")
        rows = []
        for tau in taus:
            p, rk = trial(pop, tau, M, seed=7)
            diff, risk = p - base, rk["profit"][:, -1].std()
            rows.append((p.mean(), 2 * diff.std() / np.sqrt(M), p.mean() / risk))
            print(f"   tau {tau:4.1f}:    expected profit {p.mean():+.2f}/market, vs forecaster {diff.mean():+.3f}"
                  f" +- {diff.std() / np.sqrt(M):.3f}, risk {risk:.1f}, profit/risk {p.mean() / risk:.3f}")
        g, name = np.array(rows), f.removesuffix(".npz")
        ax[0].errorbar(taus, g[:, 0], yerr=g[:, 1], marker="o", capsize=3, color=c, label=f"{name} Kalman")
        ax[0].axhline(base.mean(), color=c, ls="--", lw=1, label=f"{name} forecaster")
        ax[1].plot(taus, g[:, 2], marker="o", color=c, label=f"{name} Kalman")
        ax[1].axhline(base.mean() / risk0, color=c, ls="--", lw=1, label=f"{name} forecaster")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("tau: how noisy the Kalman trader thinks the price is (log-odds)")
    ax[0].set_title("expected profit per market (+-2 s.e. of the difference)")
    ax[1].set_title("profit per unit of risk (mean / s.d. per market)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_kalman.png", dpi=120)
    print("chart saved to figures/phase4_kalman.png")
