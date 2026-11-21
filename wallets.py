"""Phase 4, step 3: on-chain transparency.

On Polymarket every trade settles on a public blockchain: anyone can see every wallet's position
and trades under a pseudonymous address, but not who is behind it or what their strategy is. The
simulator works the same way: every order function receives `shares`, all wallets' positions
after the last tick's trades. A wallet is a column; a bot may read any wallet's positions and its
public track record (profit on past markets), never its type or genome.

Does that ledger hold exploitable information? This tests the simplest user of it: a wallet
watcher that copies (or fades) the wallets at the top (or bottom) of a public leaderboard.

Run `python wallets.py` for the checks and the experiment (needs the pop*.npz files from evolve.py).
"""
import numpy as np

from evolve import N_EVO, N_PUNT, drop_in, labels
from market import simulate

WALLETS = N_EVO + N_PUNT  # watchable wallets; a dropped-in trader sits in the column after them


def idle(base, err, sigma, size):
    """make_agent for drop_in: a trader that never trades."""
    def fn(t, price_hist, truth, shares, rng):
        d, a, e = base(t, price_hist, truth, shares, rng)
        d[:, -1], a[:, -1] = 0, 0
        return d, a, e
    return fn


def leaderboard(pop, n_markets, seed):
    """Public track record from a separate batch of past markets: every wallet's total profit,
    and its profit per unit of risk (mean / s.d. of its profit per market)."""
    _, r = drop_in(pop, idle, n_markets, seed)
    p = r["profit"][:, :WALLETS]
    return {"profit": p.sum(0), "profit per risk": p.mean(0) / (p.std(0) + 1e-9)}


def watcher(watch, sign):
    """make_agent for drop_in: hold sign x the average position of the watched wallets, as last
    seen on chain (after the previous tick's trades). sign +1 copies them, -1 fades them."""
    def make(base, err, sigma, size):
        def fn(t, price_hist, truth, shares, rng):
            d, a, e = base(t, price_hist, truth, shares, rng)
            d[:, -1] = sign * shares[:, watch].mean(1) - shares[:, -1]  # market order to the target
            a[:, -1] = 0  # not price-sensitive (base fills this column as an insider slot)
            return d, a, e
        return fn
    return make


if __name__ == "__main__":
    import glob

    # Check: a watcher copying (fading) a wallet that always holds +10 ends up holding +10 (-10).
    def holder(t, price_hist, truth, shares, rng):
        d, z = np.zeros(shares.shape), np.zeros(shares.shape)
        d[:, 0] = 10 - shares[:, 0]
        return d, z, z.copy()
    for sign in [1, -1]:
        r = simulate(watcher([0], sign)(holder, None, None, None), 2, n_markets=50, seed=0)
        assert np.allclose(r["shares"][:, 1], 10 * sign), "watcher doesn't mirror the wallet"
    print("wallet checks passed")

    # Experiment: in each evolved population, rank wallets on 2000 past markets, then drop in one
    # watcher on 2000 new markets. Same seed for every watcher, so differences are paired.
    pops = {f: dict(np.load(f)) for f in sorted(glob.glob("pop*.npz")) if "fee" not in f}  # no-fee populations
    assert pops, "run evolve.py first (it saves the final population)"
    M = 2000
    names = np.array(["momentum", "contrarian", "forecaster", "punter"])
    for f, pop in pops.items():
        boards = leaderboard(pop, M, seed=100)
        kind = np.r_[labels(pop), np.full(N_PUNT, 3)]
        print(f"== {f}")
        for by, board in boards.items():
            rank = np.argsort(-board)  # best first
            for k in [1, 10]:
                for side, watch in [("top", rank[:k]), ("bottom", rank[-k:])]:
                    who = ", ".join(f"{n} {c}" for n, c in zip(*np.unique(names[kind[watch]], return_counts=True)))
                    for sign, verb in [(1, "copy"), (-1, "fade")]:
                        p, r = drop_in(pop, watcher(watch, sign), M, seed=7)
                        print(f"   by {by}: {verb} {side} {k:2d} ({who}): expected profit {p.mean():+.2f}"
                              f" +- {p.std() / np.sqrt(M):.2f}/market, risk {r['profit'][:, -1].std():.1f}", flush=True)
