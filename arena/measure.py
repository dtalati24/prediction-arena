"""Phase 3: measuring how good the market is.

measure(r) turns one simulate() result into numbers. Because the simulator knows the true
probability, most measures compare the price with the truth directly instead of waiting for
outcomes, which is far less noisy.

Run `python -m arena.measure` to check every measure on fake markets where the answer is known.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import kurtosis

from arena.market import lo

HORIZON = 20  # ticks after news to follow the price


def autocorr(x, k):
    """Autocorrelation at lag k along time (axis 0), pooled across markets."""
    x = x - x.mean()
    return (x[k:] * x[:-k]).sum() / (x * x).sum()


def measure(r):
    p, v = r["prices"], r["truth"]
    T = len(p) - 1
    inner = slice(1, T)  # skip tick 0 (everything is 0.5) and the end (truth is exactly 0 or 1)
    lp, lv = lo(p[inner]), lo(v[inner])
    out = {"gap": np.abs(p - v).mean()}

    # Calibration: when the price says x, is the true probability x? Weighted over 10 price bins.
    bins = np.minimum((p[inner] * 10).astype(int), 9).ravel()
    out["calibration"] = np.abs(np.bincount(bins, (v[inner] - p[inner]).ravel(), 10)).sum() / bins.size

    # Favourite-longshot bias. Slope of true log-odds on price log-odds:
    # > 1 prices too timid (longshots overpriced, favourites underpriced), < 1 prices too extreme.
    out["extremity_slope"] = np.polyfit(lp.ravel(), lv.ravel(), 1)[0]
    cheap = p[inner] < 0.1
    out["longshot_overpricing"] = (p[inner] - v[inner])[cheap].mean() if cheap.any() else np.nan

    # News: how much of a news jump in the truth the price has absorbed k ticks later
    # (regression of price move on truth jump). 1 = fully priced in, > 1 = overreaction.
    # news[t] lands in truth[t+1]; bots first see it at tick t+1, so prices[t+2] is the first reaction.
    t, m = np.nonzero(r["news"][:T - HORIZON - 1])
    jump = lo(v[t + 1, m]) - lo(v[t, m])
    base = lo(p[t + 1, m])
    resp = np.array([((lo(p[t + 1 + k, m]) - base) * jump).sum() / (jump ** 2).sum()
                     for k in range(1, HORIZON + 1)])
    out["news_response"] = resp
    out["news_1tick"] = resp[0]

    # Stylised facts of real returns (here: tick-to-tick change in price log-odds).
    ret = np.diff(lp, axis=0)
    out["kurtosis"] = kurtosis(ret.ravel())                                   # fat tails: > 0
    out["return_autocorr"] = autocorr(ret, 1)                                 # predictability: ~0
    out["vol_clustering"] = np.mean([autocorr(np.abs(ret), k) for k in range(1, 11)])  # > 0
    return out


def perfect(r):
    """The same markets, priced by perfect bots: the price is the true probability one tick late."""
    return {**r, "prices": np.vstack([r["prices"][:1], r["truth"][:-1]])}


SCALARS = ["gap", "calibration", "extremity_slope", "longshot_overpricing", "news_1tick",
           "kurtosis", "return_autocorr", "vol_clustering"]


def dashboard(h, path):
    """h: per-generation history with measure() keys and 'ref_' keys for the perfect market."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = dict(h)
    for pre in ["", "ref_"]:  # older histories don't store news_1tick; derive it
        h[pre + "news_1tick"] = np.asarray(h[pre + "news_response"])[:, 0]
    gens = np.arange(len(h["gap"]))
    w = min(10, len(gens))

    def smooth(y):  # 10-generation moving average, for readability
        return np.convolve(y, np.ones(w) / w, mode="valid")

    panels = [("calibration", "calibration error (0 = perfect)"),
              ("extremity_slope", "favourite-longshot slope (1 = unbiased, >1 = longshots overpriced)"),
              ("news_1tick", "share of a news jump priced in after 1 tick (1 = all)"),
              ("kurtosis", "fat tails: excess kurtosis of returns"),
              ("vol_clustering", "volatility clustering: autocorr of |returns|, lags 1-10"),
              ("return_autocorr", "predictability: lag-1 autocorr of returns")]
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    ax = ax.ravel()
    for a, (k, title) in zip(ax, panels):
        a.plot(gens, h[k], color="tab:blue", alpha=0.25, lw=0.8)
        a.plot(gens[w - 1:], smooth(h[k]), color="tab:blue", label="evolving market")
        a.axhline(np.nanmean(h["ref_" + k]), color="black", ls="--", lw=1, label="perfect bots")
        a.set_title(title, fontsize=9)
    ax[0].legend(fontsize=8)
    resp = np.asarray(h["news_response"])
    ks = np.arange(1, resp.shape[1] + 1)
    for g, c in [(0, "tab:red"), (len(resp) // 2, "tab:orange"), (len(resp) - 1, "tab:green")]:
        ax[6].plot(ks, resp[max(g - 4, 0):g + 5].mean(0), color=c, label=f"generation {g}")
    ax[6].plot(ks, np.asarray(h["ref_news_response"]).mean(0), "k--", lw=1, label="perfect bots")
    ax[6].axhline(1, color="grey", lw=0.5)
    ax[6].set_title("share of a news jump priced in, k ticks later", fontsize=9)
    ax[6].set_xlabel("ticks after news")
    ax[6].legend(fontsize=8)
    ax[7].plot(gens, h["longshot_overpricing"], color="tab:blue", alpha=0.25, lw=0.8)
    ax[7].plot(gens[w - 1:], smooth(h["longshot_overpricing"]), color="tab:blue")
    ax[7].axhline(0, color="black", ls="--", lw=1)
    ax[7].set_title("longshot overpricing: price - truth when price < 10%", fontsize=9)
    ax[8].plot(gens, h["gap"], color="black", lw=0.8)
    ax[8].set_title("price-to-truth gap", fontsize=9)
    for a in ax[[3, 4, 5, 7, 8]]:
        a.set_xlabel("generation")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    from arena.market import simulate

    # Real markets from the simulator (punters only); we then swap in prices whose answer we know.
    r = simulate(lambda t, ph, v, sh, rng: (5 * rng.standard_normal(sh.shape), np.zeros(sh.shape), np.zeros(sh.shape)),
                 50, n_markets=2000, seed=3)
    good = perfect(r)
    timid = {**good, "prices": expit(0.5 * lo(good["prices"]))}  # log-odds squashed toward 50%
    slow = good["prices"].copy()                                 # moves halfway to the truth each tick
    for t in range(1, len(slow)):
        slow[t] = expit(lo(slow[t - 1]) + 0.5 * (lo(r["truth"][t - 1]) - lo(slow[t - 1])))
    slow = {**good, "prices": slow}

    res = {"perfect": measure(good), "timid": measure(timid), "slow": measure(slow), "punters only": measure(r)}
    for name, m in res.items():
        print(f"{name:>12}: " + ", ".join(f"{k} {m[k]:.3f}" for k in SCALARS)
              + f", news response k=1/5/20 {m['news_response'][[0, 4, 19]].round(2)}")

    g, ti, sl = res["perfect"], res["timid"], res["slow"]
    assert g["calibration"] < 0.01 and abs(g["extremity_slope"] - 1) < 0.05
    assert abs(g["news_response"][0] - 1) < 0.15 and abs(g["news_response"][-1] - 1) < 0.15
    assert ti["extremity_slope"] > 1.8 and ti["longshot_overpricing"] > 0.02 and ti["calibration"] > 0.05
    assert 0.35 < sl["news_response"][0] < 0.65 and sl["news_response"][-1] > 0.9
    print("Phase 3 measure checks passed")
