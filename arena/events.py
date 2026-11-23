"""The events behind the order-book markets: a hidden lead, news jumps, and the exact probability.

The lead starts somewhere (some events open as near-certainties), does a random walk with rare news
jumps, and the event happens if it ends above 0. truth[t] is the exact chance of that, given the
lead at tick t. Each market also opens with a public reference price: the exact chance given one
public, noisy look at the starting lead.

Timing: news[t] lands between tick t and tick t + 1, so nobody may be shown it before tick t + 1.

    python -m arena.events    # the checks (docs/order-book-design.md, stage 1)
"""
import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.stats import binom

T = 200
NEWS_PROB, NEWS_SIZE = 0.02, 5.0
START_SD = np.sqrt(300)  # spread of starting leads: one whole market's typical movement
OPEN_SD = 3.07           # the public look at the start, lead points: puts its price 0.3 of log-odds off
STREAM = 1_000_003       # the events' own random stream, whatever seed the caller uses for the bots

# JUMP_ODDS[n, k]: chance of k news jumps in n ticks. More than MAX_JUMPS has chance below 1e-14.
MAX_JUMPS = int(binom.isf(1e-14, T, NEWS_PROB)) + 1
JUMP_ODDS = binom.pmf(np.arange(MAX_JUMPS + 1), np.arange(T + 1)[:, None], NEWS_PROB)


def chance(lead, P, left):
    """Exact chance that the lead ends above 0 after `left` (1..T) more ticks, if it is now normal
    with mean `lead` and variance P.

    With k jumps to come the rest of the walk is normal, variance left + NEWS_SIZE^2 k, so average
    over k. (arena.market lumps the jumps into one normal, up to 4.3 points wrong near the end.)
    """
    k = np.arange(min(left, MAX_JUMPS) + 1)
    sd = np.sqrt(np.asarray(P)[..., None] + left + NEWS_SIZE**2 * k)
    p = ndtr(np.asarray(lead)[..., None] / sd) @ JUMP_ODDS[left, k]
    return np.minimum(p, 1.0)  # the sum can round to a hair above 1


def probability(lead, left):
    """The truth: the chance when the lead is known."""
    return chance(lead, 0.0, left)


def draw(M, seed):
    """M events: lead and truth (T+1, M), news (T, M), outcome and opening reference price (M,).

    seen and open_var (M,): the public's belief about the starting lead (mean and variance).
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(STREAM,)))
    start = START_SD * rng.standard_normal(M)
    steps = rng.standard_normal((T, M))
    news = rng.random((T, M)) < NEWS_PROB
    steps += news * NEWS_SIZE * rng.standard_normal((T, M))
    lead = start + np.vstack([np.zeros(M), steps.cumsum(0)])
    outcome = lead[-1] > 0
    truth = np.vstack([probability(lead[t], T - t) for t in range(T)] + [outcome])
    shrink = START_SD**2 / (START_SD**2 + OPEN_SD**2)
    seen, open_var = shrink * (start + OPEN_SD * rng.standard_normal(M)), np.full(M, START_SD**2 * (1 - shrink))
    return dict(lead=lead, truth=truth, news=news, outcome=outcome, opening=chance(seen, open_var, T),
                seen=seen, open_var=open_var)


if __name__ == "__main__":
    from math import comb, erf, sqrt

    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}")
        if not ok:
            failed.append(name)

    def worst_z(p, y):
        """Worst gap between p and how often it happened, in standard errors, over bins of p."""
        b = np.digitize(p, [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
        var = np.bincount(b, p * (1 - p), 12)
        enough = var > 10  # for the normal approximation
        return np.max(np.abs(np.bincount(b, y - p, 12)[enough]) / np.sqrt(var[enough]))

    # 1. The formula, against the plain sum over every possible number of jumps.
    def by_hand(x, n):
        return sum(comb(n, k) * 0.02**k * 0.98**(n - k) * 0.5 * (1 + erf(x / sqrt(2 * (n + 25 * k))))
                   for k in range(n + 1))
    cases = [(x, n) for n in (1, 2, 10, 50, 100, 200) for x in (-60.0, -8.0, -1.5, 0.0, 1.5, 8.0, 60.0)]
    off = max(abs(probability(np.array([x]), n)[0] - by_hand(x, n)) for x, n in cases)
    check("exact formula", off < 1e-10, f"largest error {off:.1e}")
    def in_doubt(x, P, n):  # the lead now is x + N(0, P): add P to every variance
        return sum(comb(n, k) * 0.02**k * 0.98**(n - k) * 0.5 * (1 + erf(x / sqrt(2 * (P + n + 25 * k))))
                   for k in range(n + 1))
    off = max(abs(chance(np.array([x]), P, n)[0] - in_doubt(x, P, n)) for x, n in cases for P in (0.0, 3.0, 300.0))
    check("exact formula with doubt about the lead", off < 1e-10 and np.array_equal(chance(x := np.linspace(-9, 9, 7), 0, 50), probability(x, 50)),
          f"largest error {off:.1e}")
    x = np.linspace(-150, 150, 3001)
    ps = [probability(x, n) for n in (1, 2, 10, 100, 200)]
    check("p within 0..1, rising in the lead, symmetric",
          all(p.min() >= 0 and p.max() <= 1 and (np.diff(p) >= -1e-15).all()
              and np.allclose(p + p[::-1], 1, atol=1e-12) for p in ps))

    # Many markets, keeping only what the checks need.
    CHECK_AT = [0, 100, 190, 199]  # 200, 100, 10 and 1 ticks left
    kept = {k: [] for k in ("p", "y", "gain", "after", "after_y", "open", "start")}
    jumps, calm, n_news = [], [], 0
    for i in range(10):
        e = draw(20_000, [7, i])
        p, y, news = e["truth"], e["outcome"], e["news"]
        kept["p"].append(p[CHECK_AT])
        kept["y"].append(y)
        kept["start"].append(e["lead"][0])
        kept["open"].append(e["opening"])
        # real minus expected profit of buying the favourite, one share a tick
        kept["gain"].append((np.sign(p[:T] - 0.5) * (y - p[:T])).sum(0))
        late = news[100:T - 1]  # each market's first news from tick 100 on, valued once it has landed
        has = late.any(0)
        kept["after"].append(p[late.argmax(0)[has] + 101, np.flatnonzero(has)])
        kept["after_y"].append(y[has])
        step = np.diff(e["lead"][:21], axis=0)
        jumps.append(step[news[:20]])
        calm.append(step[~news[:20]])
        n_news += news.sum()
        assert np.array_equal(p[T], y), "truth must end at the outcome"
    k = {name: np.concatenate(v, axis=-1) for name, v in kept.items()}
    M = len(k["y"])

    # 2. The gate: outcomes match p within sampling error, starts included.
    for j, t in enumerate(CHECK_AT):
        z = worst_z(k["p"][j], k["y"])
        check(f"outcomes match p, {T - t} ticks left", z < 4, f"worst bin {z:.1f} standard errors")
    z = worst_z(k["after"], k["after_y"])
    check("p is fair once news has landed", z < 4, f"worst bin {z:.1f} standard errors")
    gain, se = k["gain"].mean(), k["gain"].std() / np.sqrt(M)
    check("expected profit equals real profit", abs(gain) < 4 * se, f"difference {gain:+.3f} +- {se:.3f} a market")

    # 3. The events are the ones in the design.
    sd = k["start"].std()
    check("starting leads spread by sqrt(300)", abs(sd / np.sqrt(300) - 1) < 0.01, f"s.d. {sd:.2f}")
    p0 = k["p"][0]
    outside, decided = np.mean(abs(p0 - 0.5) > 0.4), np.mean(abs(p0 - 0.5) > 0.49)
    check("realistic openings", abs(outside - 0.20) < 0.01 and decided < 0.03,
          f"{outside:.1%} open outside 10-90c, {decided:.1%} beyond 1-99c")
    rate = n_news / (T * M)
    check("news on 2% of ticks", abs(rate - 0.02) < 4 * np.sqrt(0.02 * 0.98 / (T * M)), f"{rate:.4f}")
    v_news, v_calm = np.concatenate(jumps).var(), np.concatenate(calm).var()
    check("a step varies by 1, a news step by 26", abs(v_calm - 1) < 0.01 and abs(v_news / 26 - 1) < 0.03,
          f"{v_calm:.3f}, {v_news:.2f}")

    # 4. The public opening price is the start plus noise, and tells nothing more.
    inner = abs(p0 - 0.5) < 0.4999999
    noise = logit(np.clip(k["open"], 1e-15, 1 - 1e-15))[inner] - logit(p0[inner])
    middle = (abs(p0 - 0.5) < 0.4)[inner]  # OPEN_SD is set on 10-90c events (the research design)
    size, everywhere = np.sqrt(np.mean(noise[middle] ** 2)), np.sqrt(np.mean(noise**2))
    check("opening price is off by 0.3 of log-odds", abs(size - 0.3) < 0.01 and abs(noise.mean()) < 0.005,
          f"r.m.s. {size:.3f} at 10-90c ({everywhere:.3f} everywhere), mean {noise.mean():+.4f}")
    z = worst_z(k["open"], k["y"])
    check("opening price is a fair price", z < 4, f"worst bin {z:.1f} standard errors")
    luck = (k["y"] - p0)[inner]
    check("opening noise doesn't predict the outcome",
          abs((noise * luck).mean()) <= 4 * (noise * luck).std() / np.sqrt(len(noise)))

    # 5. Seeds: reproducible, different seeds differ, and no numbers shared with a caller's generator.
    a, b, c = draw(400, [3, 1]), draw(400, [3, 1]), draw(400, [3, 2])
    check("same seed, same events", all(np.array_equal(a[key], b[key]) for key in a) and not np.array_equal(a["lead"], c["lead"]))
    own = draw(400, 5)["lead"]
    theirs = np.random.default_rng(5).standard_normal((T, 400))
    shared = max(abs(np.corrcoef(np.diff(own, axis=0)[0], theirs[0])[0, 1]),
                 abs(np.corrcoef(own[0] + 1e-9 * np.arange(400), theirs[0])[0, 1]))
    check("random numbers not shared with the caller", shared < 0.25, f"correlation {shared:.2f}")

    assert not failed, f"{len(failed)} event checks failed: " + "; ".join(failed)
    print("event checks passed")
