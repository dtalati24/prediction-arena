"""Retail's numbers for the order-book ecology, measured from one month of Polymarket fills.

Data (both CC BY 4.0; not in git, put them in data/):
- B. Qin and R. Yang, "Polymarket-v1 Database" (arXiv 2606.04217): OrderFilled/2024_07.parquet, 34,056,680 bytes,
  https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1 (revision 62f49f61), sha256 03dc8c08aa30edaa...
- C. G. Kurg, Vultax Research Note 2026-017: realized-cost-daily.csv, 946,182 bytes,
  https://zenodo.org/records/24292871, md5 03cceea170b3182abce7aef217a5a39c. Only a cross-check.

Nothing is computed unless the cleaning reproduces the second file's July 2024 figures. Retail: wallets
under $10,000 in the month, mostly takers, not machine-like (at most 50 fills a day, 1,000 a month).
The tolerance is a distance from the last price (spread and price moves included), not what retail
would have paid, and sizes are what filled. Kalshi's API isn't used: its terms don't allow this.

    python -I arena/retail_data.py data/polymarket_v1/OrderFilled_2024_07.parquet \
        data/vultax_cost/realized-cost-daily.csv runs/retail_numbers.json    # needs pyarrow
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EXCHANGES = ["0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e", "0xc5d563a36ae78145c45a50134d48a1215220f80a"]

t = pq.read_table(sys.argv[1])
parts = pc.split_pattern(t["id"], "_")
block = pc.list_element(parts, 1).cast(pa.int64()).to_numpy()
log = pc.list_element(parts, 2).cast(pa.int64()).to_numpy()
t = t.take(np.lexsort((log, block)))
block = np.sort(block)  # the same order
summary = pc.is_in(pc.utf8_lower(t["taker"]), value_set=pa.array(EXCHANGES)).to_numpy(zero_copy_only=False)
R = len(summary)
both = pc.dictionary_encode(pa.concat_arrays([t["maker"].combine_chunks(), t["taker"].combine_chunks()]))
codes = both.indices.to_numpy()
maker, taker, n_wallets = codes[:R], codes[R:], len(both.dictionary)
enc = pc.dictionary_encode(t["token_asset_id"].combine_chunks())
token, n_tokens = enc.indices.to_numpy(), len(enc.dictionary)
q, usd, price, ts = (t[c].to_numpy() for c in ("token_amount", "usdc_amount", "price", "block_timestamp"))
maker_buys = pc.equal(t["maker_asset_id"], "0").to_numpy(zero_copy_only=False)

# Taker orders: a summary row and the fills before it in the block with that wallet as taker.
order_of, pending = np.full(R, -1), defaultdict(list)
heads, last = [], -1
for i, (b, s, mk, tk) in enumerate(zip(block.tolist(), summary.tolist(), maker.tolist(), taker.tolist())):
    if b != last:
        pending.clear()
        last = b
    if not s:
        pending[tk].append(i)
        continue
    got = pending.pop(mk, None)
    if got:
        order_of[got] = len(heads)
        heads.append(i)
heads, F = np.array(heads), np.flatnonzero(~summary)
ok = order_of[F] >= 0
O = order_of[F][ok]
filled = np.bincount(O, q[F][ok], len(heads))
adds_up = np.abs(filled - q[heads]) <= 1e-3 + 1e-3 * q[heads]

# YES and NO tokens: one order filled on both.
tf, th = token[F][ok], token[heads][O]
cross = tf != th
pairs, count = np.unique(np.sort(np.c_[tf[cross], th[cross]], 1), axis=0, return_counts=True)
partners = np.bincount(pairs.ravel(), minlength=n_tokens)
rep = np.arange(n_tokens)
done = np.zeros(n_tokens, bool)
for (a, b) in pairs[np.argsort(-count)]:
    if not done[a] and not done[b]:
        rep[b], done[a], done[b] = a, True, True

f = F[ok]
market = rep[tf]
p = np.where(tf == market, price[f], 1 - price[f])
d = np.where(maker_buys[f], -1, 1) * np.where(tf == market, 1, -1)  # +1: the taker bought this axis
taker_paid = q[f] * np.where(tf == th, price[f], 1 - price[f])

K = len(heads)
low, high = np.full(K, 9.0), np.full(K, -9.0)
np.minimum.at(low, O, p)
np.maximum.at(high, O, p)
size = filled
avg = np.bincount(O, p * q[f], K) / size
o_dir, o_market = np.zeros(K, int), np.zeros(K, np.int64)
o_dir[O], o_market[O] = d, market
one_way = np.abs(np.bincount(O, d, K)) == np.bincount(O, minlength=K)
walk = 100 * np.where(o_dir > 0, avg - low, high - avg)
band = (avg >= .1) & (avg <= .9)
bucket = np.digitize(size, [100, 1000, 10_000])

print("== sealed sanity checks")
print(f"fills {len(F):,}; in an order {ok.mean():.3%}; orders {K:,} of {summary.sum():,} summaries; add up {adds_up.mean():.3%}; one direction {one_way.mean():.3%}")
print(f"tokens {n_tokens:,}; paired {done.mean():.1%} ({done[tf].mean():.1%} of fills); more than one partner {np.sum(partners > 1)}")
BUCKETS = ["under 100", "100 to 1,000", "1,000 to 10,000", "10,000 and over"]
theirs = defaultdict(lambda: np.zeros(3))
with open(sys.argv[2], newline="") as fh:
    for r in csv.DictReader(fh):
        if r["date"].startswith("2024-07"):
            n = float(r["orders"])
            key = r["price_band"] == "10c to 90c", BUCKETS.index(r["size_bucket_contracts"])
            theirs[key] += n, n * float(r["cost_mean_cents"] or 0), float(r["orders_two_or_more_levels"])
close = True
for inside in (True, False):
    for b, name in enumerate(BUCKETS):
        on = (band == inside) & (bucket == b)
        n, c, two = theirs[(inside, b)]
        print(f"  {'10-90c ' if inside else 'outside'} {name:16s} orders {on.sum():8,} / theirs {n:8,.0f} | mean walk {walk[on].mean():.3f}c / {c / n:.3f}c "
              f"| two or more levels {np.mean(high[on] - low[on] > 1e-7):.1%} / {two / n:.1%}")
        close &= abs(on.sum() / n - 1) < 0.05 and abs(walk[on].mean() - c / n) < 0.05
good = ok.mean() > .99 and adds_up.mean() > .99 and one_way.mean() > .99 and np.sum(partners > 1) < .002 * n_tokens and close
if not good:
    sys.exit("sanity checks failed: none of the sealed numbers computed")
print("sanity checks passed\n")

# Wallets.
mk, tk, qf = maker[f], taker[f], q[f]
traded = np.bincount(mk, usd[f], n_wallets) + np.bincount(tk, taker_paid, n_wallets)
took, made = np.bincount(tk, qf, n_wallets), np.bincount(mk, qf, n_wallets)
day = (ts[f] - ts.min() // 86400 * 86400) // 86400
per_day = (np.bincount(tk * 31 + day, minlength=31 * n_wallets) + np.bincount(mk * 31 + day, minlength=31 * n_wallets)).reshape(n_wallets, 31)
active = took + made > 0
human = (per_day.max(1) <= 50) & (per_day.sum(1) <= 1000)
mostly_taker = took > 0.5 * (took + made)

vol = np.bincount(market, qf, n_tokens)
eligible = vol >= 1000


def wq(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.array([x[o][min(np.searchsorted(c, v * c[-1]), len(x) - 1)] for v in qs])


# A tick is 1/200 of each market's month; the price is the average trade of the last earlier tick with trades.
first, final = np.full(n_tokens, 2**62), np.zeros(n_tokens, np.int64)
np.minimum.at(first, market, ts[f])
np.maximum.at(final, market, ts[f])
span = np.maximum(final - first, 1)
tick = np.minimum((ts[f] - first[market]) * 200 // span[market], 199)
key = market.astype(np.int64) * 200 + tick
ticks, inv = np.unique(key, return_inverse=True)
mean_p = np.bincount(inv, p * qf) / np.bincount(inv, qf)
o_key, o_wallet, o_inv = np.zeros(K, np.int64), np.zeros(K, np.int64), np.zeros(K, np.int64)
o_key[O], o_wallet[O], o_inv[O] = key, tk, inv
before = np.maximum(o_inv - 1, 0)
has_ref = (o_inv > 0) & (ticks[before] // 200 == o_market)
ref = np.where(has_ref, mean_p[before], np.nan)
worst = np.clip(np.where(o_dir > 0, high, low), .0005, .9995)
lo = lambda x: np.log(x / (1 - x))
with np.errstate(invalid="ignore"):
    past = o_dir * (lo(worst) - lo(ref))

out, thresholds = {}, (1_000, 10_000, 100_000)
for limit in thresholds:
    retail = active & human & mostly_taker & (traded < limit)
    share = (np.bincount(market, qf * retail[tk], n_tokens) / np.maximum(vol, 1e-9))[eligible]
    on = retail[o_wallet] & eligible[o_market]
    w = 1 / np.bincount(o_market[on], minlength=n_tokens)[o_market[on]]
    s = size[on]
    keep = s <= np.percentile(s, 99.9)
    median = wq(s, w, [.5])[0]
    mean = np.average(s[keep], weights=w[keep])
    use = on & has_ref
    use[use] = (ref[use] >= .1) & (ref[use] <= .9)
    w2 = 1 / np.bincount(o_market[use], minlength=n_tokens)[o_market[use]]
    tol = wq(past[use], w2, [.25, .5, .75, .9, .95])
    out[str(limit)] = dict(share=float(np.median(share)), order_median=float(median), order_mean=float(mean),
                           tolerance_p75=float(tol[2]), tolerance=float(tol[3]), tolerance_p95=float(tol[4]))
    print(f"== retail under ${limit:,}{'  <- SEALED' if limit == 10_000 else '  (sensitivity)'}")
    print(f"  wallets {retail.sum():,} of {active.sum():,} ({retail.sum() / active.sum():.1%}); their share of money traded {traded[retail].sum() / traded.sum():.1%}")
    print(f"  share of a market's contracts: median {np.median(share):.1%} (quartiles {np.percentile(share, 25):.1%}, {np.percentile(share, 75):.1%}); "
          f"all markets pooled {np.bincount(market, qf * retail[tk], n_tokens)[eligible].sum() / vol[eligible].sum():.1%}")
    print(f"  order: median {median:.1f}, mean {mean:.1f} contracts ({on.sum():,} orders)")
    print(f"  past the price, log-odds p25/50/75/90/95: {tol.round(3)} ({use.sum():,} orders); p90 at 50c = {50 * np.tanh(tol[3] / 2):.1f}c")

mk_m = eligible
print(f"== held out: eligible markets {mk_m.sum():,}; median volume {np.median(vol[mk_m]):,.0f} contracts; mean {vol[mk_m].mean():,.0f}")
for name, g in (("maker", made), ("taker", took)):
    g = np.sort(g[g > 0])[::-1]
    print(f"   top 1% of {name} wallets: {g[:len(g) // 100].sum() / g.sum():.1%} of {name} contracts")
print(f"   machine-like wallets {np.mean(~human[active]):.1%} of wallets, {traded[~human].sum() / traded.sum():.1%} of money traded")
out["markets"] = dict(eligible=int(eligible.sum()), median_volume=float(np.median(vol[eligible])))
with open(sys.argv[3], "w") as fh:
    json.dump(out, fh, indent=1)
