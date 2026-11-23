"""The order-book ecology: every role of a prediction market, trading through arena.book.

Retail, institutions and (in stress tests) insiders are fixed. 100 professionals are six kinds of
hedge fund plus market makers; here their genes are frozen (evolution is stage 4).

Units: a trader's capital in a market, in dollars, is also its unit of size in contracts, the same
for YES and NO. A tick's decisions use only what is known then: the public opening, when news landed,
and for those who pay for research, noisy looks at the hidden lead (docs/research-error-design.md).

    python -m arena.ecology           # the checks (docs/order-book-design.md, stage 3)
    python -m arena.ecology search    # maker settings, each at the same retail volume (overwrites the last one)
    python -m arena.ecology gate      # the stage 3 gate on fresh seeds, speed included
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import expit

from arena import RUNS, events
from arena.book import MAKER, MARKET, PRICE, QTY, SIDE, TAKER, Books, arrival
from arena.market import lo
from arena.measure import measure

T, MARKETS = events.T, 400
MOMENTUM, CONTRARIAN, FORECASTER, NEWS, COPIER, FADER, MARKET_MAKER = range(7)
KINDS = ["momentum", "contrarian", "forecaster", "news", "copier", "fader", "maker"]
EACH_FUND, N_MAKERS, PROS = 15, 10, 100
INSTITUTIONS, INSIDERS, FIXED = slice(100, 103), slice(103, 106), 106  # columns; retail's order slots follow

# Capital in dollars (assumptions: see the design).
FUND_MEDIAN, FUND_SD, MAKER_CAPITAL = 10_000, 1.0, 25_000
RETAIL_ACCOUNTS, RETAIL_MEDIAN, RETAIL_SD = 2750, 100, 2.0
INST_CAPITAL, INSIDER_CAPITAL = 250_000, 5_000
MIN_STAKE = 5  # dollars: a professional enters fewer markets rather than less than this in each

# The fixed roles. Retail's numbers come from `python -I arena/retail_data.py` on one month of Polymarket
# fills (July 2024; Qin and Yang, Polymarket-v1, CC BY 4.0): wallets under $10k, mostly takers, no bots.
# Fills don't show intentions: using their sizes as intended sizes, and their p90 distance as a limit,
# are assumptions (the design's audit has what that gets wrong).
ORDER_MEDIAN, ORDER_MEAN = 20.0, 205.3                    # filled sizes there, intended sizes here
ORDER_SD = np.sqrt(2 * np.log(ORDER_MEAN / ORDER_MEDIAN))
RETAIL_SHARE = 0.1664                                     # retail's part of the median market's contracts
RETAIL_LIMIT = 0.759                                      # log-odds: p90 distance from the last price, moves included
INST_LIMIT = 0.2                                          # log-odds past the price when the order arrived
INST_MARKETS, INST_SHARE = 20, 0.25                       # each works 25% of its capital across 20 markets
INSIDER_MARKETS, INSIDER_LEAD = 4, 10

# Frozen genes, written down before the first run. Only MAKER and retail's order count are varied.
TREND_SIZE, TREND_LIMIT = 0.05, 0.1
INFORMED_SIZE, AGGRESSION, SPEND = 0.9, 0.5, 0.25
COPY_SIZE, COPY_LAG, COPY_LIMIT = 0.25, 200, 0.1
MAKER_GENES = dict(half=2.0, size=0.2, shade=5.0, response=2.0, widen=50.0, pull=1)
PULL = 50                                                 # widening this far means pulling the quotes
LOOK_SD = 3.73  # lead points, a look's noise at SPEND: today's average error of 0.057 (the research design)
STEP_VAR = 1 + events.NEWS_SIZE**2  # a tick's move when news lands
ERR_CORR, LUCK_SD, ESTIMATE_DELAY = 0.5, 0.2, 2
EPS = 1e-9  # rounding to cents has to mirror exactly between YES and NO

# The random streams (events has its own, so every setting meets the same events).
POPULATION, RETAIL, INST, ENTRY, ERRORS, ARRIVALS, NEWS_ERRORS = range(1, 8)


def stream(seed, k):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))


def population(seed, maker=None):
    """100 professionals with frozen genes, and the retail accounts. maker: MAKER_GENES to change."""
    rng = stream(seed, POPULATION)
    kind = np.repeat(np.arange(7), [EACH_FUND] * 6 + [N_MAKERS])
    is_kind = [kind == k for k in range(7)]
    trend, watch = is_kind[MOMENTUM] | is_kind[CONTRARIAN], is_kind[COPIER] | is_kind[FADER]
    informed, mm = is_kind[FORECASTER] | is_kind[NEWS], is_kind[MARKET_MAKER]
    g = {**MAKER_GENES, **(maker or {})}
    pop = dict(
        kind=kind,
        capital=np.where(mm, MAKER_CAPITAL, FUND_MEDIAN * np.exp(FUND_SD * rng.standard_normal(PROS))).astype(float),
        size=np.select([trend, informed, watch], [TREND_SIZE, INFORMED_SIZE, COPY_SIZE], float(g["size"])),
        limit=np.select([trend, watch], [TREND_LIMIT, COPY_LIMIT], 0.0),
        agg=np.where(informed, AGGRESSION, 0.0),
        spend=np.where(informed, SPEND, 0.0),
        z=LUCK_SD * rng.standard_normal(PROS),
        lag=np.where(watch, COPY_LAG, 1),
        rank=np.zeros(PROS),
        board=np.arange(PROS) % 2,
        retail=RETAIL_MEDIAN * np.exp(RETAIL_SD * rng.standard_normal(RETAIL_ACCOUNTS)),
        **{k: np.full(PROS, float(g[k])) for k in ("half", "shade", "response", "widen", "pull")},
    )
    for k in (MOMENTUM, CONTRARIAN):
        pop["lag"][is_kind[k]] = np.linspace(2, 30, EACH_FUND).round()
    pop["rank"][is_kind[COPIER]] = np.linspace(0.0, 0.3, EACH_FUND)  # copy near the top
    pop["rank"][is_kind[FADER]] = np.linspace(0.7, 1.0, EACH_FUND)   # fade near the bottom
    return pop


def look_sd(spend, z):
    """Noise of each researcher's look: 4x the spend halves its error; luck z works as exp(z) on the error."""
    return LOOK_SD * SPEND / np.where(spend > 0, spend, np.nan) * np.exp(2 * z)


def estimates(e, sd, rng, sign=1):
    """Each researcher's best guess, in log-odds, tick by tick: (M, len(sd)) per tick.

    A tick's look sees the lead ESTIMATE_DELAY ticks earlier, plus noise (half of its variance shared
    in the market). Given the public news times the lead is linear and normal, so this Kalman filter is
    exact and nothing in a researcher's own past can improve on it. sd nan: no research.
    """
    M, n = e["lead"].shape[1], len(sd)
    m, P = np.repeat(sign * e["seen"][:, None], n, 1), np.repeat(e["open_var"][:, None], n, 1)
    q, looks = 1 + events.NEWS_SIZE**2 * e["news"], np.isfinite(sd)
    sd = np.where(looks, sd, 0.0)
    for t in range(T):
        s = max(t - ESTIMATE_DELAY, 0)
        if t >= ESTIMATE_DELAY:
            if s:
                P = P + q[s - 1][:, None]
            noise = sign * sd * (np.sqrt(ERR_CORR) * rng.standard_normal((M, 1)) + np.sqrt(1 - ERR_CORR) * rng.standard_normal((M, n)))
            gain = np.where(looks, P / (P + sd**2), 0.0)
            m, P = m + gain * (sign * e["lead"][s][:, None] + noise - m), P * (1 - gain)
        yield lo(events.chance(m, P + q[s:t].sum(0)[:, None], T - t))


def lead_at(price, left):
    """The lead at which the truth would be this price (0..1), by bisection."""
    low, high = np.full(np.shape(price), -400.0), np.full(np.shape(price), 400.0)
    for _ in range(60):
        mid = (low + high) / 2
        above = events.probability(mid, left) > price
        low, high = np.where(above, low, mid), np.where(above, mid, high)
    return (low + high) / 2


def jump_move(ref, look, sd, left):
    """A news trader's view of a jump's effect, in log-odds, from one noisy look at the tick's move.

    ref (k,): the last price in cents, read as the lead it implies before the move. look (k, n): the
    move plus noise of s.d. sd (n,). left: ticks to go once it has landed. Never sees the true move.
    """
    gain = np.where(np.isfinite(sd), STEP_VAR / (STEP_VAR + np.nan_to_num(sd) ** 2), 0.0)  # nan: no research
    before = lead_at(ref / 100, left + 1)[:, None]
    return lo(events.chance(before + gain * look, STEP_VAR * (1 - gain), left)) - lo(ref / 100)[:, None]


def price_cap(ref, limit):
    """A limit in log-odds past the price (cents), as cents: the same both ways, at least 1c."""
    L = lo(ref / 100)
    return np.maximum(np.rint(50 * (expit(L + limit) - expit(L - limit))), 1)


def down(x):
    return np.clip(np.floor(x + EPS), 1, 99).astype(np.int64)


def up(x):
    return np.clip(np.ceil(x - EPS), 1, 99).astype(np.int64)


def limit_order(n, buy_at, sell_at):
    """Signed contracts as (bid, bid_n, ask, ask_n)."""
    n = np.rint(n).astype(np.int64)
    return down(buy_at), np.maximum(n, 0), up(sell_at), np.maximum(-n, 0)


def forecaster_orders(estimate, ref, held, capital, size, agg):
    """One limit order, part of the way from the price to where it would stop trading."""
    unit = size * capital  # contracts per unit of log-odds
    stop = estimate - np.divide(held, unit, out=np.zeros(unit.shape), where=unit > 0)
    now = lo(ref / 100)[:, None]
    at = 100 * expit(now + agg * (stop - now))
    return limit_order(unit * (1 - agg) * (stop - now), at, at)


def maker_quotes(centre, inventory, capital, genes, since_news):
    """Both quotes around its own centre. since_news (markets,): ticks since news last landed."""
    shift = genes["shade"] * np.divide(inventory, capital, out=np.zeros(capital.shape), where=capital > 0)
    c = np.clip(centre - shift, 2, 98)
    reacting = since_news[:, None] < genes["pull"]
    half = genes["half"] + reacting * genes["widen"]
    half = np.maximum(np.minimum(half, np.minimum(c - 1, 99 - c)), 0.5)  # narrow evenly at the edges
    n = np.rint(genes["size"] * capital * ~(reacting & (genes["widen"] >= PULL))).astype(np.int64)
    return down(c - half), n, up(c + half), n


def maker_move(centre, net, quoted_bid, quoted_ask, response):
    """net: contracts customers (everyone but the makers) bought minus sold. Never reads a price."""
    quoted = np.where(net > 0, quoted_ask, quoted_bid)
    f = np.divide(net, quoted, out=np.sign(net).astype(float), where=quoted > 0)
    return np.clip(centre + response * np.clip(f, -1, 1)[:, None], 1, 99)


def retail_money(pop, orders):
    """Each account's dollars: one turnover of the orders (50c a contract), split as pop["retail"]."""
    total = max(round(orders * MARKETS), RETAIL_ACCOUNTS) * ORDER_MEAN * 0.5 if orders else 0
    return pop["retail"] / pop["retail"].sum() * total


def retail_plan(money, orders, rng):
    """A generation of retail orders: whose, when, where (market and slot), how big and which side."""
    A = len(money)
    total = max(round(orders * MARKETS), A) if orders else 0
    who = np.arange(0)
    if total:  # everyone once, then in proportion to money
        who = rng.permutation(np.r_[np.arange(A), rng.choice(A, total - A, p=money / money.sum())])
    market = rng.integers(0, MARKETS, total)
    slot = np.zeros(total, np.int64)
    by_market = np.argsort(market, kind="stable")
    slot[by_market] = np.arange(total) - np.searchsorted(market[by_market], market[by_market])
    n = np.maximum(np.rint(ORDER_MEDIAN * np.exp(ORDER_SD * rng.standard_normal(total))), 1).astype(np.int64)
    return dict(who=who, market=market, slot=slot, n=n, tick=rng.integers(0, T, total),
                side=2 * rng.integers(0, 2, total) - 1)


def entries(capital, spend, rng):
    """Cents per market (markets, pros), research paid in cents, markets entered."""
    cents, each = np.floor(capital * 100).astype(np.int64), np.ceil(spend * 100 - EPS).astype(np.int64)
    k = np.minimum(MARKETS, cents // (MIN_STAKE * 100))
    k = np.where(each > 0, np.minimum(k, cents // np.maximum(each, 1)), k)
    research = k * each
    budget = np.zeros((MARKETS, len(cents)), np.int64)
    for j in range(len(cents)):
        chosen = rng.permutation(MARKETS)[:k[j]]
        if k[j]:
            budget[chosen, j] = (cents[j] - research[j]) // k[j]
    return budget, research, k


def targets(pop, profit, capital):
    """Each copier's and fader's wallet: the one in its spot on last generation's leaderboard."""
    ranked = [np.flatnonzero(capital > 0)]
    boards = [profit[ranked[0]], profit[ranked[0]] / capital[ranked[0]]]  # profit, return on capital
    ranked = [ranked[0][np.argsort(-b, kind="stable")] for b in boards]
    out = np.full(PROS, -1)
    for j in np.flatnonzero(np.isin(pop["kind"], (COPIER, FADER))):
        others = ranked[pop["board"][j]]
        others = others[others != j]
        if len(others):
            out[j] = others[round(pop["rank"][j] * (len(others) - 1))]
    return out


def run(pop, orders, seed, targets=None, without=(), insiders=False, event=None, mirror=False, trace=False):
    """One generation: MARKETS markets side by side, every role, settled.

    orders: retail orders a market. targets: each copier's wallet (none on a first generation).
    without: role names switched off (KINDS, "retail", "institutions"). mirror: the same generation
    with YES and NO swapped, for the symmetry check. Money is in cents except expected (dollars).
    """
    M, kind, sign = MARKETS, pop["kind"], -1 if mirror else 1
    e = events.draw(M, seed) if event is None else event
    truth, news, outcome, opening, lead = e["truth"], e["news"], e["outcome"], e["opening"], e["lead"]
    if mirror:
        truth, outcome, opening = 1 - truth, ~outcome, 1 - opening
    col = [np.flatnonzero(kind == k) for k in range(7)]
    mm, fc, nw = col[MARKET_MAKER], col[FORECASTER], col[NEWS]
    trend = np.r_[col[MOMENTUM], col[CONTRARIAN]]
    trend_sign = np.where(kind[trend] == MOMENTUM, 1, -1)
    watch = np.r_[col[COPIER], col[FADER]]
    target = np.full(PROS, -1) if targets is None else targets
    watch = watch[target[watch] >= 0]
    copy_sign = np.where(kind[watch] == COPIER, 1, -1)
    watched, which = np.unique(target[watch], return_inverse=True)

    rng = stream(seed, RETAIL)
    plan = retail_plan(pop["retail"], 0 if "retail" in without else orders, rng)
    slots = int(plan["slot"].max()) + 1 if len(plan["slot"]) else 0
    N, slot_col = FIXED + slots, FIXED + plan["slot"]
    by_tick = np.argsort(plan["tick"], kind="stable")
    by_tick = np.split(by_tick, np.searchsorted(plan["tick"][by_tick], np.arange(1, T)))

    # Money.
    budget = np.zeros((M, N), np.int64)
    off = np.isin(kind, [KINDS.index(w) for w in without if w in KINDS])
    off |= np.isin(kind, (FORECASTER, NEWS)) & (pop["spend"] <= 0)  # no research, no information
    budget[:, :PROS], research, _ = entries(pop["capital"] * ~off, pop["spend"], stream(seed, ENTRY))
    rng = stream(seed, INST)
    inst_size = round(2 * INST_SHARE * INST_CAPITAL / INST_MARKETS)  # contracts: 50c each on average
    inst_side = np.zeros((M, 3), np.int64)
    for j in range(3):
        chosen, insider_chosen = rng.permutation(M)[:INST_MARKETS], rng.permutation(M)[:INSIDER_MARKETS]
        side = sign * (2 * rng.integers(0, 2, INST_MARKETS) - 1)
        if "institutions" not in without:
            budget[chosen, INSTITUTIONS.start + j] = 100 * inst_size  # enough for either side
            inst_side[chosen, j] = side
        if insiders:
            budget[insider_chosen, INSIDERS.start + j] = 100 * INSIDER_CAPITAL // INSIDER_MARKETS
    books = Books(budget)
    cap = budget[:, :FIXED] / 100  # each one's unit of size
    funds = retail_money(pop, 0 if "retail" in without else orders)
    left = np.floor(funds * 100).astype(np.int64)  # retail accounts' free cash, shared by all markets
    start_money = left.copy()

    # Research: noisy looks at the lead (sign: the mirrored run sees the same looks the other way round).
    sd = look_sd(pop["spend"], pop["z"])
    looks, news_rng = estimates(e, sd[fc], stream(seed, ERRORS), sign), stream(seed, NEWS_ERRORS)

    speed = np.ones(N)
    speed[:PROS] = np.where(np.isin(kind, (NEWS, MARKET_MAKER)), 8.0, 2.0)
    speed[INSIDERS] = 2.0
    ioc = np.ones(N, bool)
    ioc[fc] = ioc[mm] = ioc[INSIDERS] = False
    is_mm = np.isin(np.arange(N), mm)
    genes = {k: pop[k][mm] for k in ("half", "size", "shade", "response", "widen", "pull")}

    arrivals = stream(seed, ARRIVALS)
    ref = np.clip(100 * opening, 1, 99)  # what rules call the price: the last tick's average trade
    arrived_at = ref[:, None].copy()  # the price when institutions' orders arrived
    centre = np.repeat(ref[:, None], len(mm), axis=1)
    since_news = np.full(M, T)
    logs, seen = np.empty((T, M)), np.empty((T, M, len(watched)))
    prices = np.empty((T + 1, M))
    prices[0] = ref / 100
    volume, two_sided = np.zeros(M, np.int64), np.empty((T, M), bool)
    expected = np.zeros(M * N)
    by_side = np.zeros((2, 2))  # makers, takers: sum of return per contract, contracts
    steps = []

    for t in range(T):
        sh = books.shares
        bid, ask = np.ones((M, N), np.int64), np.full((M, N), 99, np.int64)
        bid_n, ask_n = np.zeros((M, N), np.int64), np.zeros((M, N), np.int64)

        def post(cols, quote):
            bid[:, cols], bid_n[:, cols], ask[:, cols], ask_n[:, cols] = quote

        def market_order(cols, n, limit, price):
            c = price_cap(price, limit)
            post(cols, limit_order(n, price + c, price - c))

        logs[t] = lo(ref / 100)
        now = logs[t][:, None]
        landed = news[t - 1] if t else np.zeros(M, bool)
        since_news = np.where(landed, 0, since_news + 1)

        if len(trend):
            move = now - logs[np.maximum(t - pop["lag"][trend], 0)].T
            market_order(trend, trend_sign * pop["size"][trend] * cap[:, trend] * move - sh[:, trend], pop["limit"][trend], ref[:, None])
        if len(watch):
            seen[t] = sh[:, watched]
            held = seen[t][:, which] - seen[np.maximum(t - pop["lag"][watch], 0), :, which].T
            of = cap[:, target[watch]]
            share = np.divide(held, of, out=np.zeros(of.shape), where=of > 0)
            market_order(watch, copy_sign * pop["size"][watch] * cap[:, watch] * share - sh[:, watch], pop["limit"][watch], ref[:, None])
        if len(fc):
            estimate = next(looks)
            post(fc, forecaster_orders(estimate, ref, sh[:, fc], cap[:, fc], pop["size"][fc], pop["agg"][fc]))
        if len(nw):
            noise = sign * (np.sqrt(ERR_CORR) * news_rng.standard_normal((M, 1))
                            + np.sqrt(1 - ERR_CORR) * news_rng.standard_normal((M, len(nw))))
            m = np.flatnonzero(landed)
            if len(m):  # one noisy look at the move, applied to the level the last price implies
                look = sign * (lead[t, m] - lead[t - 1, m])[:, None] + noise[m] * np.nan_to_num(sd[nw])
                move = jump_move(ref[m], look, sd[nw], T - t)
                at = 100 * expit(now[m] + pop["agg"][nw] * move)
                where = np.ix_(m, nw)
                bid[where], bid_n[where], ask[where], ask_n[where] = limit_order(pop["size"][nw] * cap[where] * move, at, at)
        if insiders:
            est = lo(truth[min(t + INSIDER_LEAD, T)])[:, None]
            post(INSIDERS, forecaster_orders(est, ref, sh[:, INSIDERS], cap[:, INSIDERS], INFORMED_SIZE, 0.9))
        todo = inst_side * np.rint((inst_size - np.abs(sh[:, INSTITUTIONS])) / (T - t))
        market_order(INSTITUTIONS, todo, INST_LIMIT, arrived_at)  # never from the last trade: that's its own
        quote = maker_quotes(centre, sh[:, mm], cap[:, mm], genes, since_news)
        post(mm, quote)
        quoted_bid = np.minimum(quote[1], books.cash[:, mm] // quote[0]).sum(1)
        quoted_ask = np.minimum(quote[3], (books.cash[:, mm] + 100 * sh[:, mm]) // (100 - quote[2])).sum(1)

        # Retail: cash is put up when the order arrives, and what it didn't use goes back after the tick.
        o = by_tick[t]
        m, s, buys, who = plan["market"][o], slot_col[o], sign * plan["side"][o] > 0, plan["who"][o]
        c = price_cap(ref[m], RETAIL_LIMIT)
        at = np.where(buys, down(ref[m] + c), up(ref[m] - c))
        bid[m[buys], s[buys]], bid_n[m[buys], s[buys]] = at[buys], plan["n"][o][buys]
        ask[m[~buys], s[~buys]], ask_n[m[~buys], s[~buys]] = at[~buys], plan["n"][o][~buys]
        need, put = (plan["n"][o] * np.where(buys, at, 100 - at)).tolist(), np.zeros(len(o), np.int64)
        for i, w in enumerate(who.tolist()):  # in turn: an account may have several orders this tick
            put[i] = min(left[w], need[i])
            left[w] -= put[i]
        books.cash[m, s] += put

        if trace:
            steps.append(dict(bid=bid, bid_n=bid_n, ask=ask, ask_n=ask_n, shares=sh.copy(),
                              ref=ref.copy(), centre=centre.copy()))
        fills = books.step(bid, bid_n, ask, ask_n, ioc, arrival(arrivals, speed, M))
        if trace:
            steps[-1]["fills"] = fills

        back = books.cash[m, s] + 100 * np.minimum(sh[m, s], 0)  # all but the filled part's collateral
        books.cash[m, s] -= back
        np.add.at(left, who, back)
        budget[m, s] += put - back

        fm, maker, taker, side, price, q = fills[:, MARKET], fills[:, MAKER], fills[:, TAKER], fills[:, SIDE], fills[:, PRICE], fills[:, QTY]
        vol = np.bincount(fm, q, M)
        money = np.bincount(fm, q * price, M)
        # Order flow: whichever way a customer took (even from another customer), or rested when a maker crossed to it.
        customer = np.where(~is_mm[taker], side * q, np.where(is_mm[maker], 0, -side * q))
        net = np.bincount(fm, customer, M)
        centre = maker_move(centre, net, quoted_bid, quoted_ask, genes["response"])
        value = 100 * truth[t, fm]
        edge = side * q * (value - price) / 100
        expected += np.bincount(fm * N + taker, edge, M * N) - np.bincount(fm * N + maker, edge, M * N)
        took_yes = side > 0
        cost = np.where(took_yes, price, 100 - price)
        pays = np.where(took_yes, value, 100 - value)
        by_side += [[(q * ((100 - pays) / (100 - cost) - 1)).sum(), q.sum()], [(q * (pays / cost - 1)).sum(), q.sum()]]
        volume += vol.astype(np.int64)
        ref = np.divide(money, vol, out=ref.copy(), where=vol > 0)
        b, a = books.best.T
        two_sided[t] = both = (b > 0) & (a < 100)
        prices[t + 1] = np.where(both, (b + a) / 2, ref) / 100

    final = books.settle(outcome)
    gain = final - budget
    wallet_profit = gain[:, :FIXED].sum(0) / 100
    wallet_profit[:PROS] -= research / 100
    wallet_capital = np.r_[pop["capital"], [INST_CAPITAL] * 3, [INSIDER_CAPITAL] * 3] * budget[:, :FIXED].any(0)
    if insiders:  # someone sees the future, so a trade's value at the truth is no longer fair
        expected = gain.ravel() / 100
    account_gain = np.zeros(len(left), np.int64)
    np.add.at(account_gain, plan["who"], gain[plan["market"], slot_col])
    return dict(prices=prices, truth=truth, news=news, outcome=outcome, volume=volume, two_sided=two_sided,
                budget=budget, final=final, cash=books.cash, shares=books.shares, research=research,
                expected=expected.reshape(M, N), wallet_profit=wallet_profit, wallet_capital=wallet_capital,
                account_spent=start_money - left, account_gain=account_gain, plan=plan, by_side=by_side, trace=steps)


# --- The stage 3 procedure, fixed before the first run.

VOLUME = 9000                              # contracts in the median market (Kalshi: $8,982 staked by both sides)
RETAIL_CONTRACTS = RETAIL_SHARE * VOLUME   # what retail fills in the median market; total volume is only checked
MOST_ORDERS, FEWEST_ORDERS = 640.0, RETAIL_ACCOUNTS / MARKETS
TRAIN, VALIDATE, FRESH = (10000, 10002), (11000, 11004), (12000, 12010)  # spent by gate 2: seal new ones before a new gate
START_ORDERS, RETAIL_TOLERANCE, TRIES = 40.0, 0.03, 8
GRID = dict(half=(1, 2, 3, 4, 6, 10), size=(0.05, 0.1, 0.2, 0.5), shade=(0, 2, 5, 10, 20),
            response=(0.5, 1, 2, 4, 8), widen=(0, 3, 50), pull=(1, 3, 5))
TRIED, SHORTLIST, WORKERS = 90, 6, 4
INFORMED = ("forecaster", "news")


def trial(job):
    """One seed, one maker setting: a warm-up generation for the leaderboard, then the one measured."""
    maker, orders, seed, without = job
    pop = population(seed, maker)
    warm = run(pop, orders, [seed, 0], without=without)
    r = run(pop, orders, [seed, 1], targets=targets(pop, warm["wallet_profit"], warm["wallet_capital"]), without=without)
    m = measure({**r, "truth": np.clip(r["truth"], 0.01, 0.99)})  # prices can't leave 1..99c
    gain, kind = (r["final"] - r["budget"]) / 100, pop["kind"]
    each = {name: kind == k for k, name in enumerate(KINDS)}
    roles = {name: (r["expected"][:, :PROS][:, on].sum(1), gain[:, :PROS][:, on].sum(1)) for name, on in each.items()}
    roles["institutions"] = r["expected"][:, INSTITUTIONS].sum(1), gain[:, INSTITUTIONS].sum(1)
    roles["retail"] = r["expected"][:, FIXED:].sum(1), gain[:, FIXED:].sum(1)
    winners = np.r_[r["account_gain"] / 100, r["wallet_profit"]]
    winners = np.sort(winners[winners > 0])[::-1]
    return dict(volume=r["volume"], two_sided=r["two_sided"].mean(), gap=m["gap"], calibration=m["calibration"],
                slope=m["extremity_slope"], longshot=m["longshot_overpricing"], news=m["news_response"][[0, 4, 19]],
                retail_filled=np.abs(r["shares"][:, FIXED:]).sum(1),
                roles=roles, no_institution=~r["budget"][:, INSTITUTIONS].any(1), research={name: r["research"][on].sum() / 100 for name, on in each.items()},
                filled=r["account_spent"].sum() / 100, by_side=r["by_side"],
                losers=(r["account_gain"] < 0).mean(), top1=winners[:max(len(winners) // 100, 1)].sum() / winners.sum())


def maker_profit(runs, only=None):
    """Mean a market after research, and its standard error. only: a key of trial() picking markets.

    A seed's markets share their population, so the error comes from the seeds, not the markets.
    """
    by_seed = []
    for r in runs:
        gain = r["roles"]["maker"][0]
        by_seed.append((gain if only is None else gain[r[only]]).mean() - r["research"]["maker"] / MARKETS)
    return np.mean(by_seed), np.std(by_seed, ddof=1) / np.sqrt(len(by_seed))


def trials(pool, makers, orders, seeds, without=()):
    """One summary per maker setting, over a range of seeds. orders: retail's, for all or for each setting."""
    jobs = [(g, n, s, without) for g, n in zip(makers, np.broadcast_to(orders, len(makers))) for s in range(*seeds)]
    got, n = list(pool.map(trial, jobs)), seeds[1] - seeds[0]
    out = []
    for i, g in enumerate(makers):
        rs = got[i * n:(i + 1) * n]
        profit, se = maker_profit(rs)
        out.append(dict(maker=g, runs=rs, two_sided=np.mean([r["two_sided"] for r in rs]), gap=np.mean([r["gap"] for r in rs]),
                        volume=np.median(np.concatenate([r["volume"] for r in rs])),
                        retail=np.median(np.concatenate([r["retail_filled"] for r in rs])),
                        profit=profit, se=se))
    return out


def best_first(tried):
    """Pre-registered: most maker profit among two-sided books; if there are none, most profit."""
    return sorted(tried, key=lambda r: (r["two_sided"] > 0.8, r["profit"]), reverse=True)


def equal_retail(filled, n):
    """Orders a market for each of n settings at which retail fills its target, and whether that was reached.

    Settings can only be compared there: one that fills more retail at the same order count looks better.
    filled(orders, which) gives retail's median contracts for settings `which`.
    """
    orders, todo = np.full(n, START_ORDERS), np.arange(n)
    for _ in range(TRIES):
        ratio = filled(orders[todo], todo) / RETAIL_CONTRACTS
        miss = np.abs(ratio - 1) > RETAIL_TOLERANCE
        todo, ratio = todo[miss], ratio[miss]
        if not len(todo):
            break
        orders[todo] = np.clip(orders[todo] * np.maximum(ratio, 0.01) ** -1.1, FEWEST_ORDERS, MOST_ORDERS)
    return orders, ~np.isin(np.arange(n), todo)


def matched(pool, makers, seeds):
    """trials() of each setting at its own order count. Those that can't reach retail's target are left out."""
    last = {}

    def filled(orders, which):
        got = trials(pool, [makers[i] for i in which], orders, seeds)
        last.update({i: {**r, "orders": n} for i, n, r in zip(which, orders, got)})
        return np.array([r["retail"] for r in got])

    ok = equal_retail(filled, len(makers))[1]
    return [last[i] for i in range(len(makers)) if ok[i]]


def search():
    """Every maker setting at the same retail volume; the best are matched again on validation seeds."""
    from time import perf_counter

    start = perf_counter()
    rng = np.random.default_rng(3)
    grid, seen = [dict(MAKER_GENES)], set()
    while len(grid) <= TRIED:
        g = {k: v[rng.integers(len(v))] for k, v in GRID.items()}
        if tuple(g.values()) not in seen:
            seen.add(tuple(g.values()))
            grid.append(g)
    with ProcessPoolExecutor(WORKERS) as pool:
        tried = []
        for i in range(0, len(grid), 23):
            tried += matched(pool, grid[i:i + 23], TRAIN)
            print(f"searched {min(i + 23, len(grid))}/{len(grid)}, {len(tried)} reached retail's target, "
                  f"{perf_counter() - start:.0f}s", flush=True)
        short = best_first(tried)[:SHORTLIST]
        again = best_first(matched(pool, [r["maker"] for r in short], VALIDATE))
    if not again:
        raise SystemExit("no maker setting reached retail's target")
    for title, rows in (("training", short), ("validation", again)):
        print(f"== {title}", flush=True)
        for r in rows:
            print(f"  {r['maker']}, {r['orders']:.1f} orders: makers {r['profit']:+.2f} +- {r['se']:.2f} a market, "
                  f"two-sided {r['two_sided']:.0%}, gap {r['gap']:.3f}, volume {r['volume']:,.0f}, retail {r['retail']:,.0f}", flush=True)
    chosen, orders = {k: float(v) for k, v in again[0]["maker"].items()}, again[0]["orders"]
    for k, v in chosen.items():
        if v == max(GRID[k]):
            print(f"warning: {k} = {v:g} is the most the grid allows, so a better setting may lie beyond it")
    np.savez(RUNS / "stage3_setting.npz", orders=orders, **chosen,
             tried=np.array([[*r["maker"].values(), r["profit"], r["se"], r["two_sided"], r["gap"], r["volume"]] for r in tried], float))
    print(f"chosen {chosen}, {orders:.1f} retail orders a market: saved runs/stage3_setting.npz")


def slowdown(pop, orders):
    """A generation of every strategy against a generation of the current market."""
    from time import perf_counter

    from arena import evolve
    from arena.market import simulate

    old, ratios = dict(np.load(RUNS / "pop.npz")), []
    run(pop, orders, 0)  # compile first
    t = targets(pop, np.arange(106.0), np.r_[np.full(103, 1000.0), np.zeros(3)])
    for seed in range(8):  # alternate which market goes first
        rng = np.random.default_rng([seed, 1])
        z = (np.sqrt(evolve.ERR_CORR) * rng.standard_normal((MARKETS, 1))
             + np.sqrt(1 - evolve.ERR_CORR) * rng.standard_normal((MARKETS, evolve.N_EVO)))
        took = {}
        for name in ("now", "book") if seed % 2 else ("book", "now"):
            start = perf_counter()
            if name == "now":
                simulate(evolve.make_orders(old, z * old["sigma"]), evolve.WALLETS, n_markets=MARKETS, seed=seed)
            else:
                run(pop, orders, seed, targets=t)
            took[name] = perf_counter() - start
        ratios.append(took["book"] / took["now"])
        print(f"  speed, seed {seed}: current {took['now']:.2f}s, order book {took['book']:.2f}s", flush=True)
    return np.median(ratios), max(ratios)


def gate():
    saved = np.load(RUNS / "stage3_setting.npz")
    maker, orders = {k: float(saved[k]) for k in MAKER_GENES}, float(saved["orders"])
    print(f"makers {maker}, {orders:.1f} retail orders a market, seeds {FRESH[0]}..{FRESH[1] - 1}", flush=True)
    speed, slowest = slowdown(population(0, maker), orders)
    with ProcessPoolExecutor(WORKERS) as pool:
        r = trials(pool, [maker], orders, FRESH)[0]
        blind = trials(pool, [maker], orders, FRESH, INFORMED)[0]
    runs = r["runs"]

    def mean(key):
        return np.mean([x[key] for x in runs], 0)

    print("per market, all of a role together: expected profit +- se by seed | real | after research")
    for role in [*KINDS, "institutions", "retail"]:
        e, real = (np.array([x["roles"][role][j].mean() for x in runs]) for j in (0, 1))
        paid = np.mean([x["research"].get(role, 0) for x in runs]) / MARKETS
        print(f"  {role:12s} {e.mean():+9.2f} +- {e.std(ddof=1) / np.sqrt(len(e)):.2f} | {real.mean():+9.2f} | {e.mean() - paid:+9.2f}")
    side = np.sum([x["by_side"] for x in runs], 0)
    print(f"retail fills {r['retail']:,.0f} contracts in the median market (target {RETAIL_CONTRACTS:,.0f}) "
          f"| median volume {r['volume']:,.0f} (check, not tuned: Kalshi {VOLUME:,}) | retail money used ${mean('filled'):,.0f}")
    print(f"calibration {mean('calibration'):.3f} | slope {mean('slope'):.2f} | longshots overpriced by {mean('longshot'):+.4f} "
          f"| news priced in after 1/5/20 ticks {mean('news').round(2)}")
    print(f"never tuned to: return per contract, makers {side[0, 0] / side[0, 1]:+.1%}, takers {side[1, 0] / side[1, 1]:+.1%}")
    print(f"not like for like: retail accounts losing {mean('losers'):.0%}, top 1% of winners take {mean('top1'):.0%}")
    # The design fails if makers live on a subsidy. A role in 15% of markets once paid for all of them.
    alone, alone_se = maker_profit(runs, "no_institution")
    results = [
        ("under 5x slower, every strategy running", speed < 5, f"{speed:.2f}x (slowest {slowest:.2f}x)"),
        ("book two-sided on over 80% of ticks", r["two_sided"] > 0.8, f"{r['two_sided']:.1%}"),
        ("gap under half of that without informed traders", r["gap"] < blind["gap"] / 2, f"{r['gap']:.3f} vs {blind['gap']:.3f}"),
        ("makers pay for themselves on fresh seeds", r["profit"] > 2 * r["se"], f"{r['profit']:+.2f} +- {r['se']:.2f} a market"),
        ("...without a subsidy: also where no institution trades", alone > 2 * alone_se, f"{alone:+.2f} +- {alone_se:.2f} a market"),
    ]
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print("stage 3 gate passed" if all(ok for _, ok, _ in results) else "STAGE 3 GATE FAILED")


def run_checks():
    failed = []

    def case(f):
        try:
            f()
            print(f"ok   {f.__name__.replace('_', ' ')}", flush=True)
        except Exception as e:
            print(f"FAIL {f.__name__.replace('_', ' ')}  ({type(e).__name__}: {str(e)[:110]})", flush=True)
            failed.append(f.__name__.replace("_", " "))

    def rng(k):
        return np.random.default_rng([11, k])

    def genes(**kw):
        g = dict(half=2.0, size=0.2, shade=5.0, response=2.0, widen=50.0, pull=1)
        g.update(kw)
        return {k: np.array([v]) for k, v in g.items()}

    @case
    def population_is_the_design():
        p = population(1)
        assert np.bincount(p["kind"]).tolist() == [15] * 6 + [10], "head-counts"
        m = p["kind"] == MARKET_MAKER
        assert (p["capital"][m] == 25_000).all(), "makers start with $25k"
        funds = np.concatenate([population(s)["capital"][~m] for s in range(100)])
        assert abs(np.median(funds) / 10_000 - 1) < 0.05 and abs(np.log(funds).std() - 1) < 0.03, "funds' capital"
        retail = np.concatenate([population(s)["retail"] for s in range(20)])
        assert len(p["retail"]) == 2750 and abs(np.median(retail) / 100 - 1) < 0.05, "retail money"
        z = np.concatenate([population(s)["z"] for s in range(100)])
        assert abs(z.std() - 0.2) < 0.01, "luck"
        assert not np.array_equal(p["capital"], population(2)["capital"]), "seeds should differ"

    @case
    def price_cap_is_the_same_cents_both_ways():
        c = price_cap(np.array([50.0, 10, 90, 3, 97, 1, 99]), 0.2)
        assert c[0] == 5 and c[1] == c[2] and c[3] == c[4] and c[5] == c[6] and c.min() >= 1, c

    @case
    def maker_quotes_stand_still_without_trades_news_or_inventory_change():
        g, centre, inv, cap = genes(), np.array([[60.0]]), np.array([[30]]), np.array([[100.0]])
        first = maker_quotes(centre, inv, cap, g, np.array([T]))
        for _ in range(5):
            centre = maker_move(centre, np.zeros(1), np.full(1, 20.0), np.full(1, 20.0), g["response"])
            again = maker_quotes(centre, inv, cap, g, np.array([T]))
            assert all(np.array_equal(a, b) for a, b in zip(first, again)), "quotes moved by themselves"
        assert (first[0][0, 0] + first[2][0, 0]) / 2 < 60, "long inventory should shade down"

    @case
    def maker_centre_follows_only_order_flow():
        centre, r = np.array([[60.0], [60.0], [60.0], [60.0], [98.5]]), np.array([4.0])
        moved = maker_move(centre, np.array([50.0, -25, 500, 7, 500]), np.array([50.0, 100, 50, 0, 50]),
                           np.array([100.0, 100, 100, 0, 100]), r)
        assert np.allclose(moved[:, 0], [62, 59, 64, 64, 99]), moved[:, 0]

    @case
    def maker_news_reaction_has_no_direction():
        centre, inv, cap = np.array([[50.0]]), np.array([[0]]), np.array([[100.0]])
        calm = maker_quotes(centre, inv, cap, genes(widen=3.0, pull=2), np.array([2]))
        wide = maker_quotes(centre, inv, cap, genes(widen=3.0, pull=2), np.array([1]))
        assert (calm[0][0, 0], calm[2][0, 0], wide[0][0, 0], wide[2][0, 0]) == (48, 52, 45, 55), (calm, wide)
        gone = maker_quotes(centre, inv, cap, genes(widen=50.0, pull=2), np.array([0]))
        assert not gone[1].any() and not gone[3].any(), "should have pulled both quotes"

    @case
    def maker_quotes_mirror_and_narrow_evenly_at_the_edges():
        c = np.array([[1.0], [2], [3], [3.4], [10], [50.25], [96.6], [98], [99]])
        inv, cap = np.array([[0]] * 9), np.full((9, 1), 100.0)
        g = genes(half=4.0)
        b, n, a, m = maker_quotes(c, inv, cap, g, np.full(9, T))
        b2, n2, a2, m2 = maker_quotes(100 - c, inv, cap, g, np.full(9, T))
        assert np.array_equal(b, 100 - a2) and np.array_equal(a, 100 - b2) and np.array_equal(n, m2), "not mirror images"
        assert (b >= 1).all() and (a <= 99).all() and (a > b).all() and (b[2, 0], a[2, 0]) == (1, 5), (b, a)

    @case
    def forecaster_quotes_between_the_price_and_its_reservation():
        est = lo(np.array([[0.7]]))
        b, n, a, m = forecaster_orders(est, np.array([50.0]), np.array([[0]]), np.array([[100.0]]), np.array([0.9]), np.array([0.5]))
        p = 100 * expit(est / 2)
        assert b[0, 0] == np.floor(p[0, 0]) and n[0, 0] == round(90 * float(est[0, 0]) / 2) and not m.any(), (b, n)
        full = np.rint(90 * est).astype(np.int64)  # already at the position where 50c is its limit
        _, n, _, m = forecaster_orders(est, np.array([50.0]), full, np.array([[100.0]]), np.array([0.9]), np.array([0.5]))
        assert n.max() <= 1 and m.max() <= 1, "should stop at its reservation"

    @case
    def retail_plan_is_the_design():
        money = population(3)["retail"]
        plans = [retail_plan(money, 80, rng(s)) for s in range(10)]
        n = np.concatenate([p["n"] for p in plans])
        assert abs(np.median(n) / 20 - 1) < 0.06 and abs(n.mean() / 205.3 - 1) < 0.08, (np.median(n), n.mean())
        side = np.concatenate([p["side"] for p in plans])
        assert set(side) == {-1, 1} and abs(side.mean()) < 4 / np.sqrt(len(side)), "side should be a coin flip"
        p = plans[0]
        count = np.bincount(p["who"], minlength=2750)
        assert len(p["who"]) == 32_000 and count.min() >= 1, "everyone trades at least once"
        assert count[money > np.percentile(money, 99)].mean() > 20 * count[money < np.median(money)].mean(), "not by money"
        assert len(set(zip(p["market"], p["slot"]))) == len(p["who"]), "two orders share a slot"
        assert len(retail_plan(money, 0, rng(0))["who"]) == 0, "orders 0 switches retail off"

    @case
    def retail_money_is_one_turnover_of_its_orders():
        p = population(3)
        for orders in (20, 80):
            m = retail_money(p, orders)
            assert abs(m.sum() / (orders * MARKETS * ORDER_MEAN * 0.5) - 1) < 1e-9, "not one turnover"
            assert np.allclose(m / m.sum(), p["retail"] / p["retail"].sum()), "lost the spread between accounts"
        assert abs(retail_money(p, 1).sum() / (RETAIL_ACCOUNTS * ORDER_MEAN * 0.5) - 1) < 1e-9, "everyone trades at least once"
        assert not retail_money(p, 0).any(), "switched off"

    @case
    def entering_markets_respects_five_dollars_and_research():
        cap, spend = np.array([10_000.0, 1_000, 4, 2_000]), np.array([0.25, 0.25, 0.25, 0])
        budget, research, entered = entries(cap, spend, rng(1))
        assert entered.tolist() == [400, 200, 0, 400] and research.tolist() == [10_000, 5_000, 0, 0], (entered, research)
        assert budget.dtype.kind == "i" and (budget.sum(0) + research <= cap * 100).all(), "spent more than its capital"
        assert (budget[:, 1] > 0).sum() == 200 and budget[:, 0].min() >= 2400, "split unevenly"

    @case
    def copiers_pick_by_leaderboard_spot():
        p = population(4)
        profit, capital = np.arange(106.0), np.r_[np.full(103, 1000.0), np.zeros(3)]
        c = np.flatnonzero(p["kind"] == COPIER)[0]
        p["rank"][c], p["board"][c] = 0.0, 0
        out = targets(p, profit, capital)
        assert out[c] == 102, "best wallet that traded"
        watch = np.isin(p["kind"], (COPIER, FADER))
        assert (out[watch] >= 0).all() and (out[~watch] == -1).all() and (out[watch] != np.flatnonzero(watch)).all()
        p["board"][c], capital[7] = 1, 1.0
        assert targets(p, profit, capital)[c] == 7, "best return on capital"
        p["board"][c], profit[c] = 0, 1000.0
        assert targets(p, profit, capital)[c] == 102, "copied itself from the top of the board"

    base = population(5)

    @case
    def retail_limit_is_the_measured_one():
        r = run(base, 30, 16, trace=True)
        p, seen = r["plan"], 0
        for t, s in enumerate(r["trace"]):
            o = np.flatnonzero(p["tick"] == t)
            m, col, ref = p["market"][o], FIXED + p["slot"][o], s["ref"][p["market"][o]]
            c, buys = price_cap(ref, 0.759), p["side"][o] > 0
            assert np.array_equal(s["bid"][m, col][buys], down(ref + c)[buys]), "buy limit"
            assert np.array_equal(s["ask"][m, col][~buys], up(ref - c)[~buys]), "sell limit"
            seen += len(o)
        assert seen > 1000 and INST_LIMIT == 0.2, "institutions' limit isn't retail's"

    @case
    def the_money_balances_to_the_cent():
        r = run(base, 40, 5)
        assert np.array_equal(r["final"].sum(1), r["budget"].sum(1)), "a market created or lost money"
        assert (r["final"] >= 0).all() and (r["budget"] >= 0).all()
        assert (r["account_spent"] <= np.floor(retail_money(base, 40) * 100)).all(), "retail spent more than it has"
        assert (r["budget"][:, :PROS].sum(0) + r["research"] <= base["capital"] * 100).all(), "spent more than its capital"
        assert r["account_gain"].sum() == (r["final"] - r["budget"])[:, FIXED:].sum(), "retail accounts don't add up"
        assert r["volume"].sum() > 100_000, "barely traded"

    @case
    def each_trade_is_scored_at_the_truth_of_its_own_tick():
        r = run(base, 40, 13, trace=True)
        want = np.zeros_like(r["expected"])
        for t, s in enumerate(r["trace"]):
            m, maker, taker, side, price, q = s["fills"].T
            edge = side * q * (r["truth"][t, m] - price / 100)
            np.add.at(want, (m, taker), edge)
            np.add.at(want, (m, maker), -edge)
        assert np.allclose(want, r["expected"]), "valued at another tick's truth"

    @case
    def retail_gets_back_what_it_did_not_spend():
        r = run(base, 40, 14)
        p = r["plan"]
        paid = np.zeros(RETAIL_ACCOUNTS, np.int64)
        np.add.at(paid, p["who"], r["budget"][p["market"], FIXED + p["slot"]])
        assert np.array_equal(paid, r["account_spent"]), "accounts and order slots disagree"

    @case
    def yes_and_no_are_mirror_images():
        t = targets(base, np.arange(106.0), np.r_[np.full(103, 1000.0), np.zeros(3)])
        a, b = run(base, 40, 6, targets=t), run(base, 40, 6, targets=t, mirror=True)
        assert np.array_equal(a["volume"], b["volume"]), "volume"
        assert np.array_equal(a["final"] - a["budget"], b["final"] - b["budget"]), "profits"
        assert np.allclose(a["prices"], 1 - b["prices"]), "prices"

    @case
    def nobody_sees_one_tick_ahead():
        e = events.draw(MARKETS, 7)
        t0 = 80
        changed = {k: v.copy() for k, v in e.items()}
        changed["truth"][t0 + 1:] = 1 - changed["truth"][t0 + 1:]
        changed["lead"][t0 + 1:] = -changed["lead"][t0 + 1:]
        changed["news"][t0:] = ~changed["news"][t0:]
        changed["outcome"] = ~changed["outcome"]
        t = targets(base, np.arange(106.0), np.r_[np.full(103, 1000.0), np.zeros(3)])
        a = run(base, 40, 7, targets=t, event=e, trace=True)["trace"]
        b = run(base, 40, 7, targets=t, event=changed, trace=True)["trace"]
        for s in range(t0 + 1):
            for k in ("bid", "bid_n", "ask", "ask_n", "fills"):
                assert np.array_equal(a[s][k], b[s][k]), f"{k} at tick {s} depends on the future"
        assert any(not np.array_equal(a[s]["bid"], b[s]["bid"]) for s in range(t0 + 1, T)), "nothing reacted at all"

    @case
    def news_traders_act_the_tick_after_news_lands():
        r = run(base, 0, 8, without=("forecaster", "momentum", "contrarian", "copier", "fader", "institutions"), trace=True)
        col = np.flatnonzero(base["kind"] == NEWS)
        acted = 0
        for t, s in enumerate(r["trace"]):
            on = ((s["bid_n"][:, col] > 0) | (s["ask_n"][:, col] > 0)).any(1)
            landed = r["news"][t - 1] if t else np.zeros(MARKETS, bool)
            assert not (on & ~landed).any(), f"traded without news at tick {t}"
            acted += on.sum()
        assert acted > 1000, "never acted on news"

    @case
    def trend_bots_hold_size_times_the_move():
        r = run(base, 20, 9, trace=True)
        hist = np.array([s["ref"] for s in r["trace"]]) / 100
        cap = r["budget"][:, :PROS] / 100
        for kind, sign in ((MOMENTUM, 1), (CONTRARIAN, -1)):
            j = np.flatnonzero(base["kind"] == kind)
            for t in (40, 150):
                s = r["trace"][t]
                move = lo(hist[t])[:, None] - lo(hist[np.maximum(t - base["lag"][j], 0)]).T
                want = np.rint(sign * base["size"][j] * cap[:, j] * move - s["shares"][:, j])
                assert np.array_equal(s["bid_n"][:, j] - s["ask_n"][:, j], want), KINDS[kind]

    @case
    def copiers_hold_a_fraction_of_the_targets_position():
        p = {k: v.copy() for k, v in base.items()}
        c, f = np.flatnonzero(p["kind"] == COPIER)[0], np.flatnonzero(p["kind"] == FORECASTER)[0]
        t = np.full(PROS, -1)
        t[c] = f
        r = run(p, 20, 10, targets=t, trace=True)
        cap, s = r["budget"] / 100, r["trace"][120]
        share = np.divide(s["shares"][:, f], cap[:, f], out=np.zeros(MARKETS), where=cap[:, f] > 0)  # lag 200: the whole position
        want = np.rint(p["size"][c] * cap[:, c] * share - s["shares"][:, c])
        assert np.array_equal(s["bid_n"][:, c] - s["ask_n"][:, c], want), "wrong copy size"
        others = np.setdiff1d(np.flatnonzero(p["kind"] == COPIER), [c])
        assert not s["bid_n"][:, others].any(), "a copier with no target traded"

    @case
    def maker_centres_follow_every_customer_trade():
        r = run(base, 40, 12, trace=True)
        mm = np.flatnonzero(base["kind"] == MARKET_MAKER)
        still = moved = crossed = 0
        for t in range(T - 1):
            s, after = r["trace"][t], r["trace"][t + 1]["centre"]
            m, maker, taker, side, _, q = s["fills"].T
            took, rested = np.isin(taker, mm), np.isin(maker, mm)
            # the customer's direction: theirs as a taker, the opposite of the maker's when it crossed to them
            customer = np.where(~took, side * q, np.where(rested, 0, -side * q))
            net = np.bincount(m, customer, MARKETS)
            step = (after - s["centre"])[:, 0]
            assert not step[net == 0].any(), "moved with no customer trades"
            assert (step * net >= 0).all() and (np.abs(step) <= base["response"][mm[0]] + 1e-9).all(), "moved the wrong way"
            free = (net != 0) & (s["centre"][:, 0] > 1) & (s["centre"][:, 0] < 99)
            assert step[free].all(), "ignored a customer's trade"
            still += (net == 0).sum()
            moved += (step != 0).sum()
            crossed += (took & ~rested).sum()
        assert still and moved and crossed, "nothing tested"

    @case
    def institutions_never_chase_their_own_trades():
        r = run(base, 20, 15, trace=True)
        first, worked = r["trace"][0]["ref"][:, None], 0  # the price when their orders arrived
        c = price_cap(first, INST_LIMIT)
        for s in r["trace"]:
            buys, sells = s["bid_n"][:, INSTITUTIONS] > 0, s["ask_n"][:, INSTITUTIONS] > 0
            assert (s["bid"][:, INSTITUTIONS] <= down(first + c))[buys].all(), "paid past its limit from arrival"
            assert (s["ask"][:, INSTITUTIONS] >= up(first - c))[sells].all(), "sold past its limit from arrival"
            worked += buys.sum() + sells.sum()
        assert worked > 1000, "institutions never traded"

    @case
    def institutions_work_no_more_than_a_quarter_of_capital():
        spent, count = [], []
        for s in range(4):
            r = run(base, 20, 20 + s)
            held = r["shares"][:, INSTITUTIONS]
            cost = r["budget"][:, INSTITUTIONS] - (r["cash"][:, INSTITUTIONS] + 100 * np.minimum(held, 0))
            count += (r["budget"][:, INSTITUTIONS] > 0).sum(0).tolist()
            spent += (cost.sum(0) / 100 / 250_000).tolist()
            assert (np.abs(held) <= 6250).all(), "worked more than its order"
        assert count == [20] * 12 and np.mean(spent) < 0.29, (count[:3], np.mean(spent))

    @case
    def no_research_means_no_information():
        p = {k: v.copy() for k, v in base.items()}
        informed = np.flatnonzero(np.isin(p["kind"], (FORECASTER, NEWS)))
        p["spend"][informed] = 0
        for s in run(p, 20, 17, trace=True)["trace"]:
            assert not s["bid_n"][:, informed].any() and not s["ask_n"][:, informed].any(), "traded on research it didn't buy"
        _, paid, _ = entries(np.array([100.0]), np.array([0.001]), rng(2))
        assert paid[0] > 0, "cheap research rounded down to free"

    @case
    def forecasters_see_the_truth_two_ticks_late():
        e, t0 = events.draw(MARKETS, 18), 80
        changed = {**e, "truth": e["truth"].copy(), "lead": e["lead"].copy()}
        changed["truth"][t0 - 1:] = 1 - changed["truth"][t0 - 1:]
        changed["lead"][t0 - 1:] = -changed["lead"][t0 - 1:]
        fc = base["kind"] == FORECASTER
        a, b = (run(base, 20, 18, without=("news",), event=x, trace=True)["trace"] for x in (e, changed))
        bids = [(x["bid"][:, :PROS][:, fc], y["bid"][:, :PROS][:, fc]) for x, y in zip(a, b)]
        assert all(np.array_equal(x, y) for x, y in bids[:t0 + 1]), "saw the truth too soon"
        assert not np.array_equal(*bids[t0 + 1]), "saw the truth too late"

    @case
    def maker_settings_are_compared_at_equal_retail():
        scale = np.array([10.0, 40, 90, 0.001])  # the last can't reach the target
        def filled(orders, which):
            return scale[which] * orders**0.9
        orders, ok = equal_retail(filled, len(scale))
        assert ok.tolist() == [True, True, True, False], ok
        assert (np.abs(filled(orders[:3], np.arange(3)) / RETAIL_CONTRACTS - 1) <= 0.03).all(), "unequal retail"

    @case
    def maker_profit_is_after_research_and_judged_by_seed():
        with_inst = np.arange(MARKETS) < 100
        runs = [dict(roles=dict(maker=(np.where(with_inst, 10.0, v),) * 2), research=dict(maker=1.0 * MARKETS),
                     no_institution=~with_inst) for v in (3, 5, 7)]
        mean, se = maker_profit(runs, "no_institution")
        assert np.isclose(mean, 4) and np.isclose(se, 2 / np.sqrt(3)), (mean, se)
        assert np.isclose(maker_profit(runs)[0], 5.25), "all markets"

    research = {}

    def researched(seed, spend=SPEND, M=20_000):
        """A researcher's best guesses, every tick (T, M), with the events they were made on."""
        if (seed, spend, M) not in research:
            e = events.draw(M, seed)
            sd = look_sd(np.array([spend]), np.zeros(1))
            research[seed, spend, M] = e, np.array([x[:, 0] for x in estimates(e, sd, rng(seed))])
        return research[seed, spend, M]

    @case
    def research_leaves_nothing_to_learn_from_its_own_past():
        def X(e, est, t):  # its whole history, the public opening and the news so far
            l, n = est[:t + 1].T, e["news"][:t].sum(0)
            o, p = lo(e["opening"]), expit(est[t])
            return np.column_stack([np.ones(len(p)), l, o, expit(o), o * l[:, -1], l[:, -1] - l[:, 0] + o, expit(l[:, -1] - l[:, 0] + o),
                                    p, p**2, p**3, np.tanh(l[:, -1] / 2), n, n * (p - .5), l[:, -1] * abs(l[:, -1])])
        (a, ea), (b, eb) = researched(61), researched(62)
        for t in (5, 20, 100, 180):
            coef = np.linalg.lstsq(X(a, ea, t), a["truth"][t] - expit(ea[t]), rcond=None)[0]
            left = b["truth"][t] - expit(eb[t])
            removed = 1 - np.mean((left - X(b, eb, t) @ coef) ** 2) / np.mean(left**2)
            assert removed < 0.02, f"its own past removes {removed:.1%} of its error at tick {t}"

    @case
    def research_is_a_fair_best_guess_at_the_designed_accuracy():
        e, est = researched(61)
        p = expit(est)
        error = np.sqrt(np.mean((p - e["truth"][:T]) ** 2))
        assert abs(error - 0.057) < 0.003, f"average error {error:.4f}, designed 0.057"
        for t in (0, 100, 190, 199):
            b = np.digitize(p[t], [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
            var = np.bincount(b, p[t] * (1 - p[t]), 12)
            z = np.abs(np.bincount(b, e["outcome"] - p[t], 12)[var > 10]) / np.sqrt(var[var > 10])
            assert z.max() < 4, f"tick {t}: estimates and outcomes differ by {z.max():.1f} standard errors"

    @case
    def more_research_buys_a_better_level_and_better_moves():
        errors, moves = [], []
        for spend in (0.0, 0.025, 0.25, 2.5):
            e, est = researched(63, spend, 4000)
            d, m = lo(e["truth"][10:T:10]) - lo(e["truth"][:T - 10:10]), est[10::10] - est[:-10:10]
            errors.append(np.sqrt(np.mean((expit(est) - e["truth"][:T]) ** 2)))
            moves.append(np.corrcoef(d.ravel(), m.ravel())[0, 1] ** 2 if m.std() > 0 else 0.0)
        assert all(np.diff(errors) < 0) and all(np.diff(moves) > 0), (np.round(errors, 3), np.round(moves, 2))
        e, free = researched(63, 0.0, 4000)
        assert np.allclose(free[0], lo(e["opening"])) and moves[0] < 0.01, "no research must be the public view"

    @case
    def researchers_share_half_their_noise():
        M = 20_000  # a lead held at 0 with no news, so the estimates move only with the looks' noise
        still = dict(lead=np.zeros((T + 1, M)), news=np.zeros((T, M), bool), seen=np.zeros(M), open_var=np.full(M, 300.0))
        for t, est in enumerate(estimates(still, look_sd(np.full(2, SPEND), np.zeros(2)), rng(65))):
            if t == 100:
                break
        c = np.corrcoef(est.T)[0, 1]
        assert abs(c - ERR_CORR) < 0.03, f"errors correlate {c:.2f}, designed {ERR_CORR}"

    @case
    def news_traders_never_see_the_true_move():
        p = {k: v.copy() for k, v in base.items()}
        nw = np.flatnonzero(p["kind"] == NEWS)
        p["spend"][nw] = 1e-4  # research so poor it is worth nothing
        r = run(p, 0, 19, without=("forecaster", "momentum", "contrarian", "copier", "fader", "institutions", "maker"), trace=True)
        sizes, moves = [], []
        for t in range(1, T):
            m = np.flatnonzero(r["news"][t - 1])
            s = r["trace"][t]
            sizes += (s["bid_n"][m][:, nw] - s["ask_n"][m][:, nw]).sum(1).tolist()
            moves += (lo(r["truth"][t, m]) - lo(r["truth"][t - 1, m])).tolist()
        c = np.corrcoef(sizes, moves)[0, 1]
        assert len(moves) > 1000 and abs(c) < 0.1, f"worthless research still follows the true move: correlation {c:.2f}"

    @case
    def news_traders_shrink_a_noisy_look_at_the_jump():
        e = events.draw(MARKETS, 64)
        t, m = np.nonzero(e["news"][100:150])
        t, m = t + 101, m  # the tick each lands
        before = 100 * e["truth"][t - 1, m]
        step = e["lead"][t, m] - e["lead"][t - 1, m]
        for k in range(len(t)):
            sharp = jump_move(before[k:k + 1], step[k:k + 1, None], np.array([1e-6]), T - t[k])
            want = lo(e["truth"][t[k], m[k]]) - lo(before[k] / 100)
            assert abs(sharp[0, 0] - want) < 1e-6 or abs(want) > 5.9, "a perfect look should give the true move"
            blind = jump_move(before[k:k + 1], np.array([[-30.0, 30.0]]), np.array([1e9, np.nan]), T - t[k])
            assert abs(blind[0, 0] - blind[0, 1]) < 1e-6, "a useless look must not matter"
            flip = jump_move(100 - before[k:k + 1], -step[k:k + 1, None], np.array([2.0]), T - t[k])
            assert np.allclose(flip, -jump_move(before[k:k + 1], step[k:k + 1, None], np.array([2.0]), T - t[k])), "not mirrored"
        assert len(t) > 20

    @case
    def insiders_profit_when_switched_on():
        on = run(base, 30, 30, insiders=True)
        off = run(base, 30, 30)
        assert ((on["budget"][:, INSIDERS] > 0).sum(0) == 4).all() and not off["budget"][:, INSIDERS].any()
        assert (on["final"] - on["budget"])[:, INSIDERS].sum() > 0, "insiders lost money"
        assert np.array_equal(on["expected"], (on["final"] - on["budget"]) / 100), "must score on real profit"

    @case
    def expected_profit_equals_real_profit_for_every_role():
        diff = []
        for s in range(6):
            r = run(base, 50, 40 + s)
            d = (r["final"] - r["budget"]) / 100 - r["expected"]
            by = [d[:, :PROS][:, base["kind"] == k].sum(1) for k in range(len(KINDS))]
            diff.append(np.array(by + [d[:, INSTITUTIONS].sum(1), d[:, FIXED:].sum(1)]))
        d = np.concatenate(diff, 1)
        z = np.abs(d.mean(1)) / (d.std(1) / np.sqrt(d.shape[1]) + 1e-12)
        assert (z < 4.5).all(), f"z {z.round(1)}"

    @case
    def same_seed_same_market():
        a, b = run(base, 30, 50), run(base, 30, 50)
        assert np.array_equal(a["final"], b["final"]) and not np.array_equal(a["truth"], run(base, 30, 51)["truth"])

    assert not failed, f"{len(failed)} ecology checks failed: " + "; ".join(failed)
    print("ecology checks passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="check", choices=["check", "search", "gate"])
    dict(check=run_checks, search=search, gate=gate)[ap.parse_args().mode]()
