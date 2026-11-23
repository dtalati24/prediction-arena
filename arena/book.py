"""The order book: many markets at once, whole cents and contracts, price then time priority.

Every tick each trader arrives once, faster ones usually earlier. On arrival its old orders are
cancelled and it may post one bid and one offer (for YES; selling YES is buying NO). An order trades
at the resting order's price, best price first and oldest first; what's left rests, or is cancelled
for an immediate-or-cancel trader. Orders are fully collateralised when posted, so nobody can lose
more than their budget; there's no other position limit.

YES and NO are treated exactly alike, so the book can't favour longshots by itself.

    python -m arena.book          # the checks (docs/order-book-design.md, stage 2)
    python -m arena.book bench    # speed against the current market (needs runs/pop.npz)
"""
import numpy as np
from numba import njit

BID, ASK = 0, 1
MARKET, MAKER, TAKER, SIDE, PRICE, QTY = range(6)  # columns of a fill; SIDE +1: the taker bought YES
MAX_BUDGET = 10**15  # cents; beyond it 64-bit money could overflow


class Books:
    """M independent books for N traders. An order's slot is trader + N * side.

    budget (M, N): each trader's money in each market, in cents. cash and shares are public.
    """

    def __init__(self, budget):
        budget = np.asarray(budget)
        if budget.ndim != 2 or budget.dtype.kind not in "iu" or budget.min() < 0 or budget.max() > MAX_BUDGET:
            raise ValueError("budget must be (markets, traders) whole cents, 0 to 10^15")
        self.cash = budget.astype(np.int64)  # a copy
        M, N = budget.shape
        self.shares = np.zeros((M, N), np.int64)
        self.qty, self.px = np.zeros((M, 2 * N), np.int64), np.zeros((M, 2 * N), np.int64)
        self.next, self.prev = np.full((M, 2 * N), -1, np.int64), np.full((M, 2 * N), -1, np.int64)
        self.head, self.tail = np.full((M, 2, 100), -1, np.int64), np.full((M, 2, 100), -1, np.int64)
        self.best = np.tile(np.array([0, 100], np.int64), (M, 1))  # 0 / 100: that side is empty
        # At most 4N fills a market a tick: each one finishes off an order, and there are at most 2N
        # resting at the start of a tick and 2N posted during it.
        self.log = np.empty((4 * M * N, 6), np.int64)

    def step(self, bid, bid_n, ask, ask_n, ioc, order):
        """Everyone's quotes for one tick. Returns the fills (F, 6) in the order they happened.

        bid, bid_n, ask, ask_n (M, N): prices 1..99 and contracts (0: no order). ioc (N,): never rests.
        order (M, N): who arrives when. A trader whose bid is at or above its own offer posts neither.
        """
        M, N = self.cash.shape
        arrays = [np.asarray(x) for x in (bid, bid_n, ask, ask_n, order)]
        if any(x.shape != (M, N) or x.dtype.kind not in "iu" for x in arrays):
            raise ValueError("quotes and order must be (markets, traders) whole numbers")
        ioc = np.asarray(ioc)
        if ioc.shape != (N,) or ioc.dtype != bool:
            raise ValueError("ioc must be (traders,) booleans")
        arrays = [np.ascontiguousarray(x, np.int64) for x in arrays]
        problem = find_problem(*arrays)  # numba doesn't check bounds, so never let bad input in
        if problem:
            raise ValueError(["prices must be 1..99", "each trader must arrive exactly once"][problem - 1])
        n = step_all(self.cash, self.shares, self.qty, self.px, self.next, self.prev, self.head,
                     self.tail, self.best, self.log, *arrays[:4], ioc, arrays[4])
        return self.log[:n].copy()

    def settle(self, outcome):
        """Cents each trader ends with: a YES share pays $1 if the event happened."""
        return self.cash + 100 * self.shares * np.asarray(outcome, bool)[:, None]


def arrival(rng, speed, M):
    """Arrival order (M, N): trader i is before j with chance speed_i / (speed_i + speed_j)."""
    speed = np.asarray(speed, np.float32)
    return np.argsort(rng.standard_exponential((M, speed.shape[-1]), np.float32) / speed, axis=1)


@njit(cache=True)
def find_problem(bid, bid_n, ask, ask_n, order):
    M, N = bid.shape
    seen = np.full(N, -1)
    for m in range(M):
        for i in range(N):
            if (bid_n[m, i] > 0 and not 1 <= bid[m, i] <= 99) or (ask_n[m, i] > 0 and not 1 <= ask[m, i] <= 99):
                return 1
            j = order[m, i]
            if j < 0 or j >= N or seen[j] == m:
                return 2
            seen[j] = m
    return 0


@njit(cache=True)
def step_all(cash, shares, qty, px, next_, prev_, heads, tails, bests, log, bid, bid_n, ask, ask_n, ioc, order):
    # One function on purpose: numba reference-counts every array handed to a helper, atomically, and
    # with one call per order those counts took more time than the matching.
    M, N = cash.shape
    n = 0
    for m in range(M):
        c, h, left, at_px, nxt, prv = cash[m], shares[m], qty[m], px[m], next_[m], prev_[m]
        head, tail, best = heads[m], tails[m], bests[m]
        for i in order[m]:
            # 1. Cancel its old orders.
            for side in (BID, ASK):
                s = i + N * side
                if left[s] == 0:
                    continue
                p, before, after = at_px[s], prv[s], nxt[s]
                if before >= 0:
                    nxt[before] = after
                else:
                    head[side, p] = after
                if after >= 0:
                    prv[after] = before
                else:
                    tail[side, p] = before
                left[s], nxt[s], prv[s] = 0, -1, -1
                if head[side, p] < 0 and best[side] == p:  # that level emptied: walk to the next one
                    away = -1 if side == BID else 1
                    while 0 < p < 100 and head[side, p] < 0:
                        p += away
                    best[side] = p

            b, b_n, a, a_n = bid[m, i], bid_n[m, i], ask[m, i], ask_n[m, i]
            if b_n <= 0 and a_n <= 0:
                continue
            first = BID
            if b_n > 0 and a_n > 0:
                if b >= a:
                    continue  # crossed with itself. Dropping only one side would favour YES or NO.
                if best[BID] >= a:
                    first = ASK  # the side that trades now goes first: what it frees can back the other
            for k in range(2):
                side = first if k == 0 else 1 - first
                p, q = (b, b_n) if side == BID else (a, a_n)
                if q <= 0:
                    continue
                # 2. Collateral. Checking at posting is enough: a fill on one side only loosens the other's bound.
                if side == BID:
                    q = min(q, c[i] // p)
                else:
                    q = min(q, (c[i] + 100 * h[i]) // (100 - p))
                # 3. Trade: best price first, oldest first, at the resting order's price.
                other, sign = 1 - side, 1 if side == BID else -1
                while q > 0:
                    at = best[other]
                    if at == 0 or at == 100 or sign * (p - at) < 0:
                        break
                    s = head[other, at]
                    j = s - N * other
                    f = min(q, left[s])
                    c[i] -= sign * f * at
                    c[j] += sign * f * at
                    h[i] += sign * f
                    h[j] -= sign * f
                    if n >= len(log):
                        raise RuntimeError("fill log full")
                    log[n, MARKET], log[n, MAKER], log[n, TAKER] = m, j, i
                    log[n, SIDE], log[n, PRICE], log[n, QTY] = sign, at, f
                    n += 1
                    q -= f
                    left[s] -= f
                    if left[s] == 0:  # used up, and it was the oldest at that price
                        after, nxt[s] = nxt[s], -1
                        head[other, at] = after
                        if after >= 0:
                            prv[after] = -1
                        else:
                            tail[other, at] = -1
                            away = -1 if other == BID else 1
                            while 0 < at < 100 and head[other, at] < 0:
                                at += away
                            best[other] = at
                # 4. What's left rests.
                if q > 0 and not ioc[i]:
                    s, last = i + N * side, tail[side, p]
                    left[s], at_px[s], prv[s] = q, p, last
                    if last >= 0:
                        nxt[last] = s
                    else:
                        head[side, p] = s
                    tail[side, p] = s
                    if sign * (p - best[side]) > 0:
                        best[side] = p
    return n


def run_checks():
    failed = []

    def case(f):
        try:
            f()
            print(f"ok   {f.__name__.replace('_', ' ')}")
        except Exception as e:
            print(f"FAIL {f.__name__.replace('_', ' ')}  ({type(e).__name__}: {str(e)[:90]})")
            failed.append(f.__name__.replace("_", " "))

    def play(budget, *ticks, ioc=()):
        """One market. A tick lists (trader, bid, bid size, ask, ask size) in arrival order; others come last."""
        N = len(budget)
        books, fills = Books(np.array([budget])), []
        for tick in ticks:
            q = np.zeros((4, 1, N), np.int64)
            for i, *quote in tick:
                q[:, 0, i] = quote
            first = [i for i, *_ in tick]
            order = np.array([first + [i for i in range(N) if i not in first]])
            fills += books.step(*q, np.isin(np.arange(N), ioc), order)[:, 1:].tolist()
        return books, fills

    def resting(books, i, side):
        return int(books.qty[0, i + books.cash.shape[1] * side])

    def accounts(books, i):
        return int(books.cash[0, i]), int(books.shares[0, i])

    R = 100_000  # a budget that never binds

    @case
    def trades_at_the_resting_price_and_rests_the_remainder():
        b, f = play([R, R], [(0, 0, 0, 40, 10)], [(1, 45, 15, 0, 0), (0, 0, 0, 40, 0)])
        assert f == [[0, 1, 1, 40, 10]], f
        assert accounts(b, 1) == (R - 400, 10) and accounts(b, 0) == (R + 400, -10)
        assert resting(b, 1, BID) == 5

    @case
    def best_price_first():
        _, f = play([R] * 3, [(0, 0, 0, 44, 5), (1, 0, 0, 41, 5), (2, 45, 8, 0, 0)])
        assert f == [[1, 2, 1, 41, 5], [0, 2, 1, 44, 3]], f

    @case
    def oldest_first_at_the_same_price():
        _, f = play([R] * 3, [(1, 30, 5, 0, 0), (0, 30, 5, 0, 0), (2, 0, 0, 30, 7)])
        assert f == [[1, 2, -1, 30, 5], [0, 2, -1, 30, 2]], f

    @case
    def replacing_an_order_loses_its_place():
        _, f = play([R] * 3, [(0, 30, 5, 0, 0), (1, 30, 5, 0, 0)],
                    [(0, 30, 5, 0, 0), (2, 0, 0, 30, 5), (1, 30, 5, 0, 0)])
        assert f == [[1, 2, -1, 30, 5]], f

    @case
    def orders_are_cancelled_on_arrival():
        b, f = play([R, R], [(0, 30, 5, 70, 5)], [(0, 0, 0, 0, 0), (1, 80, 5, 0, 0)])
        assert not f and resting(b, 0, BID) == resting(b, 0, ASK) == 0

    @case
    def immediate_or_cancel_never_rests():
        b, f = play([R, R], [(0, 0, 0, 50, 3)], [(1, 50, 8, 0, 0), (0, 0, 0, 0, 0)], ioc=[1])
        assert f == [[0, 1, 1, 50, 3]] and resting(b, 1, BID) == 0, f

    @case
    def each_trader_has_its_own_budget():
        b, _ = play([1000, 5000, 0], [(0, 40, 999, 0, 0), (1, 20, 999, 0, 0), (2, 10, 9, 0, 0)])
        assert [resting(b, i, BID) for i in range(3)] == [25, 250, 0]
        b, _ = play([1000, 5000], [(0, 0, 0, 40, 999), (1, 0, 0, 80, 999)])
        assert [resting(b, i, ASK) for i in range(2)] == [16, 250], "a sale needs $1 a contract, less the price"

    @case
    def a_position_backs_its_own_closing():
        b, _ = play([1000, R], [(1, 0, 0, 40, 25), (0, 40, 25, 0, 0)], [(1, 30, 99, 0, 0), (0, 0, 0, 30, 999)])
        assert accounts(b, 0) == (1050, -10), accounts(b, 0)  # sells its 25, then goes 10 short

    @case
    def no_position_limit_but_collateral():
        b, _ = play([10**7, 10**7], [(0, 0, 0, 50, 5000), (1, 50, 5000, 0, 0)])
        assert accounts(b, 1) == (10**7 - 250_000, 5000), accounts(b, 1)

    @case
    def crossed_own_quotes_are_both_dropped():
        b, f = play([R, R], [(0, 60, 5, 60, 5), (1, 55, 5, 50, 5)])
        assert not f and not b.qty.any(), "a crossed pair was quoted"

    @case
    def the_side_that_trades_is_placed_first():
        # collateral freed by the trade backs the other quote, for buyers and sellers alike
        b, _ = play([R, 1000], [(0, 0, 0, 40, 20), (1, 40, 10, 60, 999)])
        assert (accounts(b, 1), resting(b, 1, ASK)) == ((600, 10), 40), "taker buying"
        b, _ = play([R, 1000], [(0, 60, 20, 0, 0), (1, 40, 999, 60, 10)])
        assert (accounts(b, 1), resting(b, 1, BID)) == ((1600, -10), 40), "taker selling"

    @case
    def settlement_pays_a_dollar_or_nothing():
        b, _ = play([1000, 1000], [(0, 0, 0, 40, 5), (1, 40, 5, 0, 0)])
        assert b.settle([True]).tolist() == [[700, 1300]] and b.settle([False]).tolist() == [[1200, 800]]

    @case
    def bad_input_is_refused_before_anything_changes():
        good = dict(bid=np.full((2, 3), 40), bid_n=np.full((2, 3), 5), ask=np.full((2, 3), 60),
                    ask_n=np.full((2, 3), 5), ioc=np.zeros(3, bool), order=np.tile(np.arange(3), (2, 1)))
        bad = [dict(order=np.array([[0, 1, 2], [0, 1, 1]])), dict(order=np.array([[0, 1, 2], [0, 1, 3]])),
               dict(order=np.array([[0, 1, 2], [0, 1, -1]])), dict(bid=np.array([[40, 40, 40], [40, 0, 40]])),
               dict(ask=np.array([[60, 60, 60], [60, 100, 60]])), dict(bid_n=np.full((2, 3), 5.5)),
               dict(ask_n=np.full((2, 4), 5)), dict(ioc=np.zeros(4, bool))]
        for change in bad:
            books = Books(np.full((2, 3), 1000))
            try:
                books.step(**{**good, **change})
            except ValueError:
                assert not books.qty.any(), f"changed the book before refusing {list(change)}"
            else:
                raise AssertionError(f"accepted a bad {list(change)[0]}")
        for budget in (np.full((2, 3), -1), np.full((2, 3), 10.5), np.full((2, 3), 10**16), np.full(3, 100)):
            try:
                Books(budget)
            except ValueError:
                continue
            raise AssertionError("accepted a bad budget")

    def random_ticks(rng, M, N, T):
        """Random quotes around a moving centre: all sizes, some crossed pairs, some one-sided."""
        for _ in range(T):
            centre = rng.integers(1, 100, (M, 1))
            bid = np.clip(centre + rng.integers(-6, 5, (M, N)), 1, 99)
            ask = np.clip(bid + rng.integers(-2, 10, (M, N)), 1, 99)
            sizes = np.rint(np.exp(rng.normal(3, 2, (2, M, N)))).astype(np.int64)
            sizes[rng.random((2, M, N)) < 0.02] = 10**12
            sizes[rng.random((2, M, N)) < 0.3] = 0
            yield bid, sizes[0], ask, sizes[1], rng.random((M, N)).argsort(1)

    def random_budgets(rng, M, N):
        budget = np.rint(np.exp(rng.normal(9, 2.5, (M, N)))).astype(np.int64)
        budget[rng.random((M, N)) < 0.05] = 0
        budget[rng.random((M, N)) < 0.02] = 10**12
        return budget

    @case
    def matches_the_brute_force_reference():
        rng = np.random.default_rng(0)
        M, N, T = 60, 24, 120
        budget = random_budgets(rng, M, N)
        books, ioc = Books(budget), np.arange(N) % 4 == 0
        cash, held = budget.tolist(), [[0] * N for _ in range(M)]
        book, traded = [[] for _ in range(M)], 0  # resting [side, price, trader, size], oldest first
        for bid, bid_n, ask, ask_n, order in random_ticks(rng, M, N, T):
            got = books.step(bid, bid_n, ask, ask_n, ioc, order)
            want = []
            for m in range(M):
                c, h, rest = cash[m], held[m], book[m]
                for i in order[m].tolist():
                    rest[:] = [o for o in rest if o[2] != i]
                    quotes = [(BID, int(bid[m, i]), int(bid_n[m, i])), (ASK, int(ask[m, i]), int(ask_n[m, i]))]
                    if quotes[0][2] > 0 and quotes[1][2] > 0:
                        if quotes[0][1] >= quotes[1][1]:
                            continue
                        if any(o[0] == BID and o[1] >= quotes[1][1] for o in rest):
                            quotes.reverse()
                    for side, p, q in quotes:
                        if q <= 0:
                            continue
                        q = min(q, c[i] // p if side == BID else (c[i] + 100 * h[i]) // (100 - p))
                        sign = 1 if side == BID else -1
                        while q > 0:
                            hits = [o for o in rest if o[0] != side and sign * (p - o[1]) >= 0]
                            if not hits:
                                break
                            o = (min if side == BID else max)(hits, key=lambda x: x[1])  # the first of equals
                            f = min(q, o[3])
                            c[i] -= sign * f * o[1]
                            c[o[2]] += sign * f * o[1]
                            h[i] += sign * f
                            h[o[2]] -= sign * f
                            want.append([m, o[2], i, sign, o[1], f])
                            q -= f
                            o[3] -= f
                            if not o[3]:
                                rest.remove(o)
                        if q > 0 and not ioc[i]:
                            rest.append([side, p, i, q])
            assert got.tolist() == want, "fills differ from the reference"
            traded += len(want)
            assert np.array_equal(books.cash, cash) and np.array_equal(books.shares, held), "accounts differ"
            qty = np.zeros((M, 2 * N), np.int64)
            for m in range(M):
                for side, _, i, q in book[m]:
                    qty[m, i + N * side] = q
                bids, asks = [o[1] for o in book[m] if o[0] == BID], [o[1] for o in book[m] if o[0] == ASK]
                assert books.best[m].tolist() == [max(bids, default=0), min(asks, default=100)], "best prices differ"
            assert np.array_equal(books.qty, qty), "resting orders differ"
        assert traded > 20_000, f"barely traded: {traded} fills"

    @case
    def yes_and_no_are_mirror_images():
        rng = np.random.default_rng(1)
        M, N = 200, 30
        budget = random_budgets(rng, M, N)
        a, b, ioc = Books(budget), Books(budget), np.arange(N) % 5 == 0
        traded = 0
        for bid, bid_n, ask, ask_n, order in random_ticks(rng, M, N, 100):
            f = a.step(bid, bid_n, ask, ask_n, ioc, order)
            g = b.step(100 - ask, ask_n, 100 - bid, bid_n, ioc, order)
            g[:, SIDE], g[:, PRICE] = -g[:, SIDE], 100 - g[:, PRICE]
            assert np.array_equal(f, g), "fills aren't mirror images"
            assert np.array_equal(b.shares, -a.shares) and np.array_equal(b.cash, a.cash + 100 * a.shares)
            traded += len(f)
        assert traded > 20_000, f"barely traded: {traded} fills"

    @case
    def money_balances_to_the_cent():
        rng = np.random.default_rng(2)
        M, N = 300, 50
        budget = random_budgets(rng, M, N)
        books, ioc = Books(budget), np.arange(N) % 3 == 0
        cash, held = budget.copy(), np.zeros((M, N), np.int64)
        traded = 0
        for bid, bid_n, ask, ask_n, order in random_ticks(rng, M, N, 150):
            f = books.step(bid, bid_n, ask, ask_n, ioc, order)
            m, maker, taker, side, p, q = f.T
            np.add.at(cash, (m, taker), -side * p * q)
            np.add.at(cash, (m, maker), side * p * q)
            np.add.at(held, (m, taker), side * q)
            np.add.at(held, (m, maker), -side * q)
            assert np.array_equal(cash, books.cash) and np.array_equal(held, books.shares), "the fills don't explain the accounts"
            assert (maker != taker).all() and (q > 0).all(), "a self-trade or an empty fill"
            assert (books.cash >= 0).all() and (books.cash + 100 * books.shares >= 0).all(), "collateral broken"
            assert (books.best[:, BID] < books.best[:, ASK]).all(), "book left crossed"
            assert not books.qty[:, np.r_[ioc, ioc]].any(), "an immediate-or-cancel order rested"
            traded += len(f)
        for outcome in (np.ones(M, bool), np.zeros(M, bool)):
            final = books.settle(outcome)
            assert (final >= 0).all() and np.array_equal(final.sum(1), budget.sum(1)), "settlement doesn't balance"
        assert traded > 50_000, f"barely traded: {traded} fills"

    @case
    def faster_traders_usually_arrive_first():
        speed, M = np.array([8.0, 2.0, 1.0, 1.0]), 200_000
        order = arrival(np.random.default_rng(3), speed, M)
        assert (np.sort(order, 1) == np.arange(4)).all(), "not everyone arrives once"
        place = order.argsort(1)
        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            want = speed[i] / (speed[i] + speed[j])
            got = np.mean(place[:, i] < place[:, j])
            assert abs(got - want) < 4 * np.sqrt(want * (1 - want) / M), f"{i} before {j}: {got:.3f}, not {want:.3f}"
        again = arrival(np.random.default_rng(3), speed, M)
        assert np.array_equal(order, again), "not reproducible"

    assert not failed, f"{len(failed)} order book checks failed: " + "; ".join(failed)
    print("order book checks passed")


def bench():
    """Gate: one generation's matching under 5x slower than a generation of the current market."""
    from time import perf_counter

    from arena import RUNS, events
    from arena.evolve import ERR_CORR, N_EVO, WALLETS, make_orders
    from arena.market import simulate

    M, T, PROS, SLOTS = 400, events.T, 106, 140
    N = PROS + SLOTS
    speed, ioc = np.r_[np.full(16, 8.0), np.full(90, 2.0), np.full(SLOTS, 1.0)], np.arange(N) >= PROS
    pop = dict(np.load(RUNS / "pop.npz"))

    def current(seed):
        rng = np.random.default_rng([seed, 1])
        z = np.sqrt(ERR_CORR) * rng.standard_normal((M, 1)) + np.sqrt(1 - ERR_CORR) * rng.standard_normal((M, N_EVO))
        start = perf_counter()
        simulate(make_orders(pop, z * pop["sigma"]), WALLETS, n_markets=M, seed=seed)
        return perf_counter() - start

    def book(seed, worst):
        """Only the book's own work is timed: events, arrival order, matching, settlement."""
        rng = np.random.default_rng([seed, 2])
        start = perf_counter()
        e = events.draw(M, seed)
        spent, fills, books = perf_counter() - start, 0, Books(np.full((M, N), 10**6 if worst else 10_000))
        for t in range(T):
            if worst:  # everyone quotes both sides every tick around a jumping price
                centre, size, rest = rng.integers(10, 91, (M, 1)) + rng.integers(-4, 5, (M, N)), np.full((M, N), 30), ~ioc | True
            else:      # professionals quote both sides of their view, retail sends about 90 orders a market
                centre = np.rint(np.clip(100 * e["truth"][t], 8, 92))[:, None].astype(np.int64) + rng.integers(-4, 5, (M, N))
                size, rest = np.full((M, N), 30), ~ioc
                size[:, PROS:] *= rng.random((M, SLOTS)) < 90 / SLOTS / T
            start = perf_counter()
            order = arrival(rng, speed, M)
            fills += len(books.step(centre - 1, size, centre + 1, size, ~rest, order))
            spent += perf_counter() - start
        start = perf_counter()
        books.settle(e["outcome"])
        return spent + perf_counter() - start, fills / M

    book(0, False)  # compile first
    ratios = {False: [], True: []}
    for seed in range(8):  # alternate which market goes first, so both see the same machine
        first = current(seed) if seed % 2 else None
        new = {worst: book(seed, worst) for worst in (False, True)}
        old = current(seed) if first is None else first
        for worst, (spent, fills) in new.items():
            ratios[worst].append(spent / old)
        print(f"seed {seed}: current {old:.2f}s | book {new[False][0]:.2f}s ({new[False][1]:,.0f} fills a market) "
              f"| heaviest load {new[True][0]:.2f}s ({new[True][1]:,.0f} fills)", flush=True)
    for worst, r in ratios.items():
        print(f"{'heaviest load' if worst else 'expected load'}: median {np.median(r):.2f}x, worst {max(r):.2f}x")
        assert np.median(r) < 5, "over 5x slower than the current market"
    print("speed gate passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="check", choices=["check", "bench"])
    if ap.parse_args().mode == "check":
        run_checks()
    else:
        bench()
