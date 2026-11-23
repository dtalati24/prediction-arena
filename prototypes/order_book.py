"""PROTOTYPE, throwaway: would a real limit order book work as the arena's market?

Prices are whole cents (1..99), sizes whole contracts, money whole cents, as on Kalshi. Each tick
every bot arrives once in random order, cancels its old orders and may post one bid and one offer.
An order trades at the resting order's price, best price first and oldest first; what's left rests
(or is cancelled, if immediate-or-cancel). Every order is fully collateralised when posted.

Go/no-go: under 5x slower than the current market, and prices still track the truth.

    python -m prototypes.order_book check   # matching engine against a brute-force copy
    python -m prototypes.order_book bench   # speed and accuracy against the current market
"""
import argparse


import numpy as np
from numba import njit




BID, ASK = 0, 1
LIMIT, BUDGET = 100, 100_00  # position limit in contracts, budget in cents


class Books:
    """M independent books for N bots. An order's slot is bot + N * side."""

    def __init__(self, M, N):
        self.cash = np.full((M, N), BUDGET, np.int64)
        self.shares = np.zeros((M, N), np.int64)
        self.qty, self.px = np.zeros((M, 2 * N), np.int64), np.zeros((M, 2 * N), np.int64)
        self.next, self.prev = np.full((M, 2 * N), -1, np.int64), np.full((M, 2 * N), -1, np.int64)
        self.head, self.tail = np.full((M, 2, 100), -1, np.int64), np.full((M, 2, 100), -1, np.int64)
        self.best = np.tile(np.array([0, 100], np.int64), (M, 1))  # 0 / 100: that side is empty

    def step(self, bid, bid_n, ask, ask_n, ioc, order):
        """Everyone's orders for one tick. Returns volume and traded cents, per market."""
        return step_all(self.cash, self.shares, self.qty, self.px, self.next, self.prev, self.head,
                        self.tail, self.best, bid, bid_n, ask, ask_n, ioc, order)


@njit(cache=True)
def unlink(s, side, qty, px, nxt, prv, head, tail, best):
    p, a, b = px[s], prv[s], nxt[s]
    if a >= 0:
        nxt[a] = b
    else:
        head[side, p] = b
    if b >= 0:
        prv[b] = a
    else:
        tail[side, p] = a
    qty[s], nxt[s], prv[s] = 0, -1, -1
    if best[side] == p and head[side, p] < 0:  # that level emptied: walk to the next one
        if side == BID:
            while p > 0 and head[side, p] < 0:
                p -= 1
        else:
            while p < 100 and (head[side, p] < 0):
                p += 1
        best[side] = p


@njit(cache=True)
def place(i, side, p, q, rest, cash, sh, qty, px, nxt, prv, head, tail, best):
    N = len(cash)
    if q <= 0 or p < 1 or p > 99:
        return 0, 0
    # A fill on one side only loosens the other side's bounds, so checking at posting is enough.
    if side == BID:
        q = min(q, cash[i] // p, LIMIT - sh[i])
    else:
        q = min(q, (cash[i] + 100 * sh[i]) // (100 - p), LIMIT + sh[i])
    other, sign = 1 - side, 1 if side == BID else -1
    volume, traded = 0, 0
    while q > 0:
        at = best[other]
        if at == 0 or at == 100 or sign * (p - at) < 0:
            break
        s = head[other, at]
        j = s % N
        f = min(q, qty[s])
        cash[i] -= sign * f * at
        cash[j] += sign * f * at
        sh[i] += sign * f
        sh[j] -= sign * f
        q -= f
        qty[s] -= f
        volume += f
        traded += f * at
        if qty[s] == 0:
            unlink(s, other, qty, px, nxt, prv, head, tail, best)
    if q > 0 and rest:
        s, last = i + N * side, tail[side, p]
        qty[s], px[s], prv[s] = q, p, last
        if last >= 0:
            nxt[last] = s
        else:
            head[side, p] = s
        tail[side, p] = s
        if sign * (p - best[side]) > 0:
            best[side] = p
    return volume, traded


@njit(cache=True)
def step_all(cash, sh, qty, px, nxt, prv, head, tail, best, bid, bid_n, ask, ask_n, ioc, order):
    M, N = cash.shape
    volume, traded = np.zeros(M, np.int64), np.zeros(M, np.int64)
    for m in range(M):
        c, h, q, p, nx, pv = cash[m], sh[m], qty[m], px[m], nxt[m], prv[m]
        hd, tl, bs = head[m], tail[m], best[m]
        for i in order[m]:
            for side in (BID, ASK):
                if q[i + N * side] > 0:
                    unlink(i + N * side, side, q, p, nx, pv, hd, tl, bs)
            v1, c1 = place(i, BID, bid[m, i], bid_n[m, i], not ioc[i], c, h, q, p, nx, pv, hd, tl, bs)
            v2, c2 = 0, 0
            if bid_n[m, i] <= 0 or ask[m, i] > bid[m, i]:  # never trade with yourself
                v2, c2 = place(i, ASK, ask[m, i], ask_n[m, i], not ioc[i], c, h, q, p, nx, pv, hd, tl, bs)
            volume[m] += v1 + v2
            traded[m] += c1 + c2
    return volume, traded


@njit(cache=True)
def shuffled(u):
    M, N = u.shape
    out = np.empty((M, N), np.int64)
    for m in range(M):
        for i in range(N):
            j = int(u[m, i] * (i + 1))
            if j != i:
                out[m, i] = out[m, j]
            out[m, j] = i
    return out


def check():
    """Random orders through the engine and through an obviously correct list-based copy."""
    rng = np.random.default_rng(0)
    M, N, T = 100, 16, 150
    books, ioc = Books(M, N), np.arange(N) % 4 == 0
    cash, sh = [[BUDGET] * N for _ in range(M)], [[0] * N for _ in range(M)]
    resting = [[] for _ in range(M)]  # [side, price, bot, size], oldest first
    traded_any = 0
    for _ in range(T):
        centre = rng.integers(3, 98, (M, 1))
        bid, ask = (np.clip(centre + rng.integers(-5, 6, (M, N)), 1, 99) for _ in range(2))
        bid_n, ask_n = (np.maximum(rng.integers(-20, 70, (M, N)), 0) for _ in range(2))
        order = shuffled(rng.random((M, N)))
        assert (np.sort(order, 1) == np.arange(N)).all(), "not a permutation"
        volume, traded = books.step(bid, bid_n, ask, ask_n, ioc, order)
        traded_any += volume.sum()
        for m in range(M):
            book, c, h, vol, money = resting[m], cash[m], sh[m], 0, 0
            for i in order[m]:
                book[:] = [o for o in book if o[2] != i]
                for side, p, q in ((BID, bid[m, i], bid_n[m, i]), (ASK, ask[m, i], ask_n[m, i])):
                    p, q = int(p), int(q)
                    if q <= 0 or (side == ASK and bid_n[m, i] > 0 and p <= bid[m, i]):
                        continue
                    if side == BID:
                        q = min(q, c[i] // p, LIMIT - h[i])
                    else:
                        q = min(q, (c[i] + 100 * h[i]) // (100 - p), LIMIT + h[i])
                    while q > 0:
                        hits = [o for o in book if o[0] != side and (o[1] <= p if side == BID else o[1] >= p)]
                        if not hits:
                            break
                        # min/max return the first of equals, i.e. the oldest at the best price
                        o = (min if side == BID else max)(hits, key=lambda x: x[1])
                        f, sign = min(q, o[3]), 1 if side == BID else -1
                        c[i] -= sign * f * o[1]
                        c[o[2]] += sign * f * o[1]
                        h[i] += sign * f
                        h[o[2]] -= sign * f
                        q -= f
                        o[3] -= f
                        vol += f
                        money += f * o[1]
                        if o[3] == 0:
                            book.remove(o)
                    if q > 0 and not ioc[i]:
                        book.append([side, p, int(i), q])
            assert (vol, money) == (volume[m], traded[m]), "trades differ from the reference"
            bids, asks = [o[1] for o in book if o[0] == BID], [o[1] for o in book if o[0] == ASK]
            assert list(books.best[m]) == [max(bids, default=0), min(asks, default=100)], "best prices differ"
        assert (books.cash == np.array(cash)).all() and (books.shares == np.array(sh)).all(), "accounts differ"
        assert (books.cash >= 0).all() and (books.cash + 100 * books.shares >= 0).all(), "collateral broken"
        assert (np.abs(books.shares) <= LIMIT).all(), "position limit broken"
        assert (books.cash.sum(1) == N * BUDGET).all() and (books.shares.sum(1) == 0).all(), "not conserved"
        assert (books.best[:, BID] < books.best[:, ASK]).all(), "book left crossed"
    assert traded_any > 100_000, "check barely traded"
    print(f"order book checks passed ({M * N * T:,} arrivals, {traded_any:,} contracts traded)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "bench"])
    ap.add_argument("--markets", type=int, default=400)
    args = ap.parse_args()
    if args.mode == "check":
        check()
    else:
        from prototypes.book_ecology import bench
        bench(args.markets)
