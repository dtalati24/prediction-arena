"""Phase 4, step 6: wallet lock-on. Can a population hurt one trader by targeting its wallet?

The watchers of arena.evolve follow a leaderboard spot, so they lose a trader whose spot moves.
Here watchers are handed the trained agent's wallet directly. Two questions, on the same markets
as a control where those bots are idle:
  damage  does the agent earn less?
  payoff  do the hunters earn more than 0? If not, selection has no reason to keep them.

    python -m arena.hunt pop_watch    # checks + payoff table (needs runs/ppo_pop_watch.pt)
"""
import argparse

import numpy as np
import torch

from arena import RUNS
from arena.evolve import FORECAST, WALLETS, WATCH, drop_in
from arena.ppo import Agent, rl_trader

M = 1000


def free_slots(pop, keep, k):
    """k near-idle bots (not forecasters, not on the agent's watch list) to turn into hunters."""
    slots = [j for j in np.flatnonzero(pop["kind"] != FORECAST) if j not in set(keep)]
    assert len(slots) >= k, f"only {len(slots)} free slots"
    return np.array(slots[:k])


def hunted(pop, make_agent, slots, target, alpha, lag, n_markets, seed):
    """drop_in with the bots in slots locked on to wallet target. Returns the trader's expected
    profit per market, each hunter's (M, k), positions of hunters and target by tick, and the trader's real profit."""
    p = {k: v.copy() for k, v in pop.items()}
    p["kind"][slots], p["alpha"][slots], p["lag"][slots], p["target"][slots] = WATCH, alpha, lag, target
    pos = []

    def make(*args):
        fn = make_agent(*args)

        def recorded(t, price_hist, truth, shares, rng):
            pos.append(shares[:, np.r_[slots, target]].copy())
            return fn(t, price_hist, truth, shares, rng)
        return recorded
    profit, r = drop_in(p, make, n_markets, seed)
    pos = np.array(pos + [r["shares"][:, np.r_[slots, target]]])
    gain = (np.diff(pos[:, :, :-1], axis=0) * (r["truth"][:-1] - r["fills"])[:, :, None]).sum(0)
    real = (np.diff(pos[:, :, :-1], axis=0) * (r["outcome"] - r["fills"])[:, :, None]).sum(0)
    assert np.allclose(real, r["profit"][:, slots]), "hunters' bookkeeping is off"
    return profit, gain, pos, r["profit"][:, -1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pop")
    args = ap.parse_args()
    torch.set_num_threads(1)
    pop = dict(np.load(RUNS / f"{args.pop}.npz"))
    log = np.load(RUNS / f"ppo_{args.pop}_log.npz")
    agent = Agent()
    agent.load_state_dict(torch.load(RUNS / f"ppo_{args.pop}.pt"))
    make = rl_trader(agent, log["top"], log["bottom"], greedy=True)
    keep = np.r_[log["top"], log["bottom"]]

    # Check: a small hunter holds exactly alpha x the agent's position one tick earlier, and
    # with alpha 0 the hunters never trade.
    slots = free_slots(pop, keep, 1)
    _, _, pos, _ = hunted(pop, make, slots, WALLETS, 0.05, 200, 50, 5)
    assert np.allclose(pos[1:, :, 0], 0.05 * pos[:-1, :, 1]), "hunter doesn't follow the agent"
    _, gain, pos, _ = hunted(pop, make, slots, WALLETS, 0.0, 200, 50, 5)
    assert (gain == 0).all() and (pos[:, :, 0] == 0).all(), "idle control traded"
    print("hunt checks passed", flush=True)

    top_fc = next(int(c) for c in log["top"] if c < len(pop["kind"]) and pop["kind"][c] == FORECAST)
    for k in [10, 40]:
        slots = free_slots(pop, keep, k)
        base, _, _, _ = hunted(pop, make, slots, WALLETS, 0.0, 1, M, 7)
        print(f"== {args.pop}, {k} hunters: agent alone {base.mean():+.3f}/market", flush=True)
        rows = [("agent", WALLETS, a, lag) for a in [1.0, 0.25, -0.25, -1.0] for lag in [1, 20, 200]]
        rows += [("forecaster", top_fc, a, 200) for a in [1.0, 0.25, -0.25, -1.0]]  # control: same trading, other wallet
        for who, target, alpha, lag in rows:
            p, gain, _, real = hunted(pop, make, slots, target, alpha, lag, M, 7)
            d, g = p - base, gain.mean(1)
            print(f"   {'copy' if alpha > 0 else 'fade'} {who:10s} x{abs(alpha):4.2f} lag {lag:3d}: "
                  f"agent {d.mean():+.3f} +- {d.std() / np.sqrt(M):.3f} ({100 * d.mean() / base.mean():+4.0f}%), "
                  f"each hunter {g.mean():+.3f} +- {g.std() / np.sqrt(M):.3f}"
                  f" | agent expected {p.mean():+.2f}, real {real.mean():+.2f} +- {real.std() / np.sqrt(M):.2f}", flush=True)
