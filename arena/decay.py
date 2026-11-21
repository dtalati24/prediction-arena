"""Phase 4, step 5: edge decay. Does a learned edge survive a population that adapts to it?

A trained PPO agent joins an evolving population; it never reproduces or dies. Each generation is
400 markets with the agent inside, then the population evolves on profit exactly as in evolve.py,
so whatever the agent takes from a species becomes selection pressure on that species.

Three versions per seed, from the same starting population and the same markets:
  forecaster  a plain forecaster joins instead (control: how much the market drifts anyway)
  frozen      the trained agent, never updated
  learning    the trained agent keeps training: each generation is one PPO update

All three keep the accuracy and size they started with. The agent's on-chain watch list (top and
bottom 10 wallets by profit per unit of risk) is re-ranked from the previous generation's public
results, since wallets change hands as bots die and are replaced.

    python -m arena.decay run pop --gens 200   # runs/pop.npz with runs/ppo_pop.pt, all three versions
    python -m arena.decay report               # chart from every finished run
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from arena import FIGURES, RUNS
from arena.evolve import FORECAST, INFO_COST, N_EVO, N_MARKETS, drop_in, labels, next_generation
from arena.kalman import with_kalman
from arena.ppo import LR, Agent, rl_trader, update, watch_list
from arena.wallets import WALLETS

VERSIONS = ["forecaster", "frozen", "learning"]


def run(pop, version, agent, gens, seed, log=print):
    """Co-evolve pop with one dropped-in trader for gens generations; per-generation history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fc = pop["kind"] == FORECAST
    sigma, size = np.median(pop["sigma"][fc]), np.median(np.abs(pop["alpha"][fc]))  # fixed from here on
    opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5) if version == "learning" else None
    watch = watch_list(pop)
    hist, start = defaultdict(list), time.time()
    for g in range(gens):
        rec = defaultdict(list)
        if version == "forecaster":
            make = lambda base, err, s_, a_: with_kalman(base, err, size, sigma, np.inf)  # noqa: E731
        else:
            make = rl_trader(agent, *watch, rec if version == "learning" else None, greedy=version == "frozen")
        p, r = drop_in(pop, make, N_MARKETS, [seed, g], sigma=sigma, size=size)
        if version == "learning":
            update(agent, opt, rec, r)

        prof = r["profit"]
        pw = prof[:, :WALLETS]  # public results -> next generation's watch list
        rank = np.argsort(-(pw.mean(0) / (pw.std(0) + 1e-9)))
        watch = (rank[:10], rank[-10:])
        lab = labels(pop)
        hist["agent"].append(p.mean())
        hist["agent_real"].append(prof[:, -1].mean())
        hist["share"].append(np.bincount(lab, minlength=3) / N_EVO)
        hist["type_profit"].append([prof[:, :N_EVO][:, lab == k].sum(1).mean() / max((lab == k).sum(), 1) for k in range(3)])
        hist["fc_size"].append(np.median(np.abs(pop["alpha"][lab == 2])) if (lab == 2).any() else np.nan)
        hist["gap"].append(np.abs(r["prices"] - r["truth"]).mean())
        fitness = prof[:, :N_EVO].sum(0) - (pop["kind"] == FORECAST) * N_MARKETS * INFO_COST / pop["sigma"] ** 2
        pop = next_generation(pop, fitness, rng)
        if (g + 1) % 25 == 0:
            el = time.time() - start
            log(f"{version} gen {g + 1}/{gens}: agent {np.mean(hist['agent'][-25:]):+.3f}/market (last 25), "
                f"{el / 60:.0f} min elapsed, ~{el / (g + 1) * (gens - g - 1) / 60:.0f} min left")
    return {k: np.array(v) for k, v in hist.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("pop", nargs="?", help="population name in runs/, e.g. pop or pop_seed1")
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    args = ap.parse_args()
    torch.set_num_threads(1)

    if args.mode == "run":
        name = args.pop.removesuffix(".npz")
        for version in args.versions:
            agent = Agent()
            if version != "forecaster":
                agent.load_state_dict(torch.load(RUNS / f"ppo_{name}.pt"))
            h = run(dict(np.load(RUNS / f"{name}.npz")), version, agent, args.gens, seed=1,
                    log=lambda s: print(s, flush=True))
            np.savez(RUNS / f"decay_{name}_{version}.npz", **h)
            print(f"saved runs/decay_{name}_{version}.npz", flush=True)

    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = sorted({f.stem.removeprefix("decay_").rsplit("_", 1)[0] for f in RUNS.glob("decay_*_*.npz")})
        assert names, "run some decay experiments first"
        colors = dict(forecaster="black", frozen="tab:red", learning="tab:blue")
        fig, ax = plt.subplots(2, len(names), figsize=(5 * len(names), 6.5), squeeze=False, sharex=True)
        for i, name in enumerate(names):
            print(f"== {name}")
            for v in VERSIONS:
                try:
                    h = np.load(RUNS / f"decay_{name}_{v}.npz")
                except FileNotFoundError:
                    continue
                a, G = h["agent"], len(h["agent"])
                q = max(G // 8, 1)
                print(f"   {v:10s} agent expected profit/market: first {q} gens {a[:q].mean():+.3f}, last {q} gens {a[-q:].mean():+.3f}"
                      f" | real (all gens) {h['agent_real'].mean():+.3f} | final shares mom/con/fc {h['share'][-q:].mean(0).round(2)}"
                      f" | forecaster size {np.nanmean(h['fc_size'][-q:]):.0f} | gap {h['gap'][-q:].mean():.3f}")
                w = min(10, G)
                ax[0, i].plot(np.arange(w - 1, G), np.convolve(a, np.ones(w) / w, "valid"), color=colors[v], label=v)
                ls = "-" if v == "frozen" else "--" if v == "forecaster" else None
                if ls:
                    for k, c in zip([0, 2], ["tab:orange", "tab:green"]):
                        ax[1, i].plot(h["share"][:, k], color=c, ls=ls, label=f"{['momentum', '', 'forecaster'][k]} share ({v} inside)")
            ax[0, i].axhline(0, color="grey", lw=0.5)
            ax[0, i].set_title(f"{name}: dropped-in trader's profit (10-gen avg)")
            ax[1, i].set_xlabel("generation after the trader joins")
            ax[1, i].set_ylim(0, 1)
        ax[0, 0].set_ylabel("expected profit per market")
        ax[1, 0].set_ylabel("population share")
        ax[0, 0].legend(fontsize=8)
        ax[1, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "phase4_decay.png", dpi=120)
        print("chart saved to figures/phase4_decay.png")
