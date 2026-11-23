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

In a population with wallet watchers (runs/pop_watch*.npz, from `arena.evolve --watchers`), the
dropped-in trader is a wallet on the public leaderboard like any other, so watchers can pick it.

    python -m arena.decay run pop --gens 200   # runs/pop.npz with runs/ppo_pop.pt, all three versions
    python -m arena.decay run pop_watch --gens 200   # a population with wallet watchers
    python -m arena.decay run pop_watch --lock              # watchers can lock on to one wallet
    python -m arena.decay run pop_watch --lock --plant 10   # ...and 10 start locked on to the trader
    python -m arena.decay run pop_watch --lock --plant 10 --versions frozen --watch-step 0.02 --replay 10
                                               # finer copy-size mutation + matched control (runs/rerun_*)
    python -m arena.decay report               # charts from every finished run
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from arena import FIGURES, RUNS
from arena.evolve import (FORECAST, INFO_COST, N_EVO, N_MARKETS, NAMES, WATCH, WATCHED, drop_in, labels,
                          next_generation, targets, watch_stats)
from arena.hunt import free_slots
from arena.kalman import with_kalman
from arena.ppo import LR, Agent, rl_trader, update, watch_list
from arena.wallets import WALLETS

VERSIONS = ["forecaster", "frozen", "learning"]


def matched_control(pop, make, seed, sigma, size, hunted, prev):
    """What this generation's hunters do to the trader, on the same markets.

    hunted: this generation's (expected profit, simulation). Replayed twice: with the hunters idle,
    and with them copying last generation's best forecaster, scaled to a similar position size.
    Returns hunters, the trader's profit idle / hunted / control, and the copiers' mean final
    absolute position hunted / control (all nan but the count if there's nothing to compare).
    """
    on = np.flatnonzero((pop["kind"] == WATCH) & (pop["target"] == WALLETS))
    fc = np.flatnonzero((pop["kind"] == FORECAST) & (pop["age"] > 0))
    if not len(on) or not len(fc) or prev is None:
        return [len(on)] + [np.nan] * 5
    idle = {k: v.copy() for k, v in pop.items()}
    idle["alpha"][on] = 0
    p0, r0 = drop_in(idle, make, N_MARKETS, seed, sigma=sigma, size=size)
    f = fc[np.argmax(prev[fc])]
    scale = np.abs(r0["shares"][:, -1]).mean() / (np.abs(r0["shares"][:, f]).mean() + 1e-9)
    other = {k: v.copy() for k, v in pop.items()}
    other["target"][on], other["alpha"][on] = f, pop["alpha"][on] * scale
    p1, r1 = drop_in(other, make, N_MARKETS, seed, sigma=sigma, size=size)
    p, r = hunted
    return [len(on), p0.mean(), p.mean(), p1.mean(),
            np.abs(r["shares"][:, on]).mean(), np.abs(r1["shares"][:, on]).mean()]


def run(pop, version, agent, gens, seed, log=print, lock=False, plant=0, watch_step=1.0, replay=0):
    """Co-evolve pop with one dropped-in trader for gens generations; per-generation history.

    lock: watchers get a lock-on gene (half start with it). plant: this many near-idle bots start
    as quiet hunters of the trader (copy its whole position at 0.25x), locked on to its wallet.
    watch_step: the absolute part of a watcher child's copy-size mutation. replay: every this many
    generations, also run matched_control (history "replay": generation, then what it returns).
    """
    assert not (replay and version == "learning"), "a replay would be recorded as training data"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fc = pop["kind"] == FORECAST
    sigma, size = np.median(pop["sigma"][fc]), np.median(np.abs(pop["alpha"][fc]))  # fixed from here on
    opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5) if version == "learning" else None
    watch = watch_list(pop)
    if lock:  # its own random stream, so the markets are those of the runs without the gene
        pop = {**{k: v.copy() for k, v in pop.items()},
               "lock": np.random.default_rng([seed, 1]).integers(0, 2, N_EVO)}
        if plant:
            j = free_slots(pop, np.r_[watch], plant)
            pop["kind"][j], pop["alpha"][j], pop["lag"][j] = WATCH, 0.25, 200
            pop["lock"][j], pop["target"][j] = 1, WALLETS
    hist, start, prev = defaultdict(list), time.time(), None
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
        if replay and (g + 1) % replay == 0:
            hist["replay"].append([g, *matched_control(pop, make, [seed, g], sigma, size, (p, r), prev)])
        prev = prof.mean(0)
        pw = prof[:, :WALLETS]  # public results -> next generation's watch list
        rank = np.argsort(-(pw.mean(0) / (pw.std(0) + 1e-9)))
        watch = (rank[:10], rank[-10:])
        lab = labels(pop)
        hist["agent"].append(p.mean())
        hist["agent_real"].append(prof[:, -1].mean())
        hist["share"].append(np.bincount(lab, minlength=len(NAMES)) / N_EVO)
        hist["type_profit"].append([prof[:, :N_EVO][:, lab == k].sum(1).mean() / max((lab == k).sum(), 1) for k in range(len(NAMES))])
        if "rank" in pop:
            for k, v in watch_stats(pop, lab).items():
                hist[k].append(v)
        if "lock" in pop:
            w = pop["kind"] == WATCH
            on_it = w & (pop["target"] == WALLETS)
            hist["hunters"].append(on_it.sum())
            hist["hunter_alpha"].append(np.median(pop["alpha"][on_it]) if on_it.any() else np.nan)
            hist["hunter_profit"].append(prof[:, :N_EVO][:, on_it].mean() if on_it.any() else np.nan)
            hist["locked"].append(pop["lock"][w].mean() if w.any() else np.nan)
        hist["fc_size"].append(np.median(np.abs(pop["alpha"][lab == 2])) if (lab == 2).any() else np.nan)
        hist["gap"].append(np.abs(r["prices"] - r["truth"]).mean())
        fitness = prof[:, :N_EVO].sum(0) - (pop["kind"] == FORECAST) * N_MARKETS * INFO_COST / pop["sigma"] ** 2
        pop = next_generation(pop, fitness, rng, watch_step)
        if "rank" in pop:  # public results, the dropped-in trader's included -> next leaderboard
            pop["target"] = targets(pop, prof)
        if (g + 1) % 25 == 0:
            el = time.time() - start
            on = f", watchers on it {np.mean([x[-1] for x in hist['watched'][-25:]]):.2f}" if "watched" in hist else ""
            log(f"{version} gen {g + 1}/{gens}: agent {np.mean(hist['agent'][-25:]):+.3f}/market (last 25){on}, "
                f"{el / 60:.0f} min elapsed, ~{el / (g + 1) * (gens - g - 1) / 60:.0f} min left")
    return {k: np.array(v) for k, v in hist.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("pop", nargs="?", help="population name in runs/, e.g. pop or pop_seed1")
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--lock", action="store_true", help="watchers can lock on to one wallet")
    ap.add_argument("--plant", type=int, default=0, help="hunters locked on to the trader at the start")
    ap.add_argument("--watch-step", type=float, default=1.0, help="watcher children's copy-size mutation step")
    ap.add_argument("--seed", type=int, default=1, help="seed of the markets and the evolution")
    ap.add_argument("--replay", type=int, default=0, help="run the matched control every this many generations")
    args = ap.parse_args()
    torch.set_num_threads(1)

    if args.plant and not args.lock:
        ap.error("--plant needs --lock")
    if args.replay and (not args.lock or "learning" in args.versions):
        ap.error("--replay needs --lock, and --versions without learning")
    if args.mode == "run":
        name = args.pop.removesuffix(".npz")
        out = name + ("_lock" if args.lock else "") + (str(args.plant) if args.plant else "")
        kind = "decay"
        if (args.watch_step, args.seed, args.replay) != (1.0, 1, 0):  # kept apart from the standard runs
            kind, out = "rerun", f"{out}_step{args.watch_step:g}_seed{args.seed}"
        for version in args.versions:
            agent = Agent()
            if version != "forecaster":
                agent.load_state_dict(torch.load(RUNS / f"ppo_{name}.pt"))
            h = run(dict(np.load(RUNS / f"{name}.npz")), version, agent, args.gens, seed=args.seed,
                    log=lambda s: print(s, flush=True), lock=args.lock, plant=args.plant,
                    watch_step=args.watch_step, replay=args.replay)
            np.savez(RUNS / f"{kind}_{out}_{version}.npz", **h)
            print(f"saved runs/{kind}_{out}_{version}.npz", flush=True)

    else:  # report: one chart for the original populations, one for those with wallet watchers
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_names = sorted({f.stem.removeprefix("decay_").rsplit("_", 1)[0] for f in RUNS.glob("decay_*_*.npz")})
        assert all_names, "run some decay experiments first"
        colors = dict(forecaster="black", frozen="tab:red", learning="tab:blue")
        for group in ["", "watch", "lock"]:
            names = [n for n in all_names
                     if group == ("lock" if "_lock" in n else "watch" if "watch" in n else "")]
            watch = group != ""
            if not names:
                continue
            rows = 3 if watch else 2
            fig, ax = plt.subplots(rows, len(names), figsize=(5 * len(names), 3.25 * rows), squeeze=False, sharex=True)
            for i, name in enumerate(names):
                print(f"== {name}")
                for v in VERSIONS:
                    try:
                        h = np.load(RUNS / f"decay_{name}_{v}.npz")
                    except FileNotFoundError:
                        continue
                    a, G = h["agent"], len(h["agent"])
                    q = max(G // 8, 1)
                    shares = ", ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, h["share"][-q:].mean(0)))
                    # Medians too: a copy-trading herd now and then hands one generation's traders 10-100x the usual profit.
                    print(f"   {v:10s} agent expected profit/market: first {q} gens {a[:q].mean():+.3f} (median {np.median(a[:q]):+.3f}),"
                          f" last {q} gens {a[-q:].mean():+.3f} (median {np.median(a[-q:]):+.3f})"
                          f" | real (all gens) {h['agent_real'].mean():+.3f} | final shares {shares}"
                          f" | forecaster size {np.nanmean(h['fc_size'][-q:]):.0f} | gap {h['gap'][-q:].mean():.3f}")
                    w = min(10, G)
                    ax[0, i].plot(np.arange(w - 1, G), np.convolve(a, np.ones(w) / w, "valid"), color=colors[v], label=v)
                    if "hunters" in h:
                        n = h["hunters"]
                        print(f"   {'':10s} hunters of it: at start {n[0]}, after 10 gens {n[min(10, G - 1)]}, last {q} gens"
                              f" {n[-q:].mean():.1f}, most {n.max()}; watchers with the lock gene {np.nanmean(h['locked'][-q:]):.2f}")
                    if watch:
                        on = h["watched"][:, -1]  # share of watchers watching the dropped-in trader
                        last = h["watched"][-q:]
                        last = last[last.sum(1) > 0]  # generations with any watchers
                        print(f"   {'':10s} watchers watching it: first {q} gens {on[:q].mean():.2f}, last {q} gens {on[-q:].mean():.2f}, most in any"
                              f" generation {on.max():.2f} | watching (last {q}): " + ", ".join(f"{n} {x:.2f}" for n, x in zip(WATCHED, last.mean(0))))
                        ax[2, i].plot(np.arange(w - 1, G), np.convolve(on, np.ones(w) / w, "valid"), color=colors[v], label=v)
                    ls = "-" if v == "frozen" else "--" if v == "forecaster" else None
                    if ls:
                        series = [(h["share"][:, 0], "tab:orange", "momentum"), (h["share"][:, 2], "tab:green", "forecaster")]
                        if watch:
                            series.append((h["share"][:, 3:].sum(1), "tab:purple", "watcher"))
                        for y, c, n in series:
                            ax[1, i].plot(y, color=c, ls=ls, label=f"{n} share ({v} inside)")
                ax[0, i].axhline(0, color="grey", lw=0.5)
                br = "\n" if group == "lock" else " "  # six columns: the title needs two lines
                ax[0, i].set_title(f"{name}:{br}dropped-in trader's profit (10-gen avg)")
                ax[-1, i].set_xlabel("generation after the trader joins")
                ax[1, i].set_ylim(0, 1)
                if watch:
                    ax[2, i].set_ylim(-0.02, 1.02)
                    ax[2, i].set_title("share of watchers watching the trader (10-gen avg)", fontsize=9)
            ax[0, 0].set_ylabel("expected profit per market")
            ax[1, 0].set_ylabel("population share")
            ax[0, 0].legend(fontsize=8)
            ax[1, 0].legend(fontsize=7)
            fig.tight_layout()
            path = FIGURES / f"phase4_decay{'_' + group if group else ''}.png"
            fig.savefig(path, dpi=120)
            print(f"chart saved to figures/{path.name}")
