"""Phase 4, step 4: a reinforcement-learning trader (PPO).

Same information as the forecaster and the Kalman trader: its private estimate s (with the same
kind of error), the price, its own position, and the public on-chain ledger (what the top and
bottom wallets of the leaderboard hold). Each tick a small neural network picks two numbers: how
far to shift its valuation away from its estimate, and a multiplier on its size (capped at 100,
like everyone). With both at 0 it is exactly a forecaster, and training starts there, so whatever
it learns is an improvement on the forecaster's rule.

Reward: each trade valued at the true probability when it was made (as in evolve.drop_in). The
network never sees the truth; this reward has the same expected value as real profit but far less
noise, so it learns the same thing faster.

PPO as in CleanRL (clipped objective, GAE, Adam). Each iteration runs 400 fresh markets, i.e. 400
parallel episodes of 200 ticks, against a fixed evolved population.

    python -m arena.ppo train pop      # checks, then training on runs/pop.npz; saves runs/ppo_pop.pt
    python -m arena.ppo report         # evaluates every trained agent on the 2000 test markets
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from arena import FIGURES, RUNS
from arena.evolve import drop_in
from arena.kalman import trial as kalman_trial
from arena.market import lo
from arena.wallets import leaderboard

T = 200
FEATURES = ["estimate - price", "avg past (estimate - price)", "time left", "own position",
            "price level", "price move 1 tick", "price move 5 ticks",
            "top wallets' position", "bottom wallets' position", "top wallets' last trade",
            "bottom wallets' last trade"]
N_OBS, N_ACT = len(FEATURES), 2  # actions: valuation shift (log-odds), log size multiplier
N_TRAIN = 400
LR, GAE_LAMBDA, CLIP, EPOCHS, MINIBATCH, MAX_GRAD = 3e-4, 0.95, 0.2, 4, 4096, 0.5


def layer(n_in, n_out, std=np.sqrt(2)):
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(layer(N_OBS, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))
        # Last layer starts near zero, so the untrained agent is (almost) a forecaster.
        self.actor = nn.Sequential(layer(N_OBS, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, N_ACT, 0.01))
        # Small exploration noise: noise in the valuation each tick makes it trade back and forth.
        self.log_std = nn.Parameter(torch.full((N_ACT,), -2.0))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())


def rl_trader(agent, top, bottom, record=None, greedy=False):
    """make_agent for evolve.drop_in. top/bottom: wallet columns it watches on chain.
    record (dict of lists) collects observations, actions, log-probs and values per tick."""
    def make(base, err, sigma, size):
        st = {}

        def fn(t, price_hist, truth, shares, rng):
            d, a, e = base(t, price_hist, truth, shares, rng)
            s = lo(truth[t]) + err[:, 0]
            L, L1, L5 = (lo(price_hist[max(t - k, 0)]) for k in (0, 1, 5))
            top_pos, bot_pos = shares[:, top].mean(1), shares[:, bottom].mean(1)
            if t == 0:
                st["sum"], st["top"], st["bot"] = np.zeros(len(s)), top_pos, bot_pos
            else:  # last tick's estimate vs the price that cleared on it, as the Kalman trader uses
                st["sum"] += st["prev"] - L
            obs = np.stack([s - L, st["sum"] / max(t, 1), np.full(len(s), (T - t) / T), shares[:, -1] / 100,
                            L / 6, L - L1, L - L5, top_pos / 100, bot_pos / 100,
                            (top_pos - st["top"]) / 10, (bot_pos - st["bot"]) / 10], 1)
            st["prev"], st["top"], st["bot"] = s, top_pos, bot_pos
            with torch.no_grad():
                ob = torch.as_tensor(obs, dtype=torch.float32)
                dist = agent.dist(ob)
                act = dist.mean if greedy else dist.sample()
                if record is not None:
                    record["obs"].append(ob)
                    record["act"].append(act)
                    record["logp"].append(dist.log_prob(act).sum(1))
                    record["val"].append(agent.critic(ob)[:, 0])
            act = act.numpy().astype(float)
            d[:, -1] = -shares[:, -1]
            a[:, -1] = np.clip(size * np.exp(act[:, 1]), 0, 100)
            e[:, -1] = s + act[:, 0]
            return d, a, e
        return fn
    return make


def watch_list(pop):
    """Top and bottom 10 wallets by profit per unit of risk on 2000 past markets (public data)."""
    rank = np.argsort(-leaderboard(pop, 2000, seed=100)["profit per risk"])
    return rank[:10], rank[-10:]


def update(agent, opt, rec, r):
    """One PPO update from a rollout: rec from rl_trader, r from drop_in. Returns the mean KL."""
    rew = torch.as_tensor(r["trades"] * (r["truth"][:-1] - r["fills"]), dtype=torch.float32)  # (T, M)
    obs, act, logp, val = (torch.stack(rec[k]) for k in ("obs", "act", "logp", "val"))

    # GAE with no discounting: every episode ends at tick T, value 0 after that.
    adv, last = torch.zeros_like(rew), torch.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nxt = val[t + 1] if t + 1 < T else torch.zeros(rew.shape[1])
        last = rew[t] + nxt - val[t] + GAE_LAMBDA * last
        adv[t] = last
    ret = adv + val
    obs, act, logp, adv, ret = obs.reshape(-1, N_OBS), act.reshape(-1, N_ACT), logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)

    n, kls = len(obs), []
    for epoch in range(EPOCHS):
        for mb in torch.randperm(n).split(MINIBATCH):
            dist = agent.dist(obs[mb])
            ratio = (dist.log_prob(act[mb]).sum(1) - logp[mb]).exp()
            if not kls:  # nothing updated yet: new policy = old policy, or the data is misaligned
                assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), "rollout data misaligned"
            a_ = (adv[mb] - adv[mb].mean()) / (adv[mb].std() + 1e-8)
            pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
            v_loss = 0.5 * ((agent.critic(obs[mb])[:, 0] - ret[mb]) ** 2).mean()
            loss = pg + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD)
            opt.step()
            kls.append(((ratio - 1) - ratio.log()).mean().item())
    return np.mean(kls)


def train(pop, iters, watch, seed=0, log=print):
    torch.manual_seed(seed)
    top, bottom = watch
    agent = Agent()
    opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    hist = defaultdict(list)
    start = time.time()
    for it in range(iters):
        rec = defaultdict(list)
        p, r = drop_in(pop, rl_trader(agent, top, bottom, rec), N_TRAIN, seed=10_000 + it)
        kl = update(agent, opt, rec, r)
        hist["profit"].append(p.mean())
        hist["std"].append(agent.log_std.exp().detach().numpy().copy())
        hist["kl"].append(kl)
        if it % 10 == 0 or it == iters - 1:
            el = time.time() - start
            log(f"iter {it}/{iters}: expected profit {p.mean():+.3f}/market, noise {hist['std'][-1].round(3)}, "
                f"kl {hist['kl'][-1]:.4f}, {el / 60:.0f} min elapsed, ~{el / (it + 1) * (iters - it - 1) / 60:.0f} min left")
    return agent, {k: np.array(v) for k, v in hist.items()}


def evaluate(agent, pop, watch, n_markets=2000, seed=7):
    """Greedy policy on the same test markets as the forecaster and Kalman trader."""
    rec = defaultdict(list)
    p, r = drop_in(pop, rl_trader(agent, *watch, rec, greedy=True), n_markets, seed)
    return p, r, torch.stack(rec["obs"]).numpy(), torch.stack(rec["act"]).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "report"])
    ap.add_argument("pop", nargs="?", help="population name in runs/, e.g. pop or pop_seed1")
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()
    torch.set_num_threads(2)

    if args.mode == "train":
        name = args.pop.removesuffix(".npz")
        pop = dict(np.load(RUNS / f"{name}.npz"))
        watch = watch_list(pop)
        # Check: with its last layer zeroed, the greedy agent is exactly a forecaster.
        agent = Agent()
        with torch.no_grad():
            agent.actor[-1].weight.zero_()
            agent.actor[-1].bias.zero_()
        rl, _ = drop_in(pop, rl_trader(agent, *watch, greedy=True), 200, seed=5)
        fc, _ = kalman_trial(pop, np.inf, 200, seed=5)
        assert np.array_equal(rl, fc), "untrained agent should match the forecaster exactly"
        print("PPO checks passed", flush=True)

        agent, hist = train(pop, args.iters, watch, log=lambda s: print(s, flush=True))
        torch.save(agent.state_dict(), RUNS / f"ppo_{name}.pt")
        np.savez(RUNS / f"ppo_{name}_log.npz", top=watch[0], bottom=watch[1], **hist)
        print(f"saved runs/ppo_{name}.pt")

    else:  # report: every trained agent vs forecaster and Kalman trader on the same test markets
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logs = sorted(RUNS.glob("ppo_*_log.npz"))
        assert logs, "train an agent first"
        fig, ax = plt.subplots(1, len(logs), figsize=(5 * len(logs), 3.6), squeeze=False)
        for i, f in enumerate(logs):
            name = f.name.removeprefix("ppo_").removesuffix("_log.npz")
            pop, h = dict(np.load(RUNS / f"{name}.npz")), np.load(f)
            agent = Agent()
            agent.load_state_dict(torch.load(RUNS / f"ppo_{name}.pt"))
            p, r, obs, act = evaluate(agent, pop, (h["top"], h["bottom"]))
            fc, rf = kalman_trial(pop, np.inf, 2000, seed=7)
            ka, rk = kalman_trial(pop, 0.3, 2000, seed=7)
            print(f"== {name} (2000 test markets)")
            for label, x, rr in [("forecaster", fc, rf), ("Kalman tau 0.3", ka, rk), ("PPO", p, r)]:
                diff, risk = x - fc, rr["profit"][:, -1].std()
                print(f"   {label:15s} expected profit {x.mean():+.3f}/market (vs forecaster {diff.mean():+.3f}"
                      f" +- {diff.std() / np.sqrt(len(x)):.3f}), risk {risk:.1f}, profit/risk {x.mean() / risk:.3f}")
            # What it learned: each action regressed on the (standardised) features.
            X = obs.reshape(-1, N_OBS)
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            for j, what in enumerate(["valuation shift", "log size multiplier"]):
                y = act[..., j].reshape(-1)
                coef = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
                top3 = np.argsort(-np.abs(coef[1:]))[:3]
                print(f"   {what}: mean {y.mean():+.3f}, s.d. {y.std():.3f}; biggest drivers: "
                      + ", ".join(f"{FEATURES[k]} {coef[1 + k]:+.3f}" for k in top3))
            prof = h["profit"]
            w = min(10, len(prof))
            ax[0, i].plot(prof, color="tab:blue", alpha=0.3, label="each iteration")
            ax[0, i].plot(np.arange(w - 1, len(prof)), np.convolve(prof, np.ones(w) / w, "valid"), color="tab:blue", label="10-iteration average")
            ax[0, i].axhline(fc.mean(), color="black", ls="--", lw=1, label="forecaster (test markets)")
            ax[0, i].axhline(p.mean(), color="tab:red", ls="--", lw=1, label="PPO (test markets)")
            ax[0, i].set_title(name)
            ax[0, i].set_xlabel("PPO iteration (400 markets each)")
        ax[0, 0].set_ylabel("expected profit per market")
        ax[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "phase4_ppo.png", dpi=120)
        print("chart saved to figures/phase4_ppo.png")
