"""Rollout-ground-truth comparison of RL1 vs RL2's decision QUALITY, independent
of the noisy 100-game online sample and independent of the vs-pool local score.

Walks self-play (driven by RL2, since those are the positions we most care about
validating), and at every decision point where RL1's and RL2's own greedy choices
DISAGREE, rolls out BOTH candidate actions to terminal (same continuation policy
for both -- RL2 greedy, for every seat) in the SAME single self-play world (no
belief/determinization needed: self-play means all hands are already known, so a
single rollout per candidate is an exact, noise-free comparison for that specific
world -- see generate_bomb_data.py's docstring for the same reasoning).

Reports: of the N disagreement points, how often RL2's own pick actually scores
higher by rollout outcome vs RL1's pick, and vice versa. This is the direct test
of "is RL2 making objectively better decisions than RL1" -- separate from whether
that shows up in a noisy 100-game online sample.

    ./.venv/bin/python compare_rl1_rl2_decisions.py --n-decisions 60
"""
import argparse

import numpy as np
import torch

from engine.env import Big2Env
from ppo.network_cardaware_history import CardAwareActorCriticHistory

SCALE = 13.0


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = CardAwareActorCriticHistory().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


@torch.no_grad()
def greedy(net, env, device):
    return net.greedy(env, device)


def rollout(net, env, me, device, max_steps=200):
    steps = 0
    while not env.done and steps < max_steps:
        env.step(greedy(net, env, device))
        steps += 1
    if not env.done:
        return 0.0
    return float(np.tanh(env.game.rewards[me - 1] / SCALE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl1", default="ppo/checkpoints/ppo_RL1_best.pt")
    ap.add_argument("--rl2", default="ppo/checkpoints/ppo_RL2_best.pt")
    ap.add_argument("--n-decisions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    rl1 = load(args.rl1, device)
    rl2 = load(args.rl2, device)

    disagreements = []
    env = Big2Env(); env.reset()
    checked = 0
    while len(disagreements) < args.n_decisions:
        me = env.current_player
        a1 = greedy(rl1, env, device)
        a2 = greedy(rl2, env, device)
        checked += 1
        if a1 != a2:
            e1 = env.clone(); e1.step(a1); r1 = rollout(rl2, e1, me, device)
            e2 = env.clone(); e2.step(a2); r2 = rollout(rl2, e2, me, device)
            disagreements.append({"a1": a1, "a2": a2, "r1": r1, "r2": r2,
                                   "rl2_better": r2 > r1, "rl1_better": r1 > r2, "tie": r1 == r2})
            print(f"  disagreement {len(disagreements)}/{args.n_decisions} "
                  f"(checked {checked} decisions so far) | RL1->{r1:+.2f}  RL2->{r2:+.2f}"
                  f"{'  RL2 better' if r2 > r1 else ('  RL1 better' if r1 > r2 else '  tie')}")
        env.step(greedy(rl2, env, device))
        if env.done:
            env = Big2Env(); env.reset()

    n = len(disagreements)
    rl2_wins = sum(d["rl2_better"] for d in disagreements)
    rl1_wins = sum(d["rl1_better"] for d in disagreements)
    ties = sum(d["tie"] for d in disagreements)
    disagree_rate = n / checked
    print(f"\n=== RL1 vs RL2 rollout-ground-truth comparison ===")
    print(f"decisions checked: {checked}  |  disagreement rate: {disagree_rate*100:.1f}%")
    print(f"of {n} disagreements: RL2 better {rl2_wins} ({100*rl2_wins/n:.1f}%)  "
          f"RL1 better {rl1_wins} ({100*rl1_wins/n:.1f}%)  tie {ties} ({100*ties/n:.1f}%)")
    avg_gap = np.mean([d["r2"] - d["r1"] for d in disagreements])
    print(f"average (RL2 - RL1) rollout outcome on disagreements: {avg_gap:+.3f}")


if __name__ == "__main__":
    main()
