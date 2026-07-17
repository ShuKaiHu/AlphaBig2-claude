"""Evaluate an agent against the full historical opponent pool instead of a
single fixed baseline ("vs Smart") -- vs-Smart was rejected this session
because a checkpoint can overfit to beating one specific opponent without
actually playing better in general (confirmed: policy_578's vs-Smart score
kept dropping past epoch 5 even though nothing else about the training setup
suggested a real skill regression).

Pool = every distinct policy checkpoint this project has ever produced
(human-BC, self-play league, V4PBV self-play iterations, the earlier null-
result mode1 iterations). For each evaluation game, EACH of the 3 non-agent
seats independently samples a random pool member -- so a single game can have
3 different opponents at the table, maximizing opponent diversity per game
rather than per batch.

    from ppo.pool_eval import load_pool, evaluate_vs_pool
    pool = load_pool(device)
    win_rate, avg_score = evaluate_vs_pool(greedy_fn, pool, n_games=300, device=device)
"""
import os

import numpy as np
import torch

from engine.env import Big2Env

_HERE = os.path.dirname(__file__)
_PPO_SAVED = os.path.join(_HERE, "checkpoints", "saved")
_VALUE_CKPT = os.path.join(_HERE, "..", "..", "AlphaBig2-Value", "checkpoints")

# (name, path, arch) -- every distinct policy checkpoint produced by this
# project so far. "cardaware_history"/"v6" members need their own net class;
# everything else loads as plain CardAwareActorCritic.
POOL_MEMBERS = [
    ("PPO_V1", os.path.join(_PPO_SAVED, "PPO_V1.pt"), "cardaware"),
    ("PPO_V3", os.path.join(_PPO_SAVED, "PPO_V3.pt"), "cardaware"),
    ("PPO_V4", os.path.join(_PPO_SAVED, "PPO_V4.pt"), "cardaware"),
    ("PPO_V4_init", os.path.join(_PPO_SAVED, "PPO_V4_init.pt"), "cardaware"),
    ("PPO_1266", os.path.join(_PPO_SAVED, "PPO_1266.pt"), "cardaware"),
    ("PPO_V6", os.path.join(_PPO_SAVED, "PPO_V6.pt"), "v6"),
] + [
    (f"V4PBV_Iter{i}", os.path.join(_VALUE_CKPT, f"V4PBV_Iter{i}_policy.pt"), "cardaware")
    for i in range(10)
] + [
    (f"policy_mode1_iter{i}", os.path.join(_VALUE_CKPT, f"policy_mode1_iter{i}.pt"), "cardaware")
    for i in range(1, 6)
] + [
    # 2026-07-06/07: settled results of the checkpoint-selection-criterion
    # experiment (policy_all won) and the first self-play RL iteration on top
    # of it -- added to the standard pool now that they're no longer "the
    # candidate being graded" but part of this project's growing lineage, per
    # the project's own growing-pool policy (add each new model after it's
    # trained/tested). See policy-checkpoint-selection-vs-pool and
    # self-play-rl-iteration-rl1 memories for the full results.
    ("policy_all", os.path.join(_PPO_SAVED, "policy_all.pt"), "cardaware_history"),
    ("policy_all_vs1", os.path.join(_PPO_SAVED, "policy_all_vs1.pt"), "cardaware_history"),
    ("policy_all_minloss", os.path.join(_PPO_SAVED, "policy_all_minloss.pt"), "cardaware_history"),
    ("RL1", os.path.join(_HERE, "checkpoints", "ppo_RL1_best.pt"), "cardaware_history"),
]


def _build_net(arch, device):
    if arch == "cardaware":
        from ppo.network_cardaware import CardAwareActorCritic
        return CardAwareActorCritic().to(device)
    if arch == "cardaware_history":
        from ppo.network_cardaware_history import CardAwareActorCriticHistory
        return CardAwareActorCriticHistory().to(device)
    if arch == "v6":
        from ppo.network_v6 import CardAwareV6
        return CardAwareV6().to(device)
    raise ValueError(f"unknown pool arch: {arch}")


def load_pool(device, members=None):
    """Load every pool member into a ready-to-use (name, greedy_fn) list.
    Skips (with a warning) any checkpoint that fails to load instead of
    crashing the whole pool -- the pool is meant to keep growing and members
    may move/get pruned over time."""
    pool = []
    for name, path, arch in (members or POOL_MEMBERS):
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            net = _build_net(arch, device)
            net.load_state_dict(ck["model"])
            net.eval()
        except Exception as e:
            print(f"  [pool] skip {name} ({path}): {e}")
            continue
        pool.append((name, net))
    print(f"[pool] loaded {len(pool)}/{len(members or POOL_MEMBERS)} members")
    return pool


def evaluate_vs_each(agent_fn, pool, n_games: int = 100, device="cpu"):
    """Score the agent against EACH pool member individually -- that member
    alone occupies all 3 opponent seats (isolated, not mixed with others).
    For gating a checkpoint on "must beat every archetype", not just the
    aggregate vs-pool score (which a checkpoint can pass by being merely fine
    against a diluted mix while still losing badly to one specific member).
    Returns {name: (win_rate, avg_score)}."""
    from ppo.eval_baselines import evaluate
    out = {}
    for name, net in pool:
        out[name] = evaluate(agent_fn, lambda env: net.greedy(env, device), n_games)
    return out


def evaluate_vs_pool(agent_fn, pool, n_games: int = 300, device="cpu", seed=None):
    """Roll out n_games; the agent holds one seat (rotated for balance), each
    of the other 3 seats independently samples a random pool member PER GAME
    (so one game can have 3 different opponents at the table). Returns
    (win_rate, avg_score) for the agent's seat."""
    if not pool:
        raise ValueError("empty pool")
    rng = np.random.default_rng(seed)
    env = Big2Env()
    wins = 0
    scores = []
    for g in range(n_games):
        seat = (g % 4) + 1
        seat_nets = {}
        for s in range(1, 5):
            if s == seat:
                continue
            _, net = pool[rng.integers(len(pool))]
            seat_nets[s] = net
        env.reset()
        while not env.done:
            p = env.current_player
            a = agent_fn(env) if p == seat else seat_nets[p].greedy(env, device)
            rewards, done = env.step(a)
            if done:
                sc = float(rewards[seat - 1])
                scores.append(sc)
                wins += int(sc > 0)
    return wins / n_games, float(np.mean(scores)) if scores else 0.0
