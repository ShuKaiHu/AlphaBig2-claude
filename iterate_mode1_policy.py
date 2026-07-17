"""Mode 1 — POLICY iteration with a ROLLOUT leaf (no value net).

Drop the value net; the ISMCTS leaf = greedy rollout to terminal. The thing that
iterates is now the POLICY (it is both the PUCT prior and the rollout policy).
AlphaZero-style policy improvement:
  1. self-play: all 4 seats = ISMCTS(rollout leaf, current policy, belief). At each
     decision record (public obs, normalized MCTS visit distribution π).
  2. retrain the POLICY toward π (soft cross-entropy), warm-start the current policy.
     ~50 targets/game (vs 1/game for value) → far more signal, far less data-starved
     than the value iteration.
  3. track strength: ISMCTS(rollout, new policy) vs plain V4.
  4. repeat — better policy → better rollouts + prior → better search → better π.

belief FROZEN throughout. Note: trained only on self-play visit-dists (no human
anchor) → optimises "beat V4", may drift from human-like play (fine for this Mode-1
experiment; online-vs-humans is a separate test).

    python iterate_mode1_policy.py --iters 5 --games-per-iter 80 --sims 30
"""
import bootstrap  # noqa: F401

import argparse
import os
import time

import numpy as np
import torch

import eval_alpha1 as E
from ppo.network_cardaware import CardAwareActorCritic
from ppo.action_features import ACTION_FEATURES, AF

DEV = E.DEV
CKPT_DIR = bootstrap.CKPT_DIR
E._LEAF, E._USE_BELIEF = "rollout", True          # rollout leaf → iterate the POLICY

# FROZEN V4 for the gate's opponents (E._policy evolves, so it can't be the yardstick).
_V4 = CardAwareActorCritic().to(DEV)
_V4.load_state_dict(torch.load(f"{E.SAVED}/PPO_V4.pt", map_location=DEV)["model"]); _V4.eval()


@torch.no_grad()
def _v4_greedy(env):
    legal, feats = E.legal_action_data(env)
    ob = E._to_batch([E.obs_from_env(env)], DEV)
    af = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    am = torch.ones(1, len(legal), dtype=torch.bool, device=DEV)
    logits, _ = _V4.forward(ob, af, am)
    return int(legal[int(logits[0].argmax())])


def selfplay_collect(n_games, sims, cpuct):
    """All-4 ISMCTS(rollout) self-play → list of (obs, legal, π) policy targets."""
    data, t0 = [], time.time()
    for g in range(n_games):
        env = E.Big2Env(); env.reset(); steps = 0
        while not env.done and steps < 300:
            p = env.current_player
            obs = E.obs_from_env(env)
            legal, _ = E.legal_action_data(env)
            best, visits = E.alpha1_action(env, p, sims, cpuct, return_visits=True)
            tot = sum(visits.values()) if visits else 0
            if len(legal) > 1 and tot > 0:
                pi = np.array([visits.get(int(a), 0) / tot for a in legal], dtype=np.float32)
                data.append((obs, np.asarray(legal), pi))
            env.step(best); steps += 1
        if (g + 1) % 20 == 0:
            print(f"    self-play {g+1}/{n_games} ({(time.time()-t0)/(g+1):.1f}s/game, {len(data)} targets)")
    return data


def build_policy_batch(records):
    obs = E._to_batch([r[0] for r in records], DEV)
    Lmax = max(len(r[1]) for r in records); B = len(records)
    feats = np.zeros((B, Lmax, AF), dtype=np.float32)
    mask = np.zeros((B, Lmax), dtype=bool)
    pi = np.zeros((B, Lmax), dtype=np.float32)
    for i, (_, legal, p) in enumerate(records):
        L = len(legal); feats[i, :L] = ACTION_FEATURES[legal]; mask[i, :L] = True; pi[i, :L] = p
    return (obs, torch.from_numpy(feats).to(DEV), torch.from_numpy(mask).to(DEV),
            torch.from_numpy(pi).to(DEV))


def train_policy(buffer, warm, epochs, lr=5e-4, wd=1e-4, batch=256):
    net = CardAwareActorCritic().to(DEV); net.load_state_dict(warm.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    idx = np.arange(len(buffer))
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), batch):
            obs, feats, mask, pi = build_policy_batch([buffer[i] for i in idx[s:s + batch]])
            logits, _ = net.forward(obs, feats, mask)
            loss = -(pi * torch.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def margin_vs_v4(sims, cpuct, games):
    """ISMCTS(current E._policy) at one seat vs FROZEN V4 at the other three."""
    a = []
    for i in range(games):
        seat = (i % 4) + 1
        env = E.Big2Env(); env.reset(); steps = 0
        while not env.done and steps < 300:
            p = env.current_player
            act = E.alpha1_action(env, p, sims, cpuct) if p == seat else _v4_greedy(env)
            env.step(act); steps += 1
        a.append(float(env.game.rewards[seat - 1]))
    return float(np.mean(a)) * 4.0 / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--games-per-iter", type=int, default=80)
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--buffer-iters", type=int, default=3, help="keep this many iters of targets")
    ap.add_argument("--gate-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Mode 1 POLICY iteration (rollout leaf, belief on) | sims={args.sims} | "
          f"{args.iters} iters × {args.games_per_iter} games")
    print(f"baseline V4 margin (sanity): {margin_vs_v4(args.sims, args.cpuct, args.gate_games):+.2f}")

    buffer = []
    for it in range(1, args.iters + 1):
        print(f"\n=== iteration {it}/{args.iters} ===")
        data = selfplay_collect(args.games_per_iter, args.sims, args.cpuct)
        buffer += data
        cap = args.buffer_iters * args.games_per_iter * 55       # ~55 targets/game
        buffer = buffer[-cap:]
        new = train_policy(buffer, E._policy, args.epochs)
        E._policy = new                                          # adopt
        torch.save({"model": new.state_dict(), "arch": "cardaware", "iter": it},
                   os.path.join(CKPT_DIR, f"policy_mode1_iter{it}.pt"))
        m = margin_vs_v4(args.sims, args.cpuct, args.gate_games)
        print(f"  iter {it}: {len(data)} targets ({len(buffer)} in buffer) | "
              f"ISMCTS(rollout,policy_{it})−V4 margin {m:+.2f}/game ({args.gate_games} games)")


if __name__ == "__main__":
    main()
