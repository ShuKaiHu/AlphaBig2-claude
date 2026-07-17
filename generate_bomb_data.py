"""Generate a bomb-decision training set via self-play, for fine-tuning a
CardAwareActorCriticHistory checkpoint on quad/straight-flush timing.

Unlike the earlier diagnostic (AlphaBig2-Value/test_bomb_timing.py), which used
belief-sampled multi-world averaging (needed there because it evaluates the
DEPLOYED policy under real imperfect information), this is self-play: the true
hands are dealt by us, so ground truth is free -- no belief, no world sampling.
At each decision where a legal quad/straight-flush exists, we roll out EVERY
legal action from the one true state to the end of the game (policy-greedy for
all players, including our own future turns) and record which action scored
best. That becomes the training LABEL (replacing "the human's chosen action"
used in normal BC) for a small, targeted fine-tune -- see finetune_bomb.py.

    ./.venv/bin/python generate_bomb_data.py --checkpoint ppo/checkpoints/saved/policy_1266.pt --n 400 --out ppo/data/bomb_decisions_1266.jsonl
"""
import argparse
import json
import time

import numpy as np
import torch

import enumerateOptions
import gameLogic
from engine.env import Big2Env, PASS_IDX
from ppo.network_cardaware_history import CardAwareActorCriticHistory, live_history_tensor
from ppo.network_cardaware import legal_action_data, obs_from_env, _to_batch

SCALE = 13.0


def _is_quad_or_sf(cards) -> bool:
    if cards is None or len(cards) != 5:
        return False
    arr = np.array(cards)
    return bool(gameLogic.isFourOfAKind(arr) or gameLogic.isStraightFlush(arr))


@torch.no_grad()
def _greedy_action(net, env, device):
    obs = _to_batch([obs_from_env(env)], device)
    hist = live_history_tensor(env, device)
    legal_idx, feats = legal_action_data(env)
    af = torch.from_numpy(feats).unsqueeze(0).to(device)
    am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
    logits, _ = net.forward(obs, hist, af, am)
    return int(legal_idx[int(torch.argmax(logits[0]).item())])


def _rollout(net, env, me, device, max_steps=200):
    """Continue the ONE true world to terminal, greedy policy for everyone
    (including our own future turns) -- no belief/world-sampling needed since
    this is self-play (true hands already known)."""
    steps = 0
    while not env.done and steps < max_steps:
        env.step(_greedy_action(net, env, device))
        steps += 1
    return float(np.tanh(env.game.rewards[me - 1] / SCALE))


def find_bomb_decisions(net, n_target, device, seed=0, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    env = Big2Env(); env.reset()
    t0 = time.time()
    while len(results) < n_target:
        me = env.current_player
        legal_idx, feats = legal_action_data(env)
        bomb_idx = []
        for i, a in enumerate(legal_idx):
            a = int(a)
            if a == PASS_IDX:
                continue
            cids, n = enumerateOptions.getOptionNC(a)
            if n == 5 and _is_quad_or_sf(cids):
                bomb_idx.append(i)
        if bomb_idx:
            obs = obs_from_env(env)
            history = live_history_tensor(env, "cpu")[0].numpy()  # snapshot BEFORE trying any action
            roll_avg = {}
            for a in legal_idx:
                a = int(a)
                e = env.clone()
                e.step(a)
                roll_avg[a] = _rollout(net, e, me, device)
            roll_top = max(roll_avg, key=roll_avg.get)
            bomb_actions = set(int(legal_idx[i]) for i in bomb_idx)
            pol_top = _greedy_action(net, env, device)
            results.append({
                "obs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs.items()},
                "history": history.tolist(),
                "legal": [int(a) for a in legal_idx],
                "roll_avg": {int(a): v for a, v in roll_avg.items()},
                "roll_top": roll_top,
                "bomb_actions": sorted(bomb_actions),
                "pol_top": pol_top,
                "bomb_is_roll_optimal": roll_top in bomb_actions,
                "policy_played_bomb": pol_top in bomb_actions,
            })
            if verbose and len(results) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  bomb decision {len(results)}/{n_target}  ({elapsed:.0f}s elapsed, "
                      f"~{elapsed/len(results):.1f}s/decision)")
        env.step(_greedy_action(net, env, device))
        if env.done:
            env = Big2Env(); env.reset()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = CardAwareActorCriticHistory().to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    t0 = time.time()
    results = find_bomb_decisions(net, args.n, device, seed=args.seed)
    print(f"\ngenerated {len(results)} bomb decisions in {time.time()-t0:.0f}s")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")

    bomb_optimal = sum(1 for r in results if r["bomb_is_roll_optimal"])
    policy_bombed = sum(1 for r in results if r["policy_played_bomb"])
    agree = sum(1 for r in results if r["bomb_is_roll_optimal"] == r["policy_played_bomb"])
    print(f"rollout says bomb optimal: {bomb_optimal}/{len(results)} ({100*bomb_optimal/len(results):.1f}%)")
    print(f"policy played bomb: {policy_bombed}/{len(results)} ({100*policy_bombed/len(results):.1f}%)")
    print(f"policy-rollout agreement: {agree}/{len(results)} ({100*agree/len(results):.1f}%)")


if __name__ == "__main__":
    main()
