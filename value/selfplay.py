"""Self-play game generation with the human-trained policy (AlphaGo step 1).

All four seats are played by the human-data policy, sampling from its softmax (not
greedy — greedy would replay one identical game forever) so we get a diverse corpus
of human-like games. Each game is saved in the SAME schema as the online human games
(hands + plays + scores), APPENDING to data/selfplay_games.jsonl so repeated runs
accumulate a growing, durable corpus.

    python selfplay.py --games 20000 --policy PPO_V4.pt
"""
import bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from engine.env import Big2Env, PASS_IDX
from ppo.network_cardaware import CardAwareActorCritic, obs_from_env, legal_action_data, _to_batch
from ppo.action_features import ACTION_FEATURES

OUT = os.path.join(bootstrap.DATA_DIR, "selfplay_games.jsonl")
SAVED = os.path.join(bootstrap.AB2PPO, "ppo", "checkpoints", "saved")


def load_policy(spec):
    path = spec if os.path.isabs(spec) or os.path.exists(spec) else os.path.join(SAVED, spec)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "cardaware")
    if arch == "v6":
        from ppo.network_v6 import CardAwareV6
        net = CardAwareV6()
    else:
        net = CardAwareActorCritic()
    net.load_state_dict(ck["model"])
    net.eval()
    return net, arch, os.path.basename(path)


@torch.no_grad()
def sample_action(net, env, temp):
    legal_idx, feats = legal_action_data(env)
    ob = _to_batch([obs_from_env(env)], "cpu")
    af = torch.from_numpy(feats).unsqueeze(0)
    am = torch.ones(1, len(legal_idx), dtype=torch.bool)
    logits = net.forward(ob, af, am)[0][0]
    probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)
    a_idx = int(torch.multinomial(probs, 1).item())
    return int(legal_idx[a_idx])


def play_game(net, temp):
    env = Big2Env(); env.reset()
    hands = {str(s): [int(c) for c in env.game.currentHands[s + 1]] for s in range(4)}
    plays, rewards = [], None
    while not env.done:
        acting = env.current_player                       # 1-indexed
        before = set(int(c) for c in env.game.currentHands[acting])
        a = sample_action(net, env, temp)
        rewards, _ = env.step(a)
        after = set(int(c) for c in env.game.currentHands[acting])
        cards = sorted(before - after)                    # cards played (empty = pass)
        plays.append({"seat": acting - 1,
                      "action": "pass" if a == PASS_IDX else "play",
                      "cards": cards})
    scores = {str(s): int(rewards[s]) for s in range(4)}
    winner = next(s for s in range(4) if env.game.currentHands[s + 1].size == 0)
    gid = hashlib.md5(json.dumps([hands, plays], sort_keys=True).encode()).hexdigest()[:16]
    return {"id": gid, "source": "selfplay", "hands": hands, "plays": plays,
            "scores": scores, "winner": winner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--policy", default="PPO_V4.pt", help="self-play policy (saved/ name or path)")
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temperature (diversity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT, help="output jsonl (default appends to selfplay_games.jsonl); "
                                               "set a separate file for data-scaling runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net, arch, name = load_policy(args.policy)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(bootstrap.DATA_DIR, args.out)
    print(f"self-play policy: {name} (arch={arch}) | temp={args.temp} | games={args.games}")
    print(f"writing to {out_path}")

    t0 = time.time(); n = 0
    with open(out_path, "a") as f:
        for i in range(1, args.games + 1):
            g = play_game(net, args.temp)
            g["policy"] = name
            f.write(json.dumps(g) + "\n"); n += 1
            if i % 500 == 0:
                f.flush()
                rate = i / (time.time() - t0)
                print(f"  {i}/{args.games}  ({rate:.1f} games/s, ETA {(args.games-i)/rate/60:.1f}m)")
    print(f"done: wrote {n} games in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
