"""A2 dataset: FULL-INFO value training data from REAL human games.

Positions come from real online games (in-distribution). Labels come from a
policy_4500 GREEDY rollout to terminal from each position (independent per position,
consistent with the ISMCTS greedy forward model). value = f(policy_4500, human-positions).

  python build_a2_dataset.py --games 4500 --samples 12 --out data/a2_value.npz
"""
import bootstrap  # noqa: F401  (engine+ppo path, chdir)

import argparse
import json
import os

import numpy as np
import torch
torch.set_num_threads(1)   # be a good citizen while the live online agent runs

from engine.env import Big2Env, PASS_IDX
import big2Game as B
from big2Game import handPlayed
from ppo.network_cardaware import (CardAwareActorCritic, obs_from_env,
                                   legal_action_data, _to_batch)
from ppo.parse_online_games import game_id as gid
from fullinfo_value_model import fullinfo_obs_from_env
from value_model import SCALE

PPO = "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
BUILD_SEED = 777   # pin deal-residue RNG so build_game is deterministic (see play_fixed_deals fix)

_net = CardAwareActorCritic()
_net.load_state_dict(torch.load(f"{PPO}/ppo/checkpoints/policy_4500_best.pt",
                                map_location="cpu", weights_only=False)["model"])
_net.eval()


def build_game(remaining, cardsPlayed, actor, control, trick_cards, trick_owner, passed):
    np.random.seed(BUILD_SEED)
    g = B.big2Game()
    g.currentHands = {s + 1: np.sort(np.array(remaining[s], dtype=np.int64)) for s in range(4)}
    g.cardsPlayed = cardsPlayed.copy()
    g.playersGo = actor + 1
    g.control = control
    g.mustPlayClub3 = False
    g.passedThisRound = {p: ((p - 1) in passed) for p in range(1, 5)}
    g.passCount = len(passed)
    g.lastPlayedPlayer = (trick_owner + 1) if trick_owner is not None else actor + 1
    g.gameOver = 0
    g.rewards = np.zeros(4)
    g.actionHistory = []
    g.goCounter = 0
    if control == 0 and trick_cards:
        g.goIndex = 2
        g.handsPlayed = {1: handPlayed(np.array(sorted(trick_cards), dtype=np.int64),
                                       (trick_owner or 0) + 1)}
    else:
        g.goIndex = 1
        g.handsPlayed = {}
    return g


def wrap(game):
    env = Big2Env.__new__(Big2Env)
    env._game = game
    env._done = False
    return env


@torch.no_grad()
def greedy(env):
    legal, feats = legal_action_data(env)
    if len(legal) == 0:
        return PASS_IDX
    if len(legal) == 1:
        return int(legal[0])
    ob = _to_batch([obs_from_env(env)], "cpu")
    af = torch.from_numpy(feats).unsqueeze(0)
    am = torch.ones(1, len(legal), dtype=torch.bool)
    logits = _net.forward(ob, af, am)[0][0]
    return int(legal[int(torch.argmax(logits).item())])


def greedy_rollout(env):
    steps = 0
    while not env.done and steps < 200:
        env.step(int(greedy(env))); steps += 1
    return env.game.rewards.copy()


KEYS = ["hand_ids", "trick", "seen", "opp", "counts", "passc", "opp_hands", "feat"]


def build_records(games, samples_per_game=12, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for gi, g in enumerate(games):
        plays = g["plays"]
        if not plays:
            continue
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        chosen = set(int(i) for i in rng.choice(len(plays),
                     size=min(samples_per_game, len(plays)), replace=False))
        played = {s: [] for s in range(4)}
        cp = np.zeros((4, 52), dtype=np.int64)
        tc, to, ps, consec = None, None, set(), 0
        for t, ev in enumerate(plays):
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            forced = act == "pass" and s in ps
            if t in chosen and not forced:
                rem = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
                if len(rem[s]) > 0:
                    control = 1 if tc is None else 0
                    obs = fullinfo_obs_from_env(wrap(build_game(rem, cp, s, control, tc, to, ps)))
                    rw = greedy_rollout(wrap(build_game(rem, cp, s, control, tc, to, ps)))
                    recs.append({"obs": obs, "target": float(np.tanh(rw[s] / SCALE)),
                                 "raw": float(rw[s]), "gid": gi})
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cp[s][c - 1] = 1
                tc, to = cards, s; ps.discard(s); consec = 0
            else:
                ps.add(s); consec += 1
                if consec >= 3:
                    tc, to, ps, consec = None, None, set(), 0
    return recs


def save_records(recs, path):
    arrs = {k: np.stack([r["obs"][k] for r in recs]) for k in KEYS}
    arrs["target"] = np.array([r["target"] for r in recs], dtype=np.float32)
    arrs["raw"] = np.array([r["raw"] for r in recs], dtype=np.float32)
    arrs["gid"] = np.array([r["gid"] for r in recs], dtype=np.int64)   # for leakage-free split
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_games(max_games=None, exclude_heldout=True):
    held = set(json.load(open(f"{PPO}/ppo/data/scale_heldout_ids.json"))) if exclude_heldout else set()
    out = []
    for l in open(f"{PPO}/ppo/data/online_games.jsonl"):
        g = json.loads(l)
        if (g.get("id") or gid(g)) in held:
            continue
        out.append(g)
        if max_games and len(out) >= max_games:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", default="data/a2_value.npz")
    ap.add_argument("--check", action="store_true", help="determinism self-check then exit")
    args = ap.parse_args()

    if args.check:
        g = load_games(max_games=1)[0]
        r1 = build_records([g], samples_per_game=4, seed=0)
        r2 = build_records([g], samples_per_game=4, seed=0)
        same = all(abs(a["raw"] - b["raw"]) < 1e-9 for a, b in zip(r1, r2))
        print(f"determinism self-check: {'PASS' if same else 'FAIL'}  ({len(r1)} recs)")
        for r in r1[:4]:
            print(f"  raw={r['raw']:+.0f} target={r['target']:+.3f} "
                  f"opp_hands sum={r['obs']['opp_hands'].sum():.0f} feat={np.round(r['obs']['feat'],2)}")
        return

    games = load_games(max_games=args.games)
    recs = build_records(games, samples_per_game=args.samples, seed=0)
    raw = np.array([r["raw"] for r in recs])
    print(f"A2 records: {len(recs)} from {len(games)} games")
    print(f"raw score: mean={raw.mean():+.2f} std={raw.std():.2f} "
          f"min={raw.min():+.0f} max={raw.max():+.0f} | <=-20 佔 {100*(raw<=-20).mean():.1f}%")
    save_records(recs, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
