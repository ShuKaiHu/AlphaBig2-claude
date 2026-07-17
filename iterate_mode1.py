"""Mode 1 — AlphaGo-style VALUE iteration against a FIXED reference (V4).

Loop (belief FROZEN, policy = V4 fixed; only the value net iterates):
  1. self-play: all 4 seats play the ISMCTS (PPO_V4 prior + BELIEF determinize +
     value-depth lookahead leaf using the CURRENT value). Record the games.
  2. retrain VALUE on the accumulated ISMCTS self-play games — AlphaGo style: one
     random position per game, label = that player's final score (tanh). Warm-start
     from the current value + weight decay.
  3. track strength: margin of ISMCTS(new value) vs plain V4 (the fixed yardstick).
  4. repeat — the better value should make the search a little better, whose games
     give better value targets, etc.

Honest expectation (from local diagnostics): the ISMCTS overrides V4 only ~5% of
moves, so the self-play games barely change between iterations → improvement is
likely SLOW/weak. This framework lets us measure whether it moves at all. Knobs to
make it bite: higher --sims / --cpuct (search deviates more), bigger --games-per-iter.

    python iterate_mode1.py --iters 5 --games-per-iter 80 --sims 30
"""
import bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
import torch

import eval_alpha1 as E                      # reuse loaded nets + ISMCTS agent + engine
from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records

DEV = E.DEV
MODE1_DATA = os.path.join(bootstrap.DATA_DIR, "mode1_selfplay.jsonl")
CKPT_DIR = bootstrap.CKPT_DIR

# fix the search config to the agreed scenario: value-depth-5 leaf + belief on
E._LEAF, E._VALUE_DEPTH, E._USE_BELIEF = "value", 5, True


def play_and_record(sims, cpuct):
    """One all-4-ISMCTS self-play game → dict (selfplay schema for value_dataset)."""
    env = E.Big2Env(); env.reset()
    hands = {str(s): [int(c) for c in env.game.currentHands[s + 1]] for s in range(4)}
    plays, rewards, steps = [], None, 0
    while not env.done and steps < 300:
        p = env.current_player
        before = set(int(c) for c in env.game.currentHands[p])
        a = E.alpha1_action(env, p, sims, cpuct)
        rewards, _ = env.step(a)
        after = set(int(c) for c in env.game.currentHands[p])
        plays.append({"seat": p - 1, "action": "pass" if a == E.PASS_IDX else "play",
                      "cards": sorted(before - after)})
        steps += 1
    scores = {str(s): int(rewards[s]) for s in range(4)}
    winner = next(s for s in range(4) if env.game.currentHands[s + 1].size == 0)
    return {"hands": hands, "plays": plays, "scores": scores, "winner": winner, "source": "mode1"}


def selfplay(n_games, sims, cpuct):
    t0 = time.time()
    with open(MODE1_DATA, "a") as f:
        for i in range(n_games):
            f.write(json.dumps(play_and_record(sims, cpuct)) + "\n")
            if (i + 1) % 20 == 0:
                f.flush()
                print(f"    self-play {i+1}/{n_games} ({(time.time()-t0)/(i+1):.1f}s/game)")
    total = sum(1 for _ in open(MODE1_DATA))
    print(f"    self-play done: +{n_games} → {total} games in {MODE1_DATA}")


def retrain_value(warm_net, epochs, lr=1e-3, wd=3e-4, batch=256):
    recs = build_value_records(path=MODE1_DATA, samples_per_game=1, verbose=False)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nval], idx[nval:]
    net = ValueNet().to(DEV)
    if warm_net is not None:
        net.load_state_dict(warm_net.state_dict())          # warm-start from current value
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), batch):
            b = build_value_batch([recs[i] for i in tr_i[s:s + batch]], DEV)
            loss = torch.nn.functional.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        mse, base = [], []
        for s in range(0, len(val_i), 512):
            b = build_value_batch([recs[i] for i in val_i[s:s + 512]], DEV)
            p = net(b["obs"]).cpu().numpy(); t = b["target"].cpu().numpy()
            mse.append(((p - t) ** 2).sum()); base.append(((t - t.mean()) ** 2).sum())
    return net, float(np.sum(mse) / len(val_i)), float(np.sum(base) / len(val_i)), len(recs)


def margin_vs_v4(sims, cpuct, games):
    """ISMCTS(current E._value) vs plain V4, rotating seat. Returns margin/game."""
    a = []
    for i in range(games):
        seat = (i % 4) + 1
        r = E.play_game(seat, sims, cpuct)
        a.append(float(r[seat - 1]))
    return float(np.mean(a)) * 4.0 / 3.0      # zero-sum: margin = (4/3)*mean(alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--games-per-iter", type=int, default=80)
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--gate-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Mode 1 value iteration | leaf=value vdepth=5 belief=on | sims={args.sims} | "
          f"{args.iters} iters × {args.games_per_iter} games")

    for it in range(1, args.iters + 1):
        print(f"\n=== iteration {it}/{args.iters} ===")
        selfplay(args.games_per_iter, args.sims, args.cpuct)
        new, mse, base, n = retrain_value(E._value, args.epochs)
        E._value = new                                       # adopt (no gate; track instead)
        torch.save({"model": new.state_dict(), "arch": "value", "iter": it,
                    "val_mse": mse}, os.path.join(CKPT_DIR, f"value_mode1_iter{it}.pt"))
        m = margin_vs_v4(args.sims, args.cpuct, args.gate_games)
        print(f"  iter {it}: value val_mse {mse:.4f} (base {base:.4f}, {n} games) | "
              f"ISMCTS−V4 margin {m:+.2f}/game ({args.gate_games} games)")


if __name__ == "__main__":
    main()
