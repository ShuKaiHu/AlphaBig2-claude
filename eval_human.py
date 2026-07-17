"""Distribution-shift test: does the value net (trained on PPO_V4 self-play)
predict the final score in REAL human games?

Evaluates checkpoints/VALUE.pt on the accumulated human online games
(AlphaBig2-ppo/ppo/data/online_games.jsonl, same schema as self-play). Reports
overall + by game phase, at AlphaGo 1-position/game (apples-to-apples with the
self-play val metric) and at all-positions (more samples). Note: the value net
assumes a PPO_V4 continuation, while humans play the real continuation — so this
is a genuine out-of-distribution check, not same-distribution held-out.

    python eval_human.py
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch

from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records

HUMAN = os.path.join(bootstrap.AB2PPO, "ppo", "data", "online_games.jsonl")
PHASES = [("early <13", 0, 13), ("mid 13-25", 13, 26), ("late 26-38", 26, 39), ("end 39+", 39, 99)]


def _stat(p, t):
    if len(p) < 2:
        return None
    return (len(p), float(np.mean((p - t) ** 2)), float(np.var(t)),
            float(np.corrcoef(p, t)[0, 1]), float(np.mean((p > 0) == (t > 0))))


def evaluate(net, recs, device):
    preds, tgts, played = [], [], []
    for s in range(0, len(recs), 512):
        mb = recs[s:s + 512]
        b = build_value_batch(mb, device)
        with torch.no_grad():
            preds += net(b["obs"]).cpu().numpy().tolist()
        tgts += b["target"].cpu().numpy().tolist()
        played += [int(round(float(r["obs"]["seen"].sum()))) for r in mb]
    preds, tgts, played = np.array(preds), np.array(tgts), np.array(played)
    rows = [("ALL", _stat(preds, tgts))]
    for name, lo, hi in PHASES:
        m = (played >= lo) & (played < hi)
        rows.append((name, _stat(preds[m], tgts[m])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(bootstrap.CKPT_DIR, "VALUE.pt"))
    ap.add_argument("--data", default=HUMAN)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = ValueNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    net.eval()
    print(f"value model: {os.path.basename(args.ckpt)}  |  test data: {os.path.basename(args.data)}")

    for label, spg in [("1 position / game (AlphaGo-style, comparable to val)", 1),
                       ("ALL positions / game (more samples)", 200)]:
        recs = build_value_records(path=args.data, samples_per_game=spg, seed=args.seed, verbose=False)
        print(f"\n--- {label}  [{len(recs)} samples] ---")
        print(f"{'phase':<12}{'n':>7}{'MSE':>9}{'baseline':>10}{'r':>8}{'sign%':>8}")
        for name, st in evaluate(net, recs, device):
            if st is None:
                print(f"{name:<12}{'--':>7}"); continue
            n, mse, base, r, sign = st
            print(f"{name:<12}{n:>7}{mse:>9.4f}{base:>10.4f}{r:>+8.3f}{sign*100:>7.1f}%")


if __name__ == "__main__":
    main()
