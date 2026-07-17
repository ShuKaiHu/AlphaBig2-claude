"""Ablation: which feature earns the value-net accuracy gain — the dominance probs,
the min-plays-to-empty, or both?

Same 200k records, SAME train/val split, same hyperparams. We compute the full 4-dim
feature once, then train four value nets on feature subsets and compare val MSE / r:
  plain      []            (extra_dim 0, baseline)
  dom        [P single/pair/fh nuts]   (extra_dim 3)
  minplays   [min_plays/13]            (extra_dim 1)
  both       all four                  (extra_dim 4)

    python train_value_ablation.py --max-games 200000 --epochs 30
"""
import bootstrap  # noqa: F401

import argparse

import numpy as np
import torch

from value_dataset import build_value_records
from dominance_model import DominanceNet
from train_value_aug import add_features, train, DEV, CKPT_DIR
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    recs = build_value_records(samples_per_game=1, max_games=args.max_games, seed=args.seed)
    dom = DominanceNet().to(DEV)
    dom.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dominance_best.pt"), map_location=DEV)["model"])
    dom.eval()
    print("computing dominance + min-plays features ..."); add_features(recs, dom)
    for r in recs:
        r["feat_full"] = r["feat"]

    idx = np.arange(len(recs)); np.random.shuffle(idx)          # ONE fixed split for all variants
    nval = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device {DEV}\n=== ablation ===")

    variants = [("plain", []), ("dom", [0, 1, 2]), ("minplays", [3]), ("both", [0, 1, 2, 3])]
    results = {}
    for name, cols in variants:
        for r in recs:                                          # set this variant's feature slice
            if cols:
                r["feat"] = r["feat_full"][cols]
            else:
                r.pop("feat", None)
        torch.manual_seed(args.seed)                            # same init RNG start per variant
        m, rr = train(recs, tr_i.copy(), val_i, len(cols), args.epochs, args.lr, args.wd, args.batch, f"abl_{name}")
        results[name] = (m, rr)
        for r in recs:                                          # restore full feat for next variant
            r["feat"] = r["feat_full"]

    base = results["plain"][0]
    print("\n=== ablation summary (val) ===")
    print(f"  {'variant':10s} {'MSE':>8s} {'dMSE%':>8s} {'r':>8s}")
    for name, _ in variants:
        m, rr = results[name]
        print(f"  {name:10s} {m:8.4f} {100*(base-m)/base:+8.1f} {rr:+8.3f}")


if __name__ == "__main__":
    main()
