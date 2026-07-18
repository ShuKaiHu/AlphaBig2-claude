"""Train the dominance-probability model (showdown-oracle supervised).

Reports, per label (single/pair/full-house nuts), on a held-out split:
  - positive rate (base rate)
  - accuracy @0.5
  - Brier score  (mean (p - y)^2; lower better) vs the base-rate Brier (predict the
    mean) — beating it = the model learned to discriminate
  - calibration: actual positive rate within each predicted-probability band
    (well-calibrated → predicted ≈ actual → it's a REAL probability)

    python train_dominance.py --max-games 30000 --epochs 25
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from dominance_model import DominanceNet, build_dom_batch, NOUT
from dominance_dataset import build_dominance_records
from dominance_oracle import LABELS

CKPT_DIR = bootstrap.CKPT_DIR


@torch.no_grad()
def evaluate(net, recs, ii, device):
    net.eval(); P, Y = [], []
    for s in range(0, len(ii), 512):
        b = build_dom_batch([recs[i] for i in ii[s:s + 512]], device)
        P.append(torch.sigmoid(net(b["obs"])).cpu().numpy()); Y.append(b["target"].cpu().numpy())
    net.train()
    return np.concatenate(P), np.concatenate(Y)   # (N,NOUT)


def report(P, Y):
    for j, name in enumerate(LABELS):
        p, y = P[:, j], Y[:, j]
        base = y.mean()
        brier = np.mean((p - y) ** 2); brier_base = np.mean((base - y) ** 2)
        acc = np.mean((p > 0.5) == (y > 0.5))
        print(f"  {name:<12} base {base*100:4.1f}% | acc {acc*100:4.1f}% | "
              f"Brier {brier:.4f} (base {brier_base:.4f})")
    # calibration for the full-house label (the user's example)
    j = LABELS.index("fh_nuts"); p, y = P[:, j], Y[:, j]
    print("  fh_nuts calibration  pred-band -> actual%:", end=" ")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        m = (p >= lo) & (p < lo + 0.2)
        print(f"[{lo:.1f}-{lo+0.2:.1f}]={100*y[m].mean():.0f}%(n{m.sum()})" if m.sum() else f"[{lo:.1f}]–", end=" ")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("--samples-per-game", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="dominance")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    recs = build_dominance_records(samples_per_game=args.samples_per_game, max_games=args.max_games, seed=args.seed)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * args.val_frac)); val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device {device}")

    net = DominanceNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            b = build_dom_batch([recs[i] for i in tr_i[s:s + args.batch]], device)
            loss = F.binary_cross_entropy_with_logits(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        P, Y = evaluate(net, recs, val_i, device)
        brier = float(np.mean((P - Y) ** 2))
        print(f"ep {ep:3d} | train bce {tot/nb:.4f} | val Brier {brier:.4f}")
        if ep == args.epochs or brier < best:
            best = min(best, brier)
            torch.save({"model": net.state_dict(), "arch": "dominance", "labels": LABELS, "epoch": ep},
                       os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
    print("\n=== final (val) ===")
    report(*evaluate(net, recs, val_i, device))


if __name__ == "__main__":
    main()
