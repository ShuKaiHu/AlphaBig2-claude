"""Train the value net on the AlphaGo-style (position, final-score) dataset (step 3).

Loss = MSE on tanh(score/SCALE). Reports on a held-out split:
  - val MSE            (lower better)
  - baseline MSE       = variance of targets (what predicting the mean would score);
                         val MSE well below baseline = the net learned something
  - Pearson r          between prediction and true outcome
  - sign accuracy      = does it call win-vs-lose direction right

    python train_value.py --epochs 30
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records


def evaluate(net, recs, ii, device):
    net.eval(); preds, tgts = [], []
    with torch.no_grad():
        for s in range(0, len(ii), 512):
            mb = [recs[i] for i in ii[s:s + 512]]
            b = build_value_batch(mb, device)
            preds += net(b["obs"]).cpu().numpy().tolist()
            tgts += b["target"].cpu().numpy().tolist()
    net.train()
    preds, tgts = np.array(preds), np.array(tgts)
    mse = float(np.mean((preds - tgts) ** 2))
    base = float(np.var(tgts))
    r = float(np.corrcoef(preds, tgts)[0, 1]) if len(preds) > 1 else 0.0
    sign = float(np.mean((preds > 0) == (tgts > 0)))
    return mse, base, r, sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    ap.add_argument("--patience", type=int, default=0, help="early stop after N epochs w/o val improvement (0=off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--samples-per-game", type=int, default=1, help="AlphaGo uses 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="value")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    recs = build_value_records(samples_per_game=args.samples_per_game, seed=args.seed)
    if len(recs) < 50:
        print("too few records — generate more self-play games first (selfplay.py)."); return
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * args.val_frac))
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device={device}")

    net = ValueNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"params: {sum(p.numel() for p in net.parameters()):,} | wd={args.weight_decay} | patience={args.patience}")

    best = 1e9; since_best = 0
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = build_value_batch(mb, device)
            loss = F.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        mse, base, r, sign = evaluate(net, recs, val_i, device)
        print(f"ep {ep:3d} | train mse {tot/nb:.4f} | val mse {mse:.4f} "
              f"(baseline {base:.4f}) | r {r:+.3f} | sign {sign*100:.1f}%")
        ck = {"model": net.state_dict(), "arch": "value", "epoch": ep,
              "val_mse": mse, "baseline_mse": base, "pearson": r, "sign_acc": sign}
        torch.save(ck, os.path.join(bootstrap.CKPT_DIR, f"{args.tag}_latest.pt"))
        if mse < best:
            best = mse; since_best = 0
            torch.save(ck, os.path.join(bootstrap.CKPT_DIR, f"{args.tag}_best.pt"))
            print(f"   * new best val mse {mse:.4f} (r {r:+.3f}, sign {sign*100:.1f}%) -> {args.tag}_best.pt")
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"early stop @ ep {ep} (no val improvement for {args.patience} epochs); best mse {best:.4f}")
                break


if __name__ == "__main__":
    main()
