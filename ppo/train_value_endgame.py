"""Train Big2ValueNet on SOLVED endgame labels (best-play backward induction).

The key test: Monte-Carlo labels (one noisy game outcome copied to every state)
overfit at 578 games — val never beat predict-the-mean. These labels are instead
EXACT best-play values from terminal backward induction (ppo/endgame_value.py),
and each position is independently sampled+solved, so a plain random split is
leak-free. If clean computed labels are learnable, val MSE should fall well
BELOW the predict-the-mean baseline (unlike MC).

  python -m ppo.train_value_endgame --data ppo/data/endgame_solved_k10.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from engine.model import Big2ValueNet

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
REWARD_SCALE = 13.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__),
                                                   "data", "endgame_solved_k10.npz"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="value_endgame")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = np.load(args.data)
    static, opp, target = d["static"], d["opp"], d["target"]
    n = len(static)
    idx = np.random.permutation(n)
    nval = int(n * args.val_frac)
    vi, ti = idx[:nval], idx[nval:]
    print(f"{n} solved positions | train {len(ti)} / val {len(vi)}")

    S = torch.from_numpy(static); O = torch.from_numpy(opp); Y = torch.from_numpy(target)

    base = ((Y[vi] - Y[ti].mean(0)) ** 2).mean().item()
    print(f"baseline val MSE (predict-train-mean) = {base:.4f}")

    model = Big2ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.MSELoss()

    def val_eval():
        model.eval()
        with torch.no_grad():
            p = model(S[vi], O[vi])
            mse = lossf(p, Y[vi]).item()
            ps = torch.atanh(p.clamp(-.999, .999)) * REWARD_SCALE
            ys = torch.atanh(Y[vi].clamp(-.999, .999)) * REWARD_SCALE
            mae = (ps - ys).abs().mean().item()
        model.train(); return mse, mae

    best, best_state, since = 1e9, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); np.random.shuffle(ti); tot = 0.0
        for i in range(0, len(ti), args.batch):
            b = ti[i:i + args.batch]
            p = model(S[b], O[b]); loss = lossf(p, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        tr = tot / len(ti); vmse, vmae = val_eval()
        imp = vmse < best - 1e-5
        if imp:
            best, since = vmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 10 == 0 or imp:
            print(f"[ep {ep:3d}] train {tr:.4f}  val {vmse:.4f}  val_score_MAE {vmae:.2f}"
                  f"{'  *best' if imp else ''}")
        if since >= args.patience:
            print(f"early stop @ {ep}"); break

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    torch.save({"value_state": best_state, "val_mse": best, "baseline_mse": base,
                "scale": REWARD_SCALE, "tag": args.tag}, path)
    verdict = "✓ BEATS baseline" if best < base else "✗ still >= baseline"
    print(f"saved -> {path}\nbest val MSE {best:.4f} vs baseline {base:.4f}  → {verdict}")


if __name__ == "__main__":
    main()
