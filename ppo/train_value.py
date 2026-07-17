"""Train the V10 full-info value net on REAL human showdown games.

This replaces heuristic self-play (V9b–V9e, which could never teach human-like
play) with the actual human-game distribution. At each decision we have all four
hands (showdown), so we feed the SAME V9 god-view inputs (encode_static +
encode_opp_hands) and regress toward the game's FINAL per-player score (Monte
Carlo, tanh-normalized). Architecture = engine.model.Big2ValueNet, unchanged —
only the data source changes.

At deployment the hidden opponent hands are imputed by the human-trained belief
model (ppo/checkpoints/belief_best.pt), so this value is evaluated on
belief-sampled worlds rather than the uniform-random determinization V9 used.

Split is GAME-LEVEL (whole games held out) because all states within a game
share the same target — a random split would leak the outcome. 578 games is
small, so we val-split + early-stop hard against overfitting.

Usage:
  python -m ppo.train_value --epochs 200 --batch 256 --lr 1e-3 --val-frac 0.12
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ppo.bc_dataset import build_value_records
from engine.model import Big2ValueNet

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
REWARD_SCALE = 13.0


def _split_by_game(records, val_frac, seed):
    games = sorted({r["game"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val])
    train = [r for r in records if r["game"] not in val_games]
    val = [r for r in records if r["game"] in val_games]
    return train, val


def _batch(records, device):
    static = torch.from_numpy(np.stack([r["static"] for r in records])).to(device)
    opp = torch.from_numpy(np.stack([r["opp"] for r in records])).to(device)
    target = torch.from_numpy(np.stack([r["target"] for r in records])).to(device)
    return static, opp, target


@torch.no_grad()
def _evaluate(model, val, device):
    model.eval()
    static, opp, target = _batch(val, device)
    pred = model(static, opp)
    mse = nn.functional.mse_loss(pred, target).item()
    # MAE back in raw score space (un-tanh), self/acting-agnostic average
    pred_s = torch.atanh(pred.clamp(-0.999, 0.999)) * REWARD_SCALE
    tgt_s = torch.atanh(target.clamp(-0.999, 0.999)) * REWARD_SCALE
    mae_score = (pred_s - tgt_s).abs().mean().item()
    return mse, mae_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="value_human")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    records = build_value_records(scale=REWARD_SCALE)
    train, val = _split_by_game(records, args.val_frac, args.seed)
    n_tr_games = len({r["game"] for r in train})
    n_va_games = len({r["game"] for r in val})
    print(f"train: {len(train)} states / {n_tr_games} games | "
          f"val: {len(val)} states / {n_va_games} games")

    model = Big2ValueNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # baseline: predict the mean target (sanity floor for val MSE)
    mean_tgt = np.stack([r["target"] for r in train]).mean(axis=0)
    base_mse = float(((np.stack([r["target"] for r in val]) - mean_tgt) ** 2).mean())
    print(f"baseline val MSE (predict-train-mean) = {base_mse:.4f}")

    best_val, best_state, since = float("inf"), None, 0
    os.makedirs(CKPT_DIR, exist_ok=True)
    idx = np.arange(len(train))
    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            bi = [train[j] for j in idx[i:i + args.batch]]
            static, opp, target = _batch(bi, device)
            pred = model(static, opp)
            loss = loss_fn(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        tr_mse = tot / len(train)
        val_mse, val_mae = _evaluate(model, val, device)
        improved = val_mse < best_val - 1e-5
        if improved:
            best_val, since = val_mse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 5 == 0 or improved:
            print(f"[ep {ep:3d}] train_mse={tr_mse:.4f}  val_mse={val_mse:.4f}"
                  f"  val_score_MAE={val_mae:.2f}{'  *best' if improved else ''}")
        if since >= args.patience:
            print(f"early stop @ ep {ep} (no val improvement for {args.patience})")
            break

    ck = {"value_state": best_state, "val_mse": best_val,
          "baseline_mse": base_mse, "tag": args.tag, "scale": REWARD_SCALE}
    path = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    torch.save(ck, path)
    print(f"saved -> {path}  (best val MSE {best_val:.4f} vs baseline {base_mse:.4f})")


if __name__ == "__main__":
    main()
