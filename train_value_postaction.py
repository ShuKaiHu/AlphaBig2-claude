"""Train the POST-ACTION value net (single-variable change vs VALUE_minplays.pt).

Same everything as the deployed VALUE_minplays (ValueNet extra_dim=1 = min-plays/13,
tanh(score/13) MSE target, AlphaGo 1-position-per-game decorrelation, same 200k
self-play corpus) EXCEPT the snapshot timing: build_value_records(post_action=True)
snapshots AFTER the acting player's move, so the value learns "given I JUST played
this and my hand is now H', what's my expected final score" — the in-distribution
target for scoring my own candidate actions / the ISMCTS post-action leaf.

    AB2PPO=... /path/to/ppo/.venv/bin/python3 train_value_postaction.py --epochs 30
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records
from hand_features import min_plays_to_empty


def add_minplays_feat(recs):
    """extra_dim=1: min_plays_to_empty(hand)/13 computed on the obs hand (which,
    for post_action records, is the POST-play hand H')."""
    for r in recs:
        hand = [int(c) for c in r["obs"]["hand_ids"] if c > 0]
        r["feat"] = np.array([min_plays_to_empty(hand) / 13.0], dtype=np.float32)


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
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--data", default=None, help="self-play jsonl path (default: value_dataset.DATA)")
    ap.add_argument("--post-action", action="store_true",
                    help="post-action snapshot (default OFF = pre-action, matches deployed VALUE_minplays)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="VALUE_postaction")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from value_dataset import DATA as _DEFAULT_DATA
    recs = build_value_records(path=(args.data or _DEFAULT_DATA), samples_per_game=1, seed=args.seed,
                               max_games=args.max_games, post_action=args.post_action)
    if len(recs) < 50:
        print("too few records"); return
    print("computing min-plays feature (extra_dim=1) on POST-action hands ...")
    add_minplays_feat(recs)

    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * args.val_frac))
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device={device}")

    net = ValueNet(extra_dim=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"params: {sum(p.numel() for p in net.parameters()):,} | extra_dim=1 (min-plays)")

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
        ck = {"model": net.state_dict(), "arch": "value", "extra_dim": 1, "epoch": ep,
              "val_mse": mse, "baseline_mse": base, "pearson": r, "sign_acc": sign,
              "post_action": args.post_action}
        torch.save(ck, os.path.join(bootstrap.CKPT_DIR, f"{args.tag}_latest.pt"))
        if mse < best:
            best = mse; since_best = 0
            torch.save(ck, os.path.join(bootstrap.CKPT_DIR, f"{args.tag}.pt"))
            print(f"   * new best val mse {mse:.4f} (r {r:+.3f}, sign {sign*100:.1f}%) -> {args.tag}.pt")
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"early stop @ ep {ep} (no val improvement for {args.patience}); best mse {best:.4f}")
                break


if __name__ == "__main__":
    main()
