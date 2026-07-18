"""A/B: does adding hand-strength features (dominance probs + min-plays) make the
public-info VALUE net more accurate?

For each self-play position we compute 4 extra features from PUBLIC info only:
  [P(single nuts), P(pair nuts), P(fh nuts)]  ← the trained dominance model
  [min_plays_to_empty / 13]                   ← greedy fewest-plays-to-go-out
and feed them into the value net's encoder. We train the SAME data WITH and WITHOUT
these features and compare val MSE / Pearson r / sign-accuracy. (All features are
inference-computable, so no train/test leak.)

    python train_value_aug.py --max-games 50000 --epochs 30
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records
from dominance_model import DominanceNet
from hand_features import min_plays_to_empty
from ppo.network_cardaware import _to_batch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = bootstrap.CKPT_DIR


@torch.no_grad()
def add_features(recs, dom_net):
    for s in range(0, len(recs), 512):
        chunk = recs[s:s + 512]
        dp = torch.sigmoid(dom_net(_to_batch([r["obs"] for r in chunk], DEV))).cpu().numpy()
        for i, r in enumerate(chunk):
            hand = [int(c) for c in r["obs"]["hand_ids"] if c > 0]
            mp = min_plays_to_empty(hand) / 13.0
            r["feat"] = np.array([dp[i, 0], dp[i, 1], dp[i, 2], mp], dtype=np.float32)


def evaluate(net, recs, ii):
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for s in range(0, len(ii), 512):
            b = build_value_batch([recs[i] for i in ii[s:s + 512]], DEV)
            P += net(b["obs"]).cpu().numpy().tolist(); Y += b["target"].cpu().numpy().tolist()
    net.train()
    P, Y = np.array(P), np.array(Y)
    return (float(np.mean((P - Y) ** 2)), float(np.var(Y)),
            float(np.corrcoef(P, Y)[0, 1]), float(np.mean((P > 0) == (Y > 0))))


def train(recs, tr_i, val_i, extra_dim, epochs, lr, wd, batch, tag):
    net = ValueNet(extra_dim=extra_dim).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best = 1e9; best_state = None
    for ep in range(1, epochs + 1):
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), batch):
            b = build_value_batch([recs[i] for i in tr_i[s:s + batch]], DEV)
            loss = F.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
        mse, base, r, sign = evaluate(net, recs, val_i)
        if mse < best:
            best = mse; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    mse, base, r, sign = evaluate(net, recs, val_i)
    print(f"  [{tag}] best val MSE {mse:.4f} (base {base:.4f}) | r {r:+.3f} | sign {sign*100:.1f}%")
    torch.save({"model": net.state_dict(), "arch": "value", "extra_dim": extra_dim},
               os.path.join(CKPT_DIR, f"value_{tag}.pt"))
    return mse, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    recs = build_value_records(samples_per_game=1, max_games=args.max_games, seed=args.seed)
    dom = DominanceNet().to(DEV)
    dom.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dominance_best.pt"), map_location=DEV)["model"])
    dom.eval()
    print("computing dominance + min-plays features ..."); add_features(recs, dom)

    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device {DEV}\n=== A/B ===")
    m0, r0 = train(recs, tr_i, val_i, 0, args.epochs, args.lr, args.wd, args.batch, "plain")
    m1, r1 = train(recs, tr_i, val_i, 4, args.epochs, args.lr, args.wd, args.batch, "aug")
    print(f"\nfeatures effect: MSE {m0:.4f} -> {m1:.4f} ({100*(m0-m1)/m0:+.1f}%) | "
          f"r {r0:+.3f} -> {r1:+.3f}")


if __name__ == "__main__":
    main()
