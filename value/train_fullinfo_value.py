"""Train the FULL-INFO value net on A2 data (human positions, greedy-rollout labels).

Split train/val BY GAME (gid) so positions from one game never straddle the split
(the belief-pipeline leakage lesson). Reports val MSE / R^2 / Pearson, and an A/B
vs the current public VALUE_minplays on the SAME val positions -- the whole point is
whether seeing the determinized hands makes the leaf sharper.

  python train_fullinfo_value.py --data data/a2_value.npz --epochs 40 --tag FULLINFO_a2
"""
import bootstrap  # noqa: F401

import argparse
import numpy as np
import torch
import torch.nn as nn

from fullinfo_value_model import FullInfoValueNet
from value_model import ValueNet, SCALE

DEV = "cpu"
VAL = "/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"   # bootstrap chdirs away; use abs paths
PUB_KEYS = ["hand_ids", "trick", "seen", "opp", "counts", "passc"]


def make_batch(data, idx, device, full=True):
    b = {}
    b["hand_ids"] = torch.tensor(data["hand_ids"][idx], dtype=torch.long, device=device)
    for k in ("trick", "seen", "opp", "counts", "passc"):
        b[k] = torch.tensor(data[k][idx], dtype=torch.float32, device=device)
    if full:
        b["opp_hands"] = torch.tensor(data["opp_hands"][idx], dtype=torch.float32, device=device)
        b["feat"] = torch.tensor(data["feat"][idx], dtype=torch.float32, device=device)
    else:                                          # public baseline: only OUR min_plays
        b["feat"] = torch.tensor(data["feat"][idx][:, :1], dtype=torch.float32, device=device)
    return b


def metrics(pred, tgt):
    pred, tgt = np.asarray(pred), np.asarray(tgt)
    mse = float(np.mean((pred - tgt) ** 2))
    ss = float(np.sum((tgt - tgt.mean()) ** 2))
    r2 = 1.0 - float(np.sum((pred - tgt) ** 2)) / ss if ss > 0 else 0.0
    corr = float(np.corrcoef(pred, tgt)[0, 1]) if pred.std() > 0 else 0.0
    return mse, r2, corr


@torch.no_grad()
def predict(net, data, idx, device, full=True, bs=4096):
    net.eval(); out = []
    for i in range(0, len(idx), bs):
        out.append(net(make_batch(data, idx[i:i + bs], device, full)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/a2_value.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", default="FULLINFO_a2")
    ap.add_argument("--scale", type=float, default=40.0, help="B: de-saturation scale (>13 = less tanh squish)")
    ap.add_argument("--clip", type=float, default=64.0, help="A: clip pathological greedy-tail |raw|>clip")
    ap.add_argument("--public", action="store_true", help="ablation: train PUBLIC net (no opp_hands) on same A2 data")
    args = ap.parse_args()

    data = np.load(args.data)
    raw = data["raw"]
    # A (clip pathological tail) + B (de-saturate: larger tanh scale) computed from raw
    tgt = np.tanh(np.clip(raw, -args.clip, args.clip) / args.scale).astype(np.float32)
    n = len(tgt)
    print(f"target: tanh(clip(raw,±{args.clip:.0f})/{args.scale:.0f})  "
          f"| raw std={raw.std():.1f}  clipped={100*(np.abs(raw)>args.clip).mean():.1f}%")
    gids = data["gid"]
    uniq = np.array(sorted(set(gids.tolist())))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    n_val_g = max(1, int(len(uniq) * args.val_frac))
    val_g = set(uniq[:n_val_g].tolist())
    val_idx = np.array([i for i in range(n) if gids[i] in val_g])
    tr_idx = np.array([i for i in range(n) if gids[i] not in val_g])
    print(f"records={n}  games={len(uniq)}  train={len(tr_idx)}  val={len(val_idx)}")

    FULL = not args.public
    net = (ValueNet(extra_dim=1) if args.public else FullInfoValueNet()).to(DEV)
    print(f"training {'PUBLIC (ablation, no opp_hands)' if args.public else 'FULL-INFO'} net")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()
    tgt_t = torch.tensor(tgt, dtype=torch.float32, device=DEV)

    best = {"r2": -1e9}
    for ep in range(args.epochs):
        net.train(); perm = tr_idx[np.random.permutation(len(tr_idx))]
        for i in range(0, len(perm), args.batch):
            bi = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(net(make_batch(data, bi, DEV, FULL)), tgt_t[bi])
            loss.backward(); opt.step()
        sched.step()
        vp = predict(net, data, val_idx, DEV, FULL)
        mse, r2, corr = metrics(vp, tgt[val_idx])
        tag = ""
        if r2 > best["r2"]:
            best = {"r2": r2, "mse": mse, "corr": corr, "ep": ep,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}}
            tag = " *"
        if ep % 2 == 0 or tag:
            print(f"ep{ep:2d}  val MSE={mse:.4f}  R2={r2:.3f}  corr={corr:.3f}{tag}", flush=True)

    # save best + diagnose (train-val gap) + corr-vs-RAW (transform-invariant -> fair vs public)
    net.load_state_dict(best["state"])

    def corr_raw(pred, idx):
        r = raw[idx]
        return float(np.corrcoef(pred, r)[0, 1]) if pred.std() > 0 else 0.0

    fi_corr_raw = corr_raw(predict(net, data, val_idx, DEV, FULL), val_idx)
    tsub = tr_idx if len(tr_idx) < 20000 else tr_idx[np.random.permutation(len(tr_idx))[:20000]]
    _, tr_r2, _ = metrics(predict(net, data, tsub, DEV, FULL), tgt[tsub])
    ckpt = f"{VAL}/checkpoints/{args.tag}.pt"
    torch.save({"model": best["state"], "arch": "fullinfo_value_v1",
                "val_r2": best["r2"], "val_corr": best["corr"], "corr_raw": fi_corr_raw,
                "scale": args.scale, "clip": args.clip}, ckpt)
    print(f"\nbest full-info: val R2={best['r2']:.3f}(own target)  corr_vs_raw={fi_corr_raw:.3f} (ep{best['ep']}) -> {ckpt}")
    print(f"  過擬合gap(train-val R2)={tr_r2 - best['r2']:.3f} "
          f"{'-> 資料不足,需更多' if tr_r2 - best['r2'] > 0.15 else '-> 資料量OK'}")

    # ---- A/B vs public VALUE_minplays: corr-vs-RAW is fair (transform-invariant, same val positions) ----
    try:
        pub = ValueNet(extra_dim=1).to(DEV)
        pub.load_state_dict(torch.load(f"{VAL}/checkpoints/VALUE_minplays.pt", map_location=DEV)["model"])
        pub_corr_raw = corr_raw(predict(pub, data, val_idx, DEV, full=False), val_idx)
        d = fi_corr_raw - pub_corr_raw
        verdict = "full-info 更利 ✅" if d > 0.02 else ("差不多" if d > -0.02 else "未勝 ⚠️")
        print(f"\n=== A/B on same {len(val_idx)} val positions (corr vs RAW score, 公平) ===")
        print(f"  public VALUE_minplays : corr_vs_raw={pub_corr_raw:.3f}")
        print(f"  FULL-INFO (this)      : corr_vs_raw={fi_corr_raw:.3f}")
        print(f"  -> {verdict}  (Δ={d:+.3f})")
    except Exception as e:
        print(f"(public A/B skipped: {e})")


if __name__ == "__main__":
    main()
