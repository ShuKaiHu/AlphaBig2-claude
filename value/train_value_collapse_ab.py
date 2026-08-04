"""Anti-collapse encoder A/B for the public-info value net (single-variable arms).

Same 200k records, SAME fixed train/val split, same hyperparams for every arm
(mirrors train_value_ablation.py). Arms:

  plain    ValueNet (deployed architecture, all levers off)      — baseline
  featref  ValueNet(extra_dim=1) + min_plays/13 input feature    — REFERENCE arm
           only: reproduces the VALUE_minplays recipe inside this harness so
           BCD has a same-split yardstick. The feature stays an INPUT here and
           is never used as a loss/auxiliary target anywhere (user decision
           2026-08-04: min_plays is a greedy heuristic, not a dominating factor;
           supervising toward it would bias the representation).
  B        ValueNetV2(split_emb)                                 — split tables
  C        ValueNetV2(pre_ln, bag_mean)                          — norm-decouple
  D        ValueNetV2(residual, own_count)                       — carry + count
  BCD      ValueNetV2(all of the above)

After each arm trains, the harness runs the COLLAPSE PROBE on the trained net
(20k random hands, sizes U{1..13}): closed-form ridge from hand_vec to
min_plays and to hand size (val R2), plus hand-emb mean row norm and hand_vec
variance — the exact diagnostics that exposed the collapse on value_best.pt
(probe R2 0.000, norms 0.38, variance 24x down vs init).

=== 事前註冊判讀規則(寫死,不許看到結果後修改)===
主指標 = val MSE(30 epochs 內 best);次指標 sign-acc、Pearson r、探針 R2。
1. 塌縮判定:某臂 探針R2(hand_vec→min_plays) >= 0.30 且 hand-emb 平均 row
   norm > 1.0 → 該臂記「塌縮已解」。plain 預期 ~0(復現病灶)。
2. 架構效果:BCD 的 val MSE 比 plain 低 >= 0.5%(相對)→「有效」;
   ±0.5% 內 →「null」;更差 →「有害」。
3. 對照手寫特徵:BCD 的 val MSE <= featref × 1.01 →「自學表徵已可取代手寫
   min_plays 特徵」;否則手寫特徵保留、BCD 只作為機制結論記錄。
4. 單臂 B/C/D 的相對排序只做描述性記錄(單 seed,不做採用決策)。
5. 任一裁決落在邊界 ±0.5% 內 → 先跑 --seed 1 復現方向,同向才裁決。

    AB2PPO=... python3 train_value_collapse_ab.py --max-games 200000 --epochs 30
"""
import bootstrap  # noqa: F401

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet, build_value_batch
from value_model_v2 import ValueNetV2
from value_dataset import build_value_records
from hand_features import min_plays_to_empty

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = bootstrap.CKPT_DIR

ARMS = {
    # name    -> (factory, needs_feat)
    "plain":   (lambda: ValueNet(extra_dim=0), False),
    "featref": (lambda: ValueNet(extra_dim=1), True),
    "B":       (lambda: ValueNetV2(split_emb=True), False),
    "C":       (lambda: ValueNetV2(pre_ln=True, bag_mean=True), False),
    "D":       (lambda: ValueNetV2(residual=True, own_count=True), False),
    "BCD":     (lambda: ValueNetV2(split_emb=True, pre_ln=True, bag_mean=True,
                                   residual=True, own_count=True), False),
}


def add_minplays_feat(recs):
    for r in recs:
        hand = [int(c) for c in r["obs"]["hand_ids"] if c > 0]
        r["feat"] = np.array([min_plays_to_empty(hand) / 13.0], dtype=np.float32)


# ── collapse probe (same protocol as the 2026-08 diagnosis of value_best.pt) ──
def _hand_vec_of(net, hand_ids):
    """(N,13) long -> (N,D) on DEV, for either architecture."""
    if isinstance(net, ValueNetV2):
        return net.hand_vec(hand_ids)
    pad = hand_ids == 0                                   # baseline ValueNet path
    h = net.card_emb(hand_ids)
    attn, _ = net.hand_attn(h, h, h, key_padding_mask=pad)
    keep = (~pad).float().unsqueeze(-1)
    return (attn * keep).sum(1) / keep.sum(1).clamp(min=1.0)


def _ridge_r2(X, y, lam=1e-2, val_frac=0.1, seed=1234):
    rng = np.random.RandomState(seed)
    ii = rng.permutation(len(X)); nv = int(len(X) * val_frac)
    vi, ti = ii[:nv], ii[nv:]
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-8
    Xt, Xv = (X[ti] - mu) / sd, (X[vi] - mu) / sd
    Xt = np.hstack([Xt, np.ones((len(Xt), 1))]); Xv = np.hstack([Xv, np.ones((len(Xv), 1))])
    reg = lam * np.eye(Xt.shape[1]); reg[-1, -1] = 0.0    # unregularized bias
    w = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ y[ti])
    resid = y[vi] - Xv @ w
    return 1.0 - float(np.var(resid)) / float(np.var(y[vi]))


@torch.no_grad()
def collapse_probe(net, n_hands=20000, seed=1234):
    rng = np.random.RandomState(seed)
    ids = np.zeros((n_hands, 13), dtype=np.int64)
    mp = np.zeros(n_hands, dtype=np.float32)
    size = np.zeros(n_hands, dtype=np.float32)
    for i in range(n_hands):
        s = rng.randint(1, 14)
        cards = rng.choice(np.arange(1, 53), size=s, replace=False)
        ids[i, :s] = cards
        mp[i] = min_plays_to_empty([int(c) for c in cards])
        size[i] = s
    net.eval()
    hv = []
    for a in range(0, n_hands, 1024):
        t = torch.from_numpy(ids[a:a + 1024]).to(DEV)
        hv.append(_hand_vec_of(net, t).cpu().numpy())
    net.train()
    hv = np.concatenate(hv)
    emb = net.hand_emb if isinstance(net, ValueNetV2) else net.card_emb
    return {
        "probe_minplays_r2": _ridge_r2(hv, mp),
        "probe_size_r2": _ridge_r2(hv, size),
        "hand_emb_mean_norm": float(np.linalg.norm(
            emb.weight[1:].detach().cpu().numpy(), axis=1).mean()),
        "hand_vec_var": float(hv.var(axis=0).sum()),
    }


# ── train / eval (mirrors train_value_aug.py) ────────────────────────────────
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


def train_arm(name, recs, tr_i, val_i, args):
    factory, needs_feat = ARMS[name]
    for r in recs:                                        # arm-appropriate feat slice
        if needs_feat:
            r["feat"] = r["feat_mp"]
        else:
            r.pop("feat", None)
    torch.manual_seed(args.seed)                          # same init RNG start per arm
    net = factory().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    nparam = sum(p.numel() for p in net.parameters())
    best = 1e9; best_state = None
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), args.batch):
            b = build_value_batch([recs[i] for i in tr_i[s:s + args.batch]], DEV)
            loss = F.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
        mse, base, r, sign = evaluate(net, recs, val_i)
        if mse < best:
            best = mse; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    mse, base, r, sign = evaluate(net, recs, val_i)
    pr = collapse_probe(net, n_hands=args.probe_hands, seed=1234)
    print(f"  [{name}] params {nparam:,} | best val MSE {mse:.4f} (base {base:.4f}) "
          f"| r {r:+.3f} | sign {sign*100:.1f}% | probe mp R2 {pr['probe_minplays_r2']:+.3f} "
          f"size R2 {pr['probe_size_r2']:+.3f} | emb-norm {pr['hand_emb_mean_norm']:.2f} "
          f"| hv-var {pr['hand_vec_var']:.3f}", flush=True)
    torch.save({"model": net.state_dict(), "arch": f"value_ab_{name}", "val_mse": mse,
                "pearson": r, "sign_acc": sign, **pr},
               os.path.join(CKPT_DIR, f"value_ab_{name}.pt"))
    return {"mse": mse, "r": r, "sign": sign, **pr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-hands", type=int, default=20000)
    ap.add_argument("--arms", default="plain,featref,B,C,D,BCD",
                    help="comma list from: " + ",".join(ARMS))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        assert a in ARMS, f"unknown arm {a}"

    recs = build_value_records(samples_per_game=1, max_games=args.max_games, seed=args.seed)
    print("computing min-plays feature (featref arm only) ...")
    add_minplays_feat(recs)
    for r in recs:
        r["feat_mp"] = r["feat"]

    idx = np.arange(len(recs)); np.random.shuffle(idx)    # ONE fixed split for all arms
    nval = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device {DEV}\n=== anti-collapse A/B ===")

    results = {a: train_arm(a, recs, tr_i.copy(), val_i, args) for a in arms}

    print("\n=== summary (val) ===")
    base = results.get("plain", {}).get("mse")
    print(f"  {'arm':8s} {'MSE':>8s} {'dMSE%':>7s} {'r':>7s} {'sign%':>6s} "
          f"{'mpR2':>7s} {'szR2':>7s} {'norm':>6s} {'hvVar':>7s}")
    for a in arms:
        m = results[a]
        d = f"{100*(base-m['mse'])/base:+6.1f}" if base else "   n/a"
        print(f"  {a:8s} {m['mse']:8.4f} {d:>7s} {m['r']:+7.3f} {m['sign']*100:6.1f} "
              f"{m['probe_minplays_r2']:+7.3f} {m['probe_size_r2']:+7.3f} "
              f"{m['hand_emb_mean_norm']:6.2f} {m['hand_vec_var']:7.3f}")
    print("\n判讀按檔頭事前規則;邊界 ±0.5% 內先跑 --seed 1 復現方向。")


if __name__ == "__main__":
    main()
