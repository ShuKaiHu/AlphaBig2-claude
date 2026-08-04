"""The decisive question the first trace missed: does val MSE keep improving
AFTER the hand attention path dies (plain: probe R2 +0.322 @ep1 -> +0.002 @ep5)?

If yes, the net routes around the hand path (bag/counts carry the score) and the
whole anti-collapse line is capped -- which is exactly what the A/B measured
(+1.3% for C despite probe R2 0 -> 0.658).

RESULT (2026-08-05, ~25 min run): yes — plain's best val MSE lands at ep27,
long after the hand path dies at ep5 (probe R2 +0.324@ep1 -> +0.005@ep5, W_q -> 0
by ep12); the score is carried by the bag/count paths. C stays healthy all 30 ep.
⚠️ CAVEAT (adversarially verified 2026-08-05): the mp-R2 probe here is heavily
SIZE-confounded (corr(mp, hand size)=0.87 on this distribution; size-onehot alone
scores 0.796) — "learned then forgot" is about size-bearing decodability, not
combo structure. For construct-valid mp reads use ppo/probe_mp_from_handvec.py
(mp | n_cards) on main, or control size explicitly.

Traces BOTH arms on the SAME fixed split, logging per-epoch val MSE alongside
the collapse diagnostics. Same hyperparams as train_value_collapse_ab.py.
"""


import bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet, build_value_batch, D
from value_model_v2 import ValueNetV2
from value_dataset import build_value_records
from train_value_collapse_ab import collapse_probe, evaluate

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NHEAD, HDIM = 4, D // 4


@torch.no_grad()
def entropy_of(net, ids, pad):
    if isinstance(net, ValueNetV2):
        h = net.hand_emb(ids)
        h = net.tok_ln(h) if net.tok_ln is not None else h
    else:
        h = net.card_emb(ids)
    W, b = net.hand_attn.in_proj_weight, net.hand_attn.in_proj_bias
    B, L, _ = h.shape
    q = (h @ W[:D].T + b[:D]).view(B, L, NHEAD, HDIM).transpose(1, 2)
    k = (h @ W[D:2*D].T + b[D:2*D]).view(B, L, NHEAD, HDIM).transpose(1, 2)
    logit = (q @ k.transpose(-1, -2)) / (HDIM ** 0.5)
    kmask = (~pad)[:, None, None, :]
    big = torch.where(kmask, logit, torch.tensor(-1e30, device=logit.device))
    p = torch.softmax(big, -1)
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)                  # (B,H,L)
    nval = (~pad).float().sum(1).clamp(min=2).log()[:, None, None]
    w = (~pad).float()[:, None, :].expand_as(ent)                  # FIX: expand over heads
    return float((((ent / nval) * w).sum() / w.sum()))


def trace(tag, factory, recs, tr_i, val_i, eids, epad):
    torch.manual_seed(0)
    net = factory().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-4)
    print(f"\n=== {tag} ===")
    print(f"  {'ep':>3s} {'valMSE':>8s} {'best?':>6s} {'探針R²':>8s} {'attn熵':>8s} "
          f"{'‖W_q‖':>7s} {'emb‖row‖':>9s}")
    best, best_ep = 1e9, -1
    hist = []
    for ep in range(1, 31):
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            b = build_value_batch([recs[i] for i in tr_i[s:s + 256]], DEV)
            loss = F.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
        mse, _, _, _ = evaluate(net, recs, val_i)
        mark = ""
        if mse < best:
            best, best_ep = mse, ep; mark = "  ★"
        r2 = collapse_probe(net, n_hands=4000, seed=1234)["probe_minplays_r2"]
        net.eval(); ent = entropy_of(net, eids, epad); net.train()
        W = net.hand_attn.in_proj_weight.detach()
        emb = net.hand_emb if isinstance(net, ValueNetV2) else net.card_emb
        er = float(emb.weight[1:].detach().norm(dim=1).mean())
        hist.append((ep, mse, r2))
        print(f"  {ep:3d} {mse:8.4f} {mark:>6s} {r2:+8.3f} {ent:8.4f} "
              f"{float(W[:D].norm()):7.3f} {er:9.3f}", flush=True)
    print(f"  → best val MSE {best:.4f} @ ep{best_ep}")
    return hist, best, best_ep


def main():
    torch.manual_seed(0); np.random.seed(0)
    recs = build_value_records(samples_per_game=1, max_games=200000, seed=0)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nv = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nv], idx[nv:]

    rng = np.random.RandomState(1234)
    eids = np.zeros((2000, 13), dtype=np.int64)
    for i in range(2000):
        s = rng.randint(5, 14)
        eids[i, :s] = rng.choice(np.arange(1, 53), size=s, replace=False)
    eids = torch.from_numpy(eids).to(DEV); epad = eids == 0

    print(f"device {DEV} | train {len(tr_i)} / val {nv}  (熵 1.0000 = 完全均勻 = 塌)")
    hp, bp, bep = trace("plain", lambda: ValueNet(extra_dim=0),
                        recs, tr_i.copy(), val_i, eids, epad)
    hc, bc, bec = trace("C (pre_ln + bag_mean)",
                        lambda: ValueNetV2(pre_ln=True, bag_mean=True),
                        recs, tr_i.copy(), val_i, eids, epad)

    print("\n=== 關鍵讀數 ===")
    m1 = dict((e, m) for e, m, _ in hp)
    print(f"  plain: 手牌路死亡前(ep4)val MSE {m1[4]:.4f} → 終點最佳 {bp:.4f} @ep{bep}"
          f"  ⇒ 死後仍改善 {100*(m1[4]-bp)/m1[4]:+.1f}%")
    print(f"  C    : best val MSE {bc:.4f} @ep{bec}")
    print(f"  C vs plain: {100*(bp-bc)/bp:+.2f}%")


if __name__ == "__main__":
    main()
