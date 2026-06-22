"""Train the dedicated opponent-hand belief model on human showdown data.

Pure supervised belief prediction (no policy). Reports, on a held-out val set:
  - P@count overall: rank unseen cards by P(opp holds), take top-(their count),
    fraction actually theirs.
  - P@count by GAME PHASE (#cards already played): early deals are ~random; the
    model should be strong mid/late, which is where belief matters for decisions.
  - count-prior baseline: best you can do knowing ONLY the counts (≈ chance).
    Beating it proves the model reads play patterns, not just counts.

    ./.venv/bin/python -m ppo.train_belief --target all --epochs 40
"""
import argparse
import os

import numpy as np
import torch

from ppo.belief_model import BeliefNet, belief_batch, belief_bce
from ppo.bc_dataset import build_records

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PHASES = [("early <13", 0, 13), ("mid 13-25", 13, 26), ("late 26-38", 26, 39), ("end 39+", 39, 99)]


@torch.no_grad()
def metrics(net, recs, ii, device):
    net.eval()
    ps = pn = base = bn = bce = 0.0
    ph = {p[0]: [0.0, 0] for p in PHASES}
    for s in range(0, len(ii), 512):
        mb = [recs[i] for i in ii[s:s + 512]]
        b = belief_batch(mb, device)
        logits = net(b["obs"])
        bce += belief_bce(logits, b["belief"], b["bmask"]).item() * len(mb)
        P = (torch.sigmoid(logits) * b["bmask"].unsqueeze(1)).cpu().numpy()
        T = b["belief"].cpu().numpy(); M = b["bmask"].cpu().numpy()
        for bi in range(P.shape[0]):
            nun = int(M[bi].sum())
            played = int(round(float(mb[bi]["obs"]["seen"].sum())))
            name = next(p[0] for p in PHASES if p[1] <= played < p[2])
            for o in range(3):
                k = int(T[bi, o].sum())
                if k == 0:
                    continue
                top = np.argpartition(-P[bi, o], k - 1)[:k]
                hit = float(T[bi, o, top].sum()) / k
                ps += hit; pn += 1
                base += k / max(nun, 1); bn += 1
                ph[name][0] += hit; ph[name][1] += 1
    net.train()
    return (ps / max(pn, 1), base / max(bn, 1), bce / max(len(ii), 1),
            {n: (v[0] / v[1] if v[1] else 0.0, v[1]) for n, v in ph.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["human", "winner", "all", "ours"], default="all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="belief")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    recs = build_records(args.target)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = int(len(recs) * args.val_frac)
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device={device} | target={args.target}")

    net = BeliefNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = belief_batch(mb, device)
            loss = belief_bce(net(b["obs"]), b["belief"], b["bmask"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        pc, base, vbce, ph = metrics(net, recs, val_i, device)
        phs = "  ".join(f"{n.split()[0]} {v[0]*100:.0f}%" for n, v in ph.items())
        print(f"ep {ep:3d} | loss {tot/nb:.3f} | val P@cnt {pc*100:.1f}% "
              f"(baseline {base*100:.1f}%) bce {vbce:.3f} | {phs}")
        ck = {"model": net.state_dict(), "arch": "belief", "target": args.target,
              "epoch": ep, "val_pcount": pc, "baseline": base, "phases": {n: v[0] for n, v in ph.items()}}
        torch.save(ck, os.path.join(CKPT_DIR, f"{args.tag}_latest.pt"))
        if pc > best:
            best = pc
            torch.save(ck, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            print(f"   * new best P@cnt {pc*100:.1f}% -> {args.tag}_best.pt")


if __name__ == "__main__":
    main()
