"""V6 pretrain: bigger card-aware net + opponent-hand BELIEF head, on human data.

Joint supervised objective on the showdown-reconstructed online games:
  - policy: cross-entropy toward the action the human chose (imitation)
  - belief: BCE toward the ORACLE (opponents' actual hidden cards), masked to the
            unseen pool. Oracle is loss-only — never enters the observation.

Reports: policy top-1, belief precision@count (random ~ count/unseen ≈ 33%), and
strength vs Random/Greedy/Smart. Belief-conditioned policy means a better belief
should lift play. RL (league) can follow from this warm start.

    ./.venv/bin/python -m ppo.train_v6 --target human --epochs 30
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from ppo.network_v6 import CardAwareV6, build_batch_v6, belief_loss
from ppo.bc_dataset import build_records
from ppo.eval_baselines import evaluate_vs_all

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["human", "winner", "all", "ours"], default="human")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--belief-coef", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    recs = build_records(args.target)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = int(len(recs) * args.val_frac)
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | belief_coef={args.belief_coef} | device={device}")

    net = CardAwareV6().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    @torch.no_grad()
    def metrics(ii):
        net.eval()
        correct = 0
        prec_sum, prec_n = 0.0, 0
        for s in range(0, len(ii), 512):
            mb = [recs[i] for i in ii[s:s + 512]]
            b = build_batch_v6(mb, device)
            logits, _, bl = net.forward(b["obs"], b["act_feats"], b["act_mask"])
            correct += (logits.argmax(-1) == b["pos"]).sum().item()
            # belief precision@count, per opponent, over the unseen pool
            P = torch.sigmoid(bl)                      # (B,3,52)
            tgt = b["belief"]; m = b["bmask"].unsqueeze(1)  # (B,1,52)
            Pm = (P * m).cpu().numpy(); T = tgt.cpu().numpy()
            for bi in range(Pm.shape[0]):
                for o in range(3):
                    k = int(T[bi, o].sum())
                    if k == 0:
                        continue
                    top = np.argpartition(-Pm[bi, o], k - 1)[:k]
                    prec_sum += float(T[bi, o, top].sum()) / k
                    prec_n += 1
        net.train()
        return correct / max(len(ii), 1), prec_sum / max(prec_n, 1)

    def greedy_fn(env):
        return net.greedy(env, device)

    best = -1e9; best_bp = -1.0
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i)
        tot_p = tot_b = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = build_batch_v6(mb, device)
            logits, _, bl = net.forward(b["obs"], b["act_feats"], b["act_mask"])
            lp = F.cross_entropy(logits, b["pos"])
            lb = belief_loss(bl, b["belief"], b["bmask"])
            loss = lp + args.belief_coef * lb
            opt.zero_grad(); loss.backward(); opt.step()
            tot_p += lp.item(); tot_b += lb.item(); nb += 1
        va, bp = metrics(val_i)
        print(f"ep {ep:3d} | pol {tot_p/nb:.3f} bel {tot_b/nb:.3f} | "
              f"val top1 {va*100:.1f}% | belief P@cnt {bp*100:.1f}%")
        ckpt = {"model": net.state_dict(), "arch": "v6", "bc_target": args.target,
                "epoch": ep, "val_acc": va, "belief_prec": bp}
        torch.save(ckpt, os.path.join(CKPT_DIR, f"ppo_v6_{args.target}_latest.pt"))
        if bp > best_bp:
            best_bp = bp
            torch.save(ckpt, os.path.join(CKPT_DIR, f"ppo_v6_{args.target}_belief.pt"))
        if ep % args.eval_every == 0 or ep == args.epochs:
            net.eval(); res = evaluate_vs_all(greedy_fn, args.eval_games); net.train()
            print("   eval | " + " | ".join(
                f"{n}: {wr*100:4.1f}% {sc:+.2f}{'*' if (wr > 0.25 and sc > 0) else ' '}"
                for n, (wr, sc) in res.items()))
            score = res["smart"][1]
            if score > best:
                best = score
                torch.save({"model": net.state_dict(), "arch": "v6", "bc_target": args.target,
                            "epoch": ep, "eval": res, "val_acc": va, "belief_prec": bp},
                           os.path.join(CKPT_DIR, f"ppo_v6_{args.target}_best.pt"))
                print(f"   * new best (Smart {score:+.2f}) -> ppo_v6_{args.target}_best.pt")


if __name__ == "__main__":
    main()
