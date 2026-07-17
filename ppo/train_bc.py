"""Behavioral cloning: train cardaware to imitate online (human) play.

Pure supervised learning — predict the action the human chose at each decision
(full-info reconstructed state). Reports top-1 match accuracy AND strength vs the
Random/Greedy/Smart yardstick, so we can see how far "just imitate humans" gets.

    ./.venv/bin/python -m ppo.train_bc --target human --epochs 30
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from ppo.network_cardaware import CardAwareActorCritic, build_batch
from ppo.bc_dataset import build_records
from ppo.eval_baselines import evaluate_vs_all

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["human", "winner", "all", "ours"], default="human")
    ap.add_argument("--snapshot", default=None, help="json list of game ids to restrict training to")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="checkpoint name (default ppo_bc_{target}); set for scaling runs")
    ap.add_argument("--select", choices=["smart", "valacc"], default="smart",
                    help="valacc = save best val top1 (cheap, no gameplay eval) — used for data-scaling runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    recs = build_records(args.target, snapshot=args.snapshot)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = int(len(recs) * args.val_frac)
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | device={device}")

    net = CardAwareActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def accuracy(ii):
        net.eval(); correct = 0
        with torch.no_grad():
            for s in range(0, len(ii), 512):
                mb = [recs[i] for i in ii[s:s + 512]]
                b = build_batch(mb, device)
                logits, _ = net.forward(b["obs"], b["act_feats"], b["act_mask"])
                correct += (logits.argmax(-1) == b["pos"]).sum().item()
        net.train(); return correct / max(len(ii), 1)

    def greedy_fn(env):
        return net.greedy(env, device)

    best = -1e9
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_i)
        tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = build_batch(mb, device)
            logits, _ = net.forward(b["obs"], b["act_feats"], b["act_mask"])
            loss = F.cross_entropy(logits, b["pos"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        va = accuracy(val_i)
        print(f"ep {ep:3d} | loss {tot/nb:.3f} | val top1 {va*100:.1f}%")
        tag = args.tag or f"ppo_bc_{args.target}"
        if args.select == "valacc":
            if va > best:
                best = va
                torch.save({"model": net.state_dict(), "arch": "cardaware",
                            "bc_target": args.target, "epoch": ep, "val_acc": va,
                            "snapshot": args.snapshot},
                           os.path.join(CKPT_DIR, f"{tag}_best.pt"))
                print(f"   * new best (val top1 {va*100:.1f}%) -> {tag}_best.pt")
            continue
        if ep % args.eval_every == 0 or ep == args.epochs:
            net.eval()
            res = evaluate_vs_all(greedy_fn, args.eval_games)
            net.train()
            line = "   eval | " + " | ".join(
                f"{n}: {wr*100:4.1f}% {sc:+.2f}{'*' if (wr>0.25 and sc>0) else ' '}"
                for n, (wr, sc) in res.items())
            print(line)
            score = res["smart"][1]
            if score > best:
                best = score
                torch.save({"model": net.state_dict(), "arch": "cardaware",
                            "bc_target": args.target, "epoch": ep, "eval": res,
                            "val_acc": va},
                           os.path.join(CKPT_DIR, f"{tag}_best.pt"))
                print(f"   * new best (Smart {score:+.2f}) -> {tag}_best.pt")


if __name__ == "__main__":
    main()
