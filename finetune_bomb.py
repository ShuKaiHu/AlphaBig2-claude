"""Fine-tune a CardAwareActorCriticHistory checkpoint on bomb-decision data
(from generate_bomb_data.py) -- a small, TARGETED fine-tune (low LR, few
epochs) to fix quad/straight-flush timing without disturbing general play.

Checkpoint selection uses real self-play gameplay against the historical
opponent pool (ppo/pool_eval.py::evaluate_vs_pool) -- NOT vs a single fixed
baseline like "Smart". This project directly validated the difference online
this session: three otherwise-identical policy_all variants selected by
min-loss / vs-Smart / vs-pool respectively scored avg_score -9.80 / -9.05 /
-6.78 over ~100 real games each -- vs-pool selection won cleanly, so bomb
fine-tuning (which chooses a checkpoint the exact same way) uses the same
criterion. We ALSO report the bomb-specific agreement rate (does the
fine-tuned policy now pick the roll_top action on the held-out bomb decisions)
as a secondary, targeted diagnostic -- never the selection criterion itself.

    ./.venv/bin/python finetune_bomb.py --base ppo/checkpoints/saved/policy_all.pt \
        --data ppo/data/bomb_decisions_all.jsonl --tag policy_all_bomb
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ppo.action_features import ACTION_FEATURES
from ppo.network_cardaware_history import CardAwareActorCriticHistory, build_batch_history
from ppo.pool_eval import evaluate_vs_pool, load_pool

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ppo", "checkpoints")


def load_records(path):
    records = []
    for line in open(path):
        r = json.loads(line)
        obs = {k: np.array(v, dtype=np.float32 if k != "hand_ids" else np.int64) for k, v in r["obs"].items()}
        legal = np.array(r["legal"], dtype=np.int64)
        pos = int(np.where(legal == r["roll_top"])[0][0])
        records.append({
            "obs": obs,
            "feats": ACTION_FEATURES[legal].astype(np.float32),
            "pos": pos,
            "history": np.array(r["history"], dtype=np.float32),
            "bomb_actions": set(r["bomb_actions"]),
            "roll_top": r["roll_top"],
        })
    return records


@torch.no_grad()
def bomb_agreement(net, recs, device):
    """Fraction of held-out bomb decisions where the fine-tuned policy's own
    top choice now matches the rollout-determined best action (roll_top)."""
    net.eval()
    agree = 0
    for s in range(0, len(recs), 64):
        mb = recs[s:s + 64]
        b = build_batch_history(mb, device)
        logits, _ = net.forward(b["obs"], b["history"], b["act_feats"], b["act_mask"])
        pred_pos = logits.argmax(-1).cpu().numpy()
        for i, r in enumerate(mb):
            if pred_pos[i] == r["pos"]:
                agree += 1
    return agree / max(len(recs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint to warm-start from (e.g. policy_1266.pt)")
    ap.add_argument("--data", required=True, help="bomb_decisions jsonl from generate_bomb_data.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="small on purpose -- targeted fine-tune")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="policy_bomb")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pool = load_pool(device)

    recs = load_records(args.data)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = int(len(recs) * args.val_frac)
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"bomb decisions: train {len(tr_i)} / val {nval}")

    net = CardAwareActorCriticHistory().to(device)
    base_ck = torch.load(args.base, map_location=device, weights_only=False)
    net.load_state_dict(base_ck["model"])
    print(f"warm-started from {args.base} (epoch {base_ck.get('epoch')}, "
          f"eval {base_ck.get('eval', {}).get('smart')})")

    val_recs = [recs[i] for i in val_i]
    pre_agree = bomb_agreement(net, val_recs, device)
    print(f"BEFORE fine-tune: bomb agreement on held-out = {pre_agree*100:.1f}%")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def greedy_fn(env):
        return net.greedy(env, device)

    best = -1e9
    for ep in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = build_batch_history(mb, device)
            logits, _ = net.forward(b["obs"], b["history"], b["act_feats"], b["act_mask"])
            loss = F.cross_entropy(logits, b["pos"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        agree = bomb_agreement(net, val_recs, device)
        print(f"ep {ep:3d} | loss {tot/nb:.3f} | bomb agreement (held-out) {agree*100:.1f}%")
        if ep % args.eval_every == 0 or ep == args.epochs:
            net.eval()
            wr, score = evaluate_vs_pool(greedy_fn, pool, args.eval_games, device=device, seed=ep)
            net.train()
            print(f"   pool eval | win_rate {wr*100:4.1f}% | avg_score {score:+.2f}")
            if score > best:
                best = score
                torch.save({"model": net.state_dict(), "arch": "cardaware_history",
                            "epoch": ep, "pool_score": score, "pool_win_rate": wr,
                            "bomb_agreement": agree, "bomb_agreement_pre": pre_agree,
                            "base": args.base},
                           os.path.join(CKPT_DIR, "saved", f"{args.tag}.pt"))
                print(f"   * new best (pool {score:+.2f}) -> saved/{args.tag}.pt")


if __name__ == "__main__":
    main()
