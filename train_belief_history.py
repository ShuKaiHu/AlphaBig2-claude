"""Train BeliefNetHistory (card-aware belief encoder + GRU over full play
history) on a specific snapshot of online_games.jsonl.

    ./.venv/bin/python train_belief_history.py --snapshot ppo/data/snapshot_578_ids.json --tag belief_history_578
    ./.venv/bin/python train_belief_history.py --snapshot ppo/data/snapshot_1266_ids.json --tag belief_history_1266
"""
import argparse
import json
import os

import numpy as np
import torch

from ppo.belief_model_history import BeliefNetHistory, belief_history_batch
from ppo.belief_model import belief_bce
from ppo.belief_history_dataset import DATA, HISTORY_LEN_V2, build_belief_history_records
from ppo.parse_online_games import game_id as game_content_id

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ppo", "checkpoints")
PHASES = [("early <13", 0, 13), ("mid 13-25", 13, 26), ("late 26-38", 26, 39), ("end 39+", 39, 99)]


def load_games(snapshot_path):
    all_games = [json.loads(l) for l in open(DATA)]
    if not snapshot_path:
        return all_games
    ids = set(json.load(open(snapshot_path)))
    return [g for g in all_games if (g.get("id") or game_content_id(g)) in ids]


@torch.no_grad()
def metrics(net, recs, ii, device):
    net.eval()
    ps = pn = base = bn = bce = 0.0
    ph = {p[0]: [0.0, 0] for p in PHASES}
    for s in range(0, len(ii), 256):
        mb = [recs[i] for i in ii[s:s + 256]]
        b = belief_history_batch(mb, device)
        logits = net(b["obs"], b["history"])
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
    return (ps / max(pn, 1), base / max(bn, 1), bce / max(len(ii), 1),
            {n: (v[0] / v[1] if v[1] else 0.0, v[1]) for n, v in ph.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="path to a json list of game ids to restrict training to")
    ap.add_argument("--target", choices=["human", "winner", "all", "ours"], default="all")
    ap.add_argument("--history-len", type=int, default=HISTORY_LEN_V2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="belief_history")
    # regularization / capacity knobs (all default to the ORIGINAL behavior)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--extra-games", default=None,
                    help="path to a self-play jsonl added to TRAIN ONLY (val stays pure-human)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    games = load_games(args.snapshot)
    print(f"games in snapshot: {len(games)} (snapshot={args.snapshot or 'ALL'})")
    recs = build_belief_history_records(games, target=args.target, history_len=args.history_len)
    # Split by GAME, not by decision: a game's ~45 decisions are near-duplicate
    # siblings, so a decision-level shuffle leaks the same game into both sides and
    # inflates val P@count (and picks the most-memorizing epoch).
    gids = sorted({r["gid"] for r in recs})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(gids)
    nval_g = max(1, int(len(gids) * args.val_frac))
    val_gids = set(gids[:nval_g])
    val_i = [i for i, r in enumerate(recs) if r["gid"] in val_gids]
    tr_i = [i for i, r in enumerate(recs) if r["gid"] not in val_gids]
    # optional self-play augmentation: extra games go into TRAIN ONLY, so the val
    # split (and thus checkpoint selection) stays pure-human. Self-play uses
    # target="all" (no human/bot distinction; belief target is oracle either way).
    if args.extra_games:
        n_human = len(recs)
        sp = [json.loads(l) for l in open(args.extra_games)]
        sp_recs = build_belief_history_records(sp, target="all", history_len=args.history_len)
        recs = recs + sp_recs
        tr_i = tr_i + list(range(n_human, len(recs)))
        print(f"+ {len(sp_recs)} self-play records (train-only) from {args.extra_games}")
    val_i = np.array(val_i); tr_i = np.array(tr_i)
    print(f"train {len(tr_i)} / val {len(val_i)} | val games {nval_g}/{len(gids)} "
          f"| history_len={args.history_len} | device={device}")

    net = BeliefNetHistory(gru_hidden=args.gru_hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.cosine else None)
    print(f"params: {sum(p.numel() for p in net.parameters()):,} | "
          f"dropout={args.dropout} wd={args.weight_decay} gru={args.gru_hidden} "
          f"clip={args.grad_clip} cosine={args.cosine}")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = belief_history_batch(mb, device)
            loss = belief_bce(net(b["obs"], b["history"]), b["belief"], b["bmask"])
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            tot += loss.item(); nb += 1
        if sched is not None:
            sched.step()
        pc, base, vbce, ph = metrics(net, recs, val_i, device)
        phs = "  ".join(f"{n.split()[0]} {v[0]*100:.0f}%" for n, v in ph.items())
        print(f"ep {ep:3d} | loss {tot/nb:.3f} | val P@cnt {pc*100:.1f}% "
              f"(baseline {base*100:.1f}%) bce {vbce:.3f} | {phs}")
        ck = {"model": net.state_dict(), "arch": "belief_history_v2", "target": args.target,
              "history_len": args.history_len, "epoch": ep, "val_pcount": pc, "baseline": base,
              "phases": {n: v[0] for n, v in ph.items()}, "snapshot": args.snapshot}
        torch.save(ck, os.path.join(CKPT_DIR, f"{args.tag}_latest.pt"))
        if pc > best:
            best = pc
            torch.save(ck, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            print(f"   * new best P@cnt {pc*100:.1f}% -> {args.tag}_best.pt")


if __name__ == "__main__":
    main()
