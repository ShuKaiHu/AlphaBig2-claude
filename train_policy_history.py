"""Behavioral cloning for CardAwareActorCriticHistory (policy + GRU play-history,
same per-step schema as belief_model_history.py::BeliefNetHistory).

Default trains on ALL of online_games.jsonl (no --snapshot) -- the question
this run is meant to answer: does the checkpoint with the LOWEST training loss
(most epochs, most faithful human imitation) or the checkpoint with the BEST
vs-opponent-pool score (early-stopped) actually do better against real humans
online? We can't answer that locally -- only real online avg_score can -- so
this script saves BOTH candidates and lets online testing decide:
  - {tag}.pt         : best vs-pool score seen during training (early-stop pick)
  - {tag}_minloss.pt : final epoch reached (lowest training loss)

Checkpoint selection uses REAL self-play gameplay, either against the
historical opponent pool (ppo/pool_eval.py::evaluate_vs_pool, --eval-mode
pool, default) or against a single fixed baseline (--eval-mode single, vs
"Smart" -- the exact old methodology, kept here as an explicit THIRD control
group: does optimizing against one fixed opponent generalize worse than
optimizing against a diverse pool, given otherwise identical data/architecture/
early-stopping? Suspected yes -- a checkpoint can overfit to beating one
specific opponent without playing better in general -- but only real online
testing of all three variants (min-loss / vs-pool / vs-single) can confirm
it). Neither mode uses the internal decision-level "val top1" accuracy, which
has the same train/val leakage risk that inflated the old belief numbers this
session (same-game decisions can land on both sides of a random split). val
top1 is printed for information only and never decides which checkpoint is
kept.

Early stopping is patience-based on pool score, NOT a fixed epoch cap --
training loss keeps dropping via more faithful human-imitation long after real
gameplay performance peaks (confirmed empirically: policy_578's loss fell
every epoch through ep30 while its vs-Smart score peaked at ep5 and got
steadily worse after -- an objective-mismatch, not simple overfitting: BC loss
measures "predict the exact human action," which is a different objective
from "win games," since real humans don't play optimally).

    ./.venv/bin/python train_policy_history.py --tag policy_all
    ./.venv/bin/python train_policy_history.py --eval-mode single --tag policy_all_vs1
    ./.venv/bin/python train_policy_history.py --snapshot ppo/data/snapshot_578_ids.json --tag policy_578
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ppo.belief_history_dataset import DATA, HISTORY_LEN_V2, build_belief_history_records
from ppo.eval_baselines import evaluate_vs_all
from ppo.network_cardaware_history import CardAwareActorCriticHistory, build_batch_history
from ppo.parse_online_games import game_id as game_content_id
from ppo.pool_eval import evaluate_vs_pool, load_pool

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ppo", "checkpoints")


def load_games(snapshot_path):
    all_games = [json.loads(l) for l in open(DATA)]
    if not snapshot_path:
        return all_games
    ids = set(json.load(open(snapshot_path)))
    return [g for g in all_games if (g.get("id") or game_content_id(g)) in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="path to a json list of game ids to restrict training to")
    ap.add_argument("--seat-snapshot", default=None,
                     help="path to a json list of [run, game_id, seat] triples -- restricts training to "
                          "exactly those (game, seat) decisions (e.g. one behavioral cluster's human "
                          "decisions), finer-grained than --snapshot since a single game can have human "
                          "seats belonging to different clusters. Implies target='all' filtering handled "
                          "entirely by this set.")
    ap.add_argument("--target", choices=["human", "winner", "all", "ours"], default="human")
    ap.add_argument("--history-len", type=int, default=HISTORY_LEN_V2)
    ap.add_argument("--max-epochs", type=int, default=200, help="safety cap -- early stopping should trigger first")
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many consecutive eval checkpoints (each --eval-every epochs) "
                         "with no improvement in the eval score (pool-avg or vs-single, see --eval-mode). "
                         "NOT loss-based on purpose: training loss keeps dropping via more faithful "
                         "human-imitation long after real (gameplay) performance peaks -- confirmed "
                         "empirically (policy_578's loss fell every epoch through ep30 while its vs-Smart "
                         "score peaked at ep5 and got steadily worse after).")
    ap.add_argument("--eval-mode", choices=["pool", "single"], default="pool",
                    help="pool: score = avg vs 21 randomly-sampled historical opponents (default). "
                         "single: score = avg vs Smart only -- the old methodology, kept as an explicit "
                         "control group to compare against --eval-mode pool.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="policy_history")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    pool = load_pool(device) if args.eval_mode == "pool" else None

    games = load_games(args.snapshot)
    print(f"games in snapshot: {len(games)} (snapshot={args.snapshot or 'ALL'})")
    allowed_keys = None
    if args.seat_snapshot:
        allowed_keys = {tuple(t) for t in json.load(open(args.seat_snapshot))}
        print(f"seat-snapshot: {len(allowed_keys)} (run,game_id,seat) triples from {args.seat_snapshot}")
    recs = build_belief_history_records(games, target=args.target, history_len=args.history_len,
                                         allowed_keys=allowed_keys)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = int(len(recs) * args.val_frac)
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"train {len(tr_i)} / val {nval} | history_len={args.history_len} | device={device}")

    net = CardAwareActorCriticHistory().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

    def accuracy(ii):
        net.eval(); correct = 0
        with torch.no_grad():
            for s in range(0, len(ii), 256):
                mb = [recs[i] for i in ii[s:s + 256]]
                b = build_batch_history(mb, device)
                logits, _ = net.forward(b["obs"], b["history"], b["act_feats"], b["act_mask"])
                correct += (logits.argmax(-1) == b["pos"]).sum().item()
        net.train(); return correct / max(len(ii), 1)

    def greedy_fn(env):
        return net.greedy(env, device)

    best = -1e9
    best_ep = 0
    stale = 0
    last_ck = None
    for ep in range(1, args.max_epochs + 1):
        net.train()
        np.random.shuffle(tr_i); tot = 0.0; nb = 0
        for s in range(0, len(tr_i), args.batch):
            mb = [recs[i] for i in tr_i[s:s + args.batch]]
            b = build_batch_history(mb, device)
            logits, _ = net.forward(b["obs"], b["history"], b["act_feats"], b["act_mask"])
            loss = F.cross_entropy(logits, b["pos"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        train_loss = tot / nb
        va = accuracy(val_i)
        print(f"ep {ep:3d} | loss {train_loss:.3f} | val top1 {va*100:.1f}% (leaky split, info only)")
        if ep % args.eval_every == 0 or ep == args.max_epochs:
            net.eval()
            if args.eval_mode == "pool":
                wr, score = evaluate_vs_pool(greedy_fn, pool, args.eval_games, device=device, seed=ep)
                tag_line = f"pool eval | win_rate {wr*100:4.1f}% | avg_score {score:+.2f}"
                score_key = "pool_score"
            else:
                res = evaluate_vs_all(greedy_fn, args.eval_games)
                wr, score = res["smart"][0], res["smart"][1]
                tag_line = "   eval | " + " | ".join(
                    f"{n}: {w*100:4.1f}% {s:+.2f}{'*' if (w>0.25 and s>0) else ' '}"
                    for n, (w, s) in res.items())
                score_key = "vs_smart_score"
            net.train()
            print(f"   {tag_line}")
            ck = {"model": net.state_dict(), "arch": "cardaware_history",
                  "bc_target": args.target, "epoch": ep, "train_loss": train_loss,
                  "eval_mode": args.eval_mode, score_key: score, "win_rate": wr,
                  "val_acc": va, "snapshot": args.snapshot}
            last_ck = ck
            if score > best:
                best = score; best_ep = ep; stale = 0
                torch.save(ck, os.path.join(CKPT_DIR, "saved", f"{args.tag}.pt"))
                print(f"   * new best ({args.eval_mode} {score:+.2f}) -> saved/{args.tag}.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"   * no {args.eval_mode}-score improvement for {stale} checks "
                          f"(best {best:+.2f} @ ep{best_ep}) -> stopping")
                    break

    # Always also save the FINAL epoch reached (lowest training loss), even if
    # a mid-training epoch had a better eval score -- eval score is still a
    # local proxy (this project has repeatedly confirmed local self-eval vs a
    # frozen reference is noisy/low-power), so don't let it silently discard
    # the min-loss checkpoint; keep both and let the real judge (online
    # avg_score vs humans) decide which local metric actually predicts it.
    if last_ck is not None:
        torch.save(last_ck, os.path.join(CKPT_DIR, "saved", f"{args.tag}_minloss.pt"))
        score_key = "pool_score" if args.eval_mode == "pool" else "vs_smart_score"
        print(f"   * final epoch reached (min loss, {args.eval_mode} {last_ck[score_key]:+.2f}) "
              f"-> saved/{args.tag}_minloss.pt")


if __name__ == "__main__":
    main()
