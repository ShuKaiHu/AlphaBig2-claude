"""Self-play database filling — paper-literal (v4).

Replication of "learns in advance with a fixed number of 500K games": four
seats share ONE RecordStore; after every finished game each decision becomes
a record with its successor pointer (same player's next decision) and its
npass sample (how many of the next three actors passed), so the very next
game's prediction trees walk statistics that include this game — the paper's
per-game incremental self-learning.

Usage:
    python -m planner.big2mdp.selfplay --games 500000 --level 4 \
        --save planner/big2mdp/data/records_l4.pkl --report-every 5000

Resume with --load. planner/big2mdp/data/ is gitignored.
"""
import argparse
import os
import time

import enumerateOptions
from engine.env import Big2Env
from planner.big2mdp.agent import Big2MDPAgent
from planner.big2mdp.features import PASS_KEY, action_key, state_features
from planner.big2mdp.store import RecordStore

PASS_IDX = enumerateOptions.passInd


def play_one_game(env, agents, store):
    """Play one 4-seat game; ingest every decision with successor pointers."""
    env.reset()
    decisions = []                    # (seat, feats, akey)
    while not env.done:
        me = env.current_player
        feats = state_features(env.game, me)
        a = agents[me - 1](env)
        if a == PASS_IDX:
            akey = PASS_KEY
        else:
            cards, _ = enumerateOptions.getOptionNC(int(a))
            akey = action_key(cards)
        decisions.append((me, feats, akey))
        env.step(int(a))
    rewards = env.game.rewards

    base = len(store)
    n = len(decisions)
    next_of = [-1] * n                # same-seat successor (chronological idx)
    last_seen = {}
    for j in range(n - 1, -1, -1):
        seat = decisions[j][0]
        next_of[j] = last_seen.get(seat, -1)
        last_seen[seat] = j
    for j, (seat, feats, akey) in enumerate(decisions):
        npass = sum(1 for k in range(j + 1, min(j + 4, n))
                    if decisions[k][2] is PASS_KEY)
        sc = float(rewards[seat - 1])
        succ = base + next_of[j] if next_of[j] >= 0 else -1
        store.add_record(feats, akey, won=sc > 0, score=sc,
                         npass=npass, succ=succ)
    store.n_games += 1
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--cap", type=int, default=2000,
                    help="retrieval cap per decision (tractability assumption)")
    ap.add_argument("--depth-max", type=int, default=8)
    ap.add_argument("--save", default="planner/big2mdp/data/records.pkl")
    ap.add_argument("--load", default=None)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=20000)
    args = ap.parse_args()

    store = RecordStore.load(args.load) if args.load else RecordStore()
    agents = [Big2MDPAgent(store, level=args.level, cap=args.cap,
                           depth_max=args.depth_max) for _ in range(4)]
    env = Big2Env()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    t0 = time.time()
    for gidx in range(1, args.games + 1):
        play_one_game(env, agents, store)
        if gidx % args.report_every == 0:
            rate = gidx / (time.time() - t0)
            print(f"[{gidx}/{args.games}] {rate:.1f} games/s  "
                  f"records={len(store)}  total_games={store.n_games}",
                  flush=True)
        if gidx % args.save_every == 0:
            store.save(args.save)
    store.save(args.save)
    print(f"done: {store.n_games} games, {len(store)} records → {args.save}")


if __name__ == "__main__":
    main()
