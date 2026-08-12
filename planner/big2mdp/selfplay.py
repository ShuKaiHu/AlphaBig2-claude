"""Self-play table filling — the replication of Big2MDP's "learns in advance
with a fixed number of 500K games" (all games AI-vs-AI, no human data).

Four Big2MDPAgent seats share ONE StatsStore; every finished game's decisions
are ingested (features, action key, win/loss + final score, next-3-pass), so
later games plan over statistics from earlier ones — the paper's iterative
self-learning loop.

Usage:
    python -m planner.big2mdp.selfplay --games 500000 --level 4 \
        --save planner/big2mdp/data/store_l4.pkl --report-every 5000

Resume by pointing --load at an existing store. planner/big2mdp/data/ is
gitignored (stores are large, regenerable artifacts).
"""
import argparse
import os
import time

import numpy as np

import enumerateOptions
from engine.env import Big2Env
from planner.big2mdp.agent import Big2MDPAgent
from planner.big2mdp.features import action_key, state_features, PASS_KEY
from planner.big2mdp.store import StatsStore

PASS_IDX = enumerateOptions.passInd


def play_one_game(env, agents, store):
    """Play one 4-seat game; ingest every decision into the store."""
    env.reset()
    pending = []                     # (seat, feats, akey, decision_index)
    seq = []                         # chronological pass/no-pass flags
    while not env.done:
        me = env.current_player
        feats = state_features(env.game, me)
        a = agents[me - 1](env)
        if a == PASS_IDX:
            akey = PASS_KEY
        else:
            cards, _ = enumerateOptions.getOptionNC(int(a))
            akey = action_key(cards)
        pending.append((me, feats, akey, len(seq)))
        seq.append(akey is PASS_KEY)
        env.step(int(a))
    rewards = env.game.rewards
    for seat, feats, akey, i in pending:
        nxt = seq[i + 1:i + 4]
        next3pass = len(nxt) == 3 and all(nxt)
        sc = float(rewards[seat - 1])
        store.add_record(feats, akey, won=sc > 0, score=sc, next3pass=next3pass)
    store.n_games += 1
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--save", default="planner/big2mdp/data/store.pkl")
    ap.add_argument("--load", default=None)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--no-exact-floor", action="store_true",
                    help="purist run: R_c from table statistics only")
    args = ap.parse_args()

    store = StatsStore.load(args.load) if args.load else StatsStore()
    agents = [Big2MDPAgent(store, level=args.level,
                           exact_floor=not args.no_exact_floor)
              for _ in range(4)]
    env = Big2Env()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    t0 = time.time()
    for gidx in range(1, args.games + 1):
        play_one_game(env, agents, store)
        if gidx % args.report_every == 0:
            rate = gidx / (time.time() - t0)
            print(f"[{gidx}/{args.games}] {rate:.1f} games/s  "
                  f"records={store.n_records}  total_games={store.n_games}",
                  flush=True)
        if gidx % args.save_every == 0:
            store.save(args.save)
    store.save(args.save)
    print(f"done: {store.n_games} games, {store.n_records} records → {args.save}")


if __name__ == "__main__":
    main()
