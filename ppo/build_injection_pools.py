"""Build the M3 injection pools (Track B) from the real online human games.

Reads a FROZEN snapshot of online_games.jsonl (pass --data; the live file is
appended to by the harvest job) and writes two JSONL pools under ppo/data/:

  injection_d2_pool.jsonl  -- mid-game positions: for every seat (human AND
      ours) whose game ended as a caged loss (final remaining >= 10) while
      dealt >= 1 two, replay the game with the CORRECT trick logic (same loop
      as ppo/bc_dataset.py: a play does NOT reopen the trick for already-
      passed seats; trick resets only after 3 CONSECUTIVE passes, server
      auto-passes counted) and snapshot the state at the first DECISION event
      at-or-after the moment the LAST opponent drops to <= 8 remaining cards.
  injection_d1_pool.jsonl  -- full deals where >= 1 seat's starting 13 cards
      have min_plays_to_empty <= 6 AND no bomb (canonical detectors from
      AlphaBig2-Value/hand_features.py); one record per qualifying seat.

Every D2 record is validated before being written: card conservation
(remaining + played == the dealt 52) and the seat to move has >= 1 legal
action per big2Game.returnAvailableActions on a bc_dataset._snapshot.

Dedup: identical states are dropped (D2 key = full state incl. turn + trick;
D1 key = 4 hands + strong seat). Split: 10% of GAME IDS are held out as an
evaluation-only subset (split="eval"; md5(game_id) % 10 == 0) -- training
must never inject from it (future injected-state G2 evaluation needs unseen
positions).

Record schemas are documented in ppo/injection_pools.py (the loader).

Usage (from the repo root, venv python):
  .venv/bin/python3 PPO/build_injection_pools.py --data <frozen_snapshot.jsonl>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value")

from hand_features import min_plays_to_empty, has_bomb_rank_suit  # canonical
from ppo.bc_dataset import _snapshot

TWOS = {49, 50, 51, 52}          # rank '2' = top rank index 12
CAGE_THRESHOLD = 10              # final remaining >= 10 => caged (platform 2^cage)
D2_OPP_CARDS = 8                 # trigger: last opponent drops to <= this
D1_MAX_MIN_PLAYS = 6
EVAL_MOD = 10                    # md5(game_id) % 10 == 0 -> eval split


def split_of(game_id):
    return "eval" if int(hashlib.md5(game_id.encode()).hexdigest(), 16) % EVAL_MOD == 0 else "train"


def replay_states(g):
    """Yield (event_idx, ev, state_before_ev, forced) for every event, replayed
    with the engine-matching trick logic copied from ppo/bc_dataset.build_records."""
    hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
    played = {s: [] for s in range(4)}
    trick_cards, trick_owner, passed, consec = None, None, set(), 0
    for i, ev in enumerate(g["plays"]):
        s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
        forced = act == "pass" and s in passed   # server auto-skip, not a decision
        control = 1 if trick_cards is None else 0
        remaining = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
        yield i, ev, dict(remaining=remaining, played={k: list(played[k]) for k in range(4)},
                          control=control, trick_cards=list(trick_cards) if trick_cards else None,
                          trick_owner=trick_owner, passed=sorted(passed)), forced
        if act == "play":
            played[s].extend(cards)
            trick_cards, trick_owner = cards, s
            passed.discard(s)
            consec = 0
        else:
            passed.add(s)
            consec += 1
            if consec >= 3:
                trick_cards, trick_owner, passed, consec = None, None, set(), 0


def validate_d2(rec, dealt_hands):
    """Conservation + >= 1 legal action for the seat to move. Raises on failure."""
    rem = {s: rec["hands"][s] for s in range(4)}
    ply = {s: rec["cards_played"][s] for s in range(4)}
    for s in range(4):
        assert sorted(rem[s] + ply[s]) == sorted(dealt_hands[s]), f"seat {s} cards not conserved"
    allc = sorted(c for s in range(4) for c in rem[s] + ply[s])
    assert allc == list(range(1, 53)), "52-card conservation failed"
    cardsPlayed = np.zeros((4, 52), dtype=np.int64)
    for s in range(4):
        for c in ply[s]:
            cardsPlayed[s][c - 1] = 1
    gg = _snapshot(rec["turn"], rem, cardsPlayed, rec["control"],
                   rec["trick_cards"], rec["trick_owner"], set(rec["passed"]))
    mask = gg.returnAvailableActions()
    n_legal = int(np.flatnonzero(mask).size)
    assert n_legal >= 1, "seat to move has no legal action"
    return n_legal


def build(games):
    d2, d1 = [], []
    d2_seen, d1_seen = set(), set()
    d2_dups = d1_dups = d2_no_decision = d2_invalid = 0
    for g in games:
        gid = g["id"]
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        scores = {int(k): float(v) for k, v in g["scores"].items()}
        sp = split_of(gid)

        # ---- D1: strong-deal pool (starting hands only) ----
        for s in range(4):
            mp = min_plays_to_empty(hands[s])
            pairs = [((c - 1) // 4, (c - 1) % 4) for c in hands[s]]
            if mp <= D1_MAX_MIN_PLAYS and not has_bomb_rank_suit(pairs):
                key = (tuple(tuple(sorted(hands[k])) for k in range(4)), s)
                if key in d1_seen:
                    d1_dups += 1
                    continue
                d1_seen.add(key)
                d1.append({"pool": "d1", "game_id": gid, "run": g.get("run"),
                           "split": sp, "strong_seat": s, "min_plays": mp,
                           "hands": {str(k): sorted(hands[k]) for k in range(4)}})

        # ---- D2: caged-loss mid-game pool ----
        n_plays_by = {s: sum(len(ev["cards"]) for ev in g["plays"]
                             if ev["seat"] == s and ev["action"] == "play") for s in range(4)}
        final_rem = {s: 13 - n_plays_by[s] for s in range(4)}
        targets = [s for s in range(4)
                   if final_rem[s] >= CAGE_THRESHOLD and len(set(hands[s]) & TWOS) >= 1]
        if not targets:
            continue
        states = list(replay_states(g))
        for tgt in targets:
            opps = [o for o in range(4) if o != tgt]
            # trigger: first event index after whose APPLICATION all opps <= 8.
            # state_before of event i+1 reflects events 0..i applied.
            # NOTE (verified 2026-07-18 audit): every skip counted in
            # d2_no_decision is a DOUBLE-CAGED game — a second non-target
            # opponent also finishes with >8 cards, so "ALL three opps <= 8"
            # is never satisfied at any point, including game end. It is NOT
            # "trigger reached only on the game-ending play" (structurally
            # near-impossible: the winner's final play removes <=5 cards from
            # a hand already <=5, so it cannot cross an OPPONENT below the
            # threshold). Relaxing the trigger (e.g. excluding the other caged
            # seat from the <=8 condition) could recover many of these skips
            # from existing data — that would be a spec change.
            trig = None
            for i in range(1, len(states)):
                st = states[i][2]
                if all(len(st["remaining"][o]) <= D2_OPP_CARDS for o in opps):
                    trig = i
                    break
            if trig is None:
                d2_no_decision += 1
                continue
            # first genuine decision event at-or-after the trigger moment
            dec = next(((i, ev, st) for i, ev, st, forced in states[trig:] if not forced), None)
            if dec is None:
                d2_no_decision += 1
                continue
            i, ev, st = dec
            rec = {"pool": "d2", "game_id": gid, "run": g.get("run"), "split": sp,
                   "seat": tgt, "final_score": scores.get(tgt),
                   "final_remaining": final_rem[tgt],
                   "n_twos_dealt": len(set(hands[tgt]) & TWOS),
                   "n_twos_remaining": len(set(st["remaining"][tgt]) & TWOS),
                   "event_idx": i, "turn": ev["seat"],
                   "hands": {str(k): sorted(st["remaining"][k]) for k in range(4)},
                   "cards_played": {str(k): sorted(st["played"][k]) for k in range(4)},
                   "control": st["control"], "trick_cards": st["trick_cards"],
                   "trick_owner": st["trick_owner"], "passed": st["passed"]}
            # loader-format record -> int-keyed dicts for validation
            v = {**rec, "hands": {k: sorted(st["remaining"][k]) for k in range(4)},
                 "cards_played": {k: sorted(st["played"][k]) for k in range(4)}}
            try:
                rec["n_legal"] = validate_d2(v, hands)
            except AssertionError as e:
                d2_invalid += 1
                print(f"  INVALID d2 state game={gid} seat={tgt}: {e}")
                continue
            key = json.dumps([rec["hands"], rec["cards_played"], rec["turn"], rec["control"],
                              sorted(rec["trick_cards"] or []), rec["trick_owner"],
                              rec["passed"]], sort_keys=True)
            if key in d2_seen:
                d2_dups += 1
                continue
            d2_seen.add(key)
            d2.append(rec)
    stats = dict(d2_dups=d2_dups, d1_dups=d1_dups,
                 d2_no_decision=d2_no_decision, d2_invalid=d2_invalid)
    return d2, d1, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="frozen snapshot of online_games.jsonl")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "data"))
    args = ap.parse_args()
    games = [json.loads(l) for l in open(args.data)]
    print(f"games: {len(games)}")
    d2, d1, stats = build(games)
    for name, pool in (("injection_d2_pool.jsonl", d2), ("injection_d1_pool.jsonl", d1)):
        path = os.path.join(args.outdir, name)
        with open(path, "w") as f:
            for r in pool:
                f.write(json.dumps(r) + "\n")
        n_eval = sum(r["split"] == "eval" for r in pool)
        n_games = len({r["game_id"] for r in pool})
        eval_games = len({r["game_id"] for r in pool if r["split"] == "eval"})
        print(f"{name}: {len(pool)} records ({len(pool) - n_eval} train / {n_eval} eval) "
              f"from {n_games} games ({eval_games} eval games) -> {path}")
    print(f"stats: {stats}")

    # distributions
    def hist(vals):
        from collections import Counter
        return dict(sorted(Counter(vals).items()))
    print("\nD2 n_twos_dealt:", hist(r["n_twos_dealt"] for r in d2))
    print("D2 n_twos_remaining:", hist(r["n_twos_remaining"] for r in d2))
    print("D2 target remaining cards at snapshot:", hist(len(r["hands"][str(r["seat"])]) for r in d2))
    print("D2 final_remaining:", hist(r["final_remaining"] for r in d2))
    fs = [r["final_score"] for r in d2 if r["final_score"] is not None]
    print(f"D2 final_score: mean={np.mean(fs):.2f} min={min(fs)} p25={np.percentile(fs,25):.0f} "
          f"median={np.median(fs):.0f} p75={np.percentile(fs,75):.0f} max={max(fs)}")
    print("D2 turn==target seat:", sum(r["turn"] == r["seat"] for r in d2), "/", len(d2))
    print("\nD1 min_plays histogram:", hist(r["min_plays"] for r in d1))
    print("D1 per-seat counts:", hist(r["strong_seat"] for r in d1))


if __name__ == "__main__":
    main()
