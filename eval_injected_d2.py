#!/usr/bin/env python3
"""Injected-D2 evaluator — M3 Phase 1's primary judge for 2-hoarding behavior.

Restores each of the HELD-OUT eval-split D2 positions (real online human
caged-loss states holding >= 1 dealt two; never used for training injection),
seats the CANDIDATE policy at the record's target seat, fills the other three
seats with a frozen opponent policy playing ARGMAX, and plays every game to
completion with the real engine (platform-exact scoring, incl. per-unplayed-2
doubling and the remaining>=10 cage).

Two variants are reported, both matter:
  * argmax  — 1 deterministic pass per position: the candidate also plays
    argmax. This is the mode the policy is deployed in, but a single
    deterministic trajectory hides the policy's DISTRIBUTION over hoard/dump.
  * sampled — --reps rollouts per position where ONLY the candidate samples
    from its full policy distribution (temperature 1); opponents stay argmax.
    This measures how much probability mass the policy actually puts on
    hoarding vs dumping its 2s, which argmax mode-collapse cannot show.

Determinism: rollout (record r, rep k) is seeded torch.manual_seed(
seed_base + 100000*r + k) right before play, opponents are argmax and the
engine consumes no RNG after restore_state, so runs are exactly reproducible
and two candidates evaluated with the same --seed-base are PAIRED per
(record, rep).

Headline metrics (target seat only):
  * P(>= 1 two unplayed at end | ended as caged loss)  ["hoard rate"]
    caged loss := final remaining >= 10 (the platform cage threshold)
  * mean final engine score
  * P(escaped cage) := final remaining < 10 (winning included)

Usage:
    ./.venv/bin/python3 eval_injected_d2.py \
        --ckpt ppo/checkpoints/policy_4500_best.pt \
        --out ppo/data/injected_d2_baseline.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from big2Game import big2Game                      # noqa: E402
from engine.env import Big2Env                     # noqa: E402
from ppo.injection_pools import load_d2_pool       # noqa: E402
from ppo.policies import make_policy               # noqa: E402

DEFAULT_OPP = os.path.join(_ROOT, "ppo", "checkpoints", "policy_4500_best.pt")
CAGE = 10           # platform: losses double while remaining >= 10
MAX_STEPS = 400     # a Big 2 game can never take this many decisions


def load_frozen_policy(path, device):
    """Load a policy checkpoint for greedy/sampled play. Fails fast (exit 2)
    on a nonexistent path."""
    if not os.path.isfile(path):
        print(f"ERROR: checkpoint does not exist: {path}", file=sys.stderr)
        sys.exit(2)
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = make_policy(ck.get("arch", "cardaware"), device)
    pol.net.load_state_dict(ck["model"])
    pol.eval()
    return pol


def make_env(rec):
    """Big2Env wrapping a big2Game restored from a D2 pool record.

    Pool records store 0-BASED seats; big2Game.restore_state expects 1-BASED
    engine players — the +1 conversion happens HERE (hands/cards_played dicts
    keyed 0-3 are accepted directly, restore_state maps them seat+1)."""
    hands = {s: [int(c) for c in rec["hands"][str(s)]] for s in range(4)}
    played = {s: [int(c) for c in rec["cards_played"][str(s)]] for s in range(4)}
    live_trick = int(rec["control"]) == 0
    game = big2Game.restore_state(
        hands=hands,
        whose_turn=int(rec["turn"]) + 1,
        trick_cards=([int(c) for c in rec["trick_cards"]] if live_trick else None),
        trick_owner=(int(rec["trick_owner"]) + 1) if live_trick else None,
        passed_players=[int(s) + 1 for s in rec["passed"]],
        cards_played=played,
        must_play_club3=False,
    )
    env = Big2Env.__new__(Big2Env)
    env._game = game
    env._done = False
    return env


def rollout(rec, candidate, opponent, sample_candidate, seed):
    """Play one restored game to completion; return target-seat outcome."""
    torch.manual_seed(seed)
    env = make_env(rec)
    target = int(rec["seat"]) + 1            # engine player of the D2 seat
    rewards, steps = None, 0
    while not env.done:
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(f"rollout exceeded {MAX_STEPS} steps "
                               f"(game_id={rec['game_id']}, seed={seed})")
        p = env.current_player
        if p == target:
            if sample_candidate:
                action = candidate.act(env)[0]     # temperature-1 sample
            else:
                action = candidate.greedy(env)     # deployment mode
        else:
            action = opponent.greedy(env)          # opponents ALWAYS argmax
        rewards, _done = env.step(action)

    g = env.game
    # terminal card conservation check (engine invariant, asserted per rollout)
    n_rem = sum(int(g.currentHands[p].size) for p in range(1, 5))
    n_played = int(g.cardsPlayed.sum())
    if n_rem + n_played != 52:
        raise RuntimeError(f"card conservation violated at game end: "
                           f"remaining={n_rem} played={n_played} "
                           f"(game_id={rec['game_id']}, seed={seed})")
    rem = [int(c) for c in g.currentHands[target]]
    remaining = len(rem)
    return {
        "score": float(rewards[target - 1]),
        "remaining": remaining,
        "twos_left": sum(1 for c in rem if c >= 49),   # 2s are card ids 49-52
        "caged": remaining >= CAGE,
        "escaped": remaining < CAGE,
        "won": remaining == 0,
        "steps": steps,
    }


def _agg(outs):
    """Aggregate a flat list of rollout outcome dicts."""
    n = len(outs)
    caged = [o for o in outs if o["caged"]]
    hoard_caged = [o for o in caged if o["twos_left"] >= 1]
    return {
        "n_rollouts": n,
        "p_hoard_given_caged": (len(hoard_caged) / len(caged)) if caged else None,
        "n_caged": len(caged),
        "n_caged_hoarding": len(hoard_caged),
        "p_caged": len(caged) / n,
        "p_escaped": sum(o["escaped"] for o in outs) / n,
        "p_won": sum(o["won"] for o in outs) / n,
        "mean_score": float(np.mean([o["score"] for o in outs])),
        "mean_remaining": float(np.mean([o["remaining"] for o in outs])),
        "mean_twos_left": float(np.mean([o["twos_left"] for o in outs])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="candidate policy checkpoint")
    ap.add_argument("--reps", type=int, default=30,
                    help="sampled-candidate rollouts per position (default 30)")
    ap.add_argument("--opponents", default=DEFAULT_OPP,
                    help="frozen opponent checkpoint for the 3 other seats "
                         "(argmax; default policy_4500_best.pt)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--split", default="eval", choices=["eval", "train", "all"],
                    help="D2 pool split (default eval = 34 held-out positions)")
    ap.add_argument("--out", default="", help="write full results JSON here")
    args = ap.parse_args()

    device = "cpu"      # deterministic + tiny workload; never contends with the
                        # online campaign for accelerator/threads
    torch.set_num_threads(2)

    candidate = load_frozen_policy(args.ckpt, device)
    opponent = load_frozen_policy(args.opponents, device)

    pool = load_d2_pool(args.split)
    if args.split == "eval" and len(pool) != 34:
        print(f"ERROR: expected 34 eval-split D2 positions, got {len(pool)}",
              file=sys.stderr)
        sys.exit(2)
    print(f"loaded {len(pool)} D2 positions (split={args.split}) | "
          f"candidate={os.path.basename(args.ckpt)} | "
          f"opponents={os.path.basename(args.opponents)} argmax | "
          f"reps={args.reps} seed_base={args.seed_base}")

    # sanity pass: every position restores and its legal-action count matches
    # the build-time value recorded in the pool
    n_legal_mismatch = 0
    for rec in pool:
        env = make_env(rec)
        n_legal = int(env.get_valid_actions().sum())
        if n_legal != int(rec["n_legal"]):
            n_legal_mismatch += 1
            print(f"  WARNING n_legal mismatch game_id={rec['game_id']} "
                  f"event_idx={rec['event_idx']}: engine {n_legal} != "
                  f"pool {rec['n_legal']}", file=sys.stderr)
    print(f"sanity: all {len(pool)} positions restored "
          f"(card conservation enforced by restore_state); "
          f"n_legal mismatches: {n_legal_mismatch}")

    per_position, argmax_outs, sampled_outs = [], [], []
    t0 = time.time()
    for r_idx, rec in enumerate(pool):
        seed0 = args.seed_base + 100000 * r_idx
        # argmax variant: reps are identical under all-argmax play -> 1 pass
        am = rollout(rec, candidate, opponent, sample_candidate=False,
                     seed=seed0 + 99999)
        argmax_outs.append(am)
        # sampled variant: candidate samples (T=1), opponents stay argmax
        sp = [rollout(rec, candidate, opponent, sample_candidate=True,
                      seed=seed0 + k) for k in range(args.reps)]
        sampled_outs.extend(sp)
        per_position.append({
            "game_id": rec["game_id"],
            "event_idx": rec["event_idx"],
            "seat": rec["seat"],
            "n_twos_at_snapshot": rec["n_twos_remaining"],
            "source_final_remaining": rec["final_remaining"],
            "argmax": am,
            "sampled": _agg(sp),
        })
    dt = time.time() - t0

    result = {
        "tool": "eval_injected_d2",
        "methodology": (
            "34 held-out eval-split D2 positions restored mid-game via "
            "big2Game.restore_state; candidate at the record's target seat, 3 "
            "opponents argmax. 'argmax' = 1 deterministic pass/position with "
            "the candidate also argmax (deployment mode; hides distribution). "
            "'sampled' = reps rollouts/position with ONLY the candidate "
            "sampling at temperature 1 from its policy distribution — measures "
            "the policy's probability mass on hoard vs dump, which argmax "
            "mode-collapse hides. Deterministic seeding per (record, rep): "
            "paired across candidates at equal --seed-base. Luck baseline does "
            "not apply here (raw engine scores)."),
        "candidate_ckpt": os.path.abspath(args.ckpt),
        "opponents_ckpt": os.path.abspath(args.opponents),
        "split": args.split,
        "n_positions": len(pool),
        "reps": args.reps,
        "seed_base": args.seed_base,
        "cage_threshold": CAGE,
        "n_legal_mismatches": n_legal_mismatch,
        "elapsed_sec": round(dt, 1),
        "argmax": _agg(argmax_outs),
        "sampled": _agg(sampled_outs),
        "per_position": per_position,
    }

    def _fmt(name, a):
        h = ("n/a" if a["p_hoard_given_caged"] is None
             else f"{a['p_hoard_given_caged']*100:.1f}% "
                  f"({a['n_caged_hoarding']}/{a['n_caged']})")
        print(f"  {name:7s} | P(hoard|caged) {h:16s} | mean score "
              f"{a['mean_score']:+.2f} | P(escaped cage) {a['p_escaped']*100:.1f}% "
              f"| P(won) {a['p_won']*100:.1f}% | mean 2s left "
              f"{a['mean_twos_left']:.2f} | n={a['n_rollouts']}")

    print(f"done in {dt:.1f}s")
    _fmt("argmax", result["argmax"])
    _fmt("sampled", result["sampled"])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
