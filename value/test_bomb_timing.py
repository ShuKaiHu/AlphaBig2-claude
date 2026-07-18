"""Diagnostic: when the acting player holds a LEGAL quad/straight-flush play right
now, does playing it actually score better (by rollout ground truth) than holding it
(playing something else, or passing)? Answers: is the current ~40-48% bomb-usage
rate a real weakness (bomb was the best move, policy missed it) or good judgment
(holding scored better, matching the "bombs are a threat, not an obligation" theory)?

Methodology: walk real games via greedy policy play (representative trajectory).
At each decision where the acting player has >=1 legal quad/SF action, sample
K worlds and rollout-average EVERY legal action (same ground-truth methodology as
eval_alpha1.measure_disagreement). Classify:
  A. bomb is roll-optimal AND policy played it      -> correctly bombed
  B. bomb is roll-optimal AND policy held           -> missed opportunity (real gap)
  C. bomb is NOT roll-optimal AND policy held       -> correctly held
  D. bomb is NOT roll-optimal AND policy played it  -> unnecessary bomb use

    python test_bomb_timing.py --n 30 --k-worlds 10
"""
import bootstrap  # noqa: F401

import argparse
import time
from collections import Counter

import numpy as np

import eval_alpha1 as E
from engine.env import Big2Env
import enumerateOptions

E._USE_BELIEF = True

_WIN = [tuple(range(s, s + 5)) for s in range(1, 9)] + [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]


def _rank(c):
    return (c - 1) // 4 + 1


def _suit(c):
    return (c - 1) % 4 + 1


def classify_five(cards):
    rc = Counter(_rank(c) for c in cards)
    is_quad = any(n == 4 for n in rc.values())
    suits = set(_suit(c) for c in cards)
    ranks = tuple(sorted(_rank(c) for c in cards))
    is_straight = ranks in _WIN
    is_flush = len(suits) == 1
    if is_quad:
        return "quad"
    if is_flush and is_straight:
        return "sf"
    return "other"


def _playout(e, me, search_sims, cpuct):
    """Continue a determinized world to terminal. If search_sims>0, MY future turns
    are decided by a full ISMCTS search (search_sims sims each) instead of plain
    greedy -- a deeper, less self-referential "ground truth" than pure greedy
    rollout (opponents still play greedy -- matches this codebase's standing
    modeling assumption that we search our own decisions, not opponents')."""
    steps = 0
    while not e.done and steps < 200:
        if search_sims > 0 and e.current_player == me:
            e.step(E.alpha1_action(e, me, search_sims, cpuct))
        else:
            e.step(E._greedy(e))
        steps += 1
    return float(np.tanh(e.game.rewards[me - 1] / E.SCALE))


def find_bomb_decisions(n_target, k_worlds, search_sims=0, cpuct=1.5, seed=0, verbose=True):
    """Walk greedy-policy games; whenever the acting player has a legal bomb play,
    rollout-average EVERY legal action over k_worlds shared worlds, and record which
    action(s) are bombs, which is roll-optimal, and what the policy actually chose."""
    np.random.seed(seed)
    results = []
    env = Big2Env(); env.reset()
    while len(results) < n_target:
        me = env.current_player
        legal, probs = E._policy_eval(env)
        bomb_idx = []
        for i, a in enumerate(legal):
            a = int(a)
            if a == E.PASS_IDX:
                continue
            cids, n = enumerateOptions.getOptionNC(a)
            if n == 5 and classify_five(cids) in ("quad", "sf"):
                bomb_idx.append(i)
        if bomb_idx:
            belief = E.belief_from_obs(E._belief, E.obs_from_env(env), E.DEV) if E._USE_BELIEF else None
            worlds = [E._determinize(env, me, belief) for _ in range(k_worlds)]
            roll_avg = {}
            for a in legal:
                a = int(a); rvals = []
                for w in worlds:
                    e = E._make_world(env, w); e.step(a)
                    rvals.append(_playout(e, me, search_sims, cpuct))
                roll_avg[a] = float(np.mean(rvals))
            roll_top = max(roll_avg, key=roll_avg.get)
            bomb_actions = set(int(legal[i]) for i in bomb_idx)
            pol_top = int(legal[int(np.argmax(probs))])
            results.append({
                "n_legal": int(len(legal)), "bomb_actions": bomb_actions,
                "roll_top": roll_top, "roll_avg": roll_avg, "pol_top": pol_top,
                "bomb_is_roll_optimal": roll_top in bomb_actions,
                "policy_played_bomb": pol_top in bomb_actions,
            })
            if verbose:
                print(f"  bomb decision {len(results)}/{n_target} (n_legal={len(legal)})")
        env.step(E._greedy(env))
        if env.done:
            env = Big2Env(); env.reset()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k-worlds", type=int, default=10)
    ap.add_argument("--search-sims", type=int, default=0,
                    help="if >0, MY future turns during the playout use ISMCTS search (this many "
                         "sims) instead of plain greedy -- a deeper, less self-referential ground truth")
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    results = find_bomb_decisions(args.n, args.k_worlds, args.search_sims, args.cpuct, args.seed)
    print(f"\n({time.time()-t0:.0f}s)")

    cats = Counter()
    for r in results:
        if r["bomb_is_roll_optimal"] and r["policy_played_bomb"]:
            cats["A_correctly_bombed"] += 1
        elif r["bomb_is_roll_optimal"] and not r["policy_played_bomb"]:
            cats["B_missed_opportunity"] += 1
        elif not r["bomb_is_roll_optimal"] and not r["policy_played_bomb"]:
            cats["C_correctly_held"] += 1
        else:
            cats["D_unnecessary_bomb"] += 1

    n = len(results)
    print(f"=== bomb timing diagnostic (n={n} decisions, k={args.k_worlds} worlds) ===")
    for k in ("A_correctly_bombed", "B_missed_opportunity", "C_correctly_held", "D_unnecessary_bomb"):
        c = cats.get(k, 0)
        print(f"  {k:22s}: {c}/{n} = {100*c/n:.1f}%")
    bomb_optimal = sum(1 for r in results if r["bomb_is_roll_optimal"])
    policy_bombed = sum(1 for r in results if r["policy_played_bomb"])
    print(f"\n  rollout says bomb IS optimal : {bomb_optimal}/{n} = {100*bomb_optimal/n:.1f}%")
    print(f"  policy actually played bomb  : {policy_bombed}/{n} = {100*policy_bombed/n:.1f}%")

    print("\n--- missed opportunities (bomb was best, policy held) ---")
    for r in results:
        if r["bomb_is_roll_optimal"] and not r["policy_played_bomb"]:
            gap = r["roll_avg"][r["roll_top"]] - r["roll_avg"][r["pol_top"]]
            print(f"  n_legal={r['n_legal']} gap={gap:+.3f} roll_avg={ {k: round(v,3) for k,v in r['roll_avg'].items()} }")


if __name__ == "__main__":
    main()
