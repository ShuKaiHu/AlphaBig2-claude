"""Is the policy too 'peaked' at multi-option decisions → suppressing MCTS search?

Hypothesis (user): when it's our turn with MANY legal options, PPO_V4 puts almost
all probability on one action, so the PUCT prior dominates and the search can't
explore → it just follows greedy → that's why search ≈ policy.

For every decision in greedy self-play we record, for the to-move legal set:
  - n_legal
  - max policy prob   (1.0 = all-in on one action = very peaked)
  - effective #actions = exp(entropy)  (1 = degenerate, n_legal = uniform)
At decisions with n_legal >= --thresh we also run the ISMCTS and check whether its
choice differs from greedy (override). Reported by n_legal bucket.

    python probe_policy_concentration.py --games 25 --thresh 5 --sims 60
"""
import bootstrap  # noqa: F401

import argparse
import numpy as np

import eval_alpha1 as E

E._LEAF, E._USE_BELIEF = "rollout", True   # rollout-leaf ISMCTS for the override check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    rows = []   # dict per decision
    for g in range(args.games):
        env = E.Big2Env(); env.reset(); steps = 0
        while not env.done and steps < 300:
            p = env.current_player
            legal, probs = E._policy_eval(env)
            n = len(legal)
            if n > 1:
                pr = np.clip(probs, 1e-12, 1.0)
                eff = float(np.exp(-(pr * np.log(pr)).sum()))
                overrode = None
                if n >= args.thresh and args.sims > 0:
                    best, _ = E.alpha1_action(env, p, args.sims, args.cpuct, return_visits=True)
                    overrode = int(int(best) != int(legal[int(probs.argmax())]))
                rows.append({"n": n, "maxp": float(probs.max()), "eff": eff, "ovr": overrode})
            env.step(int(legal[int(probs.argmax())]))
            steps += 1

    print(f"decisions with >1 legal: {len(rows)} over {args.games} games "
          f"(search run at n_legal>={args.thresh}, sims={args.sims})\n")
    print(f"{'n_legal':<9}{'count':>7}{'mean_maxp':>11}{'mean_eff':>10}{'override%':>11}")
    for name, lo, hi in [("2-3", 2, 4), ("4-6", 4, 7), ("7-10", 7, 11), ("11-20", 11, 21), ("21+", 21, 9999)]:
        sub = [r for r in rows if lo <= r["n"] < hi]
        if not sub:
            continue
        mp = np.mean([r["maxp"] for r in sub])
        ef = np.mean([r["eff"] for r in sub])
        ovs = [r["ovr"] for r in sub if r["ovr"] is not None]
        ovstr = f"{100*np.mean(ovs):.1f}% (n={len(ovs)})" if ovs else "—"
        print(f"{name:<9}{len(sub):>7}{mp:>11.2f}{ef:>10.1f}{ovstr:>11}")
    allmp = np.mean([r["maxp"] for r in rows])
    allov = [r["ovr"] for r in rows if r["ovr"] is not None]
    print(f"\nOVERALL: mean max_prob {allmp:.2f} | overall override (n_legal>={args.thresh}) "
          f"{100*np.mean(allov):.1f}% of {len(allov)} decisions")


if __name__ == "__main__":
    main()
