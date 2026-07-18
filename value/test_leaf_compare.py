"""Compare 3 ISMCTS leaves on identical positions + identical sampled worlds:
  rollout    — V4-greedy to terminal (slow, lowest-bias → treat as GOLD)
  value      — plain VALUE.pt
  value_aug  — dominance-aware value (VALUE + P(nuts) + min-plays)

At every multi-option decision of our seat (trajectory driven by V4 greedy, so the
visited positions are leaf-independent), we run all 3 leaves with the SAME RNG seed
→ each leaf samples the IDENTICAL sequence of belief worlds, so any difference in the
chosen action comes PURELY from the leaf. We report (a) ms/decision and (b) action
agreement, using rollout as the gold standard:
    value_aug agreeing with rollout MORE than value does  ==  the features made the
    fast value leaf behave more like the expensive gold leaf.

    python test_leaf_compare.py --games 24 --sims 64
"""
import bootstrap  # noqa: F401

import argparse
import time

import numpy as np

import eval_alpha1 as E
from engine.env import Big2Env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    E._load_aug()
    leaves = ["rollout", "value", "value_aug"]
    choices = {lf: [] for lf in leaves}
    times = {lf: 0.0 for lf in leaves}
    ndec = 0
    t_start = time.time()

    for gi in range(args.games):
        my_seat = (gi % 4) + 1
        np.random.seed(args.seed * 100000 + gi)
        env = Big2Env(); env.reset()
        steps = 0
        while not env.done and steps < 300:
            p = env.current_player
            if p == my_seat:
                legal, _ = E.legal_action_data(env)
                if len(legal) > 1:
                    node_seed = args.seed * 7919 + ndec * 13
                    for lf in leaves:                       # identical worlds per leaf
                        E._LEAF = lf
                        np.random.seed(node_seed)
                        t0 = time.time()
                        a = E.alpha1_action(env, p, args.sims, args.cpuct)
                        times[lf] += time.time() - t0
                        choices[lf].append(int(a))
                    ndec += 1
                env.step(E._greedy(env))                     # leaf-independent trajectory
            else:
                env.step(E._greedy(env))
            steps += 1
        print(f"game {gi+1}/{args.games} | decisions {ndec} | {(time.time()-t_start)/60:.1f} min")

    def agree(x, y):
        return 100.0 * np.mean([a == b for a, b in zip(choices[x], choices[y])])

    print(f"\n=== leaf comparison ({ndec} multi-option decisions, sims={args.sims}) ===")
    print("speed:")
    for lf in leaves:
        print(f"  {lf:10s} {1000*times[lf]/max(ndec,1):7.1f} ms/decision  "
              f"({times['rollout']/max(times[lf],1e-9):.1f}x faster than rollout)")
    print("\nagreement (rollout = gold standard):")
    print(f"  value      vs rollout : {agree('value','rollout'):5.1f}%")
    print(f"  value_aug  vs rollout : {agree('value_aug','rollout'):5.1f}%   "
          f"<- want this HIGHER than the line above")
    print(f"  value_aug  vs value   : {agree('value_aug','value'):5.1f}%   (how often the feature changed the move)")


if __name__ == "__main__":
    main()
