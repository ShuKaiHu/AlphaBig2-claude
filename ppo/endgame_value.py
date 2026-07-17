"""V10 value via terminal backward induction (best-play), NOT Monte-Carlo.

Near the terminal Big2 is a small PERFECT-INFO game (showdown reveals all hands),
so we can SOLVE it exactly by max^n backward induction: at every node the acting
player picks the child maximizing THEIR OWN final-score dimension; values back up
from the true terminal scores. That yields a clean best-play value label that is
independent of any single game's luck (the Monte-Carlo target overfit at 578
games because it copied one noisy outcome to every state).

This module:
  solve(env, memo)         -> exact 4-dim best-play value (with transposition memo)
  gen_records(...)         -> (encode_static, encode_opp_hands, tanh(value)) labels
                              from many independently sampled small endgames
"""
import time
import numpy as np

from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_opp_hands

REWARD_SCALE = 13.0


def _state_key(g):
    """Hashable full-info state: hands + whose turn + to-beat + who passed.
    These fully determine future legal play / outcome (cardsPlayed only affects
    features, not legality)."""
    to_beat = None
    if g.control == 0 and (g.goIndex - 1) in g.handsPlayed:
        to_beat = tuple(sorted(int(c) for c in g.handsPlayed[g.goIndex - 1].hand))
    return (
        tuple(sorted(int(c) for c in g.currentHands[1])),
        tuple(sorted(int(c) for c in g.currentHands[2])),
        tuple(sorted(int(c) for c in g.currentHands[3])),
        tuple(sorted(int(c) for c in g.currentHands[4])),
        int(g.playersGo), int(g.control), to_beat,
        tuple(sorted(p for p in range(1, 5) if g.passedThisRound.get(p))),
    )


def solve(env, memo):
    """Exact max^n best-play value (4-dim, ABSOLUTE player) of `env`'s state."""
    key = _state_key(env.game)
    cached = memo.get(key)
    if cached is not None:
        return cached
    acting = env.current_player
    valid = np.flatnonzero(env.get_valid_actions())
    if len(valid) == 0:
        valid = np.array([PASS_IDX])
    best = None
    for a in valid:
        e2 = env.clone()
        rewards, done = e2.step(int(a))
        v = np.asarray(rewards, dtype=np.float64) if done else solve(e2, memo)
        if best is None or v[acting - 1] > best[acting - 1]:
            best = v
    memo[key] = best
    return best


def _random_endgame(k_total, rng):
    """A valid full-info position reached by random legal play, stopped once the
    total cards left across all four hands is <= k_total."""
    env = Big2Env(); env.reset()
    while not env.done:
        tot = sum(env.game.currentHands[p].size for p in range(1, 5))
        if tot <= k_total:
            return env
        v = np.flatnonzero(env.get_valid_actions())
        env.step(int(rng.choice(v)) if len(v) else PASS_IDX)
    return None


def gen_records(n, k_total=10, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    records, solved, t0 = [], 0, time.time()
    pid = 0
    while len(records) < n:
        env = _random_endgame(k_total, rng)
        if env is None:
            continue
        acting = env.current_player
        memo = {}
        val = solve(env, memo)                      # exact best-play 4-dim
        g = env.game
        records.append({
            "static": encode_static(g, acting).astype(np.float32),
            "opp": encode_opp_hands(g, acting).astype(np.float32),
            "target": np.tanh(val / REWARD_SCALE).astype(np.float32),
            "pid": pid,
        })
        pid += 1
        solved += 1
        if verbose and solved % 200 == 0:
            print(f"  solved {solved}/{n}  ({(time.time()-t0)/solved*1000:.0f} ms/pos)")
    if verbose:
        print(f"generated {len(records)} solved endgames (k<= {k_total}) "
              f"in {time.time()-t0:.0f}s")
    return records


if __name__ == "__main__":
    # correctness + speed check: memoized solve must equal a no-memo solve,
    # and we time a few endgames.
    rng = np.random.default_rng(1)

    def solve_nomemo(env):
        acting = env.current_player
        valid = np.flatnonzero(env.get_valid_actions())
        if len(valid) == 0:
            valid = np.array([PASS_IDX])
        best = None
        for a in valid:
            e2 = env.clone(); rw, done = e2.step(int(a))
            v = np.asarray(rw, float) if done else solve_nomemo(e2)
            if best is None or v[acting - 1] > best[acting - 1]:
                best = v
        return best

    for k in (8, 10, 12):
        env = _random_endgame(k, rng)
        if env is None:
            continue
        memo = {}
        t0 = time.time(); v_m = solve(env, memo); t_m = time.time() - t0
        t0 = time.time(); v_n = solve_nomemo(env); t_n = time.time() - t0
        ok = np.allclose(v_m, v_n)
        print(f"k<={k}: memo {t_m*1000:.0f}ms ({len(memo)} states) | "
              f"nomemo {t_n*1000:.0f}ms | match={ok} | value={np.round(v_m,1)}")
