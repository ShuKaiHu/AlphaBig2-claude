"""Advantage-weighted BC (plan M1): per-(game, human-seat) sample weights.

Idea: keep imitating humans, but upweight the humans who scored ABOVE what
their starting hand alone predicts.  The hand-luck baseline
    baseline(hand) = E[human score | min_plays, n_twos, has_bomb]
is deliberately tiny (3 deal features) — it only removes the deal component,
not skill.  Then per (game, human seat):

    advantage = actual_score - baseline(starting hand)
    w         = clip(exp(advantage / T), lo, hi),  normalized to mean 1

Every BC decision of that seat in that game shares the weight (game-level
credit assignment — plan section 1, known limitation).

Pure numpy + stdlib; no torch/engine imports, so train_bc can import it
cheaply.  Baseline: binned cell means where a (mp, n_twos, bomb) cell has
>= min_count rows, additive ridge on one-hots as fallback for thin cells.

Usage:
    games  = load_games(path)                      # snapshot of online_games.jsonl
    rows   = human_seat_rows(games)                # 3 human seats per game
    base   = fit_baseline(rows)                    # luck model, base.r2
    wmap, diag = compute_weights(rows, base, T=20) # {(game_id, seat): w}
"""
import json
from collections import Counter

import numpy as np

CLIP_DEFAULT = (0.2, 5.0)

# ---------------------------------------------------------------- hand features
# Card id = rank_idx*4 + suit_idx + 1;  ranks 3..2 -> idx 0..12;  suits C,D,H,S.

# min_plays_to_empty copied verbatim from AlphaBig2-Value/hand_features.py
# (greedy decomposition: fives, then pairs, then singles — an upper bound).
_WIN = [tuple(range(s, s + 5)) for s in range(1, 9)] + [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]


def _rank(c):
    return (c - 1) // 4 + 1


def _extract_five(rem):
    """Return one valid 5-card combo (quad+kicker / full house / straight) or None."""
    rc = Counter(_rank(c) for c in rem)
    for r, n in rc.items():                                  # four-of-a-kind + kicker
        if n >= 4:
            quad = [c for c in rem if _rank(c) == r][:4]
            kicker = [c for c in rem if c not in quad][:1]
            if kicker:
                return quad + kicker
    trips = [r for r, n in rc.items() if n >= 3]             # full house
    for t in trips:
        for r, n in rc.items():
            if r != t and n >= 2:
                return [c for c in rem if _rank(c) == t][:3] + [c for c in rem if _rank(c) == r][:2]
    have = {}                                                # straight (covers straight-flush too)
    for c in rem:
        have.setdefault(_rank(c), []).append(c)
    for w in _WIN:
        if all(r in have for r in w):
            return [have[r][0] for r in w]
    return None


def min_plays_to_empty(cards):
    rem = [int(c) for c in cards]
    plays = 0
    while True:
        five = _extract_five(rem)
        if not five:
            break
        for c in five:
            rem.remove(c)
        plays += 1
    rc = Counter(_rank(c) for c in rem)                      # pairs
    for r in rc:
        while sum(1 for c in rem if _rank(c) == r) >= 2:
            for c in [c for c in rem if _rank(c) == r][:2]:
                rem.remove(c)
            plays += 1
    return plays + len(rem)                                  # leftover singles


def n_twos(cards):
    """Count of 2s (rank idx 12 -> card ids 49..52)."""
    return sum(1 for c in cards if int(c) >= 49)


# Straight-flush windows in rank-idx terms: 3-7 .. 10-A, A-2-3-4-5, 2-3-4-5-6.
# J-Q-K-A-2 is NOT a straight.
_SF_WINDOWS = [tuple(range(s, s + 5)) for s in range(0, 8)] + \
              [(11, 12, 0, 1, 2), (12, 0, 1, 2, 3)]


def has_bomb(cards):
    """Bomb in a starting hand: quad, or 5 same-suit cards in a legal window."""
    ranks = [(int(c) - 1) // 4 for c in cards]
    if any(n >= 4 for n in Counter(ranks).values()):
        return True
    by_suit = {}
    for c in cards:
        by_suit.setdefault((int(c) - 1) % 4, set()).add((int(c) - 1) // 4)
    for rs in by_suit.values():
        if len(rs) >= 5 and any(all(r in rs for r in w) for w in _SF_WINDOWS):
            return True
    return False


# ---------------------------------------------------------------- data loading

def load_games(path):
    return [json.loads(l) for l in open(path)]


def game_key(g):
    """Same id build_records games carry ('id' field; parse_online_games.game_id)."""
    gid = g.get("id")
    if gid is None:
        from ppo.parse_online_games import game_id as _gid
        gid = _gid(g)
    return gid


def human_seat_rows(games):
    """One row per (game, human seat): starting-hand features + final score."""
    rows = []
    for g in games:
        gid = game_key(g)
        our = g["our_seat"]
        scores = {int(k): float(v) for k, v in g["scores"].items()}
        for s in range(4):
            if s == our:
                continue
            hand = list(map(int, g["hands"][str(s)]))
            rows.append({
                "gid": gid, "seat": s,
                "mp": min_plays_to_empty(hand),
                "twos": n_twos(hand),
                "bomb": int(has_bomb(hand)),
                "score": scores[s],
                "won": int(g["winner"] == s),
            })
    return rows


# ---------------------------------------------------------------- luck baseline

MP_LO, MP_HI = 4, 10      # mp clipped into [4,10] for binning (tails are thin)
TWOS_HI = 3               # n_twos clipped at 3 (4 twos is ~1e-4 of hands)
MIN_CELL = 30             # cells with fewer rows fall back to the ridge model


class Baseline:
    """binned-mean-with-ridge-fallback E[score | mp, twos, bomb]."""

    def __init__(self, cell_mean, cell_n, ridge_coef, r2, mean_score):
        self.cell_mean = cell_mean        # {(mpc, twc, bomb): mean}
        self.cell_n = cell_n              # {(mpc, twc, bomb): count}
        self.ridge_coef = ridge_coef      # (nfeat,) additive one-hot coefs
        self.r2 = r2
        self.mean_score = mean_score

    @staticmethod
    def _clipkey(mp, twos, bomb):
        return (int(np.clip(mp, MP_LO, MP_HI)), min(int(twos), TWOS_HI), int(bomb))

    @staticmethod
    def _onehot(mp, twos, bomb):
        x = np.zeros(_NFEAT)
        x[0] = 1.0                                        # intercept
        x[1 + int(np.clip(mp, MP_LO, MP_HI)) - MP_LO] = 1.0
        x[1 + (MP_HI - MP_LO + 1) + min(int(twos), TWOS_HI)] = 1.0
        x[-1] = float(bomb)
        return x

    def predict(self, mp, twos, bomb):
        key = self._clipkey(mp, twos, bomb)
        if self.cell_n.get(key, 0) >= MIN_CELL:
            return self.cell_mean[key]
        return float(self._onehot(mp, twos, bomb) @ self.ridge_coef)


_NFEAT = 1 + (MP_HI - MP_LO + 1) + (TWOS_HI + 1) + 1      # intercept+mp+twos+bomb


def fit_baseline(rows, lam=1.0):
    y = np.array([r["score"] for r in rows], dtype=np.float64)
    keys = [Baseline._clipkey(r["mp"], r["twos"], r["bomb"]) for r in rows]
    cell_sum, cell_n = {}, {}
    for k, v in zip(keys, y):
        cell_sum[k] = cell_sum.get(k, 0.0) + v
        cell_n[k] = cell_n.get(k, 0) + 1
    cell_mean = {k: cell_sum[k] / cell_n[k] for k in cell_n}

    X = np.stack([Baseline._onehot(r["mp"], r["twos"], r["bomb"]) for r in rows])
    A = X.T @ X + lam * np.eye(_NFEAT)
    A[0, 0] -= lam                                        # don't shrink the intercept
    coef = np.linalg.solve(A, X.T @ y)

    b = Baseline(cell_mean, cell_n, coef, r2=0.0, mean_score=float(y.mean()))
    pred = np.array([b.predict(r["mp"], r["twos"], r["bomb"]) for r in rows])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    b.r2 = 1.0 - ss_res / ss_tot
    return b


# ---------------------------------------------------------------- weights

def compute_weights(rows, baseline, T, clip=CLIP_DEFAULT):
    """Returns ({(gid, seat): weight}, diagnostics).  Weights mean-1 normalized
    over the rows given (i.e. over the training corpus of seats)."""
    lo, hi = clip
    adv = np.array([r["score"] - baseline.predict(r["mp"], r["twos"], r["bomb"])
                    for r in rows], dtype=np.float64)
    raw = np.exp(adv / T)
    clipped = np.clip(raw, lo, hi)
    w = clipped / clipped.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    order = np.argsort(w)[::-1]
    top10 = order[: max(1, len(w) // 10)]
    diag = {
        "T": T, "clip": clip, "n": len(w),
        "ess": ess, "ess_frac": ess / len(w),
        "clip_lo_rate": float((raw <= lo).mean()),
        "clip_hi_rate": float((raw >= hi).mean()),
        "deciles": {f"p{p}": float(np.percentile(w, p)) for p in
                    (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)},
        "top10pct_weight_share": float(w[top10].sum() / w.sum()),
        "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
    }
    wmap = {(r["gid"], r["seat"]): float(w[i]) for i, r in enumerate(rows)}
    return wmap, diag
