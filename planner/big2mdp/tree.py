"""Route tree — the converging-reward core of Big2MDP (eq 4-9), replacing the
one-step aggregate scoring that the v0.2 smoke test falsified.

Mechanism. A Big2 win is a ROUTE: play a set, keep the free-playing right,
lead the next set, ... until the hand is empty (the FINAL set needs no hold —
playing your last cards ends the game instantly). In round abstraction the
route's success probability therefore CONVERGES FROM THE TERMINAL STATE as a
product of per-set hold-control probabilities R_c, with the weakest remaining
set scheduled last:

    p_route(a) = R_c(a) × Π_{k ∈ partition(hand−a), k ≠ weakest} R_c(k)
    (if a empties the hand: p_route = 1 — immediate win)

R_c per set comes from MP retrieval (StatsStore feature-OR matching) with the
optional exact floor (guaranteed_hold → 1.0). This is the paper's prediction
tree collapsed analytically: opponents' predicted reactions enter through
R_c ("all three passed in similar states"), interception = the complement.

Route rewards:
  R_win: if the route holds, opponents pass throughout and shed nothing →
         win score ≈ (sum of opponents' current cards) × finish multiplier of
         the LAST set (×2 bomb, ×2 principal two — engine-exact rules).
  R_lose: interception estimate — the store's aggregate R_lose for the acting
          context when available, else −(cards left after playing a), doubled
          per held 2 (the loser-side multiplier that stop-loss play manages).
"""
from collections import Counter

from planner.big2mdp.features import PASS_KEY, action_key
from planner.control import guaranteed_hold
from planner.decompose import partition_hand


def _rank(c):
    return (c - 1) // 4 + 1


def finish_multiplier(kind, cards):
    """Engine-exact winner finishing multiplier (big2Game reverse-engineered
    rules): ×2 for a bomb finish, ×2 for a PRINCIPAL two in the finishing
    combo (incidental twos — wheel A2345, FH pair, quad kicker — don't count)."""
    mult = 1
    if kind in ("quad", "straight_flush"):
        mult *= 2
    ranks = [_rank(c) for c in cards]
    if 13 in ranks:
        incidental = False
        if kind == "straight" or kind == "straight_flush":
            incidental = set(ranks) == {10, 11, 12, 13, 1} or \
                set(ranks) == {12, 13, 1, 2, 3}          # A2345 wheel forms
        if kind == "full_house":
            incidental = Counter(ranks)[13] == 2          # 2s are the pair
        if kind == "quad":
            incidental = Counter(ranks)[13] == 1          # 2 is the kicker
        if not incidental:
            mult *= 2
    return mult


class RoutePlanner:
    """Per-decision route evaluation over the candidate legal actions."""

    def __init__(self, store, exact_floor=True, min_support=32):
        self.store = store
        self.exact_floor = exact_floor
        self.min_support = min_support

    def rc(self, feats, kind, cards, unseen):
        if self.exact_floor and guaranteed_hold(kind, cards, unseen):
            return 1.0
        st = self.store.stats(feats, action_key(cards))
        if st["n"] >= self.min_support:
            return st["r_c"]
        # no reliable data: a set that is not provably safe defaults to a
        # weak prior scaled by its top rank (a ♠A lead usually holds, a 4
        # never does) — bounded well below the SP/WC thresholds.
        top = max(cards)
        return 0.35 * (top / 52.0)

    # ── control-regain recursion ────────────────────────────────────────────
    # V(sets, in_control): probability of emptying the hand.
    #   leading:   play set k → hold with rc(k) and lead again, or lose the
    #              lead (cards are shed either way) and continue from "following"
    #   following: regain by topping the table with set j (proxy: rc(j) — the
    #              same "nobody beats this" quantity), then lead the rest; each
    #              spell without control risks an opponent going out first —
    #              modelled by a survival factor from the smallest opponent
    #              hand (documented approximation of the paper's tree, which
    #              carries this through predicted opponent moves).
    def _V(self, sets, in_control, rcs, surv, memo):
        if not sets:
            return 1.0
        key = (sets, in_control)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = 0.0
        for k in range(len(sets)):
            rest = sets[:k] + sets[k + 1:]
            r = rcs[sets[k]]
            if in_control:
                v = r * self._V(rest, True, rcs, surv, memo) + \
                    (1.0 - r) * surv * self._V(rest, False, rcs, surv, memo)
            else:
                v = r * surv * self._V(rest, True, rcs, surv, memo)
            if v > best:
                best = v
        memo[key] = best
        return best

    def hand_value(self, hand, feats, unseen, in_control, min_opp):
        """Win probability for the whole hand from here (route tree root)."""
        sets = tuple(sorted(partition_hand(hand)))
        rcs = {s: self.rc(feats, s[0], list(s[1]), unseen) for s in sets}
        surv = min(1.0, max(0.15, min_opp / 8.0))
        return self._V(sets, in_control, rcs, surv, {})

    def route(self, hand, feats, unseen, cards_played, opp_sum, twos_after,
              min_opp=13):
        """Value of playing `cards_played` now from `hand`.
        Returns (p_route, r_win, r_lose)."""
        rest = [c for c in hand if c not in cards_played]
        kind = action_key(cards_played)[1]
        if not rest:                                     # final play → instant win
            return 1.0, opp_sum * finish_multiplier(kind, cards_played), 0.0
        p_hold = self.rc(feats, kind, list(cards_played), unseen)
        sets_rest = tuple(sorted(partition_hand(rest)))
        rcs = {s: self.rc(feats, s[0], list(s[1]), unseen) for s in sets_rest}
        surv = min(1.0, max(0.15, min_opp / 8.0))
        memo = {}
        p = p_hold * self._V(sets_rest, True, rcs, surv, memo) + \
            (1.0 - p_hold) * surv * self._V(sets_rest, False, rcs, surv, memo)
        weakest = min(sets_rest, key=lambda s: rcs[s])
        r_win = opp_sum * finish_multiplier(weakest[0], list(weakest[1]))
        # interception loss estimate: cards stranded ≈ rest, doubled per held 2
        r_lose = -float(len(rest)) * (2.0 ** twos_after if twos_after else 1.0)
        st = self.store.stats(feats, action_key(cards_played))
        if st["n"] >= self.min_support and st["r_lose"] < 0:
            r_lose = st["r_lose"]                        # data beats heuristic
        return p, r_win, r_lose
