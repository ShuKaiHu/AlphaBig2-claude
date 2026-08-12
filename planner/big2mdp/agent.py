"""Faithful Big2MDP agent (ToG 2025) on this engine's exact 神來也 rules.

Plays EVERY move itself — no policy / belief / value nets, no human data, no
legacy assets. Difficulty levels mirror the paper's app: 1=MDP1.0 (Rookie),
2=MDP2.0 (Normal), 3=MDP3.0 (Expert), 4=MDP4.0+WP/MP/WC/SP (Master).

v0.3: decision core = ROUTE TREE (planner/big2mdp/tree.py) — the paper's
converging-reward recursion (eq 4-9) in round abstraction — replacing the
one-step aggregate scoring that the v0.2 smoke test falsified (marginal
win-correlations burned bombs/2s early).

Heads over route values (p, r_win, r_lose):
  MDP1.0 (eq 4-6):  Q = p × r_win                       (aggressive only)
  MDP2.0 (eq 7-9):  + S_end switch (4/4/30): endgame with no winning route →
                    conservative head max (1−p) × r_lose (stop-loss, dump 2s)
  MDP3.0 (eq 10-13): aggressive Q scaled by W = 1 − d/dmax (d = sets left)
  MDP4.0 (eq 14-16): Q = p (pure win-rate route), same S_end switch, plus
  strategies WC (eq 19-24, .8/.2/.8) and SP (eq 25-27, .8/.1) — priority
  SP → WC → heads. PASS is scored as "protect the plan": full-hand route
  probability × a tempo discount (documented approximation).
"""
import numpy as np

import enumerateOptions
from planner.big2mdp.features import PASS_KEY, action_key, state_features, table_level
from planner.big2mdp.store import StatsStore
from planner.big2mdp.tree import RoutePlanner
from planner.control import unseen_cards
from planner.decompose import partition_hand

PASS_IDX = enumerateOptions.passInd

EEND = dict(alpha=4, beta=4, gamma=30)          # S_end     (Fig 7 tuning)
ECOVER = dict(alpha=0.8, beta=0.2, gamma=0.8)   # S_cover   (Table III)
ESERIES = dict(alpha=0.8, beta=0.1)             # S_series  (Table IV)
PASS_TEMPO = 0.85                               # tempo discount for PASS (doc'd approx)


def _rank(c):
    return (c - 1) // 4 + 1


class Big2MDPAgent:
    def __init__(self, store: StatsStore = None, level: int = 4,
                 exact_floor: bool = True, min_support: int = 32):
        assert level in (1, 2, 3, 4)
        self.store = store if store is not None else StatsStore()
        self.level = level
        self.planner = RoutePlanner(self.store, exact_floor=exact_floor,
                                    min_support=min_support)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _candidates(self, mask):
        out = {}
        for a in np.flatnonzero(mask == 1):
            a = int(a)
            if a == PASS_IDX:
                continue
            cards, _ = enumerateOptions.getOptionNC(a)
            key = action_key(cards)
            best = out.get(key)
            if best is None or sum(cards) < sum(best[1]):
                out[key] = (a, cards)
        return out

    def _send(self, g, me, sets):
        """S_end (eq 8). |g0| counts the non-single card sets FORMABLE from the
        hand (paper's reading: combination options, not the partition — a
        healthy opening hand forms many, so beta=4 only fires late/weak)."""
        import gameLogic
        opps = [g.currentHands[p].size for p in range(1, 5) if p != me]
        if min(opps) <= EEND["alpha"] or table_level(g) >= EEND["gamma"]:
            return True
        ho = gameLogic.handsAvailable(g.currentHands[me])
        two = enumerateOptions.twoCardOptions(ho)
        five = enumerateOptions.fiveCardOptions(ho)
        n_formable = (0 if isinstance(two, int) else len(two)) + \
                     (0 if isinstance(five, int) else len(five))
        return n_formable <= EEND["beta"]

    # ── decision ────────────────────────────────────────────────────────────
    def __call__(self, env):
        g = env.game
        me = env.current_player
        mask = env.get_valid_actions()
        legal = np.flatnonzero(mask == 1)
        if legal.size == 0:
            return PASS_IDX
        if legal.size == 1:
            return int(legal[0])

        hand = [int(c) for c in g.currentHands[me]]
        feats = state_features(g, me)
        cands = self._candidates(mask)
        if not cands:
            return PASS_IDX

        played = [int(c) for c in (np.flatnonzero((g.cardsPlayed != 0).any(axis=0)) + 1)]
        unseen = unseen_cards(hand, played)
        sets = partition_hand(hand)
        opp_counts = [g.currentHands[p].size for p in range(1, 5) if p != me]
        opp_sum = sum(opp_counts)
        min_opp = min(opp_counts)
        n_twos = sum(1 for c in hand if _rank(c) == 13)

        # ── route values for every candidate ────────────────────────────────
        scored = []                                    # (a, cards, p, r_win, r_lose)
        for key, (a, cards) in cands.items():
            twos_after = n_twos - sum(1 for c in cards if _rank(c) == 13)
            p, r_win, r_lose = self.planner.route(
                hand, feats, unseen, list(cards), opp_sum, twos_after,
                min_opp=min_opp)
            scored.append((a, cards, p, r_win, r_lose))

        # full-hand plan probability (used by SP/WC and by PASS's value)
        set_rc = [self.planner.rc(feats, k, list(cs), unseen) for k, cs in sets]
        if len(set_rc) > 1:
            weakest = min(range(len(set_rc)), key=lambda i: set_rc[i])
            p_plan = 1.0
            for i, r in enumerate(set_rc):
                if i != weakest:
                    p_plan *= r
        else:
            p_plan = 1.0

        # ── level-4 strategy overrides (priority eq 27: SP → WC) ───────────
        if self.level == 4 and len(sets) > 1:
            low = [i for i, r in enumerate(set_rc) if r <= ESERIES["beta"]]
            high = [i for i, r in enumerate(set_rc) if r >= ESERIES["alpha"]]
            if len(high) >= len(sets) - 1 and len(low) <= 1:     # SP
                a = self._play_planned(sets, set_rc, low, cands)
                if a is not None:
                    return a
                if mask[PASS_IDX] == 1:
                    return PASS_IDX                    # protect the series plan
            wc_high = [i for i, r in enumerate(set_rc) if r >= ECOVER["alpha"]]
            wc_low = [i for i, r in enumerate(set_rc) if r <= ECOVER["beta"]]
            if len(wc_high) >= 2 and len(wc_low) == 1:           # WC
                order = sorted(wc_high, key=lambda i: -set_rc[i])
                for i in order:
                    hit = cands.get(action_key(sets[i][1]))
                    if hit is not None:
                        return hit[0]

        # ── heads over route values ─────────────────────────────────────────
        if self.level >= 3:
            dmax = max(len(partition_hand([c for c in hand if c not in cards])) or 1
                       for _, cards, *_ in scored) or 1

        def q_attack(entry):
            a, cards, p, r_win, _ = entry
            q = p if self.level == 4 else p * r_win
            if self.level == 3:
                rest = [c for c in hand if c not in cards]
                d = len(partition_hand(rest)) if rest else 0
                q *= 1.0 - d / max(dmax, 1)
            return q

        def q_defend(entry):
            _, _, p, _, r_lose = entry
            return (1.0 - p) * r_lose                  # negative; max = least loss

        best = max(scored, key=q_attack)

        # PASS as plan protection (only meaningful when following): the value
        # of not responding = the whole hand's win probability from a
        # no-control state (regain cost + survival risk priced in by _V).
        if mask[PASS_IDX] == 1 and g.control == 0:
            p_pass = self.planner.hand_value(hand, feats, unseen, False, min_opp)
            if p_pass > best[2]:
                return PASS_IDX

        if self.level >= 2 and self._send(g, me, sets):
            # endgame & no winning route worth taking → stop-loss head
            # (paper: Q ≤ 0 for ALL routes; our priors never hit exactly 0,
            # so "no route" = best win probability below a small epsilon)
            if best[2] <= 0.02:
                a, *_ = max(scored, key=q_defend)
                return a

        return best[0]

    # ── strategy helpers ────────────────────────────────────────────────────
    def _play_planned(self, sets, set_rc, low, cands):
        """SP: play high-hold sets first (largest first), weak set last."""
        order = sorted((i for i in range(len(sets)) if i not in low),
                       key=lambda i: (-len(sets[i][1]), set_rc[i]))
        for i in order + low:
            if i in low and any(j not in low and cands.get(action_key(sets[j][1]))
                                for j in range(len(sets))):
                continue
            hit = cands.get(action_key(sets[i][1]))
            if hit is not None:
                return hit[0]
        return None
