"""Faithful Big2MDP agent (ToG 2025) on this engine's exact 神來也 rules.

Plays EVERY move itself — no policy / belief / value nets, no human data, no
legacy assets. Difficulty levels mirror the paper's app: 1=MDP1.0 (Rookie),
2=MDP2.0 (Normal), 3=MDP3.0 (Expert), 4=MDP4.0+WP/MP/WC/SP (Master).

Paper equations implemented (see docs/paper/BIG2MDP_METHOD_NOTES.md):
  Q1.0 (4)-(6): max-of-products P_win×R_win over table stats (their quirk —
    an optimistic per-route product, NOT an expectation — reproduced as-is at
    the aggregate level).
  Q2.0 (7)-(9): + P_lose×R_lose, gated by the S_end endgame switch (8) with
    their tuned thresholds Eend = (α=4, β=4, γ=30).
  MDP3.0 (10)-(13): toward-winning weight W = 1 − d/dmax; d(after action) =
    greedy partition size of the remaining hand (min_plays proxy for their
    shortest-path state distance — documented approximation).
  MDP4.0 (14)-(16): Qwin = P_win, Qlose = P_lose×R_lose.
  MP (17)-(18): feature-OR retrieval — lives in StatsStore.
  WC (19)-(24): Ecover = (α=.8, β=.2, γ=.8); SP (25)-(27): Eseries = (α=.8, β=.1).
  Strategy priority (27): SP → WC → win/loss via S_end.

Documented deviations (kept minimal, all engineering):
  * Candidate actions are grouped by action_key; the concrete set played for a
    chosen key is the lowest-value one (shed weakest suits first).
  * R_c per candidate: table estimate; with exact_floor=True (default) a set
    that planner/control.guaranteed_hold proves unbeatable gets R_c=1.0 (the
    frequency table converges there anyway; pass exact_floor=False for the
    purist run).
  * Cold start (empty table): play the largest-then-lowest legal set — the
    shedding behaviour their score-seeking exhibits with no data.
"""
import numpy as np

import enumerateOptions
from planner.big2mdp.features import PASS_KEY, action_key, state_features
from planner.big2mdp.store import StatsStore
from planner.control import guaranteed_hold, unseen_cards
from planner.decompose import partition_hand

PASS_IDX = enumerateOptions.passInd

EEND = dict(alpha=4, beta=4, gamma=30)          # S_end     (Fig 7 tuning)
ECOVER = dict(alpha=0.8, beta=0.2, gamma=0.8)   # S_cover   (Table III)
ESERIES = dict(alpha=0.8, beta=0.1)             # S_series  (Table IV)

_KIND = {1: "single", 2: "pair"}


def _kind_of(cards):
    k = action_key(cards)
    return k[1] if k is not PASS_KEY else "pass"


class Big2MDPAgent:
    def __init__(self, store: StatsStore = None, level: int = 4,
                 exact_floor: bool = True, min_support: int = 32):
        """min_support: table rows with fewer matched samples are treated as
        no-data (small-sample P_win=1 flukes otherwise hijack the argmax —
        the paper's 500K-game tables have no such regime; this guard changes
        nothing once counts are large)."""
        assert level in (1, 2, 3, 4)
        self.store = store if store is not None else StatsStore()
        self.level = level
        self.exact_floor = exact_floor
        self.min_support = min_support

    # ── helpers ─────────────────────────────────────────────────────────────
    def _candidates(self, mask):
        """legal non-pass actions grouped by key → {key: lowest concrete action}."""
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

    def _rc(self, feats, key, cards, unseen):
        if self.exact_floor and key is not PASS_KEY and \
                guaranteed_hold(key[1], cards, unseen):
            return 1.0
        st = self.store.stats(feats, key)
        return st["r_c"] if st["n"] >= self.min_support else 0.0

    def _send(self, g, me, sets):
        opps = [g.currentHands[p].size for p in range(1, 5) if p != me]
        from planner.big2mdp.features import table_level
        return (min(opps) <= EEND["alpha"]
                or sum(1 for k, _ in sets if k != "single") <= EEND["beta"]
                or table_level(g) >= EEND["gamma"])

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
        if not cands:                                  # only PASS is legal
            return PASS_IDX

        played = [int(c) for c in (np.flatnonzero((g.cardsPlayed != 0).any(axis=0)) + 1)]
        unseen = unseen_cards(hand, played)
        sets = partition_hand(hand)

        # ── level 4 strategies first (priority eq 27: SP → WC → heads) ─────
        if self.level == 4 and len(sets) > 1:
            set_rc = [self._rc_set(feats, k, cs, unseen) for k, cs in sets]
            low = [i for i, r in enumerate(set_rc) if r <= ESERIES["beta"]]
            high = [i for i, r in enumerate(set_rc) if r >= ESERIES["alpha"]]
            # SP (26): all but ≤1 set holds the right with high probability
            if len(high) >= len(sets) - 1 and len(low) <= 1:
                a = self._play_planned(sets, set_rc, low, cands)
                if a is not None:
                    return a
                if mask[PASS_IDX] == 1:               # protect the series plan
                    return PASS_IDX
            # WC (22): high-R_c set now, exactly-one weak set to cover next,
            # another high-R_c set after that
            if len(high) >= 2 and len(low) == 1:
                order = sorted(high, key=lambda i: -set_rc[i])
                for i in order:
                    hit = cands.get(action_key(sets[i][1]))
                    if hit is not None:
                        return hit[0]

        # ── Q heads over candidates ─────────────────────────────────────────
        scored = []
        for key, (a, cards) in cands.items():
            st = self.store.stats(feats, key)
            if st["n"] < self.min_support:             # under-supported → no data
                st = dict(st, n=0)
            scored.append((key, a, cards, st))

        if all(s[3]["n"] == 0 for s in scored):        # cold start / low support
            key, a, cards, _ = max(
                scored, key=lambda s: (len(s[2]), -sum(s[2])))
            return a
        scored = [s for s in scored if s[3]["n"] > 0]

        send = self._send(g, me, sets)

        def w_dist(cards):                             # MDP3.0 weight
            if self.level < 3:
                return 1.0
            rem = [c for c in hand if c not in cards]
            return len(partition_hand(rem)) if rem else 0

        if self.level >= 3:
            dists = {s[1]: w_dist(s[2]) for s in scored}
            dmax = max(dists.values()) or 1

        def q_win(st, a):
            if self.level == 4:
                q = st["p_win"]                        # eq 14
            else:
                q = st["p_win"] * st["r_win"]          # eq 4
            if self.level == 3:
                q *= 1.0 - dists[a] / dmax             # eq 10-11
            return q

        def q_lose(st):
            return st["p_lose"] * st["r_lose"]         # eq 15 (negative)

        best_key, best_a, best_cards, best_st = max(
            scored, key=lambda s: q_win(s[3], s[1]))

        if self.level >= 2 and send and q_win(best_st, best_a) <= 0:
            # conservative head: minimise expected loss (eq 9 / 16)
            _, a, _, _ = max(scored, key=lambda s: q_lose(s[3]))
            return a

        if self.level >= 2 and send is False and best_st["n"] == 0 \
                and mask[PASS_IDX] == 1:
            return PASS_IDX                            # no info, no urgency
        return best_a

    # ── strategy helpers ────────────────────────────────────────────────────
    def _rc_set(self, feats, kind, cards, unseen):
        return self._rc(feats, action_key(cards), list(cards), unseen)

    def _play_planned(self, sets, set_rc, low, cands):
        """SP: play guaranteed-side sets first (largest first), weak set last."""
        order = sorted((i for i in range(len(sets)) if i not in low),
                       key=lambda i: (-len(sets[i][1]), set_rc[i]))
        if len(sets) == 1:
            order = [0]
        for i in order + low:
            if i in low and len(sets) > 1 and any(j not in low for j in range(len(sets))
                                                  if cands.get(action_key(sets[j][1]))):
                continue                               # weak set only when forced/last
            hit = cands.get(action_key(sets[i][1]))
            if hit is not None:
                return hit[0]
        return None
