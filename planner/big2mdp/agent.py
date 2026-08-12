"""Faithful Big2MDP agent (ToG 2025) — v4, paper-literal decision core.

Every decision: retrieve similar historical states (eq 17) → build/evaluate
the prediction tree (planner/big2mdp/tree.py, eq 3-9) → heads:

  MDP1.0: argmax q1 (optimistic win value)                       eq 4-6
  MDP2.0: + S_end (4/4/30); endgame with q1+loss ≤ 0 for ALL
          actions → least-loss action                            eq 7-9
  MDP3.0: q1 weighted by W = 1 − d/dmax (d = tree rounds-to-win) eq 10-13
  MDP4.0: argmax p_win; endgame with p_win = 0 for all → least
          loss; strategies SP (25-27) and WC (19-24) on the
          tree's rc means, thresholds at the paper's tuned
          values; priority SP → WC → heads                       eq 14-16

PASS is an ordinary action evaluated through the same tree (the paper's
action set is {Play, Pass}).

Where the paper is silent (documented minimal choices, planner/README.md):
cold start with no retrievable data → shed the largest-then-lowest legal
set; retrieval cap / tree depth cap; a partition set absent from the data
gets rc = 0.
"""
import numpy as np

import enumerateOptions
import gameLogic
from planner.big2mdp.features import PASS_KEY, action_key, state_features, table_level
from planner.big2mdp.store import RecordStore
from planner.big2mdp.tree import evaluate
from planner.decompose import partition_hand

PASS_IDX = enumerateOptions.passInd

EEND = dict(alpha=4, beta=4, gamma=30)          # S_end     (Fig 7 tuning)
ECOVER = dict(alpha=0.8, beta=0.2, gamma=0.8)   # S_cover   (Table III)
ESERIES = dict(alpha=0.8, beta=0.1)             # S_series  (Table IV)


class Big2MDPAgent:
    def __init__(self, store: RecordStore = None, level: int = 4,
                 cap: int = 2000, depth_max: int = 8):
        assert level in (1, 2, 3, 4)
        self.store = store if store is not None else RecordStore()
        self.level = level
        self.cap = cap
        self.depth_max = depth_max
        # introspection for the online wrapper / dashboard: set on every
        # decision; never read by the decision logic itself.
        self.last = {}

    # ── helpers ─────────────────────────────────────────────────────────────
    def _candidates(self, mask):
        out = {}
        for a in np.flatnonzero(mask == 1):
            a = int(a)
            if a == PASS_IDX:
                out[PASS_KEY] = (a, [])
                continue
            cards, _ = enumerateOptions.getOptionNC(a)
            key = action_key(cards)
            best = out.get(key)
            if best is None or sum(cards) < sum(best[1]):
                out[key] = (a, cards)
        return out

    def _send(self, g, me):
        """S_end (eq 8); |g0| = non-single card sets FORMABLE from the hand."""
        opps = [g.currentHands[p].size for p in range(1, 5) if p != me]
        if min(opps) <= EEND["alpha"] or table_level(g) >= EEND["gamma"]:
            return True
        ho = gameLogic.handsAvailable(g.currentHands[me])
        two = enumerateOptions.twoCardOptions(ho)
        five = enumerateOptions.fiveCardOptions(ho)
        n_formable = (0 if isinstance(two, int) else len(two)) + \
                     (0 if isinstance(five, int) else len(five))
        return n_formable <= EEND["beta"]

    @staticmethod
    def _cold(cands):
        """Bootstrap when nothing retrievable (paper blank): shed the largest,
        then lowest-value, legal non-pass set."""
        plays = [(a, cards) for k, (a, cards) in cands.items() if k is not PASS_KEY]
        if not plays:
            return PASS_IDX
        a, _ = max(plays, key=lambda t: (len(t[1]), -sum(t[1])))
        return a

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

        feats = state_features(g, me)
        cands = self._candidates(mask)
        root = self.store.retrieve(feats, cap=self.cap)
        tree = evaluate(self.store, root, depth_max=self.depth_max) if root else {}

        self.last = {"mode": "head", "tree": tree, "root_n": len(root)}
        scored = [(k, a, cards, tree[k]) for k, (a, cards) in cands.items()
                  if k in tree]
        if not scored:
            self.last["mode"] = "cold"
            return self._cold(cands)

        # ── level-4 strategies (priority eq 27: SP → WC → heads) ───────────
        hand = [int(c) for c in g.currentHands[me]]
        sets = partition_hand(hand)
        if self.level == 4 and len(sets) > 1:
            set_rc = [tree[action_key(cs)].rc if action_key(cs) in tree else 0.0
                      for _, cs in sets]
            low = [i for i, r in enumerate(set_rc) if r <= ESERIES["beta"]]
            high = [i for i, r in enumerate(set_rc) if r >= ESERIES["alpha"]]
            if len(high) >= len(sets) - 1 and len(low) <= 1:     # SP (eq 26)
                a = self._play_planned(sets, set_rc, low, cands)
                if a is not None:
                    self.last["mode"] = "sp"
                    return a
                if mask[PASS_IDX] == 1:
                    self.last["mode"] = "sp-pass"
                    return PASS_IDX                    # protect the series plan
            wc_high = [i for i, r in enumerate(set_rc) if r >= ECOVER["alpha"]]
            wc_low = [i for i, r in enumerate(set_rc) if r <= ECOVER["beta"]]
            if len(wc_high) >= 2 and len(wc_low) == 1:           # WC (eq 22)
                order = sorted(wc_high, key=lambda i: -set_rc[i])
                for i in order:
                    hit = cands.get(action_key(sets[i][1]))
                    if hit is not None:
                        self.last["mode"] = "wc"
                        return hit[0]

        # ── heads ───────────────────────────────────────────────────────────
        if self.level == 3:
            ds = [st.d_min for *_, st in scored if st.d_min is not None]
            dmax = max(ds) if ds else 1

        def q_attack(entry):
            _, _, _, st = entry
            if self.level == 4:
                return st.p_win                        # eq 14
            q = st.q1                                  # eq 4
            if self.level == 3 and st.d_min is not None:
                q *= 1.0 - st.d_min / max(dmax, 1)     # eq 10-11
            return q

        best = max(scored, key=q_attack)

        if self.level >= 2 and self._send(g, me):
            if self.level == 4:
                no_route = all(st.p_win == 0.0 for *_, st in scored)   # eq 16
            else:
                no_route = all(st.q1 + st.loss <= 0 for *_, st in scored)  # eq 9
            if no_route:
                _, a, _, _ = max(scored, key=lambda e: e[3].loss)  # least loss
                self.last["mode"] = "stoploss"
                return a

        return best[1]

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
