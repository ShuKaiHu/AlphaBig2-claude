"""Prediction tree — paper-literal (v4).

The tree is BUILT FROM RETRIEVED RECORDS (eq 17: feature-OR selection into
the prediction tree) and evaluated by the converging-reward recursion
(eq 3-9): transitions follow each record's stored successor chain, grouped
into destination states by their feature signature; probabilities are
occurrence frequencies within the group (eq 3/18); at the acting player's
own nodes the recursion takes MAX over actions (eq 5: V = max Q); rewards
enter at terminal records (the game's final score).

Head quantities per root action (all from the same walk):
  p_win : probability of eventually winning under best-own-play  (eq 14, Q4.0)
  q1    : max-of-products optimistic win value  P×R, negatives ignored
          (eq 4-6 — the paper's Fig 2 semantics, NOT an expectation)
  loss  : expected loss-side value Σ P×R_lose                    (eq 7/15)
  rc    : mean fraction of the next three actors passing         (eq 19-20)
  d_min : fewest rounds to a winning terminal                    (eq 10, MDP3.0 d)

Documented tractability assumptions the paper does not specify: a retrieval
cap (most recent records win), a recursion depth cap with historical-outcome
fallback at the cut, and loss-side propagation through the win-maximizing
action.
"""


class ActionStats:
    __slots__ = ("p_win", "q1", "loss", "rc", "d_min", "n")

    def __init__(self, p_win=0.0, q1=0.0, loss=0.0, rc=0.0, d_min=None, n=0):
        self.p_win = p_win; self.q1 = q1; self.loss = loss
        self.rc = rc; self.d_min = d_min; self.n = n


def _outcome_stats(store, idxs):
    """Historical-outcome fallback for a cluster (depth cut)."""
    n = len(idxs)
    wins = [i for i in idxs if store.won[i]]
    losses = [i for i in idxs if not store.won[i]]
    p = len(wins) / n
    vw = sum(store.score[i] for i in wins) / len(wins) if wins else 0.0
    vl = sum(store.score[i] for i in losses) / len(losses) if losses else 0.0
    return p, vw, vl, None


def _state_sig(store, i):
    return (store.lead[i], store.level[i], store.own[i], store.opps[i])


def evaluate(store, root_idxs, depth_max=8):
    """→ {akey: ActionStats} for the root cluster."""

    def eval_state(idxs, depth):
        """State value under own-node max → (p_win, v_win, v_loss, d_min)."""
        if depth >= depth_max:
            return _outcome_stats(store, idxs)
        per_action = eval_actions(idxs, depth)
        if not per_action:
            return _outcome_stats(store, idxs)
        best = max(per_action.values(), key=lambda s: s.p_win)
        return best.p_win, best.q1, best.loss, best.d_min

    def eval_actions(idxs, depth):
        groups = {}
        for i in idxs:
            groups.setdefault(store.akey[i], []).append(i)
        out = {}
        for aid, g in groups.items():
            n = len(g)
            rc = sum(store.npass[i] for i in g) / (3.0 * n)
            term_w = [i for i in g if store.succ[i] < 0 and store.won[i]]
            term_l = [i for i in g if store.succ[i] < 0 and not store.won[i]]
            succs = [store.succ[i] for i in g if store.succ[i] >= 0]
            branches = {}
            for s in succs:
                branches.setdefault(_state_sig(store, s), []).append(s)
            p_win = len(term_w) / n
            q1 = (len(term_w) / n) * \
                 (sum(store.score[i] for i in term_w) / len(term_w)) if term_w else 0.0
            loss = (len(term_l) / n) * \
                   (sum(store.score[i] for i in term_l) / len(term_l)) if term_l else 0.0
            d_min = 1 if term_w else None
            for sig, br in branches.items():
                p = len(br) / n
                bp, bw, bl, bd = eval_state(br, depth + 1)
                p_win += p * bp
                if bw > 0:
                    q1 = max(q1, p * bw)          # max-of-products, negatives ignored
                loss += p * bl
                if bd is not None:
                    d = bd + 1
                    d_min = d if d_min is None else min(d_min, d)
            out[aid] = ActionStats(p_win, q1, loss, rc, d_min, n)
        return out

    per_action = eval_actions(list(root_idxs), 0)
    return {store.akey_of(aid): st for aid, st in per_action.items()}
