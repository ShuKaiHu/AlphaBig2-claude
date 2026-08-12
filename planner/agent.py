"""MDPLite v0 — policy/heuristic arm + two rule overrides borrowed from
Big2MDP (ToG 2025), rebuilt on this engine's exact 神來也 rules:

  SP (series preempting / 連出收尾, targets G1): if all but at most one set in
  the greedy hand partition are GUARANTEED to hold control (planner/control.py,
  worst-case-exact, not a frequency table), run the table: play guaranteed sets
  while in control, weak set last (the finishing play needs no control).

  SEND (endgame stop-loss / 輸勢丟險牌, targets G2): in the endgame, when the
  finisher line does not exist and an opponent is about to go out, dump the
  held cards that multiply the loss (the 2s) while they can still be played.

Everything else falls through to `fallback` (default: eval_baselines.smart_action;
deploy target: policy_4500 — inject via MDPLite(fallback=...)).

v0 scope notes (documented, not hidden):
  * SP triggers only from guaranteed sets — no probabilistic R_c yet (belief
    refinement is the planned v1).
  * SEND's "no winning line" proxy is `not SP-eligible ∧ min(opp cards) ≤ SEND_OPP`;
    thresholds start at Big2MDP's tuned values (E_end α=4) pending our own sweep.
  * Any evaluation of this arm must go through the standard ladder (V0–V3) with
    PREREGISTERED readout rules — this module is infrastructure, not a result.
"""
import numpy as np

import enumerateOptions
from planner.control import analyze_partition, unseen_cards
from planner.decompose import partition_hand

PASS_IDX = enumerateOptions.passInd
SEND_OPP = 4          # opponent-cards threshold (Big2MDP E_end alpha, tuned=4)
SEND_MIN_OPP_GOING_OUT = 2   # "someone is about to win" for the dump override


def _rank(c):
    return (c - 1) // 4 + 1


def _played_cards(g):
    return [int(c) for c in (np.flatnonzero((g.cardsPlayed != 0).any(axis=0)) + 1)]


def _action_of(cards, mask):
    idx = enumerateOptions.action_index_from_cards(np.array(sorted(cards)))
    return int(idx) if mask[idx] == 1 else None


class MDPLite:
    def __init__(self, fallback=None):
        if fallback is None:
            from ppo.eval_baselines import smart_action
            fallback = smart_action
        self.fallback = fallback

    def __call__(self, env):
        g = env.game
        me = env.current_player
        hand = [int(c) for c in g.currentHands[me]]
        mask = env.get_valid_actions()
        legal = np.flatnonzero(mask == 1)
        if legal.size == 0:
            return PASS_IDX
        if legal.size == 1:
            return int(legal[0])

        unseen = unseen_cards(hand, _played_cards(g))
        sets = partition_hand(hand)
        holds = analyze_partition(sets, unseen)
        weak = [i for i, h in enumerate(holds) if not h]

        # ── SP: finisher line exists ────────────────────────────────────────
        if len(weak) <= 1:
            if len(sets) == 1:
                a = _action_of(sets[0][1], mask)
                if a is not None:
                    return a
            else:
                # play guaranteed sets while we can, weak set stays for last;
                # among guaranteed, shed the largest set first (faster clock,
                # same guarantee).
                order = sorted((i for i in range(len(sets)) if i not in weak),
                               key=lambda i: (-len(sets[i][1]), sets[i][1]))
                for i in order:
                    a = _action_of(sets[i][1], mask)
                    if a is not None:
                        return a
            # finisher exists but nothing from the plan is legal right now
            # (e.g. following a bigger trick): protect the plan — pass if we may.
            if mask[PASS_IDX] == 1 and len(sets) > 1:
                return PASS_IDX

        # ── SEND: stop-loss dump of multiplier cards ────────────────────────
        opp_counts = [g.currentHands[p].size for p in range(1, 5) if p != me]
        endgame = min(opp_counts) <= SEND_OPP or \
            sum(1 for k, _ in sets if k != "single") <= 4
        if endgame and len(weak) > 1 and min(opp_counts) <= SEND_MIN_OPP_GOING_OUT:
            my_twos = [c for c in hand if _rank(c) == 13]
            for c in sorted(my_twos, reverse=True):
                a = _action_of([c], mask)
                if a is not None:
                    return a

        return self.fallback(env)


def mdplite_action(env):
    """Module-level convenience: MDPLite with the Smart fallback."""
    global _DEFAULT
    try:
        _DEFAULT
    except NameError:
        _DEFAULT = MDPLite()
    return _DEFAULT(env)
