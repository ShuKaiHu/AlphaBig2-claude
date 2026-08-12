"""Deterministic hold-control ("拿權") analysis — Big2MDP's R_c, done exactly.

guaranteed_hold(kind, cards, unseen) answers: "if I lead this combo, is it
IMPOSSIBLE for any opponent to beat it, even if one opponent holds every
relevant unseen card?" — a worst-case lower bound on the probability of
keeping the free-playing right (Big2MDP estimates the same quantity with a
frequency table; we compute it from the rules).

Correctness notes vs engine/dominance.py (which we deliberately do NOT modify
— its outputs are frozen features for deployed v6 checkpoints):
  * single ♠2 is NOT hardcoded unbeatable: a quad / straight flush can take it
    (matches big2Game.returnAvailableActions; dominance.play_strengths:182
    still has the stale ♠2=1.0 shortcut).
  * straight ordering uses gameLogic._STRAIGHT_SEQUENCES window INDEX
    (A2345 lowest … 23456 HIGHEST), not the window-top rank
    (dominance.higher_straight_possible mis-ranks the two wrap windows).
Straight-window membership is cross-checked against the value/hand_features
canon at import time (CLAUDE.md iron rule: never hand-roll windows).
"""
from collections import Counter

import numpy as np

import gameLogic
from value.hand_features import STRAIGHT_WINDOWS_RIDX

# ── self-contained primitives (planner has NO dependency on the deprecated
# engine/dominance.py or on any legacy model line) ──────────────────────────
ALL_CARDS = frozenset(range(1, 53))


def unseen_cards(my_hand, played):
    """Cards still in opponents' hands = everything minus mine minus played."""
    return ALL_CARDS - set(int(c) for c in my_hand) - set(int(c) for c in played)


def bomb_possible(unseen) -> bool:
    cnt = Counter((c - 1) // 4 + 1 for c in unseen)
    return any(v >= 4 for v in cnt.values())


def straight_flush_possible(unseen) -> bool:
    by_suit = {}
    for c in unseen:
        by_suit.setdefault((c - 1) % 4, set()).add((c - 1) // 4 + 1)
    return any(all(r in ranks for r in seq)
               for ranks in by_suit.values()
               for seq in gameLogic._STRAIGHT_SEQUENCES)

_SEQS = gameLogic._STRAIGHT_SEQUENCES  # index = strength order, tuple = rank values 1..13

# canon cross-check: same 10 windows as value/hand_features (0-indexed there)
assert {tuple(sorted(s)) for s in _SEQS} == \
       {tuple(sorted(r + 1 for r in w)) for w in STRAIGHT_WINDOWS_RIDX}, \
    "planner straight windows diverge from value/hand_features canon"


def _rank(c):
    return (c - 1) // 4 + 1


def _seq_high_value(seq):
    return 13 if seq == (13, 1, 2, 3, 4) else seq[-1]


def _straight_key(cards):
    """(window_index, high_card_id) for a straight/straight-flush, engine-exact."""
    key = gameLogic.straightRank(np.array(sorted(cards)))
    if key is None:
        raise ValueError(f"not a straight: {cards}")
    return key


def _completable(seq, un_ranks):
    return all(r in un_ranks for r in seq)


def guaranteed_hold(kind, cards, unseen):
    """True iff leading (kind, cards) cannot be beaten by ANY combination of the
    unseen cards (worst case: one opponent holds all of them)."""
    unseen = set(unseen)
    cards = [int(c) for c in cards]
    bomb = bomb_possible(unseen)
    sf = straight_flush_possible(unseen)
    un_rc = Counter(_rank(c) for c in unseen)
    un_ranks = set(un_rc)

    if kind == "single":
        c = cards[0]
        # NO ♠2 exception: bombs take any single, including ♠2.
        return (not bomb) and (not sf) and not any(u > c for u in unseen)

    if kind == "pair":
        if bomb or sf:
            return False
        r = _rank(cards[0])
        if any(rr > r and n >= 2 for rr, n in un_rc.items()):
            return False
        same = [u for u in unseen if _rank(u) == r]
        # same-rank pair compares by its highest card id (suit tiebreak)
        return not (len(same) >= 2 and max(same) > max(cards))

    if kind == "straight":
        if bomb or sf:
            return False
        idx, high = _straight_key(cards)
        for j, seq in enumerate(_SEQS):
            if not _completable(seq, un_ranks):
                continue
            if j > idx:
                return False
            if j == idx:
                hv = _seq_high_value(seq)
                if any(_rank(u) == hv and u > high for u in unseen):
                    return False
        return True

    if kind == "full_house":
        if bomb or sf:
            return False
        trip = [r for r, n in Counter(_rank(c) for c in cards).items() if n >= 3][0]
        for rr, n in un_rc.items():
            if rr > trip and n >= 3 and any(r2 != rr and n2 >= 2 for r2, n2 in un_rc.items()):
                return False
        return True

    if kind == "quad":
        if sf:
            return False
        qr = [r for r, n in Counter(_rank(c) for c in cards).items() if n >= 4][0]
        return not any(rr > qr and n >= 4 for rr, n in un_rc.items())

    if kind == "straight_flush":
        idx, high = _straight_key(cards)
        by_suit = {}
        for u in unseen:
            by_suit.setdefault((u - 1) % 4, set()).add(u)
        for suit_cards in by_suit.values():
            ranks_in_suit = {_rank(u) for u in suit_cards}
            for j, seq in enumerate(_SEQS):
                if not _completable(seq, ranks_in_suit):
                    continue
                if j > idx:
                    return False
                if j == idx:
                    hv = _seq_high_value(seq)
                    if any(_rank(u) == hv and u > high for u in suit_cards):
                        return False
        return True

    raise ValueError(f"unknown kind {kind}")


def analyze_partition(sets, unseen):
    """[(kind, cards)] → list of bools (guaranteed_hold per set)."""
    return [guaranteed_hold(k, cs, unseen) for k, cs in sets]
