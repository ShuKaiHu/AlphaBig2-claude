"""Hand decomposition into playable combos — the SP (series-preempting) substrate.

Greedy partition, IDENTICAL priority order to the canonical
value/hand_features.min_plays_to_empty (quad+kicker → full house → straight →
pairs → singles), so a partition's set-count always equals min_plays_to_empty's
play count. Greedy is an upper bound on the true minimum decomposition — good
enough for the v0 planner; an exact/beam partitioner is a later refinement.

Combo kinds: "single" | "pair" | "straight" | "full_house" | "quad" |
"straight_flush" (a straight that is also a flush is classified straight_flush;
a quad is always quad+kicker, 5 cards, per this ruleset).
"""
import numpy as np

import gameLogic
from value.hand_features import _extract_five


def _rank(c):
    return (c - 1) // 4 + 1


def _classify_five(cards):
    arr = np.array(sorted(cards))
    if gameLogic.isStraightFlush(arr.copy()):
        return "straight_flush"
    if gameLogic.isFourOfAKind(arr.copy()):
        return "quad"
    # isFullHouse sorts its argument IN PLACE (see ppo/belief_history_dataset.py
    # header for the bug this caused elsewhere) — always hand it a copy.
    if gameLogic.isFullHouse(arr.copy())[0]:
        return "full_house"
    if gameLogic.isStraight(arr.copy()):
        return "straight"
    raise ValueError(f"_extract_five returned a non-five-combo: {cards}")


def partition_hand(cards):
    """cards: iterable of card ids 1..52 → list of (kind, tuple_of_cards),
    disjoint, union == cards. Greedy fives → pairs → singles."""
    rem = [int(c) for c in cards]
    sets = []
    while True:
        five = _extract_five(rem)
        if not five:
            break
        for c in five:
            rem.remove(c)
        sets.append((_classify_five(five), tuple(sorted(five))))
    # pairs (same loop shape as min_plays_to_empty)
    from collections import Counter
    rc = Counter(_rank(c) for c in rem)
    for r in list(rc):
        while sum(1 for c in rem if _rank(c) == r) >= 2:
            pair = [c for c in rem if _rank(c) == r][:2]
            for c in pair:
                rem.remove(c)
            sets.append(("pair", tuple(sorted(pair))))
    sets.extend(("single", (c,)) for c in sorted(rem))
    return sets
