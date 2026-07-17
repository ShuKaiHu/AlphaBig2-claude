"""Cheap deterministic hand features for the value net.

min_plays_to_empty(hand): the (greedy) fewest legal Big2 plays needed to empty the
hand — a straight/full-house/quad/SF clears 5 cards in one play, a pair clears 2, a
single clears 1. Smaller = closer to going out. Greedy (extract biggest combos
first) → an upper bound on the true minimum, good enough as a feature.
"""
from collections import Counter

_WIN = [tuple(range(s, s + 5)) for s in range(1, 9)] + [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]

# Canonical straight windows in 0-indexed rank space (0='3' .. 12='2'), for the
# analysis scripts that work on rank indexes instead of card ids. EXACTLY these
# 10; J-Q-K-A-2 ({8..12}) is NOT a straight — an off-by-one (range(0, 9)) that
# includes it silently misclassifies S-J..S-2 hands as straight-flush bombs,
# which is how five scratch analysis scripts overcounted bomb exclusions.
# Import these instead of re-deriving windows by hand.
STRAIGHT_WINDOWS_RIDX = [frozenset(range(s, s + 5)) for s in range(0, 8)] \
                        + [frozenset({11, 12, 0, 1, 2}), frozenset({12, 0, 1, 2, 3})]
assert len(STRAIGHT_WINDOWS_RIDX) == 10
assert frozenset({8, 9, 10, 11, 12}) not in STRAIGHT_WINDOWS_RIDX      # J-Q-K-A-2 rejected
assert frozenset({9, 10, 11, 12, 0}) not in STRAIGHT_WINDOWS_RIDX      # Q-K-A-2-3 rejected
assert frozenset({10, 11, 12, 0, 1}) not in STRAIGHT_WINDOWS_RIDX      # K-A-2-3-4 rejected
assert frozenset({7, 8, 9, 10, 11}) in STRAIGHT_WINDOWS_RIDX           # 10-J-Q-K-A ok
assert frozenset({11, 12, 0, 1, 2}) in STRAIGHT_WINDOWS_RIDX           # A-2-3-4-5 ok
assert frozenset({12, 0, 1, 2, 3}) in STRAIGHT_WINDOWS_RIDX            # 2-3-4-5-6 ok
assert {tuple(sorted(w)) for w in STRAIGHT_WINDOWS_RIDX} == \
       {tuple(sorted(r - 1 for r in w)) for w in _WIN}                 # matches the card-id windows


def has_bomb_rank_suit(pairs):
    """True iff a hand holds a quad or a straight flush.  `pairs` = iterable of
    (rank_idx 0-12, suit) tuples — the encoding the online-log analysis scripts
    use.  This is THE bomb detector for analysis code; do not hand-roll windows."""
    pairs = list(pairs)
    rc = Counter(r for r, _ in pairs)
    if len(pairs) >= 5 and any(n >= 4 for n in rc.values()):
        return True
    by_suit = {}
    for r, s in pairs:
        by_suit.setdefault(s, set()).add(r)
    return any(w <= idx for w in STRAIGHT_WINDOWS_RIDX for idx in by_suit.values())


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
