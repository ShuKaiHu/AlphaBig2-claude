"""Cheap deterministic hand features for the value net.

min_plays_to_empty(hand): the (greedy) fewest legal Big2 plays needed to empty the
hand — a straight/full-house/quad/SF clears 5 cards in one play, a pair clears 2, a
single clears 1. Smaller = closer to going out. Greedy (extract biggest combos
first) → an upper bound on the true minimum, good enough as a feature.
"""
from collections import Counter

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
