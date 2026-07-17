"""V1 dominance score — public info only, NO belief.

'dominance' of a combo C = fraction of the combos that could beat C that are NOT
assemblable from opponents' unseen cards. 1.0 = currently unbeatable ("100%"), 0.0
= every beater is out there. Dynamic: beaters already PLAYED stop being threats, so
a low card becomes dominant once everything higher is gone.

Singles & pairs are EXACT (a beater unseen = an opponent holds it → real threat).
5-card combos are APPROXIMATE in V1 (conservative: a beater is a threat if its
cards are all unseen, ignoring whether they sit in ONE opponent's hand — that split
question is what belief K-world sampling refines in V2).

This variant: single / pair / 5-card {straight 順, full house 葫蘆, quad 鐵支,
straight-flush 同花順}. NO standalone triple, NO flush.
"""
from collections import Counter

from hand_features import _rank, _extract_five   # _rank(id)=value 1..13 (1='3',13='2')

ALL = set(range(1, 53))


# ── per-combo dominance ───────────────────────────────────────────────────────
def dom_single(c, unseen):
    """RANK among all 52 singles = 1 − (higher singles an opponent still holds)/51.
    single strength = card_id order (id52 = 2♠ = top). Dynamic: higher singles that
    are played/mine stop counting, so my card's rank rises as the field thins."""
    alive = sum(1 for x in range(c + 1, 53) if x in unseen)   # higher singles opponents hold
    return 1.0 - alive / 51.0


def dom_pair(v, unseen_rc):
    """RANK among 13 pair-values = 1 − (higher pair-values an opponent can form)/12.
    beaten only by a higher-value pair (bombs are 5-card, can't beat a pair)."""
    alive = sum(1 for w in range(v + 1, 14) if unseen_rc.get(w, 0) >= 2)
    return 1.0 - alive / 12.0


def dom_fullhouse(t, unseen_rc):
    """RANK among 13 full-house triple-values = 1 − (higher FH an opponent can form)/12
    (bombs ignored in V1 per the no-bomb simplification)."""
    alive = sum(1 for w in range(t + 1, 14) if unseen_rc.get(w, 0) >= 3)
    return 1.0 - alive / 12.0


def dom_straight(unseen_rc):
    """V1 APPROX: a straight is beaten by any full house (common) or bomb, so its
    dominance ~ 'no full house assemblable'. Higher-straight ordering ignored in V1."""
    fh_alive = sum(1 for w in range(1, 14) if unseen_rc.get(w, 0) >= 3)
    return 1.0 - fh_alive / 13.0


# ── hand decomposition (same greedy as min_plays: 5-card, then pairs, then singles)
def decompose(hand):
    rem = [int(c) for c in hand]
    combos = []
    while True:
        five = _extract_five(rem)
        if not five:
            break
        for c in five:
            rem.remove(c)
        rc = Counter(_rank(c) for c in five)
        if 4 in rc.values():
            combos.append(("bomb", five))                        # quad+kicker
        elif sorted(rc.values()) == [2, 3]:
            combos.append(("fh", five))
        else:
            suits = {(c - 1) % 4 for c in five}
            combos.append(("bomb" if len(suits) == 1 else "straight", five))  # sf = bomb
    rc = Counter(_rank(c) for c in rem)
    for r in list(rc):
        while sum(1 for c in rem if _rank(c) == r) >= 2:
            pair = [c for c in rem if _rank(c) == r][:2]
            for c in pair:
                rem.remove(c)
            combos.append(("pair", pair))
    for c in rem:
        combos.append(("single", [c]))
    return combos


def dom_of(typ, cards, unseen, unseen_rc):
    if typ == "single":
        return dom_single(cards[0], unseen)
    if typ == "pair":
        return dom_pair(_rank(cards[0]), unseen_rc)
    if typ == "fh":
        tri = [r for r, n in Counter(_rank(c) for c in cards).items() if n == 3][0]
        return dom_fullhouse(tri, unseen_rc)
    if typ == "straight":
        return dom_straight(unseen_rc)
    return 1.0                                                    # bomb (quad/SF) ~ dominant


def dom_score(hand, played, breakdown=False):
    """Weighted-by-size mean dominance of the hand's greedy decomposition. [0,1]."""
    hand = {int(c) for c in hand}
    played = {int(c) for c in played}
    unseen = ALL - hand - played
    unseen_rc = Counter(_rank(c) for c in unseen)
    combos = decompose(hand)
    tot_w = tot_s = 0.0
    rows = []
    for typ, cards in combos:
        d = dom_of(typ, cards, unseen, unseen_rc)
        sz = len(cards)
        tot_w += sz; tot_s += d * sz
        rows.append((typ, cards, round(d, 3)))
    score = tot_s / max(tot_w, 1)
    return (score, rows) if breakdown else score
