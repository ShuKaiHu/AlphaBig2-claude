"""Ground-truth dominance labels from the SHOWDOWN-reconstructed true hands.

Unlike engine/dominance.py (which is conservative — it asks "could a beater be
FORMED from the pool of all unseen cards", ignoring that a multi-card beater must
sit in ONE opponent's hand), this checks each opponent's ACTUAL hand. So a play is
'nuts' (oracle=1) iff NO single opponent really holds a combo that beats it.

That is the TRUE label per game (e.g. A-full-house with 3 twos unseen: nuts iff no
single opponent held three 2s + a pair). Across many games, a model trained on these
0/1 labels outputs P(nuts) — the soft 'dominance probability'. card_id=(rank-1)*4+suit,
rank 1=3..13=2 (2 highest).
"""
from collections import Counter

_WINDOWS = [tuple(range(s, s + 5)) for s in range(1, 9)] + [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]


def _rank(c):
    return (c - 1) // 4 + 1


def _ranks(hand):
    return Counter(_rank(int(c)) for c in hand)


def _has_bomb(hand):
    return any(n >= 4 for n in _ranks(hand).values())


def _has_sf(hand):
    by_suit = {1: set(), 2: set(), 3: set(), 4: set()}
    for c in hand:
        by_suit[(int(c) - 1) % 4 + 1].add(_rank(int(c)))
    return any(all(r in s for r in w) for s in by_suit.values() for w in _WINDOWS)


def _best_pair(hand):
    rc = _ranks(hand)
    return max([r for r, n in rc.items() if n >= 2], default=0)


def _best_fh_trip(hand):
    """Highest triple rank that can be paired with a different rank (a real full house)."""
    rc = _ranks(hand)
    trips = [r for r, n in rc.items() if n >= 3]
    out = 0
    for t in trips:
        if any(r != t and n >= 2 for r, n in rc.items()):
            out = max(out, t)
    return out


def oracle_nuts(my_hand, opp_hands):
    """opp_hands: list of the 3 opponents' true remaining hands. Returns [single,
    pair, full_house] each 1.0 if my best play of that type is unbeatable, else 0.0
    (and 0.0 if I cannot form that type)."""
    my = [int(c) for c in my_hand]
    opps = [[int(c) for c in h] for h in opp_hands]
    override = any(_has_bomb(h) or _has_sf(h) for h in opps)   # quad/SF beats any lower type

    # single: my highest card unbeaten (no opp has a higher card) and no override
    my_max = max(my) if my else 0
    opp_max = max((max(h) for h in opps if h), default=0)
    single = 1.0 if (my_max > opp_max and not override) else 0.0

    # pair: my best pair rank beats every opponent's best pair, no override
    mp = _best_pair(my)
    pair = 1.0 if (mp > 0 and not override and all(_best_pair(h) < mp for h in opps)) else 0.0

    # full house: my best fh-triple beats every opponent's, no override (quad/SF/higher-fh)
    mt = _best_fh_trip(my)
    fh = 1.0 if (mt > 0 and not override and all(_best_fh_trip(h) < mt for h in opps)) else 0.0

    return [single, pair, fh]


LABELS = ["single_nuts", "pair_nuts", "fh_nuts"]
