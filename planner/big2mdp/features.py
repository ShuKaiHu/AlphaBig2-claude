"""State features & action abstraction for the Big2MDP replication.

Paper mapping (ToG 2025):
  * MP feature set (eq 17): (1) first played card set of the state (= the
    combo currently leading the trick), (2) Table-Card Level, (3) opponents'
    remaining card counts, (4) own remaining card count.
  * Table-Card Level (2022 paper §III): every card is worth rank_value (1..13,
    face 3→1 … face 2→13) PLUS a suit extra (♣1 ♦2 ♥3 ♠4); the level is the
    cumulative played total // 10 (0..49). [Documented reading of "the suits
    can get extra values 1,2,3,4" — the paper never shows the exact sum.]
  * Actions are abstracted to (n_cards, kind, key) — the paper indexes actions
    by concrete card sets, but its own MP matching works at feature level; we
    key by combo kind + deciding rank (pair/single: rank; straight/SF: window
    index; full house: trip rank; quad: quad rank) so table statistics
    aggregate across suit variants. PASS is its own key.
"""
import numpy as np

import gameLogic

PASS_KEY = ("pass",)


def rank_of(c):
    return (c - 1) // 4 + 1


def suit_of(c):
    return (c - 1) % 4 + 1


def card_value(c):
    """Table-level value of one card: rank value + suit extra."""
    return rank_of(c) + suit_of(c)


def table_level(g):
    """Cumulative value of ALL played cards // 10 (0..49)."""
    played = np.flatnonzero((g.cardsPlayed != 0).any(axis=0)) + 1
    return min(int(sum(card_value(int(c)) for c in played)) // 10, 49)


def action_key(cards):
    """Abstract a concrete card set (list of ids) to its table key."""
    cards = sorted(int(c) for c in cards)
    n = len(cards)
    if n == 0:
        return PASS_KEY
    if n == 1:
        return (1, "single", rank_of(cards[0]))
    if n == 2:
        return (2, "pair", rank_of(cards[0]))
    arr = np.array(cards)
    if gameLogic.isStraightFlush(arr.copy()):
        return (5, "straight_flush", gameLogic.straightRank(arr)[0])
    if gameLogic.isFourOfAKind(arr.copy()):
        from collections import Counter
        qr = [r for r, k in Counter(map(rank_of, cards)).items() if k >= 4][0]
        return (5, "quad", qr)
    if gameLogic.isFullHouse(arr.copy())[0]:
        from collections import Counter
        tr = [r for r, k in Counter(map(rank_of, cards)).items() if k >= 3][0]
        return (5, "full_house", tr)
    if gameLogic.isStraight(arr.copy()):
        return (5, "straight", gameLogic.straightRank(arr)[0])
    raise ValueError(f"unclassifiable card set: {cards}")


def state_features(g, me):
    """The four MP features for the acting player `me` (1-indexed seat).
    Returns (lead_key, level, own_count, opp_counts_tuple)."""
    if g.control == 1:
        lead_key = None                      # leading: no combo to respond to
    else:
        lead_key = action_key([int(c) for c in g.handsPlayed[g.goIndex - 1].hand])
    level = table_level(g)
    own = int(g.currentHands[me].size)
    opps = tuple(int(g.currentHands[((me - 1 + k) % 4) + 1].size) for k in (1, 2, 3))
    return (lead_key, level, own, opps)
