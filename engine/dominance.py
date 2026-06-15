"""Card-dominance (the "nuts") — deterministic logic, no neural net.

Answers the human question "is my card/combo currently unbeatable?" purely from
public information (my hand + all played cards), by reasoning about the UNSEEN
cards (those still in opponents' hands).

card_id encoding (enumerateOptions._card_id):
    card_id = (rank_value-1)*4 + suit_idx
    rank_value: 1=Three .. 11=King, 12=Ace, 13=Two   (Big 2 order, 2 is highest)
    suit_idx:   1=Clubs, 2=Diamonds, 3=Hearts, 4=Spades
    → higher card_id == strictly stronger SINGLE.  card_id 52 = ♠2 = global nuts single.

House rule (rules.md): a four-of-a-kind ("鐵支") or straight-flush can beat ANY
lower combo type. So a single/pair is only TRULY unbeatable if no opponent could
hold a bomb or straight-flush either.
"""

ALL_CARDS = frozenset(range(1, 53))


def _rank(card_id: int) -> int:
    return (card_id - 1) // 4 + 1          # 1..13


def unseen_cards(my_hand, played):
    """Cards still in opponents' hands = everything minus mine minus played."""
    return ALL_CARDS - set(int(c) for c in my_hand) - set(int(c) for c in played)


def bomb_possible(unseen) -> bool:
    """An opponent could hold a four-of-a-kind iff some rank has all 4 cards unseen."""
    cnt = {}
    for c in unseen:
        cnt[_rank(c)] = cnt.get(_rank(c), 0) + 1
    return any(v >= 4 for v in cnt.values())


# 5-consecutive rank_value windows that form a straight (Big 2 order, rank_value
# 1=Three .. 12=Ace, 13=Two). 3-4-5-6-7 .. 10-J-Q-K-A, plus the low wraps
# A-2-3-4-5 and 2-3-4-5-6.
_STRAIGHT_WINDOWS = [tuple(range(s, s + 5)) for s in range(1, 9)]          # 3-7 .. 10-A
_STRAIGHT_WINDOWS += [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]                 # A2345, 23456


def straight_flush_possible(unseen) -> bool:
    """An opponent could hold a straight-flush iff some suit has 5 unseen cards
    whose ranks complete one of the straight windows."""
    by_suit = {1: set(), 2: set(), 3: set(), 4: set()}
    for c in unseen:
        suit = (c - 1) % 4 + 1
        by_suit[suit].add(_rank(c))
    for ranks in by_suit.values():
        for win in _STRAIGHT_WINDOWS:
            if all(r in ranks for r in win):
                return True
    return False


def unbeatable_singles(my_hand, played):
    """Return the sorted list of MY card_ids that cannot be beaten as a single
    play right now (no higher unseen single, and no opponent bomb / straight-flush
    that could over-trump). Typically just the top card(s)."""
    unseen = unseen_cards(my_hand, played)
    if bomb_possible(unseen) or straight_flush_possible(unseen):
        return []  # an over-trump is possible → no single is truly safe
    max_unseen = max(unseen) if unseen else 0
    return sorted(c for c in (int(x) for x in my_hand) if c > max_unseen)


def best_pair_is_nuts(my_hand, played) -> bool:
    """True if I hold a pair whose rank no opponent pair can beat (and no bomb/SF
    over-trump). Pairs compare by rank only."""
    unseen = unseen_cards(my_hand, played)
    if bomb_possible(unseen) or straight_flush_possible(unseen):
        return False
    # my pair ranks
    from collections import Counter
    my_rc = Counter(_rank(int(c)) for c in my_hand)
    my_pair_ranks = [r for r, n in my_rc.items() if n >= 2]
    if not my_pair_ranks:
        return False
    best_my = max(my_pair_ranks)
    # highest rank an opponent could make a pair with
    un_rc = Counter(_rank(c) for c in unseen)
    opp_pair_ranks = [r for r, n in un_rc.items() if n >= 2]
    best_opp = max(opp_pair_ranks) if opp_pair_ranks else 0
    return best_my > best_opp


def flush_possible(unseen) -> bool:
    """An opponent could hold a flush iff some suit has >= 5 unseen cards."""
    by_suit = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in unseen:
        by_suit[(c - 1) % 4 + 1] += 1
    return any(v >= 5 for v in by_suit.values())


def full_house_possible(unseen) -> bool:
    """An opponent could hold a full house iff some rank has >=3 unseen and a
    DIFFERENT rank has >=2 unseen (triple + pair)."""
    cnt = {}
    for c in unseen:
        cnt[_rank(c)] = cnt.get(_rank(c), 0) + 1
    triples = [r for r, n in cnt.items() if n >= 3]
    for t in triples:
        if any(n >= 2 for r, n in cnt.items() if r != t):
            return True
    return False


def _my_straight_top_ranks(my_hand):
    """Top rank_value of every straight my hand can form (each window where I
    hold all 5 ranks). Returns the set of window-top ranks (last element)."""
    have = set(_rank(int(c)) for c in my_hand)
    tops = []
    for win in _STRAIGHT_WINDOWS:
        if all(r in have for r in win):
            tops.append(win[-1])
    return tops


def higher_straight_possible(unseen, my_top_rank) -> bool:
    """Could an opponent complete a straight whose top rank beats mine?
    (Conservative: compares window-top rank only, ignores top-card suit tiebreak.)"""
    un_ranks = set(_rank(c) for c in unseen)
    for win in _STRAIGHT_WINDOWS:
        if win[-1] > my_top_rank and all(r in un_ranks for r in win):
            return True
    return False


def best_straight_is_nuts(my_hand, played) -> bool:
    """True iff I hold a straight that NO possible opponent 5-card play can beat
    right now — i.e. when I lead it, it wins for sure. A straight is beaten by a
    higher straight, ANY flush, ANY full house, a bomb, or a straight-flush, so
    all of those must be impossible from the unseen cards. (This is naturally a
    late-game signal: early on a flush/FH is almost always still possible.)"""
    my_tops = _my_straight_top_ranks(my_hand)
    if not my_tops:
        return False
    unseen = unseen_cards(my_hand, played)
    if (bomb_possible(unseen) or straight_flush_possible(unseen)
            or flush_possible(unseen) or full_house_possible(unseen)):
        return False
    return not higher_straight_possible(unseen, max(my_tops))


def dominance_features(my_hand, played):
    """Compact dominance feature vector for the model / decision logic:
       [0] has at least one unbeatable single          (1/0)
       [1] count of unbeatable singles, normalised /5
       [2] best pair is the nuts                        (1/0)
       [3] an opponent bomb/SF over-trump is possible   (1/0)
    """
    unseen = unseen_cards(my_hand, played)
    overtrump = bomb_possible(unseen) or straight_flush_possible(unseen)
    ubs = unbeatable_singles(my_hand, played)
    return [
        1.0 if ubs else 0.0,
        min(len(ubs), 5) / 5.0,
        1.0 if best_pair_is_nuts(my_hand, played) else 0.0,
        1.0 if overtrump else 0.0,
    ]
