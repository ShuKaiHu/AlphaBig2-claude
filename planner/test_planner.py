"""Unit tests for planner/ (decompose + control + agent smoke).

Run: python -m planner.test_planner
Card ids: id = (rank-1)*4 + suit; rank 1..13 = faces 3..2; suit 1♣ 2♦ 3♥ 4♠.
"""
import numpy as np

from planner.control import guaranteed_hold
from planner.decompose import partition_hand


def cid(rank, suit):
    return (rank - 1) * 4 + suit


ALL = set(range(1, 53))


def unseen_from(my, extra_seen=()):
    return ALL - set(my) - set(extra_seen)


def test_spade2_not_unbeatable_when_bomb_out():
    s2 = cid(13, 4)                       # ♠2
    my = [s2]
    unseen = unseen_from(my)              # everything else unseen → bombs possible
    assert not guaranteed_hold("single", [s2], unseen), \
        "♠2 must NOT be guaranteed while an unseen quad/SF is possible"


def test_spade2_guaranteed_when_no_bomb_possible():
    s2 = cid(13, 4)
    my = [s2]
    # opponents hold only 3 cards of distinct ranks, no SF window completable
    unseen = {cid(1, 1), cid(5, 2), cid(9, 3)}
    assert guaranteed_hold("single", [s2], unseen)


def test_23456_is_top_straight():
    # my 2-3-4-5-6 (mixed suits); unseen can complete 10-J-Q-K-A but NOT any
    # bomb/SF → must still be guaranteed (23456 outranks every other window).
    my = [cid(13, 1), cid(1, 2), cid(2, 3), cid(3, 4), cid(4, 1)]
    unseen = {cid(8, 1), cid(9, 2), cid(10, 3), cid(11, 4), cid(12, 1)}
    assert guaranteed_hold("straight", my, unseen)


def test_straight_beaten_by_higher_window():
    # my 3-4-5-6-7; unseen completes 4-5-6-7-8 (higher window), no bomb/SF
    my = [cid(1, 1), cid(2, 2), cid(3, 3), cid(4, 4), cid(5, 1)]
    unseen = {cid(2, 1), cid(3, 2), cid(4, 2), cid(5, 3), cid(6, 4)}
    assert not guaranteed_hold("straight", my, unseen)


def test_pair_suit_tiebreak():
    # my pair of Kings (♣♦); the two unseen kings (♥♠) form a higher same-rank pair
    my = [cid(11, 1), cid(11, 2)]
    unseen = {cid(11, 3), cid(11, 4), cid(1, 1)}
    assert not guaranteed_hold("pair", my, unseen)
    # holding ♥K♠K instead: same-rank beat impossible; no higher pair/bomb/SF
    my2 = [cid(11, 3), cid(11, 4)]
    unseen2 = {cid(11, 1), cid(11, 2), cid(1, 1)}
    assert guaranteed_hold("pair", my2, unseen2)


def test_quad_beaten_only_by_higher_quad_or_sf():
    my = [cid(10, 1), cid(10, 2), cid(10, 3), cid(10, 4), cid(1, 1)]  # QQQQ+3
    unseen_hi_quad = {cid(12, 1), cid(12, 2), cid(12, 3), cid(12, 4), cid(2, 1)}
    assert not guaranteed_hold("quad", my, unseen_hi_quad)
    unseen_low = {cid(2, 1), cid(3, 2), cid(4, 3)}
    assert guaranteed_hold("quad", my, unseen_low)


def test_partition_covers_hand():
    rng = np.random.default_rng(0)
    for _ in range(200):
        hand = rng.choice(np.arange(1, 53), size=13, replace=False)
        sets = partition_hand(hand)
        flat = sorted(c for _, cs in sets for c in cs)
        assert flat == sorted(int(c) for c in hand)
        from value.hand_features import min_plays_to_empty
        assert len(sets) == min_plays_to_empty(hand)


def test_big2mdp_selfplay_smoke():
    from engine.env import Big2Env
    from planner.big2mdp.agent import Big2MDPAgent
    from planner.big2mdp.selfplay import play_one_game
    from planner.big2mdp.store import RecordStore

    store = RecordStore()
    agents = [Big2MDPAgent(store, level=4) for _ in range(4)]
    env = Big2Env()
    for _ in range(15):
        rewards = play_one_game(env, agents, store)
        assert abs(float(np.sum(rewards))) < 1e-6      # zero-sum settlement
    assert store.n_games == 15 and len(store) > 15 * 20
    # successor chains stay within bounds and are strictly forward
    for i in range(len(store)):
        s = store.succ[i]
        assert s == -1 or (i < s < len(store))


def test_big2mdp_store_roundtrip_and_tree(tmpdir="/tmp"):
    import os
    from planner.big2mdp.store import RecordStore
    from planner.big2mdp.tree import evaluate

    st = RecordStore()
    feats = (None, 3, 13, (13, 13, 13))
    ak = (1, "single", 13)
    # two chains: play the 2-single then win next round / lose next round
    i0 = st.add_record(feats, ak, won=True, score=9.0, npass=3, succ=-1)
    st.add_record(feats, ak, won=False, score=-4.0, npass=0, succ=-1)
    st.add_record((None, 4, 12, (13, 13, 12)), ak, won=True, score=9.0,
                  npass=3, succ=i0)
    p = os.path.join(tmpdir, "big2mdp_records_test.pkl")
    st.save(p)
    st2 = RecordStore.load(p)
    assert len(st2) == 3 and st2.succ[2] == i0
    root = st2.retrieve(feats, cap=100)
    tree = evaluate(st2, root)
    assert ak in tree
    s = tree[ak]
    assert 0.0 < s.p_win <= 1.0 and s.rc > 0.0
    os.remove(p)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all planner tests passed")
