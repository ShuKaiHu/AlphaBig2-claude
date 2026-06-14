"""Combo-break rate: how often does the agent play a single/pair that dismantles
a 5-card combo (straight / flush / full house / quad) it was holding — when a
non-breaking alternative existed?  (User observation 2026-06-14: the agent often
breaks its straights/full houses by shedding a single or pair from them.)

Metric per self play of <5 cards, from mcts_moves.jsonl (my_hand + chosen):
  n5(H)        = # of distinct 5-card combos the hand can form (engine: lead-mode
                 returnAvailableActions, count of 5-card actions)
  n5(H - C)    = same after removing the played cards
  broke        = n5 dropped
  avoidable    = broke AND some other legal single/pair of the SAME size removed
                 from H would NOT have dropped n5 (a non-breaking shed existed)

Usage: python combo_break.py [--since 2026-06-14T16:19]
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import enumerateOptions as eo
from big2Game import big2Game

_SUIT = {"♣": 1, "♦": 2, "♥": 3, "♠": 4}
_RANK = {"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"10":8,"J":9,"Q":10,"K":11,"A":12,"2":13}


def _label_to_id(lbl: str) -> int:
    suit = _SUIT[lbl[0]]
    rank = _RANK[lbl[1:]]
    return (rank - 1) * 4 + suit


def _n5(card_ids) -> int:
    """# of legal 5-card plays this hand can lead (= 5-card combo potential)."""
    if len(card_ids) < 5:
        return 0
    g = big2Game.__new__(big2Game)
    g.currentHands = {1: np.array(sorted(card_ids), dtype=np.int64)}
    g.playersGo = 1
    g.control = 1            # leading → all combos enumerated
    g.goIndex = 1
    g.handsPlayed = {}
    g.passedThisRound = {1: False, 2: False, 3: False, 4: False}
    g.mustPlayClub3 = False
    g.club3Player = 1
    mask = g.returnAvailableActions()
    return int(sum(1 for a in np.flatnonzero(mask) if a != eo.passInd
                   and eo.getOptionNC(int(a))[1] == 5))


def _singles_pairs(card_ids):
    """All single and pair plays available from the hand (as card-id tuples)."""
    by_rank = {}
    for c in card_ids:
        by_rank.setdefault((c - 1) // 4, []).append(c)
    singles = [(c,) for c in card_ids]
    pairs = []
    for cs in by_rank.values():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pairs.append((cs[i], cs[j]))
    return singles, pairs


def measure(path, since):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("ts", "") >= since]
    plays = broke = avoidable = 0
    by_combo = {"順子/同花/葫蘆/四條(任一)": 0}
    examples = []
    for r in rows:
        ch = r.get("chosen")
        if not isinstance(ch, list) or len(ch) >= 5:
            continue
        try:
            hand = [_label_to_id(x) for x in r["my_hand"]]
            played = [_label_to_id(x) for x in ch]
        except Exception:
            continue
        if not set(played) <= set(hand):
            continue
        n5_before = _n5(hand)
        if n5_before == 0:
            continue                      # no combo to break
        plays += 1
        after = [c for c in hand if c not in set(played)]
        n5_after = _n5(after)
        if n5_after < n5_before:
            broke += 1
            # could a same-size shed have avoided the drop?
            singles, pairs = _singles_pairs(hand)
            alts = singles if len(played) == 1 else pairs
            non_breaking = any(
                _n5([c for c in hand if c not in set(alt)]) == n5_before
                for alt in alts if set(alt) != set(played))
            if non_breaking:
                avoidable += 1
                if len(examples) < 6:
                    examples.append((r["my_hand"], ch, n5_before, n5_after, r["control"]))
    print(f"{path}  (since {since})")
    print(f"  出單張/對子 且手上原有五張組合 的決策 = {plays}")
    print(f"  其中拆掉了組合(五張數下降) = {broke} ({broke/max(1,plays)*100:.0f}%)")
    print(f"  且當下有不拆的替代出法(可避免) = {avoidable} ({avoidable/max(1,plays)*100:.0f}%)")
    for h, c, b, a, ctl in examples:
        print(f"    {ctl}: 手{h} 出{c}  五張潛力 {b}→{a}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude/artifacts/mcts_moves.jsonl")
    ap.add_argument("--since", default="2026-06-14T16:19")
    a = ap.parse_args()
    measure(a.path, a.since)
