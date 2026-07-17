"""Per-action feature table for the card-aware (dot-product) policy.

Each of the ACTION_SIZE (14739) actions is encoded as a fixed feature vector.
The card-aware net runs a small MLP over these features and scores each LEGAL
action by dot product with the state embedding — so two near-identical actions
(e.g. the 1024 suit-variants of one straight) get near-identical features and
share learning, instead of being 1024 unrelated output slots in a fixed head.

Feature layout (AF = 64):
  [0:52]   multi-hot of the cards the action plays
  [52:56]  type one-hot   [single, pair, five, pass]
  [56:60]  five subtype   [straight, fullhouse, quad, straightflush]
  [60]     top rank / 13
  [61]     n_cards / 5
  [62]     n_twos / 4
  [63]     is_pass
The table is deterministic, so it's computed once and cached to disk.
"""
import os

import numpy as np

import enumerateOptions as _eo
import gameLogic

AF = 64
ACTION_SIZE = _eo.passInd + 1
PASS_IDX = _eo.passInd
_CACHE = os.path.join(os.path.dirname(__file__), "action_features.npy")


def _rank(cid: int) -> int:
    return (cid - 1) // 4 + 1


def _five_subtype(cards):
    arr = np.array(cards)
    if gameLogic.isStraightFlush(arr):
        return 3
    if gameLogic.isFourOfAKind(arr):
        return 2
    if gameLogic.isFullHouse(arr)[0]:
        return 1
    return 0  # straight (engine has no plain flush)


def _build():
    tbl = np.zeros((ACTION_SIZE, AF), dtype=np.float32)
    for a in range(ACTION_SIZE):
        if a == PASS_IDX:
            tbl[a, 55] = 1.0  # type=pass
            tbl[a, 63] = 1.0  # is_pass
            continue
        cards, nc = _eo.getOptionNC(a)
        cards = [int(c) for c in cards]
        for c in cards:
            tbl[a, c - 1] = 1.0
        if nc == 1:
            tbl[a, 52] = 1.0
        elif nc == 2:
            tbl[a, 53] = 1.0
        else:  # nc == 5
            tbl[a, 54] = 1.0
            tbl[a, 56 + _five_subtype(cards)] = 1.0
        tbl[a, 60] = max(_rank(c) for c in cards) / 13.0
        tbl[a, 61] = nc / 5.0
        tbl[a, 62] = sum(1 for c in cards if _rank(c) == 13) / 4.0
    return tbl


def load_table() -> np.ndarray:
    if os.path.exists(_CACHE):
        tbl = np.load(_CACHE)
        if tbl.shape == (ACTION_SIZE, AF):
            return tbl
    tbl = _build()
    try:
        np.save(_CACHE, tbl)
    except OSError:
        pass
    return tbl


ACTION_FEATURES = load_table()
