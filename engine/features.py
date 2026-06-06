import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gameLogic

# ── Dimensions ────────────────────────────────────────────────────────────────
# Tier-A card-dominance features are now BAKED IN (always on) — part of the model
# structure, not an external toggle. STATIC_DIM is permanently 306. (The old
# pre-dominance +7.1 baseline is 302-dim and must be loaded with an older
# checkout; it lives in checkpoints/saved/baseline_v7.1.pt for reference.)
# The BIG2_DOMINANCE env var is kept only as an escape hatch: set it to "0" to
# fall back to 302-dim if you ever need to load that legacy baseline.
import os as _os
DOMINANCE_ON = _os.environ.get("BIG2_DOMINANCE") != "0"   # default ON
_DOMINANCE_DIM = 4 if DOMINANCE_ON else 0

STATIC_DIM = 302 + _DOMINANCE_DIM   # static game-state features (306 with dominance)
HIST_STEP_DIM = 29    # per-step history encoding
HISTORY_LEN = 196     # theoretical max game length (49 plays × 4 steps/play)
GRU_HIDDEN = 128      # GRU output dimension
TOTAL_DIM = STATIC_DIM + GRU_HIDDEN

# Card rank/suit helpers
# card_id in [1..52]: rank = ceil(id/4) in [1..13], suit = id%4 in {0,1,2,3}
# rank 1=3, 2=4, ..., 12=A, 13=2  (Big 2 ordering)
# suit 0=spades(?), 1=clubs, 2=diamonds, 3=hearts — check enumerateOptions
# We don't need the exact meaning; we just encode them consistently.


def _card_rank(card_id: int) -> int:
    return int(np.ceil(card_id / 4))  # 1..13


def _card_suit(card_id: int) -> int:
    return int(card_id % 4)  # 0..3


def _hand_type_idx(hand) -> int:
    """
    Returns index 0–6:
      0=none, 1=single, 2=pair, 3=straight, 4=flush, 5=full-house, 6=four-of-kind, 7=straight-flush
    (7 dims for 8 types, idx 0 = none)
    """
    nC = len(hand)
    if nC == 0:
        return 0
    if nC == 1:
        return 1
    if nC == 2:
        return 2
    if nC == 5:
        if gameLogic.isStraightFlush(hand):
            return 7
        if gameLogic.isFourOfAKind(hand):
            return 6
        if gameLogic.isFullHouse(hand)[0]:
            return 5
        if gameLogic.isFlush(hand):
            return 4
        return 3  # straight
    return 1  # fallback (3-card etc.)


def _last_hand_features(hand) -> tuple:
    """
    Returns (type_idx, rank, suit) for the last played hand's highest card.
    type_idx: 0-7
    rank: 1-13
    suit: 0-3
    """
    nC = len(hand)
    type_idx = _hand_type_idx(hand)
    if nC == 1:
        return type_idx, _card_rank(hand[0]), _card_suit(hand[0])
    if nC == 2:
        return type_idx, _card_rank(hand[1]), _card_suit(hand[1])
    if nC == 5:
        if gameLogic.isFullHouse(hand)[0]:
            # High card is the triple's rank
            return type_idx, _card_rank(hand[2]), 0
        return type_idx, _card_rank(hand[4]), _card_suit(hand[4])
    return type_idx, _card_rank(hand[-1]), _card_suit(hand[-1])


def encode_static(game, player: int) -> np.ndarray:
    """
    Encode the static game state for `player` as a 302-dim float32 vector.

    Layout:
      [0:52]      my hand (card present bits)
      [52:260]    each player's played cards (4 × 52)
      [260:264]   remaining card counts per player, normalized by 13
      [264:268]   current player one-hot
      [268]       control flag
      [269]       must-play-club3 flag
      [270:274]   last-played-player one-hot
      [274:278]   passed-this-round flags
      [278:286]   last hand type (8 dims, one-hot, idx 0 = no hand)
      [286:299]   last hand high-card rank (13 dims one-hot)
      [299:303]   last hand high-card suit (4 dims one-hot)
    Total: 52 + 208 + 4 + 4 + 1 + 1 + 4 + 4 + 8 + 13 + 4 = 303... let me recount

    Actually:
    52 + 208 + 4 + 4 + 1 + 1 + 4 + 4 + 8 + 13 + 3 = 302
    (suit is 0-3 → 4 dims but last dim reused... let's just use 4)
    52+208+4+4+1+1+4+4+8+13+4 = 303. Round to 302 by using 3 suit dims.

    We'll use exactly: 52+208+4+4+1+1+4+4+8+13+3 = 302.
    Suit: 3 one-hot dims for clubs/diamonds/hearts; spades = all zeros.
    """
    g = game
    feat = np.zeros(STATIC_DIM, dtype=np.float32)
    idx = 0

    # My hand: 52 bits
    for card in g.currentHands[player]:
        feat[idx + card - 1] = 1.0
    idx += 52  # → 52

    # Each player's played cards: 4 × 52 = 208
    for p in range(1, 5):
        feat[idx: idx + 52] = g.cardsPlayed[p - 1].astype(np.float32)
        idx += 52  # → 260

    # Remaining counts, normalized
    for p in range(1, 5):
        feat[idx + p - 1] = g.currentHands[p].size / 13.0
    idx += 4  # → 264

    # Current player one-hot
    feat[idx + g.playersGo - 1] = 1.0
    idx += 4  # → 268

    # Control flag
    feat[idx] = float(g.control)
    idx += 1  # → 269

    # Must-play-club3 flag
    feat[idx] = float(g.mustPlayClub3)
    idx += 1  # → 270

    # Last-played-player one-hot
    feat[idx + g.lastPlayedPlayer - 1] = 1.0
    idx += 4  # → 274

    # Passed-this-round flags
    for p in range(1, 5):
        feat[idx + p - 1] = float(g.passedThisRound[p])
    idx += 4  # → 278

    # Last hand type + high-card rank + suit  (8 + 13 + 3 = 24 dims)
    if g.goIndex > 1 and (g.goIndex - 1) in g.handsPlayed:
        hand = g.handsPlayed[g.goIndex - 1].hand
        t, rank, suit = _last_hand_features(hand)
        if t < 8:
            feat[idx + t] = 1.0
        idx += 8  # → 286
        if 1 <= rank <= 13:
            feat[idx + rank - 1] = 1.0
        idx += 13  # → 299
        # suit 1=clubs,2=diamonds,3=hearts → dims 0,1,2; suit 0=spades → all zero
        if 1 <= suit <= 3:
            feat[idx + suit - 1] = 1.0
        idx += 3  # → 302
    else:
        idx += 24  # no last hand, all zeros

    # ── Tier-A card dominance (opt-in) ──────────────────────────────────────
    # 4 dims: [has unbeatable single, #unbeatable/5, best pair is nuts,
    #          opponent bomb/SF over-trump possible]. Computed from public info
    # (my hand + all played cards) — the human "will I get beaten?" signal.
    if DOMINANCE_ON:
        from engine import dominance as _dom
        my_hand = [int(c) for c in g.currentHands[player]]
        played = [c for p in range(4)
                  for c in range(1, 53) if g.cardsPlayed[p][c - 1]]
        feat[idx: idx + 4] = np.asarray(
            _dom.dominance_features(my_hand, played), dtype=np.float32)
        idx += 4

    assert idx == STATIC_DIM, f"Static dim mismatch: {idx} != {STATIC_DIM}"
    return feat


def encode_history_steps(game, history_len: int = HISTORY_LEN) -> np.ndarray:
    """
    Encode the last `history_len` action-history entries into shape (history_len, HIST_STEP_DIM).

    Per-step layout (29 dims):
      [0:4]   player one-hot
      [4]     pass flag
      [5:13]  hand type one-hot (8 dims, idx 0 = single since pass is separate)
      [13:26] high-card rank one-hot (13 dims)
      [26:29] high-card suit one-hot (3 dims, spades = all zero)

    Steps are zero-padded at the front if fewer than history_len actions exist.
    """
    steps = np.zeros((history_len, HIST_STEP_DIM), dtype=np.float32)
    history = game.actionHistory
    # Take the last history_len non-forced-skip steps (or all if fewer)
    entries = [e for e in history if not e.get('forced_skip', False)]
    entries = entries[-history_len:]

    for i, entry in enumerate(entries):
        s = steps[i]
        p = entry['player']
        s[p - 1] = 1.0  # player one-hot [0:4]

        if entry['pass']:
            s[4] = 1.0  # pass flag [4]
            # hand type / rank / suit all zero
        else:
            hand = entry['hand']
            t, rank, suit = _last_hand_features(hand)
            # hand type in [5:13] (offset 5, 8 dims)
            if t < 8:
                s[5 + t] = 1.0
            # rank in [13:26]
            if 1 <= rank <= 13:
                s[13 + rank - 1] = 1.0
            # suit in [26:29]
            if 1 <= suit <= 3:
                s[26 + suit - 1] = 1.0

    return steps
