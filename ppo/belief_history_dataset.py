"""Belief training records AUGMENTED with full play-history.

The deployed belief pipeline (BeliefNet / CardAwareActorCritic, via
`ppo/network_cardaware.py::obs_from_env`) only sees the CURRENT trick plus
cumulative per-opponent played-card masks -- it has no memory of who passed on
what in EARLIER tricks, which is one of the richest inference signals in Big 2
(a pass reveals "I hold nothing that beats X," a strong, persistent
constraint on that player's hand for the rest of the game).

This module builds records like bc_dataset.build_records, but also replays
each game's `plays` list to accumulate, at each decision point, a per-step
history entry: (which player acted, what combo was active on the table that
they had to respond to [raw card_ids, empty if they had control], what they
actually played [raw card_ids, empty if they passed]). No hand-type
classification or strength scoring of any kind is computed -- every step is
just the exact cards involved, letting the model's own card-embedding
+attention machinery (the same pattern already used for obs_from_env's
trick/seen/opp fields) learn whatever structure it needs. This sidesteps two
confirmed bugs in an earlier iteration that reused the old (superseded)
Big2Net line's `engine.features.encode_history_steps`: (1) that function's
`_last_hand_features` computes a play's "high card" via positional indexing
AFTER calling `gameLogic.isFullHouse`, which mutates the hand in place via
`.sort()` -- giving wrong results for A-2-3-4-5 straights and for
four-of-a-kinds whose kicker outranks the quad; (2) even when correct, it only
recorded this "high card" for PLAY events, hardcoding PASS events' hand-type/
rank/suit to all-zero -- discarding exactly the "declined to beat X" signal
this whole history feature exists to capture.
"""
import os

import numpy as np

import enumerateOptions as eo
from ppo.bc_dataset import _snapshot
from ppo.network_cardaware import ACTION_FEATURES, obs_from_env
from ppo.network_v6 import belief_target_from_env, unseen_mask_from_obs

DATA = os.path.join(os.path.dirname(__file__), "data", "online_games.jsonl")

HISTORY_LEN_V2 = 196          # same sequence-length cap as the old engine (49 plays x 4 seats)
HIST_STEP_DIM_V2 = 4 + 52 + 52  # player one-hot + required-combo raw cards + played-combo raw cards


class _Shim:
    __slots__ = ("game",)
    def __init__(self, g): self.game = g
    @property
    def current_player(self): return self.game.playersGo


def _cards_to_vec(card_ids):
    v = np.zeros(52, dtype=np.float32)
    for c in card_ids:
        v[c - 1] = 1.0
    return v


def encode_history_steps_v2(steps, history_len=HISTORY_LEN_V2):
    """steps: list of (seat:int 0-3, required_cards:list[int], played_cards:list[int]),
    in chronological order (one entry per REAL decision -- no forced-skip bookkeeping
    exists in this reconstruction, unlike the live engine's actionHistory).
    Returns (history_len, HIST_STEP_DIM_V2) float32, front-padded with zeros
    (real steps occupy the LAST len(steps) rows, oldest steps drop off the front
    if there are more than history_len) -- same convention as the old engine's
    encode_history_steps.
    """
    out = np.zeros((history_len, HIST_STEP_DIM_V2), dtype=np.float32)
    recent = steps[-history_len:]
    start = history_len - len(recent)
    for i, (seat, req_cards, played_cards) in enumerate(recent):
        row = out[start + i]
        row[seat] = 1.0
        row[4:56] = _cards_to_vec(req_cards)
        row[56:108] = _cards_to_vec(played_cards)
    return out


def build_belief_history_records(games, target="all", history_len=HISTORY_LEN_V2, allowed_keys=None):
    """records: {obs, feats(L,AF), pos, belief(3,52), bmask(52), history(history_len,HIST_STEP_DIM_V2)}.

    allowed_keys: optional set of (run, game_id, seat) triples -- when given, a
    decision is only included if it's ALSO in this set (ANDed with the target
    gate below), e.g. to restrict training to one behavioral-cluster's exact
    (game, seat) decisions rather than every human seat in a game (a single
    game can have human seats from DIFFERENT clusters). Pass target="all" with
    allowed_keys set to let allowed_keys do the entire filtering.
    """
    records = []
    for _gi, g in enumerate(games):
        if _gi % 500 == 0:
            print(f"  build_belief_history: {_gi}/{len(games)} games, {len(records)} records",
                  flush=True)
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        our = g["our_seat"]; winner = g["winner"]
        run = g.get("run"); gid = g.get("id")
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        # engine-matching tracker: a play keeps others' passed flags; only 3
        # CONSECUTIVE passes reset the trick. 神來也 auto-passes already-passed
        # seats (median 4ms) -- those are NOT decisions and must not enter records
        # OR the history steps (the old code cleared passed on every play, used
        # len(passed)>=3, and recorded every auto-pass as a phantom history step).
        trick_cards, trick_owner, passed, consec = None, None, set(), 0
        steps = []  # (seat, required_cards, played_cards) per REAL decision, in order
        for ev in g["plays"]:
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            forced = act == "pass" and s in passed   # server auto-skip, not a decision
            control = 1 if trick_cards is None else 0
            want = (not forced) and (target == "all" or (target == "human" and s != our)
                    or (target == "winner" and s == winner)
                    or (target == "ours" and s == our))
            if want and allowed_keys is not None:
                want = (run, gid, s) in allowed_keys
            if want:
                remaining = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
                gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                try:
                    mask = gg.returnAvailableActions()
                    legal = np.flatnonzero(mask)
                    label = eo.passInd if act == "pass" else eo.action_index_from_cards(cards)
                    pos = int(np.where(legal == label)[0][0]) if label in legal else -1
                except Exception:
                    pos = -1
                if pos >= 0:
                    shim = _Shim(gg)
                    obs = obs_from_env(shim)
                    hist = encode_history_steps_v2(steps, history_len=history_len)
                    records.append({"obs": obs,
                                    "feats": ACTION_FEATURES[legal].astype(np.float32),
                                    "pos": pos,
                                    "belief": belief_target_from_env(shim),
                                    "bmask": unseen_mask_from_obs(obs),
                                    "history": hist,
                                    "gid": gid})           # for game-level train/val split
            # record REAL decisions into the running history BEFORE applying (a
            # forced auto-pass is skipped: the seat already passed this trick, so
            # it carries no new info and would just add phantom pass rows).
            if not forced:
                required_cards = [] if control else list(trick_cards)
                played_cards = cards if act == "play" else []
                steps.append((s, required_cards, played_cards))
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cardsPlayed[s][c - 1] = 1
                trick_cards, trick_owner = cards, s
                passed.discard(s); consec = 0
            else:
                passed.add(s); consec += 1
                if consec >= 3:
                    trick_cards, trick_owner, passed, consec = None, None, set(), 0
    return records
