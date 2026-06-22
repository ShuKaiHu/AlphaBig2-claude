"""Build a behavioral-cloning dataset from the reconstructed online games.

For each decision in online_games.jsonl we know full information (every hand), so
we replay each game with a hand-tracked state machine and, at each target
decision, snapshot a big2Game for the ACTING player and produce a cardaware
training example: (obs, legal actions, label = the action the human chose).

target: "human" (all non-our-agent seats), "winner" (only the game winner),
        "all" (every seat), "ours" (our agent only).
"""
import json
import os

import numpy as np

import enumerateOptions as eo
from big2Game import big2Game, handPlayed
from ppo.network_cardaware import obs_from_env, ACTION_FEATURES
from ppo.network_v6 import belief_target_from_env, unseen_mask_from_obs

DATA = os.path.join(os.path.dirname(__file__), "data", "online_games.jsonl")


class _Shim:
    __slots__ = ("game",)
    def __init__(self, g): self.game = g
    @property
    def current_player(self): return self.game.playersGo


def _snapshot(acting, remaining, cardsPlayed, control, trick_cards, trick_owner, passed):
    """A big2Game snapshot for `acting` (0-3) enough for returnAvailableActions
    + obs_from_env. Engine players are 1-indexed (= seat+1)."""
    g = big2Game.__new__(big2Game)
    g.currentHands = {s + 1: np.sort(np.array(remaining[s], dtype=np.int64)) for s in range(4)}
    g.cardsPlayed = cardsPlayed.copy()
    g.playersGo = acting + 1
    g.control = control
    g.mustPlayClub3 = False
    g.passedThisRound = {p: ((p - 1) in passed) for p in range(1, 5)}
    g.passCount = len(passed)
    g.lastPlayedPlayer = (trick_owner + 1) if trick_owner is not None else acting + 1
    g.gameOver = 0
    g.rewards = np.zeros(4)
    if control == 0 and trick_cards:
        g.goIndex = 2
        g.handsPlayed = {1: handPlayed(np.array(trick_cards, dtype=np.int64), (trick_owner or 0) + 1)}
    else:
        g.goIndex = 1
        g.handsPlayed = {}
    return g


def build_records(target="human", verbose=True):
    games = [json.loads(l) for l in open(DATA)]
    records, n_dec, n_skip = [], 0, 0
    for g in games:
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        our = g["our_seat"]; winner = g["winner"]
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        trick_cards, trick_owner, passed = None, None, set()
        for ev in g["plays"]:
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            control = 1 if trick_cards is None else 0
            want = (target == "all" or (target == "human" and s != our)
                    or (target == "winner" and s == winner)
                    or (target == "ours" and s == our))
            if want:
                n_dec += 1
                remaining = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
                gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                try:
                    mask = gg.returnAvailableActions()
                    legal = np.flatnonzero(mask)
                    label = eo.passInd if act == "pass" else eo.action_index_from_cards(cards)
                    pos = int(np.where(legal == label)[0][0]) if label in legal else -1
                except Exception:
                    pos = -1
                    legal = np.array([])
                if pos < 0:
                    n_skip += 1
                else:
                    shim = _Shim(gg)
                    obs = obs_from_env(shim)
                    records.append({"obs": obs,
                                    "feats": ACTION_FEATURES[legal].astype(np.float32),
                                    "pos": pos,
                                    "belief": belief_target_from_env(shim),  # oracle, loss-only
                                    "bmask": unseen_mask_from_obs(obs)})
            # apply event
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cardsPlayed[s][c - 1] = 1
                trick_cards, trick_owner, passed = cards, s, set()
            else:
                passed.add(s)
                if len(passed) >= 3:
                    trick_cards, trick_owner, passed = None, None, set()
    if verbose:
        print(f"target={target}: examples={len(records)}  "
              f"(label-in-mask {100*len(records)/max(n_dec,1):.1f}% of {n_dec} target decisions; "
              f"skipped {n_skip})")
    return records


if __name__ == "__main__":
    for t in ("human", "winner", "all"):
        build_records(t)
