"""AlphaGo-style value dataset (step 2).

AlphaGo trained its value net on (position, game-outcome) pairs where each position
was sampled from a SEPARATE self-play game — exactly one position per game — to break
the heavy correlation between positions within a game (they share the same label).
We replicate that: for each self-play game pick ONE random decision ply, build the
acting player's PUBLIC observation there, and label it with that player's FINAL score
(tanh-normalized). One (obs, target) per game by default.

`samples_per_game > 1` is offered for data-hungry experiments but breaks the strict
AlphaGo decorrelation — keep it at 1 unless you know why you want more.
"""
import bootstrap  # noqa: F401

import json
import os

import numpy as np

from ppo.bc_dataset import _snapshot, _Shim
from ppo.network_cardaware import obs_from_env
from value_model import SCALE

DATA = os.path.join(bootstrap.DATA_DIR, "selfplay_games.jsonl")


def build_value_records(path=DATA, samples_per_game=1, seed=0, max_games=None, verbose=True,
                        post_action=False):
    """post_action=False (default, unchanged): snapshot the state seat s FACES,
    BEFORE applying its action -> trick = the combo s must beat (or empty). This
    is what VALUE_minplays.pt was trained on.

    post_action=True: snapshot the state AFTER s applies its action -> for a play,
    s's hand is the REDUCED hand and trick = s's OWN just-played combo (min-plays
    is then computed on the post-play hand automatically, since add_features reads
    obs['hand_ids']). This is the genuinely in-distribution training set for the
    "post-action value" use — evaluating the value of MY OWN resulting hand right
    after MY candidate action (see post-action-value-decision-design). Label is
    s's final score either way. Only this snapshot TIMING changes; corpus, ply
    sampling, feature setup, tanh target are all identical."""
    rng = np.random.default_rng(seed)
    games = [json.loads(l) for l in open(path)]
    if max_games:
        games = games[:max_games]
    recs, skip = [], 0
    for g in games:
        plays = g["plays"]
        if not plays:
            continue
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        scores = {int(k): float(v) for k, v in g["scores"].items()}
        k = min(samples_per_game, len(plays))
        chosen = set(int(i) for i in rng.choice(len(plays), size=k, replace=False))
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        trick_cards, trick_owner, passed = None, None, set()

        def take(s):
            nonlocal skip
            control = 1 if trick_cards is None else 0
            remaining = {kk: [c for c in hands[kk] if c not in set(played[kk])] for kk in range(4)}
            # An empty acting hand (post_action snapshot of the game-winning play,
            # which just emptied s's hand) has no "resulting hand to evaluate" and
            # makes the hand-attention softmax over an all-padded sequence go NaN.
            # The game is already over there; skip it. Never triggers pre-action
            # (s always holds >=1 card when it's their turn to act).
            if len(remaining[s]) == 0:
                skip += 1
                return
            try:
                gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                recs.append({"obs": obs_from_env(_Shim(gg)),
                             "target": float(np.tanh(scores.get(s, 0.0) / SCALE))})
            except Exception:
                skip += 1

        for t, ev in enumerate(plays):
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            if not post_action and t in chosen:
                take(s)
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cardsPlayed[s][c - 1] = 1
                trick_cards, trick_owner, passed = cards, s, set()
            else:
                passed.add(s)
                if len(passed) >= 3:
                    trick_cards, trick_owner, passed = None, None, set()
            if post_action and t in chosen:
                take(s)
    if verbose:
        mode = "post-action" if post_action else "pre-action"
        print(f"value records ({mode}): {len(recs)} ({samples_per_game}/game) "
              f"from {len(games)} games, skipped {skip}")
    return recs


if __name__ == "__main__":
    build_value_records()
