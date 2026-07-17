"""Build (public obs, oracle-dominance) training pairs from self-play games.

Same showdown-supervised paradigm as the belief model: replay each game, and at
sampled decisions snapshot the full (reconstructed) state, build the acting player's
PUBLIC obs, and compute the ORACLE dominance label from the true hands. The model
then learns P(nuts | public info) — the soft dominance probability.
"""
import bootstrap  # noqa: F401

import json
import os

import numpy as np

from ppo.bc_dataset import _snapshot, _Shim
from ppo.network_cardaware import obs_from_env
from dominance_oracle import oracle_nuts, LABELS

DATA = os.path.join(bootstrap.DATA_DIR, "selfplay_games.jsonl")


def build_dominance_records(path=DATA, samples_per_game=4, seed=0, max_games=None, verbose=True):
    rng = np.random.default_rng(seed)
    games = [json.loads(l) for l in open(path)]
    if max_games:
        games = games[:max_games]
    recs = []
    for g in games:
        plays = g["plays"]
        if not plays:
            continue
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        k = min(samples_per_game, len(plays))
        chosen = set(int(i) for i in rng.choice(len(plays), size=k, replace=False))
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        trick_cards, trick_owner, passed = None, None, set()
        for t, ev in enumerate(plays):
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            if t in chosen:
                control = 1 if trick_cards is None else 0
                remaining = {kk: [c for c in hands[kk] if c not in set(played[kk])] for kk in range(4)}
                if remaining[s]:                       # acting player still has cards
                    gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                    my = remaining[s]
                    opps = [remaining[kk] for kk in range(4) if kk != s]
                    try:
                        recs.append({"obs": obs_from_env(_Shim(gg)),
                                     "target": np.array(oracle_nuts(my, opps), dtype=np.float32)})
                    except Exception:
                        pass
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
        tgt = np.stack([r["target"] for r in recs])
        rates = dict(zip(LABELS, (100 * tgt.mean(0)).round(1)))
        print(f"dominance records: {len(recs)} ({samples_per_game}/game) from {len(games)} games | "
              f"positive-rate {rates}")
    return recs


if __name__ == "__main__":
    build_dominance_records()
