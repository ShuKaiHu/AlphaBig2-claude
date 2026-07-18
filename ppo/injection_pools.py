"""Loaders for the M3 injection pools (Track B).

Pools are built by PPO/build_injection_pools.py from a frozen snapshot of the
real online human games (see that file's docstring for the extraction rules)
and live in ppo/data/injection_d2_pool.jsonl / injection_d1_pool.jsonl.

Split discipline: 10% of GAME IDS are held out as split="eval" (deterministic:
md5(game_id) % 10 == 0). Training must inject ONLY from split="train"; the
eval subset is reserved for injected-state G2 evaluation on unseen positions.

D2 record (mid-game caged-loss position; all card ids 1..52, seats 0..3):
  pool             "d2"
  game_id, run     source game (online_games.jsonl id / harvest run)
  split            "train" | "eval"  (by game id)
  seat             the caged-loss target seat (final remaining >= 10, dealt
                   >= 1 two) -- the learner sits HERE when injecting
  final_score      that seat's final platform score in the source game
  final_remaining  cards that seat still held at game end (>= 10)
  n_twos_dealt / n_twos_remaining   2s (ids 49-52) dealt / still held at the
                   snapshot by the target seat
  event_idx        index into the source game's plays[] of the decision event
                   the snapshot precedes
  turn             seat to move (had >= 1 legal action -- validated at build)
  hands            {"0".."3": [remaining card ids]} at the snapshot
  cards_played     {"0".."3": [card ids that seat has already played]}
  control          1 = trick reset (turn seat leads), 0 = must beat trick_cards
  trick_cards      cards of the live trick's last play (null iff control=1)
  trick_owner      seat that played trick_cards (null iff control=1)
  passed           seats already out of the current trick (play does NOT
                   reopen; trick resets only after 3 consecutive passes)
  n_legal          number of legal actions for `turn` (build-time check value)

D1 record (strong full deal; learner sits at strong_seat, fresh game):
  pool             "d1"
  game_id, run, split   as above
  strong_seat      seat whose starting 13 cards have min_plays <= 6 and no
                   bomb (canonical AlphaBig2-Value/hand_features.py detectors)
  min_plays        min_plays_to_empty of that hand (4..6)
  hands            {"0".."3": [13 card ids]} the full original deal
"""
import json
import os

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
D2_PATH = os.path.join(_DATA_DIR, "injection_d2_pool.jsonl")
D1_PATH = os.path.join(_DATA_DIR, "injection_d1_pool.jsonl")


def _load(path, split):
    if split not in ("train", "eval", "all"):
        raise ValueError(f"split must be train|eval|all, got {split!r}")
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if split == "all" or r["split"] == split:
                out.append(r)
    return out


def load_d2_pool(split="train", path=D2_PATH):
    """Mid-game caged-loss positions. split: "train" (injectable), "eval"
    (G2-evaluation only -- never inject during training), or "all"."""
    return _load(path, split)


def load_d1_pool(split="train", path=D1_PATH):
    """Strong-deal full-deal records (learner sits at strong_seat)."""
    return _load(path, split)


def d2_state_args(rec):
    """Decode a D2 record into the argument tuple of ppo.bc_dataset._snapshot:
    (acting, remaining, cardsPlayed, control, trick_cards, trick_owner, passed).
    Feed via  _snapshot(*d2_state_args(rec))  to get a big2Game snapshot that
    supports returnAvailableActions / obs_from_env for the seat to move."""
    remaining = {s: list(map(int, rec["hands"][str(s)])) for s in range(4)}
    cardsPlayed = np.zeros((4, 52), dtype=np.int64)
    for s in range(4):
        for c in rec["cards_played"][str(s)]:
            cardsPlayed[s][int(c) - 1] = 1
    return (int(rec["turn"]), remaining, cardsPlayed, int(rec["control"]),
            rec["trick_cards"], rec["trick_owner"], set(rec["passed"]))
