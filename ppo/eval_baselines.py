"""Evaluation baselines from Patwa, *Self-Play RL under Imperfect Information in
Big 2* (arXiv 2605.28863): Random, Greedy (Algorithm 1), Smart (Algorithm 2).

These are the EVALUATION yardstick, not training opponents. Success criterion
(paper): win_rate > 0.25 AND avg_score > 0 against an opponent class.

Adaptations to our engine's ruleset / representation:
- Card ids are 1..52, id = (rank-1)*4 + suit; rank 1..13 maps to faces 3..2
  (rank 13 == the four '2's, the highest). So `rank(c)` below = paper's rank.
- Action index -> concrete cards via enumerateOptions.getOptionNC.
- Our engine has NO plain flush (removed), so Smart's BreakPenalty skips the
  "flush structure" term the paper mentions; everything else is ported faithfully.
- "low orphan" threshold and "very strong trick" are judgment calls (documented
  inline) since the paper leaves them informal.
"""
import numpy as np

import enumerateOptions as _eo
import gameLogic
from engine.env import Big2Env, PASS_IDX

getOptionNC = _eo.getOptionNC


# ── card helpers ────────────────────────────────────────────────────────────
def _rank(cid: int) -> int:
    """1..13 ; 1 == face '3' (lowest), 13 == face '2' (highest)."""
    return (cid - 1) // 4 + 1


def _is_two(cid: int) -> bool:
    return _rank(cid) == 13


def _phase(hand_size: int) -> str:
    if hand_size > 10:
        return "early"
    if hand_size >= 6:
        return "mid"
    return "late"


def _cards_of(action: int):
    cards, _nc = getOptionNC(action)
    return [int(c) for c in cards]


def _active_trick(env: Big2Env):
    """Cards of the trick to beat, or None when leading (control == 1)."""
    game = env.game
    if game.control == 1:
        return None
    return [int(c) for c in game.handsPlayed[game.goIndex - 1].hand]


def _is_quad_or_sf(cards) -> bool:
    if cards is None or len(cards) != 5:
        return False
    arr = np.array(cards)
    return bool(gameLogic.isFourOfAKind(arr) or gameLogic.isStraightFlush(arr))


def _very_strong_trick(cards) -> bool:
    """Judgment call: a 5-card trick, or one containing a '2', is 'very strong'."""
    if cards is None:
        return False
    return len(cards) == 5 or any(_is_two(c) for c in cards)


# ── Random / Greedy ─────────────────────────────────────────────────────────
def random_action(env: Big2Env) -> int:
    return env.random_action()


def greedy_action(env: Big2Env) -> int:
    """Algorithm 1: if forced, take it; else play the minimum non-pass action
    under the engine's combination ordering (singles < pairs < five; within a
    type, ascending index == ascending comparison key). = weakest legal play."""
    valid = np.flatnonzero(env.get_valid_actions())
    if len(valid) <= 1:
        return int(valid[0]) if len(valid) else PASS_IDX
    non_pass = valid[valid != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX
    return int(non_pass.min())


# ── Smart (Algorithm 2) ─────────────────────────────────────────────────────
def _break_penalty(action_cards, hand, phase: str) -> float:
    """Penalty for an action that breaks a remaining pair/triple or a potential
    five-card structure left in `hand` (hand = full current hand incl. action).
    Returns the single largest applicable penalty (paper adds one BreakPenalty)."""
    take = {}
    for c in action_cards:
        take[_rank(c)] = take.get(_rank(c), 0) + 1
    have = {}
    for c in hand:
        have[_rank(c)] = have.get(_rank(c), 0) + 1

    pen = 0.0
    # pair / triple break: took part of a >=2 group, leaving a broken remainder
    for r, t in take.items():
        before = have.get(r, 0)
        if before >= 2 and 0 < t < before and (before - t) < 2:
            pen = max(pen, 8.0 if phase == "early" else 4.0 if phase == "mid" else 0.0)

    five_pen = 20.0 if phase == "early" else 8.0 if phase == "mid" else 4.0
    # quad break: a rank with 4 in hand, action takes 1..3 of them
    for r, t in take.items():
        if have.get(r, 0) == 4 and 0 < t < 4:
            pen = max(pen, five_pen)
    # straight-window break: 5 consecutive ranks (excl. rank 13='2') each present
    # in hand, where the action removes the last copy of one of those ranks.
    after = {r: have.get(r, 0) - take.get(r, 0) for r in have}
    for lo in range(1, 9):  # windows lo..lo+4 within ranks 1..12 (exclude 13)
        window = range(lo, lo + 5)
        if all(have.get(r, 0) >= 1 for r in window):
            if any(after.get(r, 0) == 0 and take.get(r, 0) > 0 for r in window):
                pen = max(pen, five_pen)
                break
    return pen


def _low_orphans(action_cards, hand) -> int:
    """Count of LOW ranks (face 3..8, i.e. rank<=6) left as singletons after the
    action. 'low' threshold is a documented judgment call."""
    take = {}
    for c in action_cards:
        take[_rank(c)] = take.get(_rank(c), 0) + 1
    have = {}
    for c in hand:
        have[_rank(c)] = have.get(_rank(c), 0) + 1
    n = 0
    for r in range(1, 7):  # ranks 1..6 == faces 3..8
        if have.get(r, 0) - take.get(r, 0) == 1:
            n += 1
    return n


def smart_action(env: Big2Env) -> int:
    """Algorithm 2: score each non-pass action (lower = better), pick the min,
    with narrow voluntary-pass cases."""
    game = env.game
    me = env.current_player
    mask = env.get_valid_actions()
    valid = np.flatnonzero(mask)
    non_pass = [int(a) for a in valid if a != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX
    if len(valid) == 1:
        return int(valid[0])

    hand = [int(c) for c in game.currentHands[me]]
    nH = len(hand)
    phase = _phase(nH)
    trick = _active_trick(env)
    trick_is_bomb = _is_quad_or_sf(trick)
    trick_very_strong = _very_strong_trick(trick)

    scores = {}
    for a in non_pass:
        cards = _cards_of(a)
        if len(cards) == nH:
            scores[a] = -1000.0  # immediate win
            continue
        s = 0.8 * sum(_rank(c) for c in cards)
        twos = sum(1 for c in cards if _is_two(c))
        if phase == "early":
            s += 10 * twos
        elif phase == "mid":
            s += 5 * twos
        s += _break_penalty(cards, hand, phase)
        s += 6 * _low_orphans(cards, hand)
        s += -4 * len(cards)
        if phase == "late":
            s -= 10
        if trick_very_strong and phase == "late":
            s -= 10
        if trick_is_bomb:
            s += 25
        scores[a] = s

    a_star = min(non_pass, key=lambda a: scores[a])
    pass_legal = bool(mask[PASS_IDX])
    star_twos = sum(1 for c in _cards_of(a_star) if _is_two(c))

    if pass_legal and phase == "early" and star_twos >= 2 and scores[a_star] > 30:
        return PASS_IDX
    if pass_legal and trick_is_bomb:
        return PASS_IDX
    return int(a_star)


OPPONENTS = {
    "random": random_action,
    "greedy": greedy_action,
    "smart": smart_action,
}


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(agent_fn, opponent_fn, n_games: int = 1000):
    """Roll out n_games; the agent holds one seat (rotated for balance), the
    opponent_fn holds the other three. Returns (win_rate, avg_score) for the
    agent's seat. Success (paper): win_rate > 0.25 and avg_score > 0."""
    env = Big2Env()
    wins = 0
    scores = []
    for g in range(n_games):
        seat = (g % 4) + 1
        env.reset()
        while not env.done:
            p = env.current_player
            a = agent_fn(env) if p == seat else opponent_fn(env)
            rewards, done = env.step(a)
            if done:
                sc = float(rewards[seat - 1])
                scores.append(sc)
                wins += int(sc > 0)
    return wins / n_games, float(np.mean(scores)) if scores else 0.0


def evaluate_vs_all(agent_fn, n_games: int = 1000):
    """Evaluate the agent against every opponent class. Returns
    {name: (win_rate, avg_score)}."""
    return {name: evaluate(agent_fn, fn, n_games) for name, fn in OPPONENTS.items()}
