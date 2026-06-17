import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import enumerateOptions
from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_history_steps
from engine.mcts import MCTS

REWARD_SCALE = 13.0
LAMBDA_TD = 0.9


def _smart_heuristic_action(env: Big2Env) -> int:
    """A stronger, more human-like heuristic — used to stress-test the model
    against an opponent that is NOT the weak 'always lowest' pattern it may have
    overfit to.

    Lead (control==1): shed the LOWEST single first (probe cheaply, keep combos
      and high control cards like 2s for later). Only break into pairs / 5-card
      combos when no singles remain. This preserves combos and high cards rather
      than dumping a random 5-card combo early (the weak heuristic's mistake).
    Follow (control==0): minimal winning play (lowest action index already skips
      bombs since they sort highest), i.e. never overpay / waste a bomb.
    """
    import enumerateOptions as _eo
    valid = np.flatnonzero(env.get_valid_actions())
    non_pass = valid[valid != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX
    game = env.game
    if game.control == 1:
        singles = [a for a in non_pass if a < _eo.PAIR_OFFSET]
        if singles:
            return int(min(singles))                      # lowest single
        pairs = [a for a in non_pass if _eo.PAIR_OFFSET <= a < _eo.FIVE_OFFSET]
        if pairs:
            return int(min(pairs))                         # lowest pair
        return int(min(non_pass))                          # lowest 5-card
    return int(non_pass.min())                             # minimal winning


def _heuristic_action(env: Big2Env) -> int:
    """Smarter heuristic: prefer combos over singles, play minimum winning hand.

    When leading (control=1):
      5-card combo (random choice) > pair (lowest) > single (lowest)
    When following (control=0):
      minimum valid combo that beats the table (smallest winning action index)
      pass if nothing beats the table
    """
    import enumerateOptions as _eo
    valid_mask = env.get_valid_actions()
    valid = np.flatnonzero(valid_mask)
    non_pass = valid[valid != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX

    game = env.game
    if game.control == 1:
        # Leading: split by combo size
        five_card = [a for a in non_pass if a >= _eo.FIVE_OFFSET]
        pairs      = [a for a in non_pass if _eo.PAIR_OFFSET <= a < _eo.FIVE_OFFSET]
        singles    = [a for a in non_pass if a < _eo.PAIR_OFFSET]
        if five_card:
            return int(np.random.choice(five_card))   # random 5-card combo
        if pairs:
            return int(pairs[0])                       # lowest pair
        return int(singles[0])                         # lowest single
    else:
        # Following: play minimum winning hand (lowest index that beats table)
        return int(non_pass[0])


def _combo_aware_heuristic_action(env: Big2Env) -> int:
    """Like _smart_heuristic_action, but when FOLLOWING a SMALL (single/pair)
    lead it will PASS rather than break a valuable 5-card combo to win a cheap
    trick — the human 'hold your straight/full-house/bomb' instinct.

    This is the V9e enabler: pure/weak-heuristic self-play never punishes
    combo-wasting (vs opponents that can't capitalise, breaking a combo to grab
    a small trick is locally free), so V6–V9d learned to break combos / never
    pass. An opponent (and BC target) that itself HOLDS combos both demonstrates
    strategic passing AND plays better, so wasting a combo finally costs the
    learner reward.

    Lead (control==1): identical to _smart (shed lowest single, keep combos).
    Follow (control==0): lowest winning play that does NOT shatter a strong
      (>=0.5) 5-card combo; if every winning play would, hold (PASS).
    """
    import enumerateOptions as _eo
    from engine.dominance import combo_strength
    valid = np.flatnonzero(env.get_valid_actions())
    non_pass = valid[valid != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX
    game = env.game
    me = env.current_player
    if game.control == 1:
        singles = [a for a in non_pass if a < _eo.PAIR_OFFSET]
        if singles:
            return int(min(singles))
        pairs = [a for a in non_pass if _eo.PAIR_OFFSET <= a < _eo.FIVE_OFFSET]
        if pairs:
            return int(min(pairs))
        return int(min(non_pass))
    # Follow: only run the (cloning) combo check vs a small lead while actually
    # holding a strong combo — otherwise just minimal-win like _smart.
    prev_hand = game.handsPlayed[game.goIndex - 1].hand
    if len(prev_hand) <= 2:
        my_hand = [int(c) for c in game.currentHands[me]]
        played = [c for p in range(4) for c in range(1, 53)
                  if game.cardsPlayed[p][c - 1]]
        before = combo_strength(my_hand, played)
        if before >= 0.5:
            for cand in sorted(int(a) for a in non_pass):
                e2 = env.clone()
                e2.step(cand)
                after = combo_strength(
                    [int(c) for c in e2.game.currentHands[me]],
                    [c for p in range(4) for c in range(1, 53)
                     if e2.game.cardsPlayed[p][c - 1]])
                if before - after < 0.3:      # this play keeps the combo intact
                    return cand
            return PASS_IDX                    # every win shatters the combo → hold
    return int(non_pass.min())


def _model_action(model, env: Big2Env) -> int:
    """Greedy policy from model (no MCTS) for opponent pool agents."""
    game = env.game
    player = env.current_player
    static = encode_static(game, player)
    history = encode_history_steps(game)
    valid = env.get_valid_actions()
    probs, _ = model.predict(static, history, valid)
    masked = probs * valid
    total = masked.sum()
    if total == 0:
        return PASS_IDX
    return int(np.argmax(masked))


def _belief_target_for(game, player: int) -> np.ndarray:
    """Oracle belief target relative to `player`: the cards held by the 3 other
    players, in ascending absolute-index order excluding `player`."""
    belief_target = np.zeros(52 * 3, dtype=np.float32)
    others = [p for p in range(1, 5) if p != player]
    for opp_idx, opp_p in enumerate(others):
        for card in game.currentHands[opp_p]:
            c = int(card)
            if 1 <= c <= 52:
                belief_target[opp_idx * 52 + c - 1] = 1.0
    return belief_target


def run_episode(
    model,
    n_simulations: int = 50,
    temperature: float = 1.0,
    learner_player: int = 1,   # kept for API compat; reward returned is this player's
    opponent=None,             # optional dict {player:callable} to override seats
    bc_mode: bool = False,
    value_model=None,          # optional Big2ValueNet (V9): full-info MCTS leaf eval
    bc_action_fn=None,         # heuristic cloned in bc_mode (default: _heuristic_action)
) -> tuple:
    """
    Run ONE true 4-player self-play episode (max^n MCTS).

    All four seats are driven by `model` (running MCTS), and training data is
    collected for EVERY player's turn — not just one learner.  This is what
    makes the 4-dim value head see all perspectives and lets the policy head
    learn from every seat.

    Value targets are 4-dim (one normalized reward per ABSOLUTE player index).
    For self-play (MCTS) episodes they are forward-view TD(λ) returns that
    bootstrap from each state's MCTS root value (lower variance than the raw
    terminal, especially early-game).  For bc episodes they are the pure
    Monte-Carlo terminal outcome.  See the value-target block below.

    `opponent`: optional dict mapping player index (1..4) → callable(env)->action
                to override specific seats with a frozen pool model or heuristic.
                Overridden seats do NOT contribute training data (they are not
                the learner).  Default None = pure self-play, collect all seats.

    `bc_mode`: behavioral cloning — every collected seat uses the heuristic
               action and a one-hot policy target.  Bootstraps the policy.

    Returns:
        data: list of (static, history, valid_mask, policy_target,
                       value_target_vec(4,), belief_target)
        terminal_reward: float — raw terminal reward for `learner_player`.
    """
    env = Big2Env()
    env.reset()
    mcts = None if bc_mode else MCTS(model, n_simulations=n_simulations,
                                     value_model=value_model)
    overrides = opponent if isinstance(opponent, dict) else {}

    episode_steps = []     # collected (data-contributing) steps
    terminal_rewards = None

    while not env.done:
        player = env.current_player
        game = env.game

        if player in overrides:
            # Frozen / heuristic seat — acts but contributes no training data.
            action = overrides[player](env)
        else:
            static = encode_static(game, player)
            history = encode_history_steps(game)
            valid = env.get_valid_actions()

            # Forced-pass shortcut: if PASS is the only legal action there is no
            # decision to make — skip MCTS (saves compute) and DON'T collect a
            # training sample (a pass-only state teaches the policy nothing).
            va = np.flatnonzero(valid)
            if va.size == 0 or (va.size == 1 and int(va[0]) == PASS_IDX):
                rewards, done = env.step(PASS_IDX)
                if done:
                    terminal_rewards = np.asarray(rewards, dtype=np.float64).reshape(4)
                    break
                continue

            if bc_mode:
                action = (bc_action_fn or _heuristic_action)(env)
                visits = np.zeros(env.ACTION_SIZE, dtype=np.float32)
                visits[action] = 1.0
                root_value = None
            else:
                action, visits, root_value = mcts.run(
                    env, temperature=temperature, return_root_value=True)

            total_v = visits.sum()
            valid_sum = valid.sum()
            if total_v > 0:
                policy_target = visits / total_v
            elif valid_sum > 0:
                policy_target = valid / valid_sum
            else:
                policy_target = np.ones_like(valid) / len(valid)

            episode_steps.append({
                "player": player,
                "static": static.astype(np.float32),
                "history": history.astype(np.float32),
                "valid": valid.astype(np.float32),
                "policy_target": policy_target.astype(np.float32),
                "belief_target": _belief_target_for(game, player),
                # MCTS-improved 4-dim value (absolute index) at this state — the
                # TD(λ) bootstrap signal. None for bc steps (pure MC instead).
                "value_pred": (None if root_value is None
                               else np.asarray(root_value, dtype=np.float64).reshape(4)),
            })

        rewards, done = env.step(action)

        if done:
            terminal_rewards = np.asarray(rewards, dtype=np.float64).reshape(4)
            break

    # ── Per-step value targets (absolute player index, 4-dim) ────────────────
    if terminal_rewards is None:
        terminal_rewards = np.zeros(4, dtype=np.float64)
    G = np.tanh(terminal_rewards / REWARD_SCALE).astype(np.float64)  # terminal outcome
    terminal_reward = float(terminal_rewards[learner_player - 1])

    # bc episodes → pure Monte-Carlo terminal (the true outcome of that heuristic
    #   game; bootstrapping a self-play-trained value onto a heuristic trajectory
    #   would be a distribution mismatch).
    # self-play (MCTS) episodes → forward-view TD(λ).  With terminal-only reward
    #   and γ=1 this reduces to the backward recursion
    #       target[i] = (1-λ)·v[i+1] + λ·target[i+1],   target[last] = G
    #   where v[i] = the MCTS root value at collected step i.  λ→1 recovers pure
    #   MC; λ=0.9 cuts early-game target variance — the V7 lesson: fix the noisy
    #   target, do NOT up-weight the value loss.
    M = len(episode_steps)
    value_targets = [G.copy() for _ in range(M)]
    if not bc_mode and M >= 2:
        for i in range(M - 2, -1, -1):
            v_next = episode_steps[i + 1]["value_pred"]
            if v_next is None:           # safety: fall back to MC for this link
                continue
            value_targets[i] = (1.0 - LAMBDA_TD) * v_next + LAMBDA_TD * value_targets[i + 1]

    data = [
        (
            step["static"],
            step["history"],
            step["valid"],
            step["policy_target"],
            value_targets[i].astype(np.float32),
            step["belief_target"],
        )
        for i, step in enumerate(episode_steps)
    ]
    return data, terminal_reward


def make_pool_opponent(pool_model):
    """Wrap a frozen pool model as an opponent callable for run_episode."""
    def _act(env: Big2Env) -> int:
        return _model_action(pool_model, env)
    return _act
