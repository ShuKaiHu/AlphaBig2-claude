import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import enumerateOptions
from big2Game import big2Game

ACTION_SIZE = enumerateOptions.passInd + 1  # 14739
PASS_IDX = enumerateOptions.passInd


class Big2Env:
    """
    Clean Gym-like wrapper around big2Game.
    Unlike the original big2Game.step(), does NOT auto-reset on game over.
    Caller is responsible for calling reset() between episodes.
    """

    ACTION_SIZE = ACTION_SIZE

    def __init__(self):
        self._game = big2Game()
        self._done = False

    def reset(self, **kwargs):
        """No-arg call is byte-identical to legacy behavior. ADDITIVE (M3
        Phase 1): kwargs pass through to big2Game.reset (hands= for full-deal
        injection, starter=, must_play_club3=)."""
        self._game.reset(**kwargs)
        self._done = False

    def load_game(self, game):
        """ADDITIVE (M3 Phase 1): adopt an externally constructed big2Game
        (e.g. big2Game.restore_state(...) for MID-GAME injection) as the live
        episode. Does not touch reset()/step() default paths."""
        self._game = game
        self._done = False

    def step(self, action: int):
        """
        Apply action for the current player.
        Returns (rewards, done):
          - rewards: np.ndarray shape (4,) on terminal step, else None
          - done: bool
        """
        assert not self._done, "Game is over; call reset() first."
        opt, nC = enumerateOptions.getOptionNC(action)
        self._game.updateGame(opt, nC)
        if self._game.gameOver:
            self._done = True
            return self._game.rewards.copy(), True
        return None, False

    def get_valid_actions(self) -> np.ndarray:
        """Binary mask, shape (ACTION_SIZE,). 1 = valid action."""
        return self._game.returnAvailableActions()

    @property
    def current_player(self) -> int:
        """1-indexed current player (1–4)."""
        return self._game.playersGo

    @property
    def done(self) -> bool:
        return self._done

    @property
    def game(self):
        return self._game

    def clone(self) -> "Big2Env":
        """Deep-copy for MCTS tree search."""
        new = Big2Env.__new__(Big2Env)
        new._game = self._game.clone()
        new._done = self._done
        return new

    # ── Convenience action policies ──────────────────────────────────────────

    def heuristic_action(self) -> int:
        """Play the lowest-index valid non-pass action, or pass if forced."""
        valid = np.flatnonzero(self.get_valid_actions())
        non_pass = valid[valid != PASS_IDX]
        return int(non_pass[0]) if len(non_pass) > 0 else PASS_IDX

    def random_action(self) -> int:
        """Uniform random sample from valid actions."""
        valid = np.flatnonzero(self.get_valid_actions())
        return int(np.random.choice(valid)) if len(valid) > 0 else PASS_IDX
