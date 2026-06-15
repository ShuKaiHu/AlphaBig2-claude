"""Offline BEHAVIORAL combo-break rate: play MCTS games vs 3 heuristics and
measure how often the learner's own single/pair plays dismantle a 5-card combo
it held (n5 potential drops), and how often that was avoidable (a same-size shed
existed that wouldn't have dropped n5). Auto-uses the checkpoint's full-info
value net + combo features. Compares V9b (combo) vs V9a directly in real play.

Usage: python combo_break_play.py CKPT [--games 40] [--sims 80]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import enumerateOptions as eo
from engine.env import Big2Env, PASS_IDX
from engine.evaluator import _heuristic_action
from engine.mcts import MCTS
from eval_reward import load_model
from combo_break import _n5, _singles_pairs


def measure(ckpt, n_games=40, sims=80, learner=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    model, value_model = load_model(ckpt)   # sets combo dim if present
    mcts = MCTS(model, n_simulations=sims, value_model=value_model)
    plays = broke = avoidable = 0
    env = Big2Env()
    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p == learner:
                va = np.flatnonzero(env.get_valid_actions())
                if va.size == 0 or (va.size == 1 and int(va[0]) == PASS_IDX):
                    a = PASS_IDX
                else:
                    a, _ = mcts.run(env, temperature=0.0)
                    if a != PASS_IDX:
                        cards, n = eo.getOptionNC(a)
                        if n < 5:
                            hand = [int(c) for c in env.game.currentHands[learner]]
                            n5b = _n5(hand)
                            if n5b > 0:
                                plays += 1
                                played = set(int(c) for c in cards)
                                if _n5([c for c in hand if c not in played]) < n5b:
                                    broke += 1
                                    sing, pair = _singles_pairs(hand)
                                    alts = sing if n == 1 else pair
                                    if any(_n5([c for c in hand if c not in set(al)]) == n5b
                                           for al in alts if set(al) != played):
                                        avoidable += 1
            else:
                a = _heuristic_action(env)
            _, done = env.step(a)
            if done:
                break
    print(f"{ckpt}")
    print(f"  full-info value: {'YES' if value_model is not None else 'no'}  games={n_games} sims={sims}")
    print(f"  出單張/對子且手上有五張組合 = {plays}")
    print(f"  拆掉組合 = {broke} ({broke/max(1,plays)*100:.0f}%)")
    print(f"  可避免的拆 = {avoidable} ({avoidable/max(1,plays)*100:.0f}%)")
    return broke / max(1, plays), avoidable / max(1, plays)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=80)
    a = ap.parse_args()
    measure(a.checkpoint, n_games=a.games, sims=a.sims)
