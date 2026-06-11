"""Follow-waste rate: when following (control=0) and choosing to play, how often
does the agent play HIGHER than necessary (a smaller same-size legal beat
existed)? This is the V9 success metric — the V8 model wasted big cards on 40%
of online follows (A to beat a 4, a 2 to beat a 5).

Offline protocol: learner = MCTS (auto full-info value when the checkpoint has
one) vs 3 heuristics in a perfect-info env. For each learner follow-play, a
"waste" = some legal same-size action has a strictly smaller max card_id (the
env mask only contains beating actions when control=0, so all candidates beat
the table). The engine has no online "next player holds 1 card" rule, so no
exclusions are needed.

Usage: python follow_waste.py CKPT [--games 100] [--sims 80]
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


def _action_max_card(action: int) -> int:
    cards, _n = eo.getOptionNC(action)
    return int(max(cards))


def measure(ckpt, n_games=100, sims=80, learner=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    model, value_model = load_model(ckpt)
    mcts = MCTS(model, n_simulations=sims, value_model=value_model)

    follows = plays = wastes = passes = 0
    waste_gaps = []  # chosen max-card minus minimal available max-card (in card_id)
    env = Big2Env()
    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p == learner:
                va = np.flatnonzero(env.get_valid_actions())
                if va.size == 0 or (va.size == 1 and int(va[0]) == PASS_IDX):
                    # forced skip/pass — the env can return an EMPTY mask here
                    # (MCTS would degenerate to argmax(zeros)=action 0); not a
                    # decision, don't search and don't count.
                    a = PASS_IDX
                else:
                    is_follow = env.game.control == 0
                    a, _ = mcts.run(env, temperature=0.0)
                    if is_follow:
                        follows += 1
                        if a == PASS_IDX:
                            passes += 1
                        else:
                            plays += 1
                            n_cards = eo.getOptionNC(a)[1]
                            same_size = [x for x in va if x != PASS_IDX
                                         and eo.getOptionNC(x)[1] == n_cards]
                            chosen_max = _action_max_card(a)
                            min_max = min(_action_max_card(x) for x in same_size)
                            if chosen_max > min_max:
                                wastes += 1
                                waste_gaps.append(chosen_max - min_max)
            else:
                a = _heuristic_action(env)
            _, done = env.step(a)
            if done:
                break
    rate = wastes / plays if plays else float("nan")
    print(f"{ckpt}")
    print(f"  full-info value: {'YES' if value_model is not None else 'no'}  sims={sims}  games={n_games}")
    print(f"  跟牌決策={follows}  其中出牌={plays} pass={passes}")
    print(f"  浪費(有更小同型可壓卻出更高)={wastes}/{plays} = {rate:.1%}")
    if waste_gaps:
        print(f"  浪費幅度(card_id 差): 平均={np.mean(waste_gaps):.1f}  最大={max(waste_gaps)}")
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=80)
    a = ap.parse_args()
    measure(a.checkpoint, n_games=a.games, sims=a.sims)
