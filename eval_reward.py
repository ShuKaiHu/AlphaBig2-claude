"""Reward-focused evaluation for AlphaBig2 checkpoints.

The goal is to MAXIMIZE REWARD (avg_score), not win_rate. A model that
consistently places 2nd with few cards left has a good avg_score even with a
modest win_rate — that is good Big 2 play.

The old in-training eval used only 200 games, giving ~±0.7 std on avg_score,
which is large enough to make noise look like "best" checkpoints. This tool
runs many games and reports avg_score with a standard error so real progress
can be told apart from noise.

Usage:
    python eval_reward.py [checkpoint.pt] [--games 2000] [--mcts SIMS]

Defaults to engine/checkpoints/best.pt, 2000 games, greedy policy (no MCTS).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from engine.model import Big2Net
from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_history_steps
from engine.evaluator import _heuristic_action


def _greedy_action(model, env):
    g, p = env.game, env.current_player
    probs, _ = model.predict(encode_static(g, p), encode_history_steps(g), env.get_valid_actions())
    masked = probs * env.get_valid_actions()
    return int(np.argmax(masked)) if masked.sum() > 0 else PASS_IDX


def _mcts_action(mcts, env):
    action, _ = mcts.run(env, temperature=0.0)  # argmax on visit counts
    return action


def evaluate(model, n_games=2000, learner=1, sims=0, seed=0, value_model=None):
    """Model as `learner` vs 3 heuristics. Returns dict of reward stats.
    value_model: optional Big2ValueNet (V9) — full-info MCTS leaf evaluation.
    This eval is perfect-info (true env), so no determinization is involved."""
    np.random.seed(seed)
    mcts = None
    if sims > 0:
        from engine.mcts import MCTS
        mcts = MCTS(model, n_simulations=sims, value_model=value_model)
    scores, placements = [], []
    env = Big2Env()
    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p == learner:
                a = _mcts_action(mcts, env) if mcts else _greedy_action(model, env)
            else:
                a = _heuristic_action(env)
            rewards, done = env.step(a)
            if done:
                s = float(rewards[learner - 1])
                scores.append(s)
                # placement: 1 = fewest cards left (winner empties hand)
                # approximate via reward rank is not exact; use cards-left rank
                placements.append(_placement(env, learner))
                break
    scores = np.array(scores)
    pl = np.array(placements)
    return {
        "n": len(scores),
        "avg_score": float(scores.mean()),
        "stderr": float(scores.std() / np.sqrt(len(scores))),
        "win_rate": float((scores > 0).mean()),
        "first_rate": float((pl == 1).mean()),
        "placement_dist": {k: int((pl == k).sum()) for k in (1, 2, 3, 4)},
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def _placement(env, learner):
    """1..4 placement by cards left (fewest = 1)."""
    counts = [env.game.currentHands[p].size for p in range(1, 5)]
    mine = counts[learner - 1]
    return 1 + sum(1 for c in counts if c < mine)


def load_model(path):
    """Returns (model, value_model). value_model is a Big2ValueNet when the
    checkpoint carries a V9 full-info value net ("value_state"), else None."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = Big2Net()
    cur = model.state_dict()
    # Tolerate shape mismatches (e.g. evaluating an old 1-dim-value checkpoint):
    # greedy eval only uses the policy head, so a re-init value head is fine.
    compatible = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    model.eval()
    value_model = None
    if isinstance(ckpt, dict) and "value_state" in ckpt:
        from engine.model import Big2ValueNet
        value_model = Big2ValueNet()
        value_model.load_state_dict(ckpt["value_state"])
        value_model.eval()
    return model, value_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?",
                    default=os.path.join(os.path.dirname(__file__), "engine/checkpoints/best.pt"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--mcts", type=int, default=0, help="MCTS sims (0=greedy policy)")
    ap.add_argument("--seeds", type=int, default=1, help="repeat with N seeds for variance")
    args = ap.parse_args()

    model, value_model = load_model(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {'MCTS '+str(args.mcts)+' sims' if args.mcts else 'greedy policy'}"
          f"{' + full-info value (V9)' if value_model is not None and args.mcts else ''}, "
          f"{args.games} games × {args.seeds} seed(s)\n")

    avgs = []
    for seed in range(args.seeds):
        r = evaluate(model, n_games=args.games, sims=args.mcts, seed=seed,
                     value_model=value_model)
        avgs.append(r["avg_score"])
        print(f"seed {seed}: avg_score={r['avg_score']:+.3f} ± {r['stderr']:.3f}  "
              f"win={r['win_rate']:.1%}  1st={r['first_rate']:.1%}  "
              f"placements={r['placement_dist']}  range=[{r['score_min']:.0f},{r['score_max']:.0f}]")

    if args.seeds > 1:
        avgs = np.array(avgs)
        print(f"\nacross {args.seeds} seeds: avg_score={avgs.mean():+.3f} ± {avgs.std():.3f}")

    # Reference baselines (4-player Big2):
    #   random/symmetric play → avg_score ≈ 0
    #   beating 3 weak heuristics → avg_score > 0 (the more positive, the better)
    print("\nbaseline: avg_score≈0 means even with the heuristics; >0 means winning reward.")
