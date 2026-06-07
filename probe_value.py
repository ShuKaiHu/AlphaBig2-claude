"""Value-head health probe: how well does the predicted value (the acting
player's own dimension) correlate with the eventual terminal outcome?

For each state where the learner acts, we record:
  pred  = model.value(state)[learner-1]          (tanh-space, [-1,1])
  truth = tanh(terminal_reward[learner-1] / 13)  (same space as training target)

A healthy value head has high Pearson r between pred and truth, and a
calibration slope near 1. We bucket by game phase (early/mid/late) because
late-game states are far more predictable than the opening.

Usage:
  BIG2_DOMINANCE=1 python probe_value.py CKPT [--games 400]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from engine.model import Big2Net
from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_history_steps
from engine.evaluator import _heuristic_action

REWARD_SCALE = 13.0


def _greedy(model, env):
    g, p = env.game, env.current_player
    probs, _ = model.predict(encode_static(g, p), encode_history_steps(g),
                             env.get_valid_actions())
    m = probs * env.get_valid_actions()
    return int(np.argmax(m)) if m.sum() > 0 else PASS_IDX


def probe(ckpt, n_games=400, learner=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    model = Big2Net()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd); model.eval()

    rows = []  # (pred, frac_played, idx_in_game)
    env = Big2Env()
    for _ in range(n_games):
        env.reset()
        states = []  # (pred_value_self, n_cards_remaining_self)
        while not env.done:
            p = env.current_player
            if p == learner:
                g = env.game
                st = encode_static(g, learner); hs = encode_history_steps(g)
                with torch.no_grad():
                    _, val_t, _ = model.forward(
                        torch.FloatTensor(st).unsqueeze(0),
                        torch.FloatTensor(hs).unsqueeze(0),
                        torch.FloatTensor(env.get_valid_actions()).unsqueeze(0))
                pred = float(val_t.squeeze(0).numpy()[learner - 1])
                states.append((pred, int(g.currentHands[learner].size)))
                a = _greedy(model, env)
            else:
                a = _heuristic_action(env)
            rewards, done = env.step(a)
            if done:
                truth = float(np.tanh(rewards[learner - 1] / REWARD_SCALE))
                n = len(states)
                for i, (pred, rem) in enumerate(states):
                    rows.append((pred, truth, i / max(1, n - 1)))
                break
    arr = np.array(rows)  # (N,3): pred, truth, phase
    pred, truth, phase = arr[:, 0], arr[:, 1], arr[:, 2]
    def stats(mask, label):
        pr, tr = pred[mask], truth[mask]
        if len(pr) < 5 or pr.std() < 1e-6:
            print(f"  {label:6s}: n={len(pr):5d}  r=  n/a (退化)")
            return
        r = np.corrcoef(pr, tr)[0, 1]
        # calibration slope (tr = a + b*pr)
        b = np.polyfit(pr, tr, 1)[0]
        print(f"  {label:6s}: n={len(pr):5d}  r={r:+.3f}  斜率={b:+.2f}  "
              f"pred[{pr.mean():+.2f}±{pr.std():.2f}] truth[{tr.mean():+.2f}±{tr.std():.2f}]")
    print(f"\n{ckpt}")
    print(f"  總狀態數={len(arr)}  整體:")
    stats(np.ones(len(arr), bool), "ALL")
    stats(phase < 0.34, "早期")
    stats((phase >= 0.34) & (phase < 0.67), "中期")
    stats(phase >= 0.67, "晚期")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=400)
    a = ap.parse_args()
    probe(a.checkpoint, n_games=a.games)
